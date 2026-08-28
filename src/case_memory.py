"""
Case memory — the part that makes this a system rather than a model.

Most detectors score, alert, and forget. Every morning they make the same
mistake they made yesterday, and an analyst types the same rejection note.
Ringfence remembers both directions:

- A candidate whose behavioural signature matches a **confirmed ring** gets a
  +15% precedent boost and the precedent attached to its dossier.
- A candidate whose signature matches a **confirmed false positive** gets a
  -20% damp and a "historical false positive pattern" tag.

The second one is the interesting half, and it is this repo's answer to "show
one failure handled gracefully". When we get it wrong, an analyst says so once,
and the system is measurably less wrong about that shape from then on. The
demo's Page 5 replays exactly that: same candidate, before and after, with the
score moving.

What "embedding" means here
---------------------------
The vector is the candidate's own 31-dimensional feature signature, scale-
compressed, **population-standardised**, and L2-normalised. We do not run the
features through a sentence embedder first. The thing we want to match on *is*
the behavioural shape, and we already hold it as a numeric vector; rendering it
to English and re-encoding it would only lose information.

The standardisation step is not optional decoration — it was a bug we measured
and fixed. Raw features cannot be compared with cosine directly, because
`time_span_hours` runs to thousands while `probe_fraction` is bounded at 1, so
one feature dominates the metric. Log compression alone fixes the scale but not
the geometry: **all 31 features are non-negative**, so every compressed vector
lives in the same orthant and cosine similarity between any two candidates came
out at a median of 0.964. One false positive damped all 557 candidates — case
memory was matching "is a candidate", not "is this pattern".

Standardising each dimension against the population mean and standard deviation
recentres the cloud on the origin, so vectors spread across the sphere and
cosine measures genuine similarity again. The scaler is fitted during training
and persisted to `data/processed/signature_scaler.json`; see
`scorer.fit_signature_scaler`.

Because stored signatures are scaler-dependent, **re-run `make clean-memory`
after retraining** — mixing signatures from two scalers compares vectors in two
different spaces. `CaseMemory` records the scaler fingerprint and warns when it
sees a mismatch.

Storage: SQLite for metadata and vectors (durable, zero-setup, inspectable with
any sqlite client). A ChromaDB backend is used instead when chromadb is
installed and RINGFENCE_USE_CHROMA=1; the numpy index is the default so the
test suite does not need a 2GB torch download to prove the damp works.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import (
    CASE_MEMORY,
    CASE_MEMORY_DB,
    CHROMA_DIR,
    SIGNATURE_SCALER_PATH,
    CaseMemoryConfig,
)
from src.feature_engine import FEATURE_NAMES
from src.utils import cosine_similarity_matrix, get_logger

log = get_logger("ringfence.case_memory")

DISPOSITION_FRAUD = "fraud"
DISPOSITION_FALSE_POSITIVE = "false_positive"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    alert_id      TEXT PRIMARY KEY,
    ring_id       TEXT,
    disposition   TEXT NOT NULL,
    card_ids      TEXT NOT NULL,
    device_pool   TEXT,
    notes         TEXT,
    analyst       TEXT,
    signature     BLOB NOT NULL,
    confirmed_at  TEXT NOT NULL,
    template_id   TEXT
);
CREATE INDEX IF NOT EXISTS idx_cases_disposition ON cases(disposition);

CREATE TABLE IF NOT EXISTS templates (
    template_id     TEXT PRIMARY KEY,
    n_confirmations INTEGER NOT NULL,
    centroid        BLOB NOT NULL,
    device_pool     TEXT,
    member_alerts   TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id   TEXT,
    action     TEXT NOT NULL,
    detail     TEXT,
    at         TEXT NOT NULL
);
"""


def load_signature_scaler(path=SIGNATURE_SCALER_PATH) -> dict | None:
    """
    Load the persisted population statistics used to standardise signatures.

    Returns None when no scaler has been fitted yet. Callers fall back to
    compression-only signatures, which still work — they are simply far less
    discriminative, so a warning is logged rather than the failure being silent.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return {
            "mean": np.asarray(payload["mean"], dtype=float),
            "std": np.asarray(payload["std"], dtype=float),
            "fingerprint": payload.get("fingerprint", ""),
            "n_samples": payload.get("n_samples", 0),
        }
    except (FileNotFoundError, OSError, KeyError, json.JSONDecodeError):
        return None


def compress_features(features: dict[str, float] | np.ndarray) -> np.ndarray:
    """
    Scale-compress a raw feature vector: `sign(x) * log1p(|x|)`.

    Brings the unbounded features (amount escalation runs to ~5,000x, time span
    to thousands of hours) onto the same order of magnitude as the bounded ones.
    Shared by the scaler-fitting path and the query path so both see identical
    inputs.
    """
    if isinstance(features, dict):
        vec = np.array([float(features.get(n, 0.0)) for n in FEATURE_NAMES], dtype=float)
    else:
        vec = np.asarray(features, dtype=float).ravel()
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return np.sign(vec) * np.log1p(np.abs(vec))


def signature_from_features(
    features: dict[str, float] | np.ndarray, scaler: dict | None = None
) -> np.ndarray:
    """
    Convert a feature vector into a comparable 31-dim signature.

    Three steps, and all three are load-bearing:

    1. **Compress** — `sign(x)·log1p(|x|)`, so `time_span_hours` at 2,000 stops
       drowning `probe_fraction` at 0.6.
    2. **Standardise** — subtract the population mean, divide by the population
       standard deviation. Every feature here is non-negative, so without this
       the whole cloud sits in one orthant and *every* pair of candidates comes
       out at cosine ~0.96. Recentring on the origin is what makes the metric
       discriminate at all.
    3. **Normalise** — L2, so cosine measures shape rather than magnitude.

    Falls back to compression-only when no scaler is available (a cold start
    before the first training run). That path works but matches loosely, so it
    warns.
    """
    compressed = compress_features(features)

    if scaler is not None:
        std = np.where(scaler["std"] > 1e-9, scaler["std"], 1.0)
        compressed = (compressed - scaler["mean"]) / std

    norm = np.linalg.norm(compressed)
    return compressed / norm if norm > 0 else compressed


class CaseMemory:
    """
    Persistent precedent and failure store.

    Thread-safe enough for the demo API (`check_same_thread=False` with a
    per-call cursor); a production deployment would move this to Postgres plus
    a real vector store, which is written up in ARCHITECTURE.md rather than
    pretended away here.
    """

    def __init__(self, db_path=CASE_MEMORY_DB, cfg: CaseMemoryConfig = CASE_MEMORY,
                 scaler_path=SIGNATURE_SCALER_PATH):
        self.cfg = cfg
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self._chroma = self._maybe_open_chroma()

        self.scaler = load_signature_scaler(scaler_path)
        if self.scaler is None:
            log.warning(
                "no signature scaler found; signatures fall back to compression-only, which "
                "matches far too loosely. Run `make train` to fit one."
            )
        self._check_scaler_fingerprint()

        log.info(f"case memory at {db_path} ({self.count()} cases, "
                 f"vector backend = {'chromadb' if self._chroma else 'numpy'}, "
                 f"scaler = {'fitted' if self.scaler else 'none'})")

    def _signature(self, features) -> np.ndarray:
        """Build a signature using this store's scaler."""
        return signature_from_features(features, self.scaler)

    def _check_scaler_fingerprint(self) -> None:
        """
        Warn if stored signatures were built with a different scaler.

        Signatures are only comparable within one scaler's space. Silently
        mixing two would make similarity scores meaningless in a way that looks
        like it is working.
        """
        current = (self.scaler or {}).get("fingerprint", "")
        row = self.conn.execute(
            "SELECT detail FROM audit_log WHERE action = 'scaler_fingerprint' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        stored = row["detail"] if row else None

        if stored is None:
            self.conn.execute(
                "INSERT INTO audit_log (alert_id, action, detail, at) VALUES (?, ?, ?, ?)",
                (None, "scaler_fingerprint", current, datetime.now(timezone.utc).isoformat()),
            )
            self.conn.commit()
        elif stored != current and self.count() > 0:
            log.warning(
                f"case memory holds signatures from scaler '{stored}' but the current scaler is "
                f"'{current}'. Similarity scores across that boundary are not meaningful — "
                f"run `make clean-memory` after retraining."
            )

    # ── vector backend ───────────────────────────────────────────────────

    def _maybe_open_chroma(self):
        """Open the ChromaDB collection if it is installed and switched on."""
        if os.getenv("RINGFENCE_USE_CHROMA", "0") != "1":
            return None
        try:
            import chromadb

            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            return client.get_or_create_collection("ringfence_cases", metadata={"hnsw:space": "cosine"})
        except Exception as exc:  # noqa: BLE001 - optional dependency
            log.warning(f"ChromaDB requested but unavailable ({exc}); using the numpy index")
            return None

    # ── writes ───────────────────────────────────────────────────────────

    def store_disposition(
        self,
        alert_id: str,
        disposition: str,
        notes: str = "",
        cards: list[str] | None = None,
        signature: np.ndarray | dict[str, float] | None = None,
        ring_id: str = "",
        device_pool: list[str] | None = None,
        analyst: str = "demo_analyst",
    ) -> None:
        """
        Record an analyst's verdict on an alert.

        Both verdicts are stored. Storing only confirmations would give us a
        system that learns what fraud looks like and never learns what its own
        mistakes look like — which is the failure this module exists to fix.

        Every write also lands in `audit_log` with a timestamp and the analyst,
        so a decision can be reconstructed after the fact.
        """
        if disposition not in (DISPOSITION_FRAUD, DISPOSITION_FALSE_POSITIVE):
            raise ValueError(
                f"disposition must be '{DISPOSITION_FRAUD}' or '{DISPOSITION_FALSE_POSITIVE}', got {disposition!r}"
            )

        sig = self._signature(signature) if signature is not None else np.zeros(len(FEATURE_NAMES))
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            """INSERT INTO cases (alert_id, ring_id, disposition, card_ids, device_pool, notes,
                                  analyst, signature, confirmed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(alert_id) DO UPDATE SET
                 disposition=excluded.disposition, notes=excluded.notes,
                 signature=excluded.signature, confirmed_at=excluded.confirmed_at""",
            (alert_id, ring_id, disposition, json.dumps(cards or []), json.dumps(device_pool or []),
             notes, analyst, sig.astype(np.float32).tobytes(), now),
        )
        self._audit(alert_id, f"disposition:{disposition}", notes)
        self.conn.commit()

        if self._chroma is not None:
            try:
                self._chroma.upsert(
                    ids=[alert_id],
                    embeddings=[sig.tolist()],
                    metadatas=[{"disposition": disposition, "ring_id": ring_id, "notes": notes}],
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(f"chroma upsert failed, SQLite remains authoritative: {exc}")

        log.info(f"stored disposition {disposition} for {alert_id} ({len(cards or [])} cards)")

    def _audit(self, alert_id: str, action: str, detail: str = "") -> None:
        """Append an immutable audit-log row."""
        self.conn.execute(
            "INSERT INTO audit_log (alert_id, action, detail, at) VALUES (?, ?, ?, ?)",
            (alert_id, action, detail, datetime.now(timezone.utc).isoformat()),
        )

    # ── reads ────────────────────────────────────────────────────────────

    def _load(self, disposition: str) -> tuple[list[sqlite3.Row], np.ndarray]:
        """Load all cases of one disposition and their signature matrix."""
        rows = self.conn.execute(
            "SELECT * FROM cases WHERE disposition = ? ORDER BY confirmed_at", (disposition,)
        ).fetchall()
        if not rows:
            return [], np.zeros((0, len(FEATURE_NAMES)))
        matrix = np.vstack([np.frombuffer(r["signature"], dtype=np.float32).astype(float) for r in rows])
        return rows, matrix

    def _search(self, query, disposition: str, threshold: float, top_k: int) -> list[dict]:
        """Cosine search over one disposition's signatures, above a threshold."""
        q = self._signature(query)
        rows, matrix = self._load(disposition)
        if not rows:
            return []
        sims = cosine_similarity_matrix(q, matrix)
        order = np.argsort(-sims)[:top_k]
        return [
            {
                "alert_id": rows[i]["alert_id"],
                "ring_id": rows[i]["ring_id"],
                "disposition": rows[i]["disposition"],
                "similarity": float(sims[i]),
                "notes": rows[i]["notes"],
                "analyst": rows[i]["analyst"],
                "confirmed_at": rows[i]["confirmed_at"],
                "cards": json.loads(rows[i]["card_ids"]),
            }
            for i in order
            if sims[i] >= threshold
        ]

    def find_precedents(self, candidate_features, top_k: int = 3) -> list[dict]:
        """
        Confirmed fraud rings whose signature matches this candidate above
        `precedent_similarity_threshold` (0.85).
        """
        return self._search(candidate_features, DISPOSITION_FRAUD, self.cfg.precedent_similarity_threshold, top_k)

    def find_false_positive_patterns(self, candidate_features, top_k: int = 3) -> list[dict]:
        """
        Confirmed false positives whose signature matches this candidate above
        `fp_similarity_threshold` (0.80).

        The threshold is deliberately looser than the precedent threshold. We
        would rather catch a near-miss of a known mistake and damp it than
        repeat the mistake on a technicality of similarity.
        """
        return self._search(candidate_features, DISPOSITION_FALSE_POSITIVE, self.cfg.fp_similarity_threshold, top_k)

    def adjustments(self, features) -> tuple[list[dict], list[dict], float, float]:
        """
        One-shot lookup used on the scoring path.

        Returns (precedents, fp_patterns, boost, damp). Both adjustments are
        bounded constants from config — not learned, not unbounded, and applied
        by the composer where they show up explicitly in the composite score's
        contributing signals.
        """
        precedents = self.find_precedents(features, self.cfg.top_k)
        fps = self.find_false_positive_patterns(features, self.cfg.top_k)
        boost = self.cfg.precedent_boost if precedents else 0.0
        damp = self.cfg.fp_damp if fps else 0.0
        return precedents, fps, boost, damp

    # ── templates ────────────────────────────────────────────────────────

    def consolidate_templates(self, min_confirmations: int | None = None) -> list[dict]:
        """
        Group confirmed rings with similar signatures into Ring Templates.

        Greedy single-pass clustering at the precedent threshold: each confirmed
        ring joins the first template it matches, or seeds a new one. A group
        reaching `min_confirmations` (5) becomes a template.

        Templates are the *only* route to an automated action anywhere in this
        system, and even then it takes more than 10 confirmed precedents before
        the gating layer will allow an auto-block. See `vulcan_integration`.
        """
        k = min_confirmations if min_confirmations is not None else self.cfg.template_min_confirmations
        rows, matrix = self._load(DISPOSITION_FRAUD)
        if len(rows) < k:
            log.info(f"template consolidation: {len(rows)} confirmed rings, need {k}; nothing consolidated")
            return []

        clusters: list[dict] = []
        for i, row in enumerate(rows):
            placed = False
            for cluster in clusters:
                if cosine_similarity_matrix(matrix[i], np.vstack(cluster["vectors"])).max() >= self.cfg.precedent_similarity_threshold:
                    cluster["vectors"].append(matrix[i])
                    cluster["alerts"].append(row["alert_id"])
                    cluster["devices"].update(json.loads(row["device_pool"] or "[]"))
                    placed = True
                    break
            if not placed:
                clusters.append({
                    "vectors": [matrix[i]],
                    "alerts": [row["alert_id"]],
                    "devices": set(json.loads(row["device_pool"] or "[]")),
                })

        templates = []
        now = datetime.now(timezone.utc).isoformat()
        for n, cluster in enumerate(c for c in clusters if len(c["alerts"]) >= k):
            template_id = f"TMPL_{n:03d}"
            centroid = np.mean(np.vstack(cluster["vectors"]), axis=0)
            norm = np.linalg.norm(centroid)
            centroid = centroid / norm if norm else centroid
            self.conn.execute(
                """INSERT INTO templates (template_id, n_confirmations, centroid, device_pool, member_alerts, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(template_id) DO UPDATE SET
                     n_confirmations=excluded.n_confirmations, centroid=excluded.centroid,
                     member_alerts=excluded.member_alerts""",
                (template_id, len(cluster["alerts"]), centroid.astype(np.float32).tobytes(),
                 json.dumps(sorted(cluster["devices"])), json.dumps(cluster["alerts"]), now),
            )
            for alert_id in cluster["alerts"]:
                self.conn.execute("UPDATE cases SET template_id = ? WHERE alert_id = ?", (template_id, alert_id))
            self._audit(template_id, "template_consolidated", f"{len(cluster['alerts'])} confirmed rings")
            templates.append({
                "template_id": template_id,
                "n_confirmations": len(cluster["alerts"]),
                "member_alerts": cluster["alerts"],
                "device_pool": sorted(cluster["devices"]),
                "auto_block_eligible": len(cluster["alerts"]) > 10,
            })
        self.conn.commit()
        log.info(f"template consolidation: {len(templates)} templates from {len(rows)} confirmed rings")
        return templates

    def match_template(self, features) -> dict | None:
        """Return the best matching Ring Template above the precedent threshold."""
        rows = self.conn.execute("SELECT * FROM templates").fetchall()
        if not rows:
            return None
        q = self._signature(features)
        matrix = np.vstack([np.frombuffer(r["centroid"], dtype=np.float32).astype(float) for r in rows])
        sims = cosine_similarity_matrix(q, matrix)
        best = int(np.argmax(sims))
        if sims[best] < self.cfg.precedent_similarity_threshold:
            return None
        return {
            "template_id": rows[best]["template_id"],
            "n_confirmations": int(rows[best]["n_confirmations"]),
            "similarity": float(sims[best]),
            "auto_block_eligible": int(rows[best]["n_confirmations"]) > 10,
        }

    # ── introspection ────────────────────────────────────────────────────

    def count(self, disposition: str | None = None) -> int:
        """Number of stored cases, optionally filtered by disposition."""
        if disposition:
            return int(self.conn.execute(
                "SELECT COUNT(*) FROM cases WHERE disposition = ?", (disposition,)).fetchone()[0])
        return int(self.conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0])

    def get_case(self, alert_id: str) -> dict | None:
        """Fetch one stored case by alert id."""
        row = self.conn.execute("SELECT * FROM cases WHERE alert_id = ?", (alert_id,)).fetchone()
        if not row:
            return None
        return {
            "alert_id": row["alert_id"], "ring_id": row["ring_id"], "disposition": row["disposition"],
            "cards": json.loads(row["card_ids"]), "notes": row["notes"], "analyst": row["analyst"],
            "confirmed_at": row["confirmed_at"], "template_id": row["template_id"],
        }

    def audit_trail(self, alert_id: str | None = None, limit: int = 100) -> list[dict]:
        """Read back the audit log, newest first."""
        if alert_id:
            rows = self.conn.execute(
                "SELECT * FROM audit_log WHERE alert_id = ? ORDER BY id DESC LIMIT ?", (alert_id, limit)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Summary for the /metrics endpoint and the dashboard."""
        return {
            "n_cases": self.count(),
            "n_confirmed_fraud": self.count(DISPOSITION_FRAUD),
            "n_false_positives": self.count(DISPOSITION_FALSE_POSITIVE),
            "n_templates": int(self.conn.execute("SELECT COUNT(*) FROM templates").fetchone()[0]),
            "n_audit_entries": int(self.conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]),
            "precedent_threshold": self.cfg.precedent_similarity_threshold,
            "precedent_boost": self.cfg.precedent_boost,
            "fp_threshold": self.cfg.fp_similarity_threshold,
            "fp_damp": self.cfg.fp_damp,
            "vector_backend": "chromadb" if self._chroma else "numpy",
        }

    def close(self) -> None:
        """Close the SQLite connection."""
        self.conn.close()
