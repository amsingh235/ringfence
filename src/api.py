"""
FastAPI service.

Design notes worth reading before the code:

**Nothing here can move money.** Every endpoint either observes or recommends.
There is no capability to test a card, probe a merchant, block a payment, or
generate fraudulent traffic — by construction, not by policy.

**The scoring path is synchronous, the narrative is not.** `/detect/candidate`
computes features, scores, consults case memory, composes with Vulcan and
builds the fully-cited dossier inline, because all of that is deterministic and
fits in the latency budget. The LLM narrative and the durable write are
background tasks: a slow model must not be able to slow down a fraud decision.

**Degraded mode is a first-class state, not an exception.** If the model
artifact is missing, the graph is unavailable, or a card is unknown, the API
does not 500. It returns a scored response with `degraded: true` and forces the
configured safe default — REVIEW. A detector that cannot see escalates to a
human; it never quietly approves. `/health` reports which components are live.

Endpoints
---------
POST /ingest/transaction              accept a transaction, return ack
POST /detect/candidate                score a card set, return composite + dossier
GET  /alerts                          recent alerts, paginated and filterable
GET  /alerts/{id}/dossier             full dossier with evidence
POST /alerts/{id}/disposition         analyst verdict -> case memory
GET  /alerts/{id}/failure-recovery    how a false positive changed future scoring
GET  /health                          component status and model version
GET  /metrics                         Prometheus-style metrics
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from src.config import (
    ALERTS_DB,
    CASE_MEMORY,
    GATING,
    MERCHANTS_CSV,
    MODEL_VERSION,
    ensure_dirs,
)
from src.dossier_builder import Dossier, build_dossier, dossier_corpus_metrics, generate_llm_summary
from src.feature_engine import compute_candidate_meta, compute_features
from src.utils import LATENCY_RECORDER, clear_request_id, get_logger, new_alert_id, set_request_id
from src.vulcan_integration import VulcanComposer

log = get_logger("ringfence.api")

_ALERT_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id        TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    cards           TEXT NOT NULL,
    n_cards         INTEGER NOT NULL,
    ringfence_score REAL NOT NULL,
    composite_score REAL NOT NULL,
    recommendation  TEXT NOT NULL,
    disposition     TEXT,
    degraded        INTEGER NOT NULL DEFAULT 0,
    dossier         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC);
"""


# ──────────────────────────────────────────────────────────────────────────
# Request / response models
# ──────────────────────────────────────────────────────────────────────────


class TransactionIn(BaseModel):
    """One inbound transaction. Mirrors the generated schema."""

    transaction_id: str
    card_id: str
    merchant_id: str
    amount: float = Field(ge=0)
    timestamp: datetime
    device_fingerprint: str
    ip_address: str
    email_hash: str
    vulcan_score: float = Field(default=0.0, ge=0.0, le=1.0)


class CandidateIn(BaseModel):
    """A candidate card set submitted for ring scoring."""

    cards: list[str] = Field(min_length=1, max_length=500)
    vulcan_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Per-transaction Vulcan score; derived from the cluster's own activity if omitted",
    )
    include_dossier: bool = True
    use_llm: bool = False

    @field_validator("cards")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        """Reject an empty set after de-duplication; order is not significant."""
        out = sorted(set(v))
        if not out:
            raise ValueError("cards must contain at least one unique card id")
        return out


class DispositionIn(BaseModel):
    """An analyst's verdict on an alert."""

    disposition: Literal["fraud", "false_positive"]
    notes: str = ""
    analyst: str = "demo_analyst"


# ──────────────────────────────────────────────────────────────────────────
# Application state
# ──────────────────────────────────────────────────────────────────────────


class AppState:
    """
    Everything loaded once at start-up.

    Each component is loaded independently and its failure recorded rather than
    raised, so a missing model does not prevent the graph endpoints from
    working — and `/health` can say precisely which half is down.
    """

    def __init__(self) -> None:
        self.graph = None
        self.index = None
        self.scorer = None
        self.memory = None
        self.composer = VulcanComposer(GATING)
        self.errors: dict[str, str] = {}
        self.ingest_buffer: list[dict] = []
        self.started_at = datetime.now(timezone.utc)
        self.counters = {"detect": 0, "ingest": 0, "alerts": 0, "dispositions": 0, "degraded_responses": 0}
        ensure_dirs()
        self.alerts_db = sqlite3.connect(str(ALERTS_DB), check_same_thread=False)
        self.alerts_db.row_factory = sqlite3.Row
        self.alerts_db.executescript(_ALERT_SCHEMA)
        self.alerts_db.commit()

    def load(self) -> None:
        """Load graph, transaction index, model and case memory. Never raises."""
        from src.case_memory import CaseMemory
        from src.data_generator import load_transactions
        from src.feature_engine import build_index
        from src.graph_builder import load_graph
        from src.scorer import RingfenceScorer

        try:
            self.graph = load_graph()
            log.info(f"graph loaded: {self.graph.number_of_nodes():,} nodes")
        except Exception as exc:  # noqa: BLE001
            self.errors["graph"] = str(exc)
            log.warning(f"graph unavailable: {exc}")

        try:
            txns = load_transactions()
            merchants = pd.read_csv(MERCHANTS_CSV) if MERCHANTS_CSV.exists() else None
            self.index = build_index(txns, merchants)
        except Exception as exc:  # noqa: BLE001
            self.errors["transactions"] = str(exc)
            log.warning(f"transaction index unavailable: {exc}")

        self.scorer = RingfenceScorer()
        if not self.scorer.available:
            self.errors["model"] = "model artifact not found; scoring in degraded mode"

        try:
            self.memory = CaseMemory()
        except Exception as exc:  # noqa: BLE001
            self.errors["case_memory"] = str(exc)
            log.warning(f"case memory unavailable: {exc}")

    @property
    def ready(self) -> bool:
        """True when the full scoring path is available."""
        return self.graph is not None and self.index is not None and bool(self.scorer and self.scorer.available)

    def save_alert(self, dossier: Dossier) -> None:
        """Persist an alert and its dossier."""
        self.alerts_db.execute(
            """INSERT INTO alerts (alert_id, created_at, cards, n_cards, ringfence_score,
                                   composite_score, recommendation, disposition, degraded, dossier)
               VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
               ON CONFLICT(alert_id) DO UPDATE SET dossier=excluded.dossier""",
            (
                dossier.alert_id, dossier.generated_at.isoformat(), json.dumps(dossier.candidate_cards),
                len(dossier.candidate_cards), dossier.ringfence_score, dossier.vulcan_composite_score,
                dossier.recommendation, int(dossier.degraded), dossier.model_dump_json(),
            ),
        )
        self.alerts_db.commit()
        self.counters["alerts"] += 1

    def get_alert(self, alert_id: str) -> sqlite3.Row | None:
        """Fetch one stored alert row."""
        return self.alerts_db.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)).fetchone()


STATE = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts on start-up, close connections on shutdown."""
    log.info("Ringfence API starting")
    STATE.load()
    log.info(f"ready={STATE.ready} errors={STATE.errors or 'none'}")
    yield
    STATE.alerts_db.close()
    if STATE.memory:
        STATE.memory.close()
    log.info("Ringfence API stopped")


app = FastAPI(
    title="Ringfence — Abuse-Ring Sentinel",
    description=(
        "Network-layer abuse-ring detection for Razorpay. Strictly defensive: this service "
        "observes and recommends. It cannot block, decline, or move money."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def trace_requests(request: Request, call_next):
    """Bind a UUID to every request and record its latency."""
    request_id = set_request_id(request.headers.get("X-Request-ID"))
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        ms = (time.perf_counter() - t0) * 1000
        LATENCY_RECORDER.record(f"api{request.url.path}", ms)
        clear_request_id()
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-ms"] = f"{ms:.2f}"
    return response


# ──────────────────────────────────────────────────────────────────────────
# Ingest
# ──────────────────────────────────────────────────────────────────────────


@app.post("/ingest/transaction", tags=["ingest"])
def ingest_transaction(txn: TransactionIn) -> dict:
    """
    Accept a transaction and acknowledge it.

    Buffered, not scored inline. The identity graph is a batch artifact —
    rebuilt on a schedule, not per transaction — and pretending otherwise would
    misrepresent the architecture. The production path (Kafka -> Flink windowed
    features -> Redis graph cache) is written up in ARCHITECTURE.md; this
    endpoint is the boundary that path plugs into.
    """
    STATE.ingest_buffer.append(txn.model_dump())
    STATE.counters["ingest"] += 1
    return {
        "status": "accepted",
        "transaction_id": txn.transaction_id,
        "buffered": len(STATE.ingest_buffer),
        "note": "Buffered for the next batch graph rebuild; identity edges are not updated inline.",
    }


# ──────────────────────────────────────────────────────────────────────────
# Detect
# ──────────────────────────────────────────────────────────────────────────


def _upgrade_summary_async(alert_id: str, dossier_json: str) -> None:
    """
    Background task: replace the deterministic template summary with the LLM
    narrative, if one can be produced.

    Runs after the response has been sent. If it fails, the stored dossier keeps
    its deterministic summary and remains complete — the narrative is never on
    the critical path for a fraud decision.
    """
    try:
        dossier = Dossier.model_validate_json(dossier_json)
        summary, status = generate_llm_summary(dossier.evidence)
        if summary:
            dossier.llm_summary = summary
            dossier.llm_status = status
            STATE.save_alert(dossier)
            log.info(f"upgraded dossier {alert_id} with LLM narrative")
    except Exception as exc:  # noqa: BLE001
        log.warning(f"background summary upgrade failed for {alert_id}: {exc}")


@app.post("/detect/candidate", tags=["detect"])
def detect_candidate(payload: CandidateIn, background: BackgroundTasks) -> dict:
    """
    Score a candidate card set and return the composite risk plus its dossier.

    Degrades rather than fails: unknown cards, a missing model, or an empty
    transaction history all produce a scored response flagged `degraded` with
    the safe-default recommendation, never a 5xx.
    """
    t0 = time.perf_counter()
    STATE.counters["detect"] += 1

    degraded_reasons: list[str] = []
    if STATE.graph is None or STATE.index is None:
        degraded_reasons.append("identity graph or transaction index unavailable")
    if not (STATE.scorer and STATE.scorer.available):
        degraded_reasons.append("model artifact unavailable")

    known = [c for c in payload.cards if STATE.graph is not None and c in STATE.graph]
    if STATE.graph is not None and len(known) < len(payload.cards):
        degraded_reasons.append(f"{len(payload.cards) - len(known)} of {len(payload.cards)} cards unknown to the graph")

    # --- features -------------------------------------------------------
    if STATE.graph is not None and STATE.index is not None and known:
        with LATENCY_RECORDER.time("api.features"):
            features = compute_features(known, STATE.graph, STATE.index)
        meta = compute_candidate_meta(known, STATE.index)
    else:
        from src.feature_engine import FEATURE_NAMES

        features = {n: 0.0 for n in FEATURE_NAMES}
        meta = {"vulcan_mean": 0.0, "vulcan_cashout_mean": 0.0, "n_transactions": 0}

    degraded = bool(degraded_reasons)
    if degraded:
        STATE.counters["degraded_responses"] += 1

    # --- score ----------------------------------------------------------
    with LATENCY_RECORDER.time("api.inference"):
        ringfence_score = STATE.scorer.score(features) if (STATE.scorer and STATE.scorer.available) else 0.0

    # --- case memory ----------------------------------------------------
    precedents, fp_patterns, boost, damp = [], [], 0.0, 0.0
    template = None
    if STATE.memory is not None and not degraded:
        precedents, fp_patterns, boost, damp = STATE.memory.adjustments(features)
        template = STATE.memory.match_template(features)

    # --- compose --------------------------------------------------------
    vulcan = payload.vulcan_score
    if vulcan is None:
        vulcan = float(meta.get("vulcan_cashout_mean") or meta.get("vulcan_mean", 0.0))

    composite = STATE.composer.compute_composite(
        vulcan_score=vulcan,
        ringfence_score=ringfence_score,
        precedent_boost=boost,
        fp_damp=damp,
        n_template_precedents=int(template["n_confirmations"]) if template else 0,
        degraded=degraded,
    )

    response: dict = {
        "alert_id": None,
        "cards": payload.cards,
        "cards_scored": known,
        "ringfence_score": round(ringfence_score, 4),
        "threshold": STATE.scorer.threshold if STATE.scorer else 0.5,
        "exceeds_threshold": bool(STATE.scorer and ringfence_score >= STATE.scorer.threshold),
        "composite": composite,
        "template_match": template,
        "degraded": degraded,
        "degraded_reasons": degraded_reasons,
        "model_version": STATE.scorer.model_version if STATE.scorer else "none",
    }

    # --- dossier --------------------------------------------------------
    if payload.include_dossier:
        alert_id = new_alert_id()
        with LATENCY_RECORDER.time("api.dossier"):
            dossier = build_dossier(
                cards=known or payload.cards,
                features=features,
                meta=meta,
                G=STATE.graph if STATE.graph is not None else _empty_graph(payload.cards),
                ringfence_score=ringfence_score,
                composite=composite,
                threshold=STATE.scorer.threshold if STATE.scorer else 0.5,
                precedents=precedents,
                fp_patterns=fp_patterns,
                alert_id=alert_id,
                use_llm=False,  # narrative is upgraded in the background
                degraded=degraded,
                degraded_reason="; ".join(degraded_reasons) or None,
            )
        STATE.save_alert(dossier)
        response["alert_id"] = alert_id
        response["dossier"] = json.loads(dossier.model_dump_json())
        response["dossier_quality"] = dossier.quality_metrics()

        if payload.use_llm:
            background.add_task(_upgrade_summary_async, alert_id, dossier.model_dump_json())
            response["llm_summary_pending"] = True

    response["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return response


def _empty_graph(cards: list[str]):
    """A graph containing only the requested cards, for the fully-degraded path."""
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(cards)
    return G


# ──────────────────────────────────────────────────────────────────────────
# Alerts
# ──────────────────────────────────────────────────────────────────────────


@app.get("/alerts", tags=["alerts"])
def list_alerts(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    recommendation: Optional[Literal["APPROVE", "REVIEW", "BLOCK"]] = None,
    disposition: Optional[Literal["fraud", "false_positive"]] = None,
) -> dict:
    """List recent alerts, newest first, with pagination and filters."""
    where, params = [], []
    if recommendation:
        where.append("recommendation = ?")
        params.append(recommendation)
    if disposition:
        where.append("disposition = ?")
        params.append(disposition)
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    total = STATE.alerts_db.execute(f"SELECT COUNT(*) FROM alerts {clause}", params).fetchone()[0]
    rows = STATE.alerts_db.execute(
        f"SELECT alert_id, created_at, n_cards, ringfence_score, composite_score, recommendation, "
        f"disposition, degraded FROM alerts {clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        [*params, limit, offset],
    ).fetchall()

    return {
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "alerts": [dict(r) for r in rows],
    }


@app.get("/alerts/{alert_id}/dossier", tags=["alerts"])
def get_dossier(alert_id: str) -> dict:
    """Return the full dossier: evidence, citations, feature snapshot, summary."""
    row = STATE.get_alert(alert_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"alert {alert_id} not found")
    dossier = json.loads(row["dossier"])
    parsed = Dossier.model_validate(dossier)
    return {
        "alert_id": alert_id,
        "disposition": row["disposition"],
        "dossier": dossier,
        "quality": parsed.quality_metrics(),
    }


@app.post("/alerts/{alert_id}/disposition", tags=["alerts"])
def set_disposition(alert_id: str, payload: DispositionIn) -> dict:
    """
    Record an analyst's verdict. This is the write that makes the system learn.

    A `fraud` verdict becomes a precedent that boosts similar future candidates.
    A `false_positive` verdict becomes a damp that lowers them. Both land in the
    case-memory audit log with a timestamp and an analyst.
    """
    row = STATE.get_alert(alert_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"alert {alert_id} not found")
    if STATE.memory is None:
        raise HTTPException(status_code=503, detail="case memory unavailable")

    dossier = Dossier.model_validate_json(row["dossier"])

    STATE.alerts_db.execute("UPDATE alerts SET disposition = ? WHERE alert_id = ?", (payload.disposition, alert_id))
    STATE.alerts_db.commit()

    STATE.memory.store_disposition(
        alert_id=alert_id,
        disposition=payload.disposition,
        notes=payload.notes,
        cards=dossier.candidate_cards,
        signature=dossier.feature_snapshot,
        device_pool=[
            e.value.get("value") for e in dossier.evidence
            if e.citation_type == "graph_edge" and isinstance(e.value, dict)
            and e.value.get("attribute") == "device_fingerprint"
        ],
        analyst=payload.analyst,
    )
    STATE.counters["dispositions"] += 1

    templates = STATE.memory.consolidate_templates()
    effect = (
        f"Future candidates matching this signature above "
        f"{CASE_MEMORY.precedent_similarity_threshold:.2f} similarity will be boosted by "
        f"{CASE_MEMORY.precedent_boost:.0%}."
        if payload.disposition == "fraud"
        else
        f"Future candidates matching this signature above "
        f"{CASE_MEMORY.fp_similarity_threshold:.2f} similarity will be damped by "
        f"{CASE_MEMORY.fp_damp:.0%} and tagged 'historical false positive pattern'."
    )

    return {
        "alert_id": alert_id,
        "disposition": payload.disposition,
        "stored": True,
        "effect_on_future_scoring": effect,
        "templates_consolidated": templates,
        "case_memory": STATE.memory.stats(),
    }


@app.get("/alerts/{alert_id}/failure-recovery", tags=["alerts"])
def failure_recovery(alert_id: str) -> dict:
    """
    Show, on real numbers, how a false-positive disposition changed scoring.

    Replays the alert's own feature signature through the composer twice: once
    ignoring case memory, once with it. The delta is the failure being handled —
    not a claim in a README but the two scores side by side.
    """
    row = STATE.get_alert(alert_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"alert {alert_id} not found")
    if STATE.memory is None:
        raise HTTPException(status_code=503, detail="case memory unavailable")

    dossier = Dossier.model_validate_json(row["dossier"])
    features = dossier.feature_snapshot

    precedents, fp_patterns, boost, damp = STATE.memory.adjustments(features)
    vulcan = dossier.vulcan_transaction_score

    before = STATE.composer.compute_composite(vulcan, dossier.ringfence_score, 0.0, 0.0)
    after = STATE.composer.compute_composite(vulcan, dossier.ringfence_score, boost, damp)

    return {
        "alert_id": alert_id,
        "disposition": row["disposition"],
        "analyst_notes": (STATE.memory.get_case(alert_id) or {}).get("notes", ""),
        "before_case_memory": {
            "composite_score": before["composite_score"],
            "recommendation": before["recommendation"],
        },
        "after_case_memory": {
            "composite_score": after["composite_score"],
            "recommendation": after["recommendation"],
            "precedent_boost_applied": boost,
            "fp_damp_applied": damp,
        },
        "score_delta": round(after["composite_score"] - before["composite_score"], 4),
        "matched_false_positive_patterns": fp_patterns,
        "matched_precedents": precedents,
        "explanation": (
            "An analyst marked this pattern a false positive. Case memory stored the 31-dimensional "
            f"signature; any future candidate within {CASE_MEMORY.fp_similarity_threshold:.2f} cosine "
            f"similarity now has its composite damped by {CASE_MEMORY.fp_damp:.0%} and carries a "
            "'historical false positive pattern' tag in its dossier."
            if fp_patterns else
            "No false-positive pattern matches this signature yet. Mark an alert as a false positive "
            "to see the damp take effect."
        ),
        "audit_trail": STATE.memory.audit_trail(alert_id),
    }


# ──────────────────────────────────────────────────────────────────────────
# Operational
# ──────────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["ops"])
def health() -> JSONResponse:
    """Component-level health. 200 when ready, 503 when degraded but serving."""
    payload = {
        "status": "ok" if STATE.ready else "degraded",
        "model_version": STATE.scorer.model_version if STATE.scorer else "none",
        "threshold": STATE.scorer.threshold if STATE.scorer else None,
        "components": {
            "identity_graph": STATE.graph is not None,
            "transaction_index": STATE.index is not None,
            "model": bool(STATE.scorer and STATE.scorer.available),
            "case_memory": STATE.memory is not None,
        },
        "errors": STATE.errors,
        "uptime_s": round((datetime.now(timezone.utc) - STATE.started_at).total_seconds(), 1),
        "defense_only": True,
    }
    return JSONResponse(payload, status_code=200 if STATE.ready else 503)


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics() -> str:
    """Prometheus text-format metrics: counters, latency percentiles, memory stats."""
    lines: list[str] = []

    def emit(name: str, value, help_text: str, mtype: str = "gauge") -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {mtype}")
        lines.append(f"{name} {value}")

    emit("ringfence_up", 1 if STATE.ready else 0, "1 when the full scoring path is available")
    for key, value in STATE.counters.items():
        emit(f"ringfence_{key}_total", value, f"Total {key} operations", "counter")

    for stage, pct in LATENCY_RECORDER.summary().items():
        safe = stage.replace("/", "_").replace(".", "_").replace("-", "_")
        for q in ("p50", "p95", "p99"):
            emit(f"ringfence_latency_{safe}_{q}_ms", round(pct[q], 3), f"{stage} latency {q} in ms")

    if STATE.memory is not None:
        for key, value in STATE.memory.stats().items():
            if isinstance(value, (int, float)):
                emit(f"ringfence_memory_{key}", value, f"Case memory: {key}")

    total_alerts = STATE.alerts_db.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    emit("ringfence_alerts_stored", total_alerts, "Alerts persisted")
    return "\n".join(lines) + "\n"


@app.get("/dossier-quality", tags=["ops"])
def dossier_quality(limit: int = Query(200, ge=1, le=1000)) -> dict:
    """
    Corpus-level dossier quality: uncited claims and references per claim.

    `uncited_claims` is structurally always 0 — the validator makes an uncited
    dossier unconstructible — and is reported rather than asserted.
    """
    rows = STATE.alerts_db.execute(
        "SELECT dossier FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    dossiers = [Dossier.model_validate_json(r["dossier"]) for r in rows]
    return dossier_corpus_metrics(dossiers)


@app.get("/", tags=["ops"])
def root() -> dict:
    """Service banner."""
    return {
        "service": "Ringfence — Abuse-Ring Sentinel",
        "track": "02 — AI Risk Manager",
        "version": MODEL_VERSION,
        "posture": "strictly defensive: observes and recommends; cannot block, decline or move money",
        "docs": "/docs",
    }
