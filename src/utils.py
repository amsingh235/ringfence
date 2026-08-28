"""
Shared helpers: structured logging, latency timers, determinism, small math.

Everything here is boring on purpose. The interesting code should be able to
assume that logging is JSON, that timers record percentiles, and that seeding
is done once and done everywhere.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from src.config import RANDOM_SEED

# ──────────────────────────────────────────────────────────────────────────
# Structured logging
# ──────────────────────────────────────────────────────────────────────────

_REQUEST_ID: str | None = None


class JsonFormatter(logging.Formatter):
    """
    Emit one JSON object per log line.

    Observability requirement: logs must be machine-parseable and carry the
    request UUID so a decision can be traced from HTTP call through graph,
    candidate, score and dossier.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if _REQUEST_ID:
            payload["request_id"] = _REQUEST_ID
        # Anything attached via logger.info("...", extra={"extra_fields": {...}})
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger. Human-readable by default, JSON when
    RINGFENCE_JSON_LOGS=1 (which is what the container sets).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    if os.getenv("RINGFENCE_JSON_LOGS", "0") == "1":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s %(name)-22s %(message)s", "%H:%M:%S")
        )
    logger.addHandler(handler)
    logger.setLevel(os.getenv("RINGFENCE_LOG_LEVEL", "INFO"))
    logger.propagate = False
    return logger


def set_request_id(request_id: str | None = None) -> str:
    """Bind a UUID to this execution context so every log line is traceable."""
    global _REQUEST_ID
    _REQUEST_ID = request_id or uuid.uuid4().hex[:16]
    return _REQUEST_ID


def clear_request_id() -> None:
    """Unbind the request UUID once a request finishes."""
    global _REQUEST_ID
    _REQUEST_ID = None


log = get_logger("ringfence.utils")


# ──────────────────────────────────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────────────────────────────────


def seed_everything(seed: int = RANDOM_SEED) -> np.random.Generator:
    """
    Seed every source of randomness we touch and hand back a Generator.

    Reproducibility is a submission bar, not a nicety: the panel should be
    able to clone the repo and land on the same numbers we quote.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)


# ──────────────────────────────────────────────────────────────────────────
# Latency instrumentation
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class LatencyRecorder:
    """
    Collects wall-clock samples per named stage and reports percentiles.

    We quote p99 in the README, so we measure p99 — not a mean dressed up as
    a guarantee.
    """

    samples: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def record(self, stage: str, millis: float) -> None:
        """Record one timing sample, in milliseconds, for `stage`."""
        self.samples[stage].append(millis)

    @contextmanager
    def time(self, stage: str) -> Iterator[None]:
        """Context manager that records how long the enclosed block took."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter() - t0) * 1000.0)

    def percentiles(self, stage: str) -> dict[str, float]:
        """Return p50/p95/p99/max/mean/n for one stage, in milliseconds."""
        vals = self.samples.get(stage) or []
        if not vals:
            return {"n": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
        arr = np.asarray(vals, dtype=float)
        return {
            "n": int(arr.size),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }

    def summary(self) -> dict[str, dict[str, float]]:
        """Percentiles for every recorded stage."""
        return {stage: self.percentiles(stage) for stage in self.samples}

    def reset(self) -> None:
        """Drop all samples (used between benchmark runs)."""
        self.samples.clear()


LATENCY_RECORDER = LatencyRecorder()


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """
    Time a block, log it, and record it globally.

    Rule 7 of this repo: no silent scripts. Every meaningful stage announces
    how long it took.
    """
    logger = logger or log
    t0 = time.perf_counter()
    yield
    ms = (time.perf_counter() - t0) * 1000.0
    LATENCY_RECORDER.record(label, ms)
    logger.info(f"{label} finished in {ms / 1000:.2f}s", extra={"extra_fields": {"stage": label, "ms": round(ms, 2)}})


# ──────────────────────────────────────────────────────────────────────────
# Small set / math helpers
# ──────────────────────────────────────────────────────────────────────────


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """
    Jaccard similarity |A n B| / |A u B|. Used for candidate deduplication.

    Returns 0.0 for two empty sets rather than raising — an empty candidate is
    never similar to anything, it is just discarded upstream.
    """
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def overlap_fraction(subset: Iterable[str], reference: Iterable[str]) -> float:
    """
    |subset n reference| / |reference|.

    This — not Jaccard — is the ring-recall criterion: a candidate "covers" a
    planted ring when it contains more than half that ring's cards, regardless
    of how much unrelated padding the candidate also carries. A candidate that
    finds all 5 ring cards plus 40 innocents has still *found the ring*; the
    padding is the scorer's problem, not the generator's.
    """
    ref = set(reference)
    if not ref:
        return 0.0
    return len(set(subset) & ref) / len(ref)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, safe on zero vectors."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity of one query vector against every row of `matrix`.

    Vectorised because case memory is queried on the request path and a Python
    loop over precedents would eat the latency budget as the store grows.
    """
    if matrix.size == 0:
        return np.zeros(0, dtype=float)
    qn = np.linalg.norm(query)
    mn = np.linalg.norm(matrix, axis=1)
    denom = qn * mn
    denom[denom == 0.0] = np.inf
    return (matrix @ query) / denom


def shannon_entropy(values: Sequence[float] | np.ndarray, bins: int = 12) -> float:
    """
    Shannon entropy (bits) of a value distribution, over log-spaced bins.

    Log bins because transaction amounts span INR 1 to INR 60,000; linear bins
    would put every probe in bucket zero and tell us nothing.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return 0.0
    edges = np.logspace(np.log10(max(arr.min(), 1e-6)), np.log10(max(arr.max(), 1e-6) + 1e-6), bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns `default` instead of raising or returning inf."""
    if denominator == 0 or not np.isfinite(denominator):
        return default
    result = numerator / denominator
    return float(result) if np.isfinite(result) else default


def new_alert_id() -> str:
    """Generate a short, sortable-enough alert identifier."""
    return f"RF-{uuid.uuid4().hex[:12].upper()}"


def write_json(path, payload: Any) -> None:
    """Write JSON to disk, creating parent directories as needed."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path) -> Any:
    """Read JSON from disk."""
    from pathlib import Path

    return json.loads(Path(path).read_text(encoding="utf-8"))
