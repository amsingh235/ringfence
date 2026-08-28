"""
Central configuration for Ringfence.

Every magic number in this system lives here or in an environment variable.
Nothing downstream hardcodes a threshold, a cost, or a path — that is a hard
rule, because the panel will ask "where does 0.7 come from?" and the answer
has to be a single greppable line, not a literal buried in a function.

Sections
--------
1. Paths            — where artifacts land on disk
2. Reproducibility  — the one seed used everywhere
3. GenerationConfig — synthetic data shape (spec-calibrated, test-shrinkable)
4. GraphConfig      — identity edge rules and collision capping
5. CandidateConfig  — recall-first generator knobs
6. CostConfig       — the economics that pick the threshold (NOT F1)
7. GatingConfig     — APPROVE / REVIEW / BLOCK bands and bounded-action rules
8. CaseMemoryConfig — precedent boost and false-positive damp
9. LatencyBudget    — the numbers we assert against in tests
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────────
# 1. PATHS
# ──────────────────────────────────────────────────────────────────────────

# src/config.py -> src/ -> ringfence/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = Path(os.getenv("RINGFENCE_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
GROUND_TRUTH_DIR: Path = DATA_DIR / "ground_truth"

TRANSACTIONS_CSV: Path = RAW_DIR / "transactions.csv"
CARDS_CSV: Path = RAW_DIR / "cards.csv"
MERCHANTS_CSV: Path = RAW_DIR / "merchants.csv"
RINGS_CSV: Path = GROUND_TRUTH_DIR / "rings.csv"

GRAPH_PICKLE: Path = PROCESSED_DIR / "identity_graph.gpickle"
CANDIDATES_PARQUET: Path = PROCESSED_DIR / "candidates.json"
FEATURES_CSV: Path = PROCESSED_DIR / "features.csv"
MODEL_PATH: Path = PROCESSED_DIR / "ringfence_model.pkl"
METRICS_JSON: Path = PROCESSED_DIR / "metrics.json"
CASE_MEMORY_DB: Path = PROCESSED_DIR / "case_memory.sqlite"
SIGNATURE_SCALER_PATH: Path = PROCESSED_DIR / "signature_scaler.json"
CHROMA_DIR: Path = PROCESSED_DIR / "chroma"
ALERTS_DB: Path = PROCESSED_DIR / "alerts.sqlite"


def ensure_dirs() -> None:
    """Create every output directory. Idempotent; safe to call at import time."""
    for d in (RAW_DIR, PROCESSED_DIR, GROUND_TRUTH_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# 2. REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────

RANDOM_SEED: int = 42

MODEL_VERSION: str = "ringfence-0.1.0"

# ──────────────────────────────────────────────────────────────────────────
# 3. DATA GENERATION
# ──────────────────────────────────────────────────────────────────────────


class GenerationConfig(BaseModel):
    """
    Shape of the synthetic transaction universe.

    Defaults are the spec-calibrated production numbers. `small()` returns a
    shrunk universe used by the test suite so `make test` finishes in seconds
    instead of minutes — the *code paths* are identical, only the scale differs.
    """

    n_cards: int = 2396
    n_merchants: int = 690
    n_transactions: int = 737_000
    n_worlds: int = 5
    n_rings: int = 120

    ring_size_min: int = 3
    ring_size_max: int = 8
    ring_device_pool_min: int = 2
    ring_device_pool_max: int = 3

    # Observation window
    days: int = 90

    # Legit amount distribution (lognormal, INR)
    legit_amount_mu: float = 6.4  # exp(6.4) ~= INR 600 median ticket
    legit_amount_sigma: float = 1.1

    # Ring behavioural signature
    probe_amount_min: float = 1.0
    probe_amount_max: float = 10.0
    cashout_amount_min: float = 18_000.0
    cashout_amount_max: float = 60_000.0
    probe_burst_seconds: int = 300      # probes land inside a 5-minute window
    probe_to_cashout_min_s: int = 60    # cashout follows 1 min ...
    probe_to_cashout_max_s: int = 7_200  # ... to 2 hours later

    # Identity layer, calibrated to IEEE-CIS device-cardinality quantiles.
    # One generic UA string must collide across ~1,492 accounts: that single
    # value is what forces collision capping to exist at all.
    generic_device_string: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    generic_device_accounts: int = 1_492
    generic_device_txn_fraction: float = 0.06  # share of a card's txns on it

    household_device_share: float = 0.18  # devices shared by 2-4 cards
    nat_ip_pools: int = 40                # ISP NAT IPs shared by many cards
    nat_ip_txn_fraction: float = 0.12

    # --- Hard negatives ---------------------------------------------------
    # A share of households shop in a *coordinated burst*: several small
    # purchases inside five minutes, then a larger one an hour or two later.
    # Same device, same IP, same compressed timing, same escalation direction
    # as a ring. Without these the identity graph would make ring detection
    # trivial and any precision we reported would be a lie.
    burst_household_share: float = 0.35
    burst_small_amount_max: float = 60.0
    burst_large_amount_min: float = 1_500.0
    burst_large_amount_max: float = 9_000.0

    # A share of ring members run their own fresh device instead of the ring
    # pool — real rings are not perfectly disciplined. This degrades identity
    # coverage and is what stops ring recall being a free 1.000.
    ring_defector_prob: float = 0.15

    # Simulated Vulcan per-transaction scores (0.0-1.0)
    vulcan_legit_alpha: float = 2.0
    vulcan_legit_beta: float = 12.0   # mean ~0.14
    vulcan_probe_lo: float = 0.10     # individually plausible
    vulcan_probe_hi: float = 0.30
    vulcan_cashout_lo: float = 0.30   # medium — Vulcan alone would not block
    vulcan_cashout_hi: float = 0.50

    @classmethod
    def small(cls) -> "GenerationConfig":
        """Test-scale universe: same generators, ~1/40th the volume."""
        return cls(
            n_cards=300,
            n_merchants=90,
            n_transactions=24_000,
            n_worlds=5,
            n_rings=25,
            days=30,
            generic_device_accounts=180,
            nat_ip_pools=8,
        )


# ──────────────────────────────────────────────────────────────────────────
# 4. GRAPH
# ──────────────────────────────────────────────────────────────────────────


class GraphConfig(BaseModel):
    """
    Identity-graph construction rules.

    Cards are nodes. Edges come from shared device / IP / email ONLY.
    There are deliberately no merchant edges — see `get_collision_stats`,
    which logs how completely merchant co-occurrence swamps real signal.
    """

    cap_percentile: float = 95.0

    # A hard floor under the percentile cap. The 95th percentile of device
    # cardinality is typically 3-5 cards; if we capped there we would shred
    # the 8-card rings we are trying to detect. The cap exists to stop a
    # 1,492-account generic UA from fusing the graph into one blob, not to
    # punish legitimate small cliques. Floor must exceed ring_size_max.
    min_collision_cap: int = 12

    # Rarity-weighted edge strength: an identity shared by k cards contributes
    # 1 / log2(k + 1), so a 2-card device is worth 0.63 and a 1,492-card
    # generic UA is worth 0.095. Type multipliers reflect spoofability.
    weight_device: float = 1.00
    weight_email: float = 1.00
    weight_ip: float = 0.70

    identity_attributes: tuple[str, ...] = ("device_fingerprint", "ip_address", "email_hash")


# ──────────────────────────────────────────────────────────────────────────
# 5. CANDIDATE GENERATION
# ──────────────────────────────────────────────────────────────────────────


class CandidateConfig(BaseModel):
    """
    Recall-first candidate generation. Propose every ring; let the GBM rank.

    Three generators are ensembled because each fails differently:
      - Louvain finds cohesive blocks but smears at low resolution
      - Connected components find infrastructure islands but merge at low weight
      - Identity-ego sets propose the exact shared-device set — the ring itself
    """

    louvain_resolutions: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    cc_weight_thresholds: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)

    min_candidate_size: int = 2
    max_candidate_size: int = 60  # above this it is a population, not a ring

    jaccard_dedup_threshold: float = 0.5

    # A candidate is labelled positive when it covers >50% of a planted ring.
    ring_overlap_threshold: float = 0.5


# ──────────────────────────────────────────────────────────────────────────
# 6. COST — this is what picks the threshold
# ──────────────────────────────────────────────────────────────────────────


class CostConfig(BaseModel):
    """
    The economics of being wrong. We do NOT optimise F1.

    fp_cost — INR 2.3L: analyst review time + merchant friction + the
              reputational cost of a false decline on a good customer.
    fn_cost — INR 8.5L: average ring cashout value + chargeback handling
              + merchant churn after the ring drains them.

    ring_incidence is an OPTIONAL reweighting knob, not a multiplier we apply
    blindly. Set it to reweight the negative class when the deployed candidate
    mix differs from the offline mix; leave it None to use the empirical
    out-of-fold distribution, which is the honest default. See
    `scorer.optimise_threshold` for exactly how it is applied.
    """

    fp_cost: float = Field(default=230_000.0, description="INR cost of one false positive")
    fn_cost: float = Field(default=850_000.0, description="INR cost of one missed ring")
    ring_incidence: float | None = Field(
        default=None,
        description="P(ring) in the deployed candidate population; None = empirical",
    )

    @property
    def cost_ratio(self) -> float:
        """How many false positives one miss is worth. Currently ~3.7."""
        return self.fn_cost / self.fp_cost


# ──────────────────────────────────────────────────────────────────────────
# 7. GATING — bounded actions
# ──────────────────────────────────────────────────────────────────────────


class GatingConfig(BaseModel):
    """
    Composite-score bands and the bounded-action rule.

    Ringfence never moves money. It emits a recommendation. Even a BLOCK
    recommendation requires a human unless the pattern matches a Ring Template
    with more than `auto_block_min_precedents` confirmed precedents — and that
    is the only path to an automated action anywhere in this codebase.
    """

    approve_below: float = 0.4
    block_at_or_above: float = 0.7

    auto_block_min_precedents: int = 10

    # Noisy-OR composition weights. vulcan_weight scales Vulcan's contribution,
    # ringfence_weight scales ours, before the OR. Both default to 1.0 = trust
    # each signal at face value.
    vulcan_weight: float = 1.0
    ringfence_weight: float = 1.0

    # Safe default when anything upstream is unavailable. Never APPROVE on
    # failure — a degraded detector must escalate to a human, not wave it through.
    safe_default_recommendation: str = "REVIEW"


# ──────────────────────────────────────────────────────────────────────────
# 8. CASE MEMORY
# ──────────────────────────────────────────────────────────────────────────


class CaseMemoryConfig(BaseModel):
    """Precedent matching and — the important half — failure memory."""

    precedent_similarity_threshold: float = 0.85
    precedent_boost: float = 0.15  # +15% relative on confirmed-fraud match

    fp_similarity_threshold: float = 0.80
    fp_damp: float = 0.20  # -20% relative on confirmed false-positive match

    template_min_confirmations: int = 5
    top_k: int = 3


# ──────────────────────────────────────────────────────────────────────────
# 9. LATENCY BUDGET
# ──────────────────────────────────────────────────────────────────────────


class LatencyBudget(BaseModel):
    """Numbers the test suite asserts against, not aspirations in a README."""

    feature_p99_ms: float = 50.0
    inference_p99_ms: float = 10.0
    dossier_p99_ms: float = 200.0
    api_score_p99_ms: float = 100.0
    api_dossier_p99_ms: float = 300.0


# ──────────────────────────────────────────────────────────────────────────
# Module-level singletons — import these, do not re-instantiate
# ──────────────────────────────────────────────────────────────────────────

GENERATION = GenerationConfig()
GRAPH = GraphConfig()
CANDIDATES = CandidateConfig()
COSTS = CostConfig()
GATING = GatingConfig()
CASE_MEMORY = CaseMemoryConfig()
LATENCY = LatencyBudget()

# LLM summary layer. Absent key is not an error — the dossier is deterministic
# and complete without it; the narrative is a convenience on top.
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY") or None
OPENAI_MODEL: str = os.getenv("RINGFENCE_LLM_MODEL", "gpt-4o-mini")
LLM_TIMEOUT_S: float = float(os.getenv("RINGFENCE_LLM_TIMEOUT", "8"))

API_HOST: str = os.getenv("RINGFENCE_API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("RINGFENCE_API_PORT", "8000"))
API_BASE_URL: str = os.getenv("RINGFENCE_API_URL", "http://localhost:8000")

ensure_dirs()
