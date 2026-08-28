"""
Vulcan integration layer — how Ringfence feeds the foundation model.

Ringfence does not replace Vulcan. Vulcan scores 3,000 signals per transaction
and is very good at the question "is this transaction odd?". The ring pattern
does not live inside any one transaction, though — it lives in the space
between merchants. A INR 5 charge at a recharge merchant is not odd. A INR
47,500 charge at an electronics merchant is not odd. The two of them on the same
device pool ninety minutes apart, across four merchants that will never compare
notes, is the entire crime.

So the composition is between two genuinely independent opinions. This matters
mechanically, not just rhetorically: Vulcan's score is **not** a feature of the
Ringfence model (see `feature_engine`), so combining them is not counting one
signal twice.

Composition
-----------
Noisy-OR: `composite = 1 - (1 - v)(1 - r)`.

Chosen because it has the right behaviour at the edges. Either signal alone
being high is enough to raise the composite; neither being high leaves it low;
and — the case that matters — two individually unremarkable signals compose
into something worth looking at. Vulcan 0.32 with a Ringfence ring score of
0.85 gives 0.90. A mean would have given 0.59 and buried it in the REVIEW pile.
An OR-max would have thrown away Vulcan's contribution entirely.

Bounded and gated
-----------------
This layer emits a *recommendation*. It cannot move money. `auto_action_allowed`
is False unless the candidate matches a Ring Template with more than
`auto_block_min_precedents` (10) confirmed precedents — meaning a human has
independently confirmed that exact pattern more than ten times. Everything
else, including every BLOCK on a novel pattern, requires an analyst.
"""

from __future__ import annotations

from typing import Literal

from src.config import GATING, GatingConfig
from src.utils import get_logger

log = get_logger("ringfence.vulcan_integration")

Recommendation = Literal["APPROVE", "REVIEW", "BLOCK"]


class VulcanComposer:
    """
    Composes Vulcan's per-transaction score with the Ringfence ring score.

    Vulcan scores 3,000 signals per transaction (0.0-1.0).
    Ringfence scores the candidate ring (0.0-1.0).
    The composite elevates risk when both signals align.

    Simulated for the demo. In production this is the boundary where Ringfence
    would post a ring score against Razorpay's test-mode API and receive the
    per-transaction score back; the composition maths and the gating bands are
    unchanged by that swap, which is why they live in their own class.
    """

    def __init__(self, cfg: GatingConfig = GATING):
        self.cfg = cfg

    # ── gating ───────────────────────────────────────────────────────────

    def _gate(self, composite: float) -> Recommendation:
        """Map a composite score onto the bounded action bands."""
        if composite < self.cfg.approve_below:
            return "APPROVE"
        if composite < self.cfg.block_at_or_above:
            return "REVIEW"
        return "BLOCK"

    def compute_composite(
        self,
        vulcan_score: float,
        ringfence_score: float,
        precedent_boost: float = 0.0,
        fp_damp: float = 0.0,
        n_template_precedents: int = 0,
        degraded: bool = False,
    ) -> dict:
        """
        Combine the two signals and return a bounded, explained recommendation.

        Returns:
            {
              "composite_score": float,
              "recommendation": "APPROVE" | "REVIEW" | "BLOCK",
              "contributing_signals": [...],
              "confidence": float,
              ...breakdown fields for the dossier...
            }

        `degraded=True` (no model loaded, missing data) forces the configured
        safe default — REVIEW — regardless of the arithmetic. A detector that
        cannot see must escalate to a human, never wave a transaction through.
        """
        v = float(min(max(vulcan_score, 0.0), 1.0)) * self.cfg.vulcan_weight
        r = float(min(max(ringfence_score, 0.0), 1.0)) * self.cfg.ringfence_weight
        v, r = min(v, 1.0), min(r, 1.0)

        base = 1.0 - (1.0 - v) * (1.0 - r)

        # Case-memory adjustments, applied multiplicatively and in this order so
        # that a known false-positive pattern can always pull a score down even
        # when a precedent has pushed it up.
        after_boost = base * (1.0 + precedent_boost)
        after_damp = after_boost * (1.0 - fp_damp)
        composite = float(min(max(after_damp, 0.0), 1.0))

        signals = ["vulcan", "ringfence"]
        if precedent_boost:
            signals.append("precedent")
        if fp_damp:
            signals.append("false_positive_memory")

        recommendation: Recommendation = self._gate(composite)
        if degraded:
            recommendation = self.cfg.safe_default_recommendation  # type: ignore[assignment]

        # Confidence is agreement between the two independent opinions, nudged
        # up when case memory has seen this shape before. Two signals that
        # disagree sharply produce a low-confidence composite even if the
        # number itself is high — which is exactly when an analyst should look.
        agreement = 1.0 - abs(v - r)
        confidence = float(min(1.0, 0.55 * agreement + 0.45 * max(v, r) + (0.05 if precedent_boost else 0.0)))
        if degraded:
            confidence = 0.0

        auto_action_allowed = bool(
            recommendation == "BLOCK"
            and not degraded
            and n_template_precedents > self.cfg.auto_block_min_precedents
        )

        result = {
            "composite_score": round(composite, 4),
            "recommendation": recommendation,
            "contributing_signals": signals,
            "confidence": round(confidence, 4),
            # --- explainability breakdown, carried into the dossier ---------
            "vulcan_score": round(v, 4),
            "ringfence_score": round(r, 4),
            "base_composite": round(base, 4),
            "precedent_boost_applied": round(precedent_boost, 4),
            "fp_damp_applied": round(fp_damp, 4),
            "composition": "noisy_or: 1 - (1 - vulcan) * (1 - ringfence)",
            "gate_bands": {
                "APPROVE": f"< {self.cfg.approve_below}",
                "REVIEW": f"{self.cfg.approve_below} - {self.cfg.block_at_or_above}",
                "BLOCK": f">= {self.cfg.block_at_or_above}",
            },
            "auto_action_allowed": auto_action_allowed,
            "human_review_required": not auto_action_allowed,
            "bounded_action_rule": (
                f"Automated action requires a Ring Template with more than "
                f"{self.cfg.auto_block_min_precedents} analyst-confirmed precedents. "
                f"This candidate has {n_template_precedents}. "
                + ("Auto-action permitted." if auto_action_allowed
                   else "Recommendation only; a human must action it.")
            ),
            "degraded": degraded,
        }

        log.info(
            f"composite: vulcan={v:.3f} ringfence={r:.3f} -> base={base:.3f} "
            f"boost=+{precedent_boost:.0%} damp=-{fp_damp:.0%} -> {composite:.3f} "
            f"=> {recommendation} (auto_action={auto_action_allowed}, degraded={degraded})"
        )
        return result

    def explain_elevation(self, vulcan_score: float, ringfence_score: float) -> str:
        """
        One-line narrative of what the composition did — the pitch's core claim,
        generated from the actual numbers rather than asserted.
        """
        composite = 1.0 - (1.0 - vulcan_score) * (1.0 - ringfence_score)
        return (
            f"Vulcan scored this transaction {vulcan_score:.2f} — individually plausible, "
            f"{self._gate(vulcan_score)} on its own. Ringfence scored the surrounding card cluster "
            f"{ringfence_score:.2f} as an abuse ring. Composite risk is {composite:.2f}: "
            f"{self._gate(composite)}."
        )
