"""
Case memory tests: precedent boost, false-positive damp, template consolidation,
and the audit trail.

The false-positive damp is the one the submission bar names — "one failure
handled gracefully" — so it is tested from both ends: that the damp fires on a
similar candidate, and that it does *not* fire on an unrelated one.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.case_memory import (
    DISPOSITION_FALSE_POSITIVE,
    DISPOSITION_FRAUD,
    CaseMemory,
    signature_from_features,
)
from src.config import CASE_MEMORY
from src.feature_engine import FEATURE_NAMES


def _signature(**overrides) -> dict[str, float]:
    """A feature dict with a plausible ring shape, tweakable per test."""
    base = {n: 0.0 for n in FEATURE_NAMES}
    base.update({
        "cand_size": 5.0, "cand_density": 0.8, "cand_ego_overlap_ratio": 1.0,
        "n_unique_devices": 2.0, "max_device_reuse": 5.0,
        "probe_fraction": 0.6, "cashout_fraction": 0.3,
        "amount_escalation": 4200.0, "burst_compression": 0.7,
        "time_span_hours": 2.0, "velocity_txn_per_hour": 9.0,
    })
    base.update(overrides)
    return base


# ── signatures ───────────────────────────────────────────────────────────


def test_signature_is_unit_length():
    """L2 normalisation makes cosine measure shape rather than magnitude."""
    sig = signature_from_features(_signature())
    assert sig.shape == (len(FEATURE_NAMES),)
    assert np.linalg.norm(sig) == pytest.approx(1.0, abs=1e-9)


def test_signature_compresses_unbounded_features():
    """
    Log compression stops one huge feature dominating the metric.

    Without it, `amount_escalation` at 4,200 would swamp all 30 other
    dimensions and every long-tailed candidate would look identical.
    """
    small = signature_from_features(_signature(amount_escalation=100.0))
    large = signature_from_features(_signature(amount_escalation=50_000.0))
    assert float(small @ large) > 0.9


def test_signature_handles_nan_and_inf():
    """Degenerate feature values must not produce a NaN signature."""
    sig = signature_from_features(_signature(amount_escalation=float("inf"), cand_density=float("nan")))
    assert np.isfinite(sig).all()


def test_signature_separates_different_shapes():
    """Two genuinely different behavioural shapes are not similar."""
    ring = signature_from_features(_signature())
    quiet = signature_from_features({n: 0.0 for n in FEATURE_NAMES} | {"cand_size": 40.0, "geo_spread": 0.9})
    assert float(ring @ quiet) < CASE_MEMORY.fp_similarity_threshold


# ── the scaler: signatures must actually discriminate ────────────────────


def test_training_fits_and_persists_a_signature_scaler(trained):
    """The trainer fits the population statistics case memory needs."""
    from src.case_memory import load_signature_scaler

    scaler = load_signature_scaler()
    assert scaler is not None, "no signature scaler written during training"
    assert scaler["mean"].shape == (len(FEATURE_NAMES),)
    assert scaler["std"].shape == (len(FEATURE_NAMES),)
    assert scaler["fingerprint"]
    assert trained["artifact"]["signature_scaler_fingerprint"] == scaler["fingerprint"]


def test_signatures_are_not_saturated(trained, features):
    """
    Regression guard for a real bug.

    All 31 features are non-negative, so log-compressed signatures all lived in
    one orthant and pairwise cosine similarity came out at a **median of 0.964**
    across the whole candidate population. Every candidate looked like every
    other one, so a single false-positive disposition damped all 557 candidates
    — case memory was matching "is a candidate", not "is this pattern".

    Standardising against population statistics fixes it (measured: median
    similarity −0.07, ~4% of pairs above the damp threshold). If this assertion
    ever fails again, precedent matching has silently stopped discriminating
    while still appearing to work.
    """
    from src.case_memory import load_signature_scaler

    scaler = load_signature_scaler()
    rows = features[list(FEATURE_NAMES)].to_dict("records")
    sigs = np.vstack([signature_from_features(r, scaler) for r in rows])

    sim = sigs @ sigs.T
    np.fill_diagonal(sim, np.nan)
    flat = sim[~np.isnan(sim)]

    median = float(np.nanpercentile(flat, 50))
    saturated = float((flat >= CASE_MEMORY.fp_similarity_threshold).mean())

    assert median < 0.5, f"signature space is saturated (median pairwise cosine {median:.3f})"
    assert saturated < 0.35, (
        f"{saturated:.1%} of all candidate pairs exceed the damp threshold — "
        f"one false positive would damp most of the population"
    )


def test_precedent_boost_beats_the_base_rate(trained, features, labels):
    """
    A precedent boost should fire on a mostly-ring set.

    Before the scaler fix the boost fired on everything, so its precision was
    exactly the base rate — a boost that boosts uniformly is not a boost. This
    asserts it carries real information.
    """
    from src.case_memory import load_signature_scaler

    scaler = load_signature_scaler()
    data = labels.merge(features, on="candidate_id")
    y = data["is_fraud_ring"].to_numpy()
    sigs = np.vstack([signature_from_features(r, scaler) for r in data[list(FEATURE_NAMES)].to_dict("records")])

    sim = sigs @ sigs.T
    np.fill_diagonal(sim, np.nan)

    precisions = []
    for idx in np.where(y == 1)[0]:
        hit = sim[idx] >= CASE_MEMORY.precedent_similarity_threshold
        if hit.sum() >= 3:
            precisions.append((hit & (y == 1)).sum() / hit.sum())

    assert precisions, "no ring signature matched any other — the space is over-separated"
    assert float(np.mean(precisions)) > y.mean() * 1.3, (
        f"precedent boost precision {np.mean(precisions):.2f} is not meaningfully above "
        f"the {y.mean():.2f} base rate"
    )


def test_case_memory_warns_on_a_stale_scaler(tmp_path, caplog):
    """
    Signatures are only comparable within one scaler's space.

    After retraining, a case store built with the old scaler holds vectors in a
    different space. That must be a loud warning, not a silent wrong answer.
    """
    import logging

    db = tmp_path / "stale.sqlite"
    good = tmp_path / "scaler_a.json"
    other = tmp_path / "scaler_b.json"
    for path, tag in ((good, "A"), (other, "B")):
        path.write_text(json.dumps({
            "mean": [0.0] * len(FEATURE_NAMES), "std": [1.0] * len(FEATURE_NAMES),
            "fingerprint": f"fingerprint-{tag}", "n_samples": 10,
        }), encoding="utf-8")

    m1 = CaseMemory(db_path=db, scaler_path=good)
    m1.store_disposition("RF-STALE", DISPOSITION_FRAUD, "ring", signature=_signature())
    m1.close()

    with caplog.at_level(logging.WARNING, logger="ringfence.case_memory"):
        m2 = CaseMemory(db_path=db, scaler_path=other)
        m2.close()
    assert any("clean-memory" in r.message for r in caplog.records), "stale scaler was not flagged"


# ── writes ───────────────────────────────────────────────────────────────


def test_store_and_read_back_a_disposition(memory):
    """A stored case round-trips with its notes and analyst."""
    memory.store_disposition("RF-001", DISPOSITION_FRAUD, "confirmed by ops",
                             cards=["CARD_A", "CARD_B"], signature=_signature(), ring_id="RING_0001")
    case = memory.get_case("RF-001")
    assert case["disposition"] == DISPOSITION_FRAUD
    assert case["cards"] == ["CARD_A", "CARD_B"]
    assert case["notes"] == "confirmed by ops"


def test_invalid_disposition_is_rejected(memory):
    """Only 'fraud' and 'false_positive' are accepted."""
    with pytest.raises(ValueError, match="disposition must be"):
        memory.store_disposition("RF-BAD", "maybe", signature=_signature())


def test_disposition_is_idempotent(memory):
    """Re-dispositioning updates rather than duplicating."""
    memory.store_disposition("RF-002", DISPOSITION_FRAUD, "first", signature=_signature())
    memory.store_disposition("RF-002", DISPOSITION_FALSE_POSITIVE, "actually no", signature=_signature())
    assert memory.count() == 1
    assert memory.get_case("RF-002")["disposition"] == DISPOSITION_FALSE_POSITIVE


# ── precedent boost ──────────────────────────────────────────────────────


def test_precedent_boost_fires_on_a_similar_confirmed_ring(memory):
    """A near-identical signature to a confirmed ring earns the +15% boost."""
    memory.store_disposition("RF-100", DISPOSITION_FRAUD, "classic probe-cashout",
                             cards=["C1", "C2"], signature=_signature(), ring_id="RING_0007")

    similar = _signature(cand_size=6.0, probe_fraction=0.62)
    precedents = memory.find_precedents(similar)
    assert precedents, "no precedent matched a near-identical signature"
    assert precedents[0]["similarity"] >= CASE_MEMORY.precedent_similarity_threshold
    assert precedents[0]["ring_id"] == "RING_0007"

    _, _, boost, damp = memory.adjustments(similar)
    assert boost == CASE_MEMORY.precedent_boost
    assert damp == 0.0


def test_precedent_boost_does_not_fire_on_an_unrelated_shape(memory):
    """No boost without a genuine match."""
    memory.store_disposition("RF-101", DISPOSITION_FRAUD, "ring", signature=_signature())
    unrelated = {n: 0.0 for n in FEATURE_NAMES} | {"cand_size": 45.0, "weekend_fraction": 1.0}
    _, _, boost, _ = memory.adjustments(unrelated)
    assert boost == 0.0


# ── false-positive damp — "one failure handled gracefully" ───────────────


def test_false_positive_damp_fires_on_a_similar_candidate(memory):
    """
    The failure-recovery bar.

    An analyst rejects one candidate; a similar future candidate is damped 20%
    and carries the matched case in its dossier.
    """
    memory.store_disposition("RF-200", DISPOSITION_FALSE_POSITIVE,
                             "household sharing one tablet", cards=["C1", "C2"], signature=_signature())

    similar = _signature(cand_size=5.0, probe_fraction=0.58)
    fps = memory.find_false_positive_patterns(similar)
    assert fps, "false-positive pattern did not match"
    assert fps[0]["similarity"] >= CASE_MEMORY.fp_similarity_threshold
    assert fps[0]["alert_id"] == "RF-200"
    assert "household" in fps[0]["notes"]

    _, _, boost, damp = memory.adjustments(similar)
    assert damp == CASE_MEMORY.fp_damp
    assert boost == 0.0


def test_false_positive_damp_lowers_the_composite_score(memory):
    """End to end: the damp actually moves the composite score down 20%."""
    from src.vulcan_integration import VulcanComposer

    composer = VulcanComposer()
    sig = _signature()
    before = composer.compute_composite(0.30, 0.90)

    memory.store_disposition("RF-201", DISPOSITION_FALSE_POSITIVE, "not a ring", signature=sig)
    _, fps, boost, damp = memory.adjustments(sig)
    after = composer.compute_composite(0.30, 0.90, boost, damp)

    assert fps
    assert after["composite_score"] < before["composite_score"]
    assert after["composite_score"] == pytest.approx(before["composite_score"] * 0.8, rel=1e-6)
    assert "false_positive_memory" in after["contributing_signals"]


def test_damp_is_bounded(memory):
    """
    A bad memory can move a score by at most 20%.

    Even with several matching false positives the damp does not compound —
    the correction is a bounded constant, not an unbounded learned weight.
    """
    for i in range(5):
        memory.store_disposition(f"RF-30{i}", DISPOSITION_FALSE_POSITIVE, "fp", signature=_signature())
    _, fps, _, damp = memory.adjustments(_signature())
    assert len(fps) <= CASE_MEMORY.top_k
    assert damp == CASE_MEMORY.fp_damp


def test_fp_threshold_is_looser_than_the_precedent_threshold():
    """
    Deliberate asymmetry: we would rather catch a near-miss of a known mistake
    than repeat it on a technicality.
    """
    assert CASE_MEMORY.fp_similarity_threshold < CASE_MEMORY.precedent_similarity_threshold


# ── templates ────────────────────────────────────────────────────────────


def test_templates_need_the_minimum_confirmations(memory):
    """Fewer than five confirmed rings consolidate into nothing."""
    for i in range(3):
        memory.store_disposition(f"RF-40{i}", DISPOSITION_FRAUD, "ring", signature=_signature())
    assert memory.consolidate_templates(min_confirmations=5) == []


def test_templates_consolidate_at_five_confirmations(memory):
    """Five similar confirmed rings become one Ring Template."""
    for i in range(6):
        memory.store_disposition(f"RF-50{i}", DISPOSITION_FRAUD, "ring",
                                 signature=_signature(cand_size=5.0 + i * 0.1),
                                 device_pool=[f"DEV_{i}"])
    templates = memory.consolidate_templates(min_confirmations=5)
    assert len(templates) == 1
    assert templates[0]["n_confirmations"] >= 5
    assert not templates[0]["auto_block_eligible"], "6 precedents must not unlock auto-action"


def test_auto_block_requires_more_than_ten_precedents(memory):
    """
    The bounded-action rule.

    Automated action is gated behind more than ten analyst confirmations of the
    same pattern; this is the only automated path anywhere in the system.
    """
    for i in range(12):
        memory.store_disposition(f"RF-60{i:02d}", DISPOSITION_FRAUD, "ring",
                                 signature=_signature(cand_size=5.0 + i * 0.05))
    templates = memory.consolidate_templates(min_confirmations=5)
    assert templates[0]["auto_block_eligible"]

    match = memory.match_template(_signature())
    assert match is not None
    assert match["auto_block_eligible"]


def test_template_match_returns_none_when_empty(memory):
    """No templates, no match, no crash."""
    assert memory.match_template(_signature()) is None


# ── audit trail ──────────────────────────────────────────────────────────


def test_every_disposition_is_audited(memory):
    """The audit trail records who decided what, and when."""
    memory.store_disposition("RF-700", DISPOSITION_FRAUD, "confirmed", signature=_signature(),
                             analyst="riya")
    trail = memory.audit_trail("RF-700")
    assert trail
    assert trail[0]["action"] == "disposition:fraud"
    assert trail[0]["at"]


def test_stats_reports_both_dispositions(memory):
    """Stats separate confirmations from rejections — both are the product."""
    memory.store_disposition("RF-800", DISPOSITION_FRAUD, "", signature=_signature())
    memory.store_disposition("RF-801", DISPOSITION_FALSE_POSITIVE, "", signature=_signature())
    s = memory.stats()
    assert s["n_confirmed_fraud"] == 1
    assert s["n_false_positives"] == 1
    assert s["fp_damp"] == CASE_MEMORY.fp_damp


def test_empty_memory_returns_no_adjustments(memory):
    """A cold start produces no boost and no damp, and does not raise."""
    precedents, fps, boost, damp = memory.adjustments(_signature())
    assert precedents == [] and fps == []
    assert boost == 0.0 and damp == 0.0


def test_memory_persists_across_reconnects(tmp_path):
    """Case memory survives a restart — it is durable, not a process cache."""
    path = tmp_path / "persist.sqlite"
    m1 = CaseMemory(db_path=path)
    m1.store_disposition("RF-900", DISPOSITION_FALSE_POSITIVE, "keep me", signature=_signature())
    m1.close()

    m2 = CaseMemory(db_path=path)
    assert m2.count() == 1
    assert m2.find_false_positive_patterns(_signature())
    m2.close()
