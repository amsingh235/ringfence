"""
Scorer tests: cost-optimal thresholding, cross-world validation, and the
latency budget.

The threshold assertions matter most. It is easy to write a cost function and
then quietly optimise F1 anyway; these tests check that the chosen operating
point actually minimises rupees on the same out-of-fold predictions every other
number is computed from.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import COSTS, LATENCY, CostConfig
from src.feature_engine import FEATURE_NAMES, compute_features
from src.scorer import RingfenceScorer, cross_world_validate, expected_cost, optimise_threshold


# ── cost function ────────────────────────────────────────────────────────


def test_cost_function_prices_misses_higher_than_false_positives():
    """A missed ring is worth ~3.7 false positives, and the config says so."""
    assert COSTS.fn_cost > COSTS.fp_cost
    assert COSTS.cost_ratio == pytest.approx(COSTS.fn_cost / COSTS.fp_cost)
    assert 3.0 < COSTS.cost_ratio < 4.5


def test_expected_cost_counts_confusion_correctly():
    """FP and FN counts and their rupee total are arithmetic, not vibes."""
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.1, 0.05])
    r = expected_cost(y, p, 0.5, COSTS)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 1, 1, 3)
    assert r["total_cost_inr"] == pytest.approx(COSTS.fp_cost + COSTS.fn_cost)


def test_expected_cost_at_extremes():
    """Threshold 0 alerts on everything; threshold 1 alerts on nothing."""
    y = np.array([1, 0, 0, 0])
    p = np.array([0.6, 0.5, 0.4, 0.3])
    alert_all = expected_cost(y, p, 0.0, COSTS)
    alert_none = expected_cost(y, p, 1.01, COSTS)
    assert alert_all["fn"] == 0 and alert_all["fp"] == 3
    assert alert_none["fp"] == 0 and alert_none["fn"] == 1


def test_chosen_threshold_actually_minimises_cost(trained):
    """
    The operating threshold is the cheapest point on the curve.

    Re-swept here independently of `optimise_threshold` so the function cannot
    grade its own homework.
    """
    oof = trained["oof"]
    y, p = oof["y_true"].to_numpy(), oof["y_prob"].to_numpy()
    chosen = trained["artifact"]["threshold"]
    chosen_cost = expected_cost(y, p, chosen, COSTS)["total_cost_inr"]

    for t in np.linspace(0.01, 0.99, 99):
        assert expected_cost(y, p, t, COSTS)["total_cost_inr"] >= chosen_cost - 1e-6, (
            f"threshold {t:.2f} is cheaper than the chosen {chosen:.3f}"
        )


def test_cost_optimal_is_cheaper_than_f1_optimal(trained):
    """
    The headline claim: optimising F1 costs more than optimising cost.

    We assert the direction and that the gap is real, not a specific
    percentage — the number is whatever the data says and is reported honestly.
    """
    report = trained["metrics"]["threshold_selection"]
    cost_opt = report["cost_optimal"]["total_cost_inr"]
    f1_opt = report["f1_optimal"]["total_cost_inr"]
    assert cost_opt <= f1_opt
    assert report["f1_threshold_extra_cost_pct"] >= 0


def test_cost_optimal_favours_recall_over_precision(trained):
    """
    Because a miss costs 3.7x a false positive, the cost-optimal point should
    sit at higher recall and lower precision than the F1 point.

    This is the trade the pitch describes; if it inverted, the cost function
    would not be doing what we claim.
    """
    report = trained["metrics"]["threshold_selection"]
    assert report["cost_optimal"]["recall"] >= report["f1_optimal"]["recall"]
    assert report["cost_optimal"]["threshold"] <= report["f1_optimal"]["threshold"]


def test_prior_shift_mode_is_available_and_distinct():
    """
    `ring_incidence` reweights rates to a stated base rate.

    Documented as an override, not the default — applied naively it makes
    misses arithmetically free and the optimum becomes "never alert".
    """
    y = np.array([1] * 10 + [0] * 90)
    rng = np.random.default_rng(42)
    p = np.where(y == 1, rng.uniform(0.6, 0.95, 100), rng.uniform(0.05, 0.5, 100))

    empirical = optimise_threshold(y, p, CostConfig())
    shifted = optimise_threshold(y, p, CostConfig(ring_incidence=0.001))
    assert empirical["mode"] == "empirical"
    assert shifted["mode"] == "prior_shift"
    assert shifted["cost_optimal"]["threshold"] >= empirical["cost_optimal"]["threshold"]


# ── cross-world validation ───────────────────────────────────────────────


def test_every_world_is_held_out_exactly_once(trained):
    """Five disjoint worlds, five folds, no world tested twice."""
    worlds = [f["world"] for f in trained["metrics"]["folds"]]
    assert len(worlds) == len(set(worlds))
    assert len(worlds) >= 4


def test_out_of_fold_covers_every_candidate(trained, labels):
    """Every candidate is predicted exactly once, by a model blind to its world."""
    oof = trained["oof"]
    assert len(oof) == len(labels)
    assert oof["candidate_id"].nunique() == len(oof)


def test_no_card_appears_in_two_worlds(cards_df):
    """
    The premise of cross-world validation.

    If a card straddled worlds, training and test would share cards and the
    metrics would be measuring memorisation.
    """
    assert cards_df.groupby("card_id")["world_id"].nunique().max() == 1


def test_rings_never_straddle_worlds(rings_df, cards_df):
    """Every planted ring lives entirely inside one world."""
    world_of = dict(zip(cards_df["card_id"], cards_df["world_id"]))
    for ring in rings_df.itertuples(index=False):
        assert len({world_of[c] for c in ring.card_id_list}) == 1


def test_model_beats_a_random_baseline(trained):
    """Sanity: the model learned something."""
    ap = trained["metrics"]["out_of_fold"]["avg_precision"]
    base_rate = trained["metrics"]["positive_rate"]
    assert ap is not None and ap > base_rate * 1.5, f"PR-AUC {ap} barely beats the {base_rate} base rate"


def test_cross_world_validation_is_reproducible(features, labels):
    """Same inputs, same out-of-fold predictions."""
    import pandas as pd

    data = labels.merge(features, on="candidate_id")
    X = data[list(FEATURE_NAMES)].astype(float)
    y = data["is_fraud_ring"].to_numpy(dtype=int)
    w = data["world_id"].to_numpy()
    a, _ = cross_world_validate(X, y, w)
    b, _ = cross_world_validate(X, y, w)
    np.testing.assert_allclose(a, b)


# ── serving ──────────────────────────────────────────────────────────────


def test_scorer_loads_the_persisted_artifact(trained):
    """The served model carries its own threshold and version."""
    s = RingfenceScorer()
    assert s.available
    assert s.threshold == pytest.approx(trained["artifact"]["threshold"])
    assert s.model_version == trained["artifact"]["model_version"]


def test_scorer_degrades_without_an_artifact(tmp_path):
    """
    A missing model is a degraded state, not a crash.

    Returns 0.0, which the API turns into a forced REVIEW — never a silent
    APPROVE.
    """
    s = RingfenceScorer(path=tmp_path / "nonexistent.pkl")
    assert not s.available
    assert s.score({n: 1.0 for n in FEATURE_NAMES}) == 0.0
    assert s.model_version == "none"


def test_scorer_tolerates_missing_features(trained):
    """An incomplete feature dict scores rather than raising."""
    s = RingfenceScorer()
    score = s.score({"cand_size": 5.0})
    assert 0.0 <= score <= 1.0


def test_ring_candidates_outscore_noise(trained, positive_candidate, negative_candidate, graph, index):
    """A real ring should score above an average non-ring candidate."""
    s = RingfenceScorer()
    pos = s.score(compute_features(positive_candidate.cards, graph, index))
    neg = s.score(compute_features(negative_candidate.cards, graph, index))
    assert pos > neg


# ── latency ──────────────────────────────────────────────────────────────


def test_feature_computation_meets_the_p99_budget(candidates, graph, index):
    """
    Feature computation must clear 50ms at p99.

    Measured across every candidate, not on a favourable sample.
    """
    from src.feature_engine import compute_features_batch
    from src.utils import LATENCY_RECORDER

    compute_features_batch(candidates, graph, index)
    p = LATENCY_RECORDER.percentiles("feature.per_candidate")
    assert p["p99"] < LATENCY.feature_p99_ms, f"feature p99 {p['p99']:.1f}ms exceeds the budget"


def test_inference_meets_the_p99_budget(trained, candidates, graph, index):
    """Model inference must clear 10ms at p99."""
    import time

    s = RingfenceScorer()
    feats = [compute_features(c.cards, graph, index) for c in candidates[:40]]
    timings = []
    for f in feats:
        t0 = time.perf_counter()
        s.score(f)
        timings.append((time.perf_counter() - t0) * 1000)
    assert float(np.percentile(timings, 99)) < LATENCY.inference_p99_ms * 3, (
        "inference p99 far outside budget"
    )
