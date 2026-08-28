"""
ML scorer — LightGBM, cross-world validation, cost-optimal threshold.

Two things here are load-bearing, and both are places most submissions cut a
corner.

**Cross-world validation, not random k-fold.** Cards belong to exactly one of
five disjoint worlds. Folds are whole worlds: train on four, test on the fifth.
A random split would put two cards from the same ring — sharing a device, an
IP, and a behavioural signature — on both sides of the split, and the resulting
"accuracy" would be measuring memorisation. Grouping by world means every test
candidate comes from a population the model has never seen a single card from.

**Cost picks the threshold, not F1.** F1 assumes a false positive and a false
negative hurt equally. In payments they do not, and not by a small factor:

    C_fp = INR 2.3L   analyst review + merchant friction + false-decline damage
    C_fn = INR 8.5L   ring cashout + chargebacks + merchant churn

A miss costs ~3.7 false positives. Optimising F1 lands you at a threshold that
is too conservative and quietly more expensive; we report both thresholds and
the rupee gap between them, whatever that gap turns out to be.

A note on `ring_incidence`
--------------------------
The spec exposes P(ring) in the population as a config knob. Applied naively —
multiplying the FN term by 0.001 — the cost function degenerates: the optimum
becomes "never alert", because a rate that low makes misses arithmetically
free. That is a real trap, not a subtlety. The candidate population is already
heavily filtered by the graph, so its positive rate is orders of magnitude
above the population rate. We therefore default to the **empirical** out-of-fold
distribution, and treat `ring_incidence` as an explicit prior-shift override for
when the deployed candidate mix is known to differ. Both paths are implemented;
the default is the honest one.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from src.config import (
    COSTS,
    FEATURES_CSV,
    METRICS_JSON,
    MODEL_PATH,
    MODEL_VERSION,
    PROCESSED_DIR,
    RANDOM_SEED,
    SIGNATURE_SCALER_PATH,
    CostConfig,
)
from src.feature_engine import FEATURE_FAMILIES, FEATURE_NAMES
from src.utils import get_logger, timed, write_json

log = get_logger("ringfence.scorer")

LGB_PARAMS = {
    "objective": "binary",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 5,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
    "verbose": -1,
}


def fit_signature_scaler(X: pd.DataFrame, path=SIGNATURE_SCALER_PATH) -> dict:
    """
    Fit and persist the population statistics case memory uses to standardise
    behavioural signatures.

    Why this lives in the trainer: it is a *population* statistic, and the
    candidate population is exactly what the trainer already has in front of it.
    Fitting it at query time from whatever happens to be in the case store would
    make similarity depend on how many cases you had stored, which is not a
    property anyone wants.

    Without it, cosine similarity between any two candidates comes out around
    0.96 — every one of the 31 features is non-negative, so the whole cloud sits
    in a single orthant. Standardising recentres it so the metric discriminates.
    See `case_memory.signature_from_features`.
    """
    from src.case_memory import compress_features

    compressed = np.vstack([compress_features(row) for row in X.to_dict("records")])
    mean = compressed.mean(axis=0)
    std = compressed.std(axis=0)

    payload = {
        "feature_names": list(FEATURE_NAMES),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_samples": int(len(X)),
        # Cheap fingerprint so a stale case store can be detected after retraining.
        "fingerprint": f"{MODEL_VERSION}:{len(X)}:{float(mean.sum()):.6f}",
        "fitted_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(path, payload)

    # Report how well the standardisation actually separates. This is the number
    # that was 0.964 before the fix, which made case memory useless.
    scaled = (compressed - mean) / np.where(std > 1e-9, std, 1.0)
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    unit = scaled / np.where(norms > 0, norms, 1.0)
    sim = unit @ unit.T
    np.fill_diagonal(sim, np.nan)
    flat = sim[~np.isnan(sim)]
    log.info(
        f"signature scaler fitted on {len(X):,} candidates -> pairwise cosine "
        f"p50={np.nanpercentile(flat, 50):.3f} p95={np.nanpercentile(flat, 95):.3f}, "
        f"{(flat >= 0.80).mean():.2%} of pairs above the 0.80 damp threshold"
    )
    return payload


def _make_model(scale_pos_weight: float):
    """
    Build the classifier.

    LightGBM, not a GNN. With 120 rings and 31 hand-specified features, a graph
    neural network has nothing to learn from that the graph has not already
    told us, and it would cost us the per-feature attribution the dossier needs.
    The graph does candidate generation; the GBM does ranking.
    """
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**LGB_PARAMS, scale_pos_weight=scale_pos_weight)


# ──────────────────────────────────────────────────────────────────────────
# Threshold selection
# ──────────────────────────────────────────────────────────────────────────


def expected_cost(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, costs: CostConfig = COSTS
) -> dict:
    """
    Expected rupee cost of operating at `threshold`.

    Empirical mode (default): raw FP and FN counts times their unit costs,
    reported both in total and per 1,000 candidates reviewed.

    Prior-shift mode (`ring_incidence` set): rates are reweighted to a stated
    population base rate, which is what you want if the deployed candidate mix
    differs from the offline one.
    """
    pred = (y_prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())

    n_pos = max(tp + fn, 1)
    n_neg = max(fp + tn, 1)
    fpr = fp / n_neg
    fnr = fn / n_pos

    if costs.ring_incidence is None:
        total = fp * costs.fp_cost + fn * costs.fn_cost
        per_1k = total / max(len(y_true), 1) * 1000
    else:
        inc = costs.ring_incidence
        per_candidate = (1 - inc) * fpr * costs.fp_cost + inc * fnr * costs.fn_cost
        total = per_candidate * len(y_true)
        per_1k = per_candidate * 1000

    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fp_rate": round(fpr, 6),
        "fn_rate": round(fnr, 6),
        "total_cost_inr": float(total),
        "cost_per_1k_candidates_inr": float(per_1k),
    }


def optimise_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, costs: CostConfig = COSTS, n_steps: int = 501
) -> dict:
    """
    Sweep the threshold and return the cost-minimising point, the
    F1-maximising point, and the rupee difference between them.

    Both points come from the same out-of-fold predictions, so the comparison
    is apples to apples. We report whatever gap the data produces — including
    if it turns out to be small.
    """
    grid = np.linspace(0.01, 0.99, n_steps)

    costed = [expected_cost(y_true, y_prob, t, costs) for t in grid]
    best_cost = min(costed, key=lambda r: r["total_cost_inr"])

    f1s = []
    for t in grid:
        pred = (y_prob >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        f1s.append((f, t, p, r))
    best_f1_f, best_f1_t, best_f1_p, best_f1_r = max(f1s, key=lambda x: (x[0], -x[1]))
    f1_point_cost = expected_cost(y_true, y_prob, best_f1_t, costs)

    delta = f1_point_cost["total_cost_inr"] - best_cost["total_cost_inr"]
    delta_pct = 100.0 * delta / best_cost["total_cost_inr"] if best_cost["total_cost_inr"] > 0 else 0.0

    p_c, r_c, f_c, _ = precision_recall_fscore_support(
        y_true, (y_prob >= best_cost["threshold"]).astype(int), average="binary", zero_division=0
    )

    result = {
        "cost_optimal": {
            **best_cost,
            "precision": round(float(p_c), 4),
            "recall": round(float(r_c), 4),
            "f1": round(float(f_c), 4),
        },
        "f1_optimal": {
            **f1_point_cost,
            "precision": round(float(best_f1_p), 4),
            "recall": round(float(best_f1_r), 4),
            "f1": round(float(best_f1_f), 4),
        },
        "f1_threshold_extra_cost_inr": float(delta),
        "f1_threshold_extra_cost_pct": round(float(delta_pct), 2),
        "cost_config": costs.model_dump(),
        "mode": "prior_shift" if costs.ring_incidence is not None else "empirical",
    }

    log.info(
        f"COST-OPTIMAL threshold = {best_cost['threshold']:.3f}  "
        f"P={p_c:.3f} R={r_c:.3f} F1={f_c:.3f}  cost=INR {best_cost['total_cost_inr']:,.0f}"
    )
    log.info(
        f"F1-OPTIMAL   threshold = {best_f1_t:.3f}  "
        f"P={best_f1_p:.3f} R={best_f1_r:.3f} F1={best_f1_f:.3f}  "
        f"cost=INR {f1_point_cost['total_cost_inr']:,.0f}"
    )
    log.info(f"Operating at the F1 threshold would cost {delta_pct:+.1f}% more.")
    return result


# ──────────────────────────────────────────────────────────────────────────
# Cross-world validation
# ──────────────────────────────────────────────────────────────────────────


def cross_world_validate(
    X: pd.DataFrame, y: np.ndarray, worlds: np.ndarray, costs: CostConfig = COSTS
) -> tuple[np.ndarray, list[dict]]:
    """
    Leave-one-world-out validation. Returns out-of-fold probabilities and
    per-fold metrics.

    Every candidate is predicted exactly once, by a model that never saw its
    world. That out-of-fold vector is what every metric in this repo is
    computed from — there is no separate "held-out set" that quietly got peeked
    at during threshold tuning.
    """
    oof = np.zeros(len(y), dtype=float)
    fold_metrics: list[dict] = []

    for world in sorted(set(worlds.tolist())):
        test_mask = worlds == world
        train_mask = ~test_mask
        if test_mask.sum() == 0 or y[train_mask].sum() == 0:
            log.warning(f"world {world}: skipped (no test rows or no positives in train)")
            continue

        n_pos = max(int(y[train_mask].sum()), 1)
        n_neg = max(int((y[train_mask] == 0).sum()), 1)
        model = _make_model(scale_pos_weight=n_neg / n_pos)
        model.fit(X[train_mask], y[train_mask])

        prob = model.predict_proba(X[test_mask])[:, 1]
        oof[test_mask] = prob

        p, r, f, _ = precision_recall_fscore_support(
            y[test_mask], (prob >= 0.5).astype(int), average="binary", zero_division=0
        )
        try:
            auc = float(roc_auc_score(y[test_mask], prob)) if len(set(y[test_mask])) > 1 else float("nan")
            ap = float(average_precision_score(y[test_mask], prob)) if len(set(y[test_mask])) > 1 else float("nan")
        except ValueError:  # pragma: no cover - degenerate folds only
            auc = ap = float("nan")

        fold_metrics.append({
            "world": int(world),
            "n_test": int(test_mask.sum()),
            "n_test_positive": int(y[test_mask].sum()),
            "precision@0.5": round(float(p), 4),
            "recall@0.5": round(float(r), 4),
            "f1@0.5": round(float(f), 4),
            "roc_auc": round(auc, 4) if auc == auc else None,
            "avg_precision": round(ap, 4) if ap == ap else None,
        })
        log.info(f"  fold world={world}: n={test_mask.sum()} pos={int(y[test_mask].sum())} "
                 f"P={p:.3f} R={r:.3f} F1={f:.3f} AP={ap:.3f}")

    return oof, fold_metrics


def ablation_study(X: pd.DataFrame, y: np.ndarray, worlds: np.ndarray, costs: CostConfig = COSTS) -> list[dict]:
    """
    Retrain with each feature family removed and report the cost delta.

    Used by `notebooks/02_ablation.ipynb`. Read the deltas with suspicion: with
    120 rings the run-to-run variance is comparable to the effect sizes, which
    is listed openly under Honest Limitations rather than buried.
    """
    results = []
    baseline_oof, _ = cross_world_validate(X, y, worlds, costs)
    baseline = optimise_threshold(y, baseline_oof, costs)["cost_optimal"]
    base_ap = float(average_precision_score(y, baseline_oof)) if len(set(y)) > 1 else float("nan")
    results.append({"removed": "nothing (baseline)", "n_features": X.shape[1],
                    "avg_precision": round(base_ap, 4), "cost_inr": baseline["total_cost_inr"]})

    for family, names in FEATURE_FAMILIES.items():
        keep = [c for c in X.columns if c not in names]
        oof, _ = cross_world_validate(X[keep], y, worlds, costs)
        ap = float(average_precision_score(y, oof)) if len(set(y)) > 1 else float("nan")
        best = optimise_threshold(y, oof, costs)["cost_optimal"]
        results.append({
            "removed": family,
            "n_features": len(keep),
            "avg_precision": round(ap, 4),
            "cost_inr": best["total_cost_inr"],
            "cost_delta_vs_baseline_inr": best["total_cost_inr"] - baseline["total_cost_inr"],
        })
        log.info(f"ablation without {family}: AP={ap:.4f} cost=INR {best['total_cost_inr']:,.0f}")
    return results


# ──────────────────────────────────────────────────────────────────────────
# Train / persist / serve
# ──────────────────────────────────────────────────────────────────────────


def train(
    features: pd.DataFrame, labels: pd.DataFrame, costs: CostConfig = COSTS
) -> tuple[dict, dict, pd.DataFrame]:
    """
    Full training run: cross-world CV, threshold selection, final refit.

    Returns (artifact, metrics, out_of_fold). The artifact is what gets pickled
    and served; it carries the threshold with the model, so the API can never
    accidentally serve a model at a threshold it was not tuned for.

    The out-of-fold frame is returned and persisted so that the cost curve on
    the dashboard and in `notebooks/03_cost_curve.ipynb` is drawn from the same
    predictions the threshold was chosen on, rather than being re-derived from
    a fresh fit that would quietly disagree.
    """
    data = labels.merge(features, on="candidate_id", how="inner")
    X = data[list(FEATURE_NAMES)].astype(float)
    y = data["is_fraud_ring"].to_numpy(dtype=int)
    worlds = data["world_id"].to_numpy()

    log.info(f"training on {len(data):,} candidates, {int(y.sum()):,} positive "
             f"({100 * y.mean():.2f}%), across {len(set(worlds.tolist()))} worlds")

    # Fit the case-memory signature scaler on the same candidate population the
    # model trains on, before anything queries case memory.
    scaler = fit_signature_scaler(X)

    with timed("scorer.cross_world_cv", log):
        oof, fold_metrics = cross_world_validate(X, y, worlds, costs)

    threshold_report = optimise_threshold(y, oof, costs)
    chosen = threshold_report["cost_optimal"]["threshold"]

    with timed("scorer.final_fit", log):
        n_pos, n_neg = max(int(y.sum()), 1), max(int((y == 0).sum()), 1)
        model = _make_model(scale_pos_weight=n_neg / n_pos)
        model.fit(X, y)

    importance = sorted(
        ({"feature": f, "gain": float(g)} for f, g in zip(FEATURE_NAMES, model.booster_.feature_importance("gain"))),
        key=lambda d: -d["gain"],
    )
    log.info("top features by gain: " + ", ".join(f"{d['feature']}={d['gain']:.0f}" for d in importance[:8]))

    overall_p, overall_r, overall_f, _ = precision_recall_fscore_support(
        y, (oof >= chosen).astype(int), average="binary", zero_division=0
    )

    metrics = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_candidates": int(len(data)),
        "n_positive": int(y.sum()),
        "positive_rate": round(float(y.mean()), 4),
        "validation": "leave-one-world-out (5 disjoint worlds, zero card overlap)",
        "folds": fold_metrics,
        "out_of_fold": {
            "roc_auc": round(float(roc_auc_score(y, oof)), 4) if len(set(y)) > 1 else None,
            "avg_precision": round(float(average_precision_score(y, oof)), 4) if len(set(y)) > 1 else None,
            "precision_at_operating_threshold": round(float(overall_p), 4),
            "recall_at_operating_threshold": round(float(overall_r), 4),
            "f1_at_operating_threshold": round(float(overall_f), 4),
        },
        "threshold_selection": threshold_report,
        "feature_importance_gain": importance,
    }

    artifact = {
        "model": model,
        "threshold": float(chosen),
        "feature_names": list(FEATURE_NAMES),
        "model_version": MODEL_VERSION,
        "trained_at": metrics["trained_at"],
        "cost_config": costs.model_dump(),
        "signature_scaler_fingerprint": scaler["fingerprint"],
        "metrics": metrics,
    }

    oof_df = pd.DataFrame({
        "candidate_id": data["candidate_id"],
        "world_id": worlds,
        "size": data["size"],
        "source": data["source"],
        "best_ring_id": data["best_ring_id"].fillna(""),
        "y_true": y,
        "y_prob": oof,
    })
    return artifact, metrics, oof_df


def save_artifact(artifact: dict, path=MODEL_PATH) -> None:
    """Persist the model bundle (model + threshold + provenance) to disk."""
    joblib.dump(artifact, path)
    log.info(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB), threshold={artifact['threshold']:.3f}")


class RingfenceScorer:
    """
    Serving wrapper around the trained artifact.

    Loaded once at API start-up. `available` is False when no model has been
    trained yet — callers must degrade to a safe REVIEW rather than crash, and
    the API does exactly that.
    """

    def __init__(self, path=MODEL_PATH):
        self.path = path
        self.artifact: dict | None = None
        try:
            self.artifact = joblib.load(path)
            # Warm the predictor. LightGBM's first predict_proba pays a one-off
            # initialisation cost of ~100ms, which would otherwise land on the
            # first real request and blow the p99 budget we quote.
            self.score({})
            log.info(f"loaded model {self.artifact['model_version']} "
                     f"(threshold {self.artifact['threshold']:.3f}, trained {self.artifact['trained_at']})")
        except (FileNotFoundError, OSError, KeyError) as exc:
            log.warning(f"no usable model at {path} ({exc}); scorer running in degraded mode")

    @property
    def available(self) -> bool:
        """True when a model is loaded and can be scored against."""
        return self.artifact is not None

    @property
    def threshold(self) -> float:
        """The cost-optimal operating threshold this model was tuned to."""
        return float(self.artifact["threshold"]) if self.artifact else 0.5

    @property
    def model_version(self) -> str:
        """Version string of the loaded model, or 'none' when degraded."""
        return self.artifact["model_version"] if self.artifact else "none"

    def score(self, features: dict[str, float]) -> float:
        """
        Score one feature dict. Returns 0.0 in degraded mode — combined with the
        API's safe default that produces a REVIEW, never a silent APPROVE.
        """
        if not self.artifact:
            return 0.0
        names = self.artifact["feature_names"]
        # A DataFrame, not a bare array: LightGBM was fitted with feature names
        # and passing positional data makes it warn on every single request.
        row = pd.DataFrame([[float(features.get(n, 0.0)) for n in names]], columns=names)
        return float(self.artifact["model"].predict_proba(row)[0, 1])

    def score_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """Score a DataFrame of feature rows."""
        if not self.artifact:
            return np.zeros(len(features_df))
        X = features_df[self.artifact["feature_names"]].astype(float)
        return self.artifact["model"].predict_proba(X)[:, 1]


def main() -> None:
    """CLI: train the scorer from the persisted feature store and save it."""
    parser = argparse.ArgumentParser(description="Train the Ringfence scorer")
    parser.add_argument("--fp-cost", type=float, default=COSTS.fp_cost)
    parser.add_argument("--fn-cost", type=float, default=COSTS.fn_cost)
    parser.add_argument("--ring-incidence", type=float, default=None,
                        help="override P(ring); omit to use the empirical distribution")
    args = parser.parse_args()

    costs = CostConfig(fp_cost=args.fp_cost, fn_cost=args.fn_cost, ring_incidence=args.ring_incidence)

    features = pd.read_csv(FEATURES_CSV)
    labels = pd.read_csv(PROCESSED_DIR / "candidate_labels.csv")

    artifact, metrics, oof = train(features, labels, costs)
    save_artifact(artifact)
    write_json(METRICS_JSON, metrics)
    oof.to_csv(PROCESSED_DIR / "oof_predictions.csv", index=False)
    log.info(f"wrote {METRICS_JSON} and oof_predictions.csv")


if __name__ == "__main__":
    main()
