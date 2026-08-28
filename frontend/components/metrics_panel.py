"""
Page 4 — Metrics.

Honest numbers, computed from the persisted out-of-fold predictions rather than
retyped from a README. Four blocks:

  - Ring recall vs candidate precision, and what each generator contributes
  - The cost curve, with the cost-optimal and F1-optimal points both marked
  - Feature importance by LightGBM gain
  - Latency percentiles against the stated budget

The cost curve is the page that answers "your precision is low". It is low on
purpose, and this chart is the argument: at ₹2.3L per false positive and ₹8.5L
per miss, the cheapest place to stand is not where F1 peaks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import COSTS, LATENCY


def _cost_curve(oof: pd.DataFrame) -> tuple[go.Figure, dict]:
    """Sweep the threshold over out-of-fold predictions and plot cost vs F1."""
    from src.scorer import expected_cost
    from sklearn.metrics import precision_recall_fscore_support

    y = oof["y_true"].to_numpy(dtype=int)
    p = oof["y_prob"].to_numpy(dtype=float)
    grid = np.linspace(0.01, 0.99, 199)

    costs, f1s, precisions, recalls = [], [], [], []
    for t in grid:
        costs.append(expected_cost(y, p, t, COSTS)["total_cost_inr"])
        pr, rc, f1, _ = precision_recall_fscore_support(
            y, (p >= t).astype(int), average="binary", zero_division=0)
        f1s.append(f1)
        precisions.append(pr)
        recalls.append(rc)

    costs = np.asarray(costs)
    f1s = np.asarray(f1s)
    i_cost, i_f1 = int(np.argmin(costs)), int(np.argmax(f1s))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=costs / 1e5, name="expected cost (₹ lakh)",
                             line=dict(color="#f87171", width=3), yaxis="y"))
    fig.add_trace(go.Scatter(x=grid, y=f1s, name="F1", line=dict(color="#38bdf8", width=2, dash="dot"), yaxis="y2"))
    fig.add_trace(go.Scatter(x=grid, y=precisions, name="precision",
                             line=dict(color="#a78bfa", width=1.5), yaxis="y2", opacity=0.7))
    fig.add_trace(go.Scatter(x=grid, y=recalls, name="recall",
                             line=dict(color="#4ade80", width=1.5), yaxis="y2", opacity=0.7))

    fig.add_trace(go.Scatter(
        x=[grid[i_cost]], y=[costs[i_cost] / 1e5], mode="markers+text",
        marker=dict(size=16, color="#fbbf24", symbol="star", line=dict(width=1, color="#0f172a")),
        text=["  cost-optimal ←we operate here"], textposition="middle right",
        name="cost-optimal", yaxis="y"))
    fig.add_trace(go.Scatter(
        x=[grid[i_f1]], y=[costs[i_f1] / 1e5], mode="markers+text",
        marker=dict(size=13, color="#94a3b8", symbol="x"),
        text=["  F1-optimal"], textposition="middle right", name="F1-optimal", yaxis="y"))

    fig.update_layout(
        height=430, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="decision threshold"),
        yaxis=dict(title="expected cost (₹ lakh)", side="left"),
        yaxis2=dict(title="F1 / precision / recall", overlaying="y", side="right", range=[0, 1.02]),
        legend=dict(orientation="h", y=1.16, x=0),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )

    delta = costs[i_f1] - costs[i_cost]
    summary = {
        "cost_threshold": float(grid[i_cost]), "cost_at_cost_opt": float(costs[i_cost]),
        "f1_threshold": float(grid[i_f1]), "cost_at_f1_opt": float(costs[i_f1]),
        "delta_inr": float(delta),
        "delta_pct": float(100 * delta / costs[i_cost]) if costs[i_cost] > 0 else 0.0,
        "precision_at_cost_opt": float(precisions[i_cost]), "recall_at_cost_opt": float(recalls[i_cost]),
        "precision_at_f1_opt": float(precisions[i_f1]), "recall_at_f1_opt": float(recalls[i_f1]),
    }
    return fig, summary


def render_metrics_page(backend, artifacts: dict) -> None:
    """Render the metrics dashboard page."""
    st.title("Metrics")
    st.caption("Everything below is computed from persisted out-of-fold predictions — "
               "leave-one-world-out, zero card overlap between train and test.")

    metrics = artifacts.get("metrics") or {}
    cand_metrics = artifacts.get("candidate_metrics") or {}
    oof = artifacts.get("oof")

    if not metrics:
        st.error("No metrics found. Run `make all` first.")
        return

    # ── headline ────────────────────────────────────────────────────────
    ts = metrics.get("threshold_selection", {})
    cost_opt = ts.get("cost_optimal", {})
    f1_opt = ts.get("f1_optimal", {})
    oof_metrics = metrics.get("out_of_fold", {})

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Ring recall", f"{cand_metrics.get('ring_recall', 0):.3f}",
              help=f"{cand_metrics.get('n_rings', 0)} planted rings, "
                   f">50% card overlap with at least one candidate")
    k2.metric("Precision @ cost-optimal", f"{cost_opt.get('precision', 0):.3f}")
    k3.metric("Recall @ cost-optimal", f"{cost_opt.get('recall', 0):.3f}")
    k4.metric("PR-AUC (out-of-fold)", f"{oof_metrics.get('avg_precision') or 0:.3f}")
    k5.metric("F1-threshold cost penalty", f"{ts.get('f1_threshold_extra_cost_pct', 0):+.0f}%",
              help="How much more it would cost to operate at the F1-maximising threshold")

    st.divider()

    # ── candidate generation ────────────────────────────────────────────
    st.subheader("Candidate generation — recall first")
    c1, c2 = st.columns([1, 1])
    with c1:
        by_gen = cand_metrics.get("recall_by_generator_alone", {})
        if by_gen:
            fig = go.Figure(go.Bar(
                x=[v for v in by_gen.values()] + [cand_metrics.get("ring_recall", 0)],
                y=[f"{k} alone" for k in by_gen] + ["<b>all three (ensembled)</b>"],
                orientation="h",
                marker_color=["#64748b"] * len(by_gen) + ["#fbbf24"],
                text=[f"{v:.2f}" for v in by_gen.values()] + [f"{cand_metrics.get('ring_recall', 0):.2f}"],
                textposition="outside",
            ))
            fig.update_layout(height=250, margin=dict(l=10, r=40, t=10, b=10),
                              xaxis=dict(range=[0, 1.1], title="ring recall"),
                              template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The identity-ego generator carries recall; Louvain and connected "
                       "components add coverage where the ego sets fragment.")
    with c2:
        size = cand_metrics.get("candidate_size", {})
        st.markdown(f"""
        **{cand_metrics.get('n_candidates', 0):,} candidates** proposed for
        **{cand_metrics.get('n_rings', 0)} planted rings**.

        | | |
        |---|---|
        | Candidate size p50 | {size.get('p50', 0):.0f} cards |
        | Candidate size p95 | {size.get('p95', 0):.0f} cards |
        | Candidate size max | {size.get('max', 0)} cards |
        | Positive rate | {metrics.get('positive_rate', 0):.1%} |
        """)
        missed = cand_metrics.get("rings_missed", [])
        if missed:
            st.caption(f"Missed rings: `{'`, `'.join(missed[:6])}`. These are rings whose members "
                       f"mostly used their own devices — no identity edge ever formed. "
                       f"That is the honest ceiling, not a bug.")

    st.divider()

    # ── cost curve ──────────────────────────────────────────────────────
    st.subheader("Cost curve — why the threshold is not where F1 peaks")
    if oof is not None and len(oof):
        fig, summary = _cost_curve(oof)
        st.plotly_chart(fig, use_container_width=True)
        a, b = st.columns(2)
        a.markdown(f"""
        **Cost-optimal — where we operate**
        - threshold `{summary['cost_threshold']:.3f}`
        - precision `{summary['precision_at_cost_opt']:.3f}` · recall `{summary['recall_at_cost_opt']:.3f}`
        - expected cost **₹{summary['cost_at_cost_opt'] / 1e5:,.1f}L**
        """)
        b.markdown(f"""
        **F1-optimal — where a naive tuner would stop**
        - threshold `{summary['f1_threshold']:.3f}`
        - precision `{summary['precision_at_f1_opt']:.3f}` · recall `{summary['recall_at_f1_opt']:.3f}`
        - expected cost **₹{summary['cost_at_f1_opt'] / 1e5:,.1f}L** ({summary['delta_pct']:+.0f}%)
        """)
        st.info(
            f"C_fp = ₹{COSTS.fp_cost:,.0f} · C_fn = ₹{COSTS.fn_cost:,.0f} — a miss is worth "
            f"{COSTS.cost_ratio:.1f} false positives. We would rather review "
            f"{COSTS.cost_ratio:.0f} extra cases than miss one ring, and the curve says so in rupees."
        )
    else:
        st.warning("`oof_predictions.csv` not found — re-run `make train`.")

    st.divider()

    # ── per-fold ────────────────────────────────────────────────────────
    st.subheader("Cross-world validation — per fold")
    folds = metrics.get("folds", [])
    if folds:
        st.dataframe(pd.DataFrame(folds), use_container_width=True, hide_index=True)
        st.caption("Each fold trains on four worlds and tests on the fifth. Worlds share zero cards, "
                   "so no ring straddles the split. One cherry-picked fold proves nothing — all five are here.")

    # ── feature importance ──────────────────────────────────────────────
    st.subheader("Feature importance (LightGBM gain)")
    imp = metrics.get("feature_importance_gain", [])[:18]
    if imp:
        from src.feature_engine import FEATURE_FAMILIES

        family_of = {f: fam for fam, names in FEATURE_FAMILIES.items() for f in names}
        colour = {"structural": "#38bdf8", "identity": "#a78bfa", "behavioural": "#fbbf24"}
        fig = go.Figure(go.Bar(
            x=[d["gain"] for d in reversed(imp)],
            y=[d["feature"] for d in reversed(imp)],
            orientation="h",
            marker_color=[colour.get(family_of.get(d["feature"], ""), "#64748b") for d in reversed(imp)],
        ))
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(title="gain"), template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔵 structural · 🟣 identity · 🟡 behavioural — "
                   "identity finds the cluster, behaviour decides whether it is a ring.")

    # ── latency ─────────────────────────────────────────────────────────
    st.subheader("Latency")
    lat = artifacts.get("feature_latency") or {}
    api_metrics = backend.get("/metrics")
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Feature p50", f"{lat.get('p50', 0):.1f} ms")
    l2.metric("Feature p95", f"{lat.get('p95', 0):.1f} ms")
    l3.metric("Feature p99", f"{lat.get('p99', 0):.1f} ms",
              delta=f"budget {LATENCY.feature_p99_ms:.0f} ms",
              delta_color="normal" if lat.get("p99", 0) <= LATENCY.feature_p99_ms else "inverse")
    l4.metric("Samples", int(lat.get("n", 0)))
    if lat:
        st.progress(min(lat.get("p99", 0) / LATENCY.feature_p99_ms, 1.0),
                    text=f"feature p99 is {lat.get('p99', 0):.1f} ms of the {LATENCY.feature_p99_ms:.0f} ms budget")
    if api_metrics:
        with st.expander("Live API metrics (Prometheus format)"):
            st.code(api_metrics if isinstance(api_metrics, str) else str(api_metrics), language="text")

    # ── merchant edge rule ──────────────────────────────────────────────
    stats = (artifacts.get("collision_stats") or {}).get("merchant", {})
    if stats:
        st.divider()
        st.subheader("Why there are no merchant edges")
        m1, m2, m3 = st.columns(3)
        m1.metric("Card pairs sharing a merchant", f"{stats.get('card_pairs_sharing_a_merchant', 0):,}")
        m2.metric("Card pairs sharing an identity", f"{stats.get('card_pairs_sharing_an_identity', 0):,}")
        m3.metric("Merchant pair saturation", f"{stats.get('merchant_pair_saturation', 0):.1%}")
        st.caption(stats.get("verdict", ""))
