"""
Page 5 — Failure Recovery.

This page exists to hit one bar: *show the audit trail and one failure handled
gracefully*.

It walks a single false positive end to end:

  1. The candidate that fired — a household sharing one tablet, which is
     structurally indistinguishable from a small ring
  2. The analyst marking it a false positive, with notes
  3. Case memory storing the 31-dimensional signature and the verdict
  4. The same pattern scored again, damped 20%, and tagged
  5. The audit trail, verbatim from SQLite

Nothing here is staged. The buttons write to the real case-memory database and
the second score is a real second scoring pass. If you clear
`data/processed/case_memory.sqlite`, this page resets and you can run it again.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import CASE_MEMORY


def _pick_false_positive_candidate(artifacts: dict) -> dict | None:
    """
    Find a good false-positive exhibit: a candidate the model scores highly
    that is NOT actually a planted ring.

    A real false positive, chosen by the labels — not a cherry-picked example
    written into the demo by hand. If there are none, we say so rather than
    inventing one.
    """
    labels, oof = artifacts.get("labels"), artifacts.get("oof")
    candidates = artifacts.get("candidates") or []
    if oof is None or not candidates:
        return None

    negatives = oof[oof["y_true"] == 0].sort_values("y_prob", ascending=False)
    if negatives.empty:
        return None

    top = negatives.iloc[0]
    cand = next((c for c in candidates if c.candidate_id == top["candidate_id"]), None)
    if cand is None:
        return None
    return {"candidate": cand, "score": float(top["y_prob"]), "world": int(top["world_id"])}


def _score_gauge(before: float, after: float) -> go.Figure:
    """Before/after composite score, as a two-bar comparison."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Before analyst feedback", "After analyst feedback"],
        y=[before, after],
        marker_color=["#f87171", "#4ade80"],
        text=[f"{before:.3f}", f"{after:.3f}"], textposition="outside",
        width=[0.45, 0.45],
    ))
    fig.add_hline(y=0.7, line=dict(color="#f87171", dash="dash"),
                  annotation_text="BLOCK threshold (0.70)", annotation_position="right")
    fig.add_hline(y=0.4, line=dict(color="#fbbf24", dash="dash"),
                  annotation_text="REVIEW threshold (0.40)", annotation_position="right")
    fig.update_layout(
        height=340, margin=dict(l=10, r=140, t=30, b=10),
        yaxis=dict(range=[0, 1.08], title="composite score"),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    return fig


def render_failure_recovery(backend, artifacts: dict, badge) -> None:
    """Render the failure recovery demo page."""
    st.title("Failure Recovery")
    st.markdown(
        "> **Most detectors score, alert, and forget.** They make the same mistake tomorrow that an "
        "analyst corrected today. This page follows one false positive from the alert queue into "
        "case memory and back out again, with the score moving."
    )

    from src.case_memory import CaseMemory, DISPOSITION_FALSE_POSITIVE
    from src.feature_engine import compute_candidate_meta, compute_features
    from src.vulcan_integration import VulcanComposer

    G, index, scorer = artifacts.get("graph"), artifacts.get("index"), artifacts.get("scorer")
    if G is None or index is None:
        st.error("Pipeline artifacts unavailable. Run `make all`.")
        return

    exhibit = _pick_false_positive_candidate(artifacts)
    if exhibit is None:
        st.warning("No labelled false positive available. Run `make all` to build the pipeline.")
        return

    cand = exhibit["candidate"]
    cards = list(cand.cards)
    memory = CaseMemory()
    composer = VulcanComposer()

    feats = compute_features(cards, G, index)
    meta = compute_candidate_meta(cards, index)
    rf_score = scorer.score(feats) if scorer and scorer.available else 0.0
    vulcan = float(meta.get("vulcan_cashout_mean") or meta.get("vulcan_mean", 0.0))

    already_stored = memory.get_case(cand.candidate_id)
    precedents, fp_patterns, boost, damp = memory.adjustments(feats)

    before = composer.compute_composite(vulcan, rf_score, 0.0, 0.0)
    after = composer.compute_composite(vulcan, rf_score, boost, damp)

    # ── Step 1 ───────────────────────────────────────────────────────────
    st.subheader("1 · The alert that was wrong")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidate", cand.candidate_id)
    c2.metric("Cards", len(cards))
    c3.metric("Ringfence score", f"{rf_score:.3f}")
    with c4:
        st.markdown("<div class='rf-kpi-label'>Original call</div>", unsafe_allow_html=True)
        st.markdown(badge(before["recommendation"]), unsafe_allow_html=True)

    st.error(
        f"**This is not a ring.** Ground truth says these {len(cards)} cards are unrelated — "
        f"they share infrastructure because of a household device, an ISP NAT pool, or the stock "
        f"user-agent. Ringfence scored it {rf_score:.3f} anyway. "
        f"At ₹2.3L per false positive, this is a real cost we own."
    )
    st.markdown("**Cards:** " + " ".join(f"`{c}`" for c in cards[:24]) + (" …" if len(cards) > 24 else ""))

    with st.expander("Why it looked like a ring — the features that fired"):
        interesting = ["cand_size", "cand_density", "cand_ego_overlap_ratio", "max_device_reuse",
                       "probe_fraction", "burst_compression", "amount_escalation", "cashout_fraction",
                       "small_merchant_concentration"]
        st.dataframe(
            pd.DataFrame([{"feature": k, "value": round(feats[k], 4)} for k in interesting]),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Note `burst_compression` and `probe_fraction`: a family buying four recharges in three "
            "minutes fires the same features as a probe burst. What separates them is "
            "`amount_escalation` and `cashout_fraction` — and on this candidate those were "
            "ambiguous enough to get it wrong."
        )

    st.divider()

    # ── Step 2 ───────────────────────────────────────────────────────────
    st.subheader("2 · The analyst says no")
    if already_stored and already_stored["disposition"] == DISPOSITION_FALSE_POSITIVE:
        st.success(f"Already dispositioned as **false positive** on `{already_stored['confirmed_at']}` "
                   f"by `{already_stored['analyst']}` — “{already_stored['notes']}”")
    else:
        notes = st.text_input(
            "Analyst notes",
            value="Household sharing one tablet. Small purchases are recharges for family members, "
                  "not card probing. Not a ring.",
        )
        if st.button("🚫 Mark as False Positive", type="primary"):
            memory.store_disposition(
                alert_id=cand.candidate_id,
                disposition=DISPOSITION_FALSE_POSITIVE,
                notes=notes,
                cards=cards,
                signature=feats,
                analyst="demo_analyst",
            )
            st.cache_data.clear()
            st.rerun()
        st.info("Click the button to write this verdict into case memory, then watch step 3 change.")

    st.divider()

    # ── Step 3 ───────────────────────────────────────────────────────────
    st.subheader("3 · Case memory damps the next one like it")

    if not fp_patterns:
        st.info(
            "No false-positive pattern is matching yet. Complete step 2 and this section fills in "
            "with the real before/after."
        )
    else:
        st.plotly_chart(_score_gauge(before["composite_score"], after["composite_score"]),
                        use_container_width=True)

        a, b, c = st.columns(3)
        a.metric("Composite before", f"{before['composite_score']:.3f}", help=before["recommendation"])
        b.metric("Composite after", f"{after['composite_score']:.3f}",
                 delta=f"{after['composite_score'] - before['composite_score']:+.3f}")
        c.metric("Damp applied", f"−{damp:.0%}")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("<div class='rf-kpi-label'>Before</div>", unsafe_allow_html=True)
            st.markdown(badge(before["recommendation"]), unsafe_allow_html=True)
        with r2:
            st.markdown("<div class='rf-kpi-label'>After</div>", unsafe_allow_html=True)
            st.markdown(badge(after["recommendation"]), unsafe_allow_html=True)

        for f in fp_patterns:
            st.warning(
                f"🗂️ Matched case `{f['alert_id']}` — confirmed **false positive** at "
                f"**{f['similarity']:.1%}** cosine similarity on the 31-dim behavioural signature. "
                f"Tagged *historical false positive pattern*. Analyst note: “{f['notes']}”"
            )

        st.caption(
            f"Rule: any candidate within {CASE_MEMORY.fp_similarity_threshold:.2f} cosine similarity of a "
            f"confirmed false positive is damped {CASE_MEMORY.fp_damp:.0%}. The damp is a bounded "
            f"constant from `src/config.py`, not a learned weight — a bad memory can move a score by "
            f"at most {CASE_MEMORY.fp_damp:.0%}, and it is always visible in the dossier as a "
            f"contributing signal."
        )

    st.divider()

    # ── Step 4 ───────────────────────────────────────────────────────────
    st.subheader("4 · The audit trail")
    trail = memory.audit_trail(cand.candidate_id) or memory.audit_trail(limit=20)
    if trail:
        st.dataframe(pd.DataFrame(trail)[["at", "alert_id", "action", "detail"]],
                     use_container_width=True, hide_index=True)
        st.caption("Straight out of `case_memory.sqlite`. Every disposition and every template "
                   "consolidation is appended with a timestamp and an analyst.")
    else:
        st.info("Audit trail is empty — complete step 2.")

    stats = memory.stats()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Cases stored", stats["n_cases"])
    s2.metric("Confirmed fraud", stats["n_confirmed_fraud"])
    s3.metric("Confirmed false positives", stats["n_false_positives"])
    s4.metric("Ring templates", stats["n_templates"])

    st.divider()
    st.markdown(
        f"""
        ### What this demonstrates

        - **The failure is stored, not swallowed.** The signature and the analyst's reasoning both persist.
        - **The correction is bounded.** {CASE_MEMORY.fp_damp:.0%}, from config, capped, and never enough
          to silently flip a genuine ring to APPROVE on its own.
        - **The correction is explainable.** It appears in the dossier as a cited claim with the case ID
          and the similarity behind it.
        - **The system does not crash on bad input.** Unknown cards or a missing model produce a
          `degraded` response with a forced REVIEW — never a silent approve, never a 500.
        """
    )
