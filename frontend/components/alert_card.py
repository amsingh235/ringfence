"""
Page 1 — Live Alert Queue.

What an analyst opens in the morning: recent alerts, newest first, filterable
by recommendation and disposition, with a one-click path into the dossier.

When the API is offline the queue is generated locally from the persisted
candidate set, so the demo still shows a populated queue instead of an empty
table and an apology.
"""

from __future__ import annotations

import streamlit as st


def _score_candidates_locally(artifacts: dict, limit: int = 40) -> list[dict]:
    """
    Score the top candidates directly, without the API.

    Ordered by descending Ringfence score so the queue leads with what an
    analyst would actually triage first.
    """
    from src.feature_engine import compute_candidate_meta, compute_features
    from src.vulcan_integration import VulcanComposer

    G, index, scorer = artifacts.get("graph"), artifacts.get("index"), artifacts.get("scorer")
    candidates = artifacts.get("candidates") or []
    if G is None or index is None or not candidates:
        return []

    composer = VulcanComposer()
    rows = []
    for cand in candidates:
        feats = compute_features(cand.cards, G, index)
        meta = compute_candidate_meta(cand.cards, index)
        score = scorer.score(feats) if scorer and scorer.available else 0.0
        composite = composer.compute_composite(
            vulcan_score=meta.get("vulcan_cashout_mean") or meta.get("vulcan_mean", 0.0),
            ringfence_score=score,
        )
        rows.append({
            "alert_id": cand.candidate_id,
            "n_cards": cand.size,
            "ringfence_score": round(score, 4),
            "composite_score": composite["composite_score"],
            "recommendation": composite["recommendation"],
            "source": cand.source,
            "world_id": cand.world_id,
            "disposition": None,
            "degraded": 0,
            "_cards": list(cand.cards),
        })
    rows.sort(key=lambda r: -r["composite_score"])
    return rows[:limit]


@st.cache_data(show_spinner="Scoring candidate queue…")
def _cached_local_queue(_artifacts: dict, limit: int) -> list[dict]:
    """Cache the locally-scored queue; the underscore stops Streamlit hashing the graph."""
    return _score_candidates_locally(_artifacts, limit)


def render_alert_queue(backend, artifacts: dict, badge) -> None:
    """Render the alert queue page."""
    st.title("Live Alert Queue")
    st.caption(
        "Every row is a **candidate card cluster**, not a transaction. "
        "The composite score combines Vulcan's per-transaction view with Ringfence's ring score."
    )

    col_f1, col_f2, col_f3 = st.columns([2, 2, 3])
    with col_f1:
        rec_filter = st.selectbox("Recommendation", ["All", "BLOCK", "REVIEW", "APPROVE"])
    with col_f2:
        disp_filter = st.selectbox("Disposition", ["All", "Confirmed Fraud", "False Positive", "Undecided"])
    with col_f3:
        st.write("")
        st.write("")
        if st.button("↻ Refresh", use_container_width=False):
            st.cache_data.clear()
            st.rerun()

    # --- fetch ----------------------------------------------------------
    rows: list[dict] = []
    payload = backend.get("/alerts", limit=200)
    if payload and payload.get("alerts"):
        rows = payload["alerts"]
        source_note = f"{payload['total']} alerts from the API"
    else:
        # Two different situations land here and they are not the same thing:
        # the API is down, or the API is up and its alert store is empty (a
        # fresh clone, or straight after `clean-memory`). Saying "API offline"
        # in the second case contradicts the green badge in the sidebar.
        rows = _cached_local_queue(artifacts, 40)
        reason = "no alerts stored yet" if payload is not None else "API offline"
        source_note = f"{len(rows)} candidates scored locally — {reason}"

    # --- filter ---------------------------------------------------------
    if rec_filter != "All":
        rows = [r for r in rows if r.get("recommendation") == rec_filter]
    if disp_filter == "Confirmed Fraud":
        rows = [r for r in rows if r.get("disposition") == "fraud"]
    elif disp_filter == "False Positive":
        rows = [r for r in rows if r.get("disposition") == "false_positive"]
    elif disp_filter == "Undecided":
        rows = [r for r in rows if not r.get("disposition")]

    st.caption(source_note)

    # --- summary --------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alerts shown", len(rows))
    c2.metric("BLOCK", sum(1 for r in rows if r.get("recommendation") == "BLOCK"))
    c3.metric("REVIEW", sum(1 for r in rows if r.get("recommendation") == "REVIEW"))
    c4.metric("Dispositioned", sum(1 for r in rows if r.get("disposition")))

    st.divider()

    if not rows:
        st.info("No alerts match these filters. Run `make all` to build the pipeline, then `make run`.")
        return

    # --- table ----------------------------------------------------------
    header = st.columns([2.6, 1, 1.2, 1.4, 1.4, 1.2])
    for col, label in zip(header, ["Alert", "Cards", "Ringfence", "Composite", "Recommendation", ""]):
        col.markdown(f"**{label}**")

    for row in rows[:60]:
        cols = st.columns([2.6, 1, 1.2, 1.4, 1.4, 1.2])
        cols[0].markdown(f"`{row['alert_id']}`")
        if row.get("disposition"):
            cols[0].caption(f"↳ analyst: **{row['disposition'].replace('_', ' ')}**")
        cols[1].write(row.get("n_cards", "—"))
        cols[2].write(f"{row.get('ringfence_score', 0):.3f}")
        cols[3].write(f"**{row.get('composite_score', 0):.3f}**")
        cols[4].markdown(
            badge(row.get("recommendation", "REVIEW"), bool(row.get("degraded"))),
            unsafe_allow_html=True,
        )
        if cols[5].button("Open →", key=f"open_{row['alert_id']}"):
            st.session_state.selected_alert = row["alert_id"]
            st.session_state.selected_cards = row.get("_cards")
            st.success(f"Selected `{row['alert_id']}` — switch to **2 · Dossier Viewer**.")

    st.divider()
    st.caption(
        "Precision at the operating point is intentionally not maximised. "
        "A miss costs ₹8.5L and a false positive ₹2.3L, so the threshold is set where "
        "**cost** is minimised, not where F1 is. See page 4."
    )
