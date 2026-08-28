"""
Page 2 — Dossier Viewer.

The audit trail, rendered. Card list with shared devices highlighted, the
probe -> cashout timeline, the evidence table with every citation visible, the
narrative summary, precedent and false-positive matches, and the two analyst
disposition buttons that write back into case memory.

The evidence table is the point of this page. Every row is a claim with the
graph edge, transaction ID or feature value that supports it. There is no row
without a citation, because `Dossier` cannot be constructed with one.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_CITATION_LABEL = {
    "graph_edge": "🔗 graph edge",
    "transaction_id": "🧾 transaction",
    "feature_value": "📐 feature",
    "corpus_passage": "🗂️ case memory",
    "vulcan_score": "⚡ vulcan",
}


def _build_dossier_locally(artifacts: dict, cards: list[str]) -> dict | None:
    """Build a dossier without the API, using the same builder the API calls."""
    from src.case_memory import CaseMemory
    from src.dossier_builder import build_dossier_for_candidate
    from src.vulcan_integration import VulcanComposer

    G, index, scorer = artifacts.get("graph"), artifacts.get("index"), artifacts.get("scorer")
    if G is None or index is None:
        return None
    try:
        memory = CaseMemory()
    except Exception:  # noqa: BLE001
        memory = None
    dossier = build_dossier_for_candidate(
        cards=cards, G=G, index=index, scorer=scorer,
        composer=VulcanComposer(), memory=memory, use_llm=False,
    )
    return dossier.model_dump(mode="json")


def _timeline(artifacts: dict, cards: list[str]) -> go.Figure | None:
    """
    Plot the cluster's shared-infrastructure transactions on a log-amount axis.

    Log scale because the story is four orders of magnitude wide: a ₹5 probe
    and a ₹47,500 cashout do not coexist on a linear axis, and the escalation
    is the whole point.
    """
    index = artifacts.get("index")
    if index is None:
        return None

    from src.feature_engine import (CASHOUT_AMOUNT_MIN, PROBE_AMOUNT_MAX,
                                    _gather_rows, _shared_identity_codes)
    import numpy as np

    known = [c for c in cards if c in index.card_to_slice]
    rows = _gather_rows(known, index)
    if rows.size == 0:
        return None

    shared_dev = _shared_identity_codes(known, index.card_devices)
    shared_ip = _shared_identity_codes(known, index.card_ips)
    mask = np.isin(index.device_code[rows], shared_dev) | np.isin(index.ip_code[rows], shared_ip)
    focus = rows[mask] if mask.any() else rows

    amounts = index.amount[focus]
    times = pd.to_datetime(index.ts[focus], unit="s")
    kinds = np.where(amounts < PROBE_AMOUNT_MAX, "probe (<₹10)",
                     np.where(amounts > CASHOUT_AMOUNT_MIN, "cashout (>₹10k)", "other"))

    colours = {"probe (<₹10)": "#38bdf8", "cashout (>₹10k)": "#f87171", "other": "#94a3b8"}
    fig = go.Figure()
    for kind in ["other", "probe (<₹10)", "cashout (>₹10k)"]:
        sel = kinds == kind
        if not sel.any():
            continue
        fig.add_trace(go.Scatter(
            x=times[sel], y=amounts[sel], mode="markers", name=kind,
            marker=dict(size=11 if kind != "other" else 7, color=colours[kind],
                        line=dict(width=1, color="rgba(0,0,0,0.35)")),
            customdata=np.stack([index.txn_id[focus][sel], index.merchant_id[focus][sel]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>merchant %{customdata[1]}"
                          "<br>₹%{y:,.2f}<br>%{x}<extra></extra>",
        ))

    fig.update_layout(
        height=330, margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(type="log", title="amount (₹, log scale)"),
        xaxis=dict(title=None),
        legend=dict(orientation="h", y=1.14, x=0),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_dossier_page(backend, artifacts: dict, badge) -> None:
    """Render the dossier viewer page."""
    st.title("Dossier Viewer")
    st.caption("A case file, not a score. Every claim below carries a citation — enforced at construction.")

    alert_id = st.session_state.get("selected_alert")
    cards = st.session_state.get("selected_cards")

    picker = st.container()
    with picker:
        candidates = artifacts.get("candidates") or []
        options = [c.candidate_id for c in candidates]
        if options:
            default = options.index(alert_id) if alert_id in options else 0
            chosen = st.selectbox("Candidate / alert", options, index=default,
                                  help="Or click 'Open →' on the Alert Queue page")
            if chosen != alert_id:
                alert_id = chosen
                cards = list(next(c for c in candidates if c.candidate_id == chosen).cards)
                st.session_state.selected_alert, st.session_state.selected_cards = alert_id, cards

    if not alert_id:
        st.info("Pick a candidate above, or open one from the Alert Queue.")
        return

    # --- fetch or build --------------------------------------------------
    dossier = None
    api_payload = backend.get(f"/alerts/{alert_id}/dossier")
    if api_payload:
        dossier = api_payload["dossier"]
    elif cards:
        with st.spinner("Building dossier…"):
            dossier = _build_dossier_locally(artifacts, cards)

    if not dossier:
        st.error("Could not load or build a dossier for this alert.")
        return

    cards = dossier["candidate_cards"]

    # --- header ----------------------------------------------------------
    c1, c2, c3, c4 = st.columns([1.4, 1.2, 1.2, 1.6])
    c1.metric("Ringfence score", f"{dossier['ringfence_score']:.3f}",
              help=f"operating threshold {dossier['threshold_used']:.3f}")
    c2.metric("Vulcan (per-txn)", f"{dossier['vulcan_transaction_score']:.3f}")
    c3.metric("Composite", f"{dossier['vulcan_composite_score']:.3f}")
    with c4:
        st.markdown("<div class='rf-kpi-label'>Recommendation</div>", unsafe_allow_html=True)
        st.markdown(badge(dossier["recommendation"], dossier.get("degraded", False)), unsafe_allow_html=True)
        if not dossier.get("auto_action_allowed", False):
            st.caption("🔒 Human review required — bounded action rule")

    if dossier.get("degraded"):
        st.warning(f"Degraded response: {dossier.get('degraded_reason')}. Safe default applied.")

    st.divider()

    # --- summary ---------------------------------------------------------
    st.subheader("Summary")
    st.info(dossier.get("llm_summary") or "—")
    st.caption(
        f"Generated from the cited evidence below · `llm_status={dossier.get('llm_status')}` · "
        "the narrative is derived from the citations, never the other way round."
    )

    # --- cards -----------------------------------------------------------
    st.subheader(f"Cards in cluster ({len(cards)})")
    shared_devices = sorted({
        e["value"]["value"] for e in dossier["evidence"]
        if e["citation_type"] == "graph_edge" and isinstance(e.get("value"), dict)
        and e["value"].get("attribute") == "device_fingerprint"
    })
    if shared_devices:
        st.markdown("**Shared devices:** " + " ".join(f"`{d}`" for d in shared_devices[:6]))
    st.markdown(" ".join(f"`{c}`" for c in cards[:60]) + (" …" if len(cards) > 60 else ""))

    # --- timeline --------------------------------------------------------
    st.subheader("Transaction timeline — probe → cashout")
    fig = _timeline(artifacts, cards)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Transaction index unavailable; timeline not rendered.")

    # --- evidence --------------------------------------------------------
    st.subheader("Evidence")
    q = dossier.get("quality") or {}
    ev_rows = [
        {
            "Claim": e["claim"],
            "Citation type": _CITATION_LABEL.get(e["citation_type"], e["citation_type"]),
            "Citation": e["citation_id"],
            "Refs": 1 + len(e.get("supporting_citations", [])),
        }
        for e in dossier["evidence"]
    ]
    st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True,
                 column_config={"Claim": st.column_config.TextColumn(width="large")})

    n_claims = len(dossier["evidence"])
    total_refs = sum(1 + len(e.get("supporting_citations", [])) for e in dossier["evidence"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Claims", n_claims)
    m2.metric("Uncited claims", 0, help="Structurally guaranteed: Dossier() raises on an uncited claim")
    m3.metric("Refs per claim", f"{total_refs / max(n_claims, 1):.2f}")

    with st.expander("Supporting citations, per claim"):
        for e in dossier["evidence"]:
            st.markdown(f"**{e['claim']}**")
            refs = [e["citation_id"], *e.get("supporting_citations", [])]
            st.markdown("<span class='rf-cite'>" + " · ".join(refs) + "</span>", unsafe_allow_html=True)
            st.write("")

    # --- memory ----------------------------------------------------------
    if dossier.get("precedents"):
        st.subheader("Precedent matches")
        for p in dossier["precedents"]:
            st.success(f"Matches confirmed ring `{p.get('ring_id') or p.get('alert_id')}` at "
                       f"{p['similarity']:.0%} similarity — score boosted 15%. {p.get('notes', '')}")

    if dossier.get("false_positive_patterns"):
        st.subheader("⚠️ Historical false positive pattern")
        for f in dossier["false_positive_patterns"]:
            st.warning(f"Matches case `{f['alert_id']}`, previously confirmed a FALSE POSITIVE, at "
                       f"{f['similarity']:.0%} similarity — score damped 20%. {f.get('notes', '')}")

    # --- disposition -----------------------------------------------------
    st.divider()
    st.subheader("Analyst disposition")
    st.caption("Writes to case memory and changes how similar candidates score from here on.")
    d1, d2, _ = st.columns([1, 1, 3])

    def _dispose(verdict: str, notes: str) -> None:
        result = backend.post(f"/alerts/{alert_id}/disposition",
                              {"disposition": verdict, "notes": notes})
        if result and "_error" not in result:
            st.success(result["effect_on_future_scoring"])
        else:
            from src.case_memory import CaseMemory

            memory = CaseMemory()
            memory.store_disposition(alert_id=alert_id, disposition=verdict, notes=notes,
                                     cards=cards, signature=dossier.get("feature_snapshot"))
            memory.consolidate_templates()
            st.success(f"Stored `{verdict}` in case memory (local write — API offline).")
        st.cache_data.clear()

    notes = st.text_input("Notes", placeholder="e.g. household sharing one tablet, not a ring")
    if d1.button("✅ Confirm Fraud", use_container_width=True):
        _dispose("fraud", notes or "confirmed by analyst")
    if d2.button("🚫 False Positive", use_container_width=True):
        _dispose("false_positive", notes or "not a ring")
