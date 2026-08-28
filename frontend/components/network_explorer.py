"""
Page 3 — Network Explorer.

The identity graph behind a candidate, drawn. Nodes are cards, edges are shared
identity values, and edge colour encodes *which* attribute did the linking —
which is the fastest way to see the difference between a ring (thick device and
email edges) and a NAT-pool artefact (nothing but thin IP edges).

Layout is spring-embedded with a fixed seed so the picture does not rearrange
itself every rerun; a demo where the graph jumps on each click is a demo nobody
can point at.
"""

from __future__ import annotations

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

_EDGE_COLOUR = {
    "device_fingerprint": "#f472b6",
    "email_hash": "#a78bfa",
    "ip_address": "#38bdf8",
}
_EDGE_LABEL = {
    "device_fingerprint": "shared device",
    "email_hash": "shared email hash",
    "ip_address": "shared IP",
}


def _subgraph_figure(G: nx.Graph, cards: list[str], hop: int = 0) -> go.Figure:
    """Render the candidate's induced subgraph, optionally with a 1-hop halo."""
    nodes = set(cards)
    if hop:
        for c in cards:
            nodes.update(G.neighbors(c))
    sub = G.subgraph(nodes)
    pos = nx.spring_layout(sub, seed=42, k=0.7 / max(len(sub) ** 0.5, 1))

    fig = go.Figure()

    # One trace per identity type so the legend is meaningful and toggleable.
    for attr, colour in _EDGE_COLOUR.items():
        xs, ys = [], []
        for u, v, data in sub.edges(data=True):
            if attr not in data.get("shared_types", []):
                continue
            xs += [pos[u][0], pos[v][0], None]
            ys += [pos[u][1], pos[v][1], None]
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=_EDGE_LABEL[attr],
                line=dict(width=2 if attr != "ip_address" else 1, color=colour),
                hoverinfo="skip", opacity=0.75,
            ))

    in_cand = [n for n in sub.nodes if n in set(cards)]
    halo = [n for n in sub.nodes if n not in set(cards)]

    for group, name, colour, size in (
        (halo, "1-hop neighbour", "#475569", 9),
        (in_cand, "candidate card", "#fbbf24", 15),
    ):
        if not group:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in group], y=[pos[n][1] for n in group],
            mode="markers", name=name,
            marker=dict(size=size, color=colour, line=dict(width=1.5, color="#0f172a")),
            customdata=[[n, sub.degree(n), G.nodes[n].get("n_transactions", 0),
                         G.nodes[n].get("world_id", -1)] for n in group],
            hovertemplate="<b>%{customdata[0]}</b><br>degree %{customdata[1]}"
                          "<br>%{customdata[2]} transactions<br>world %{customdata[3]}<extra></extra>",
        ))

    fig.update_layout(
        height=560, margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        legend=dict(orientation="h", y=1.06, x=0),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_network_page(backend, artifacts: dict) -> None:
    """Render the network explorer page."""
    st.title("Network Explorer")
    st.caption(
        "Cards are nodes. Edges are shared identity — device, email hash or IP. "
        "**There are no merchant edges**: 99%+ of card pairs share a merchant, so a merchant edge "
        "would say nothing about fraud. See page 4 for the measured numbers."
    )

    G = artifacts.get("graph")
    candidates = artifacts.get("candidates") or []
    if G is None or not candidates:
        st.error("Identity graph unavailable. Run `make all` to build the pipeline.")
        return

    options = [c.candidate_id for c in candidates]
    selected = st.session_state.get("selected_alert")
    default = options.index(selected) if selected in options else 0

    c1, c2 = st.columns([3, 1])
    chosen = c1.selectbox("Candidate", options, index=default)
    hop = c2.selectbox("Show neighbours", [0, 1], format_func=lambda h: "candidate only" if h == 0 else "+ 1 hop")

    cand = next(c for c in candidates if c.candidate_id == chosen)
    cards = list(cand.cards)

    m1, m2, m3, m4 = st.columns(4)
    sub = G.subgraph(cards)
    m1.metric("Cards", len(cards))
    m2.metric("Internal edges", sub.number_of_edges())
    m3.metric("Components", nx.number_connected_components(sub))
    m4.metric("Proposed by", cand.source)
    st.caption(f"Generator detail: `{cand.source_detail}`")

    st.plotly_chart(_subgraph_figure(G, cards, hop), use_container_width=True)

    with st.expander("Edge detail — every link and what created it"):
        rows = []
        for u, v, data in sub.edges(data=True):
            for shared in data.get("shared", []):
                rows.append({
                    "Card A": u, "Card B": v,
                    "Linked by": _EDGE_LABEL.get(shared["attribute"], shared["attribute"]),
                    "Value": shared["value"],
                    "Network-wide cards on this value": shared["cardinality"],
                    "Edge weight": round(data.get("weight", 0), 3),
                })
        if rows:
            import pandas as pd

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(
                "Edge weight is rarity-weighted: `1 / log2(k+1)` where k is how many cards use that "
                "value network-wide. A device shared by 2 cards scores 0.63; the stock user-agent "
                "shared by ~1,492 scores 0.095."
            )
        else:
            st.info("This candidate has no internal edges — it was proposed by a community "
                    "generator rather than by shared infrastructure.")
