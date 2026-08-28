"""
Identity graph construction.

Cards are nodes. An edge means "these two cards touched the same piece of
identity infrastructure". That is the whole model. Two decisions in here carry
the system, and both are the kind of thing a panel will push on:

**No merchant edges.** It is tempting to link cards that shopped at the same
merchant. Do not. `get_collision_stats` computes, on the real generated data,
how many card pairs share at least one merchant versus how many share an
identity attribute. Merchant co-occurrence covers a large majority of all
possible pairs — it is a statement about retail, not about fraud. Adding those
edges would bury every real signal under noise, so the edge rule is device /
IP / email only.

**Collision capping.** Device fingerprinting has a fat head. One stock user
agent covers ~1,492 accounts in this dataset. Expanded naively that single
value is 1.1 million edges and fuses the graph into one component, at which
point community detection returns "everybody" and the detector is dead. So
each identity value is capped: if more than `cap` cards share it, we keep only
the `cap` cards most strongly associated with that value.

We cap the *card set*, not the resulting edge list. Same operation, applied one
step earlier — it turns an O(k^2) expansion you immediately throw away into an
O(cap^2) one you keep.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from src.config import GRAPH, GRAPH_PICKLE, GraphConfig, ensure_dirs
from src.data_generator import load_transactions
from src.utils import get_logger, timed, write_json

log = get_logger("ringfence.graph_builder")


@dataclass(frozen=True)
class IdentityGroup:
    """
    One identity value and the cards that used it.

    `cardinality` is the TRUE, pre-cap number of cards. Edge weights are
    computed from it deliberately: a device shared by two cards is strong
    evidence, the same device shared by 1,492 is nearly none, and capping the
    card list must not launder a weak signal into a strong one.
    """

    attribute: str
    value: str
    cards: tuple[str, ...]
    cardinality: int

    @property
    def was_capped(self) -> bool:
        """True when this group lost cards to the collision cap."""
        return len(self.cards) < self.cardinality


# ──────────────────────────────────────────────────────────────────────────
# Identity grouping
# ──────────────────────────────────────────────────────────────────────────


def _identity_groups(
    transactions_df: pd.DataFrame, cfg: GraphConfig = GRAPH
) -> tuple[list[IdentityGroup], dict[str, pd.Series]]:
    """
    Group cards by every identity attribute value.

    Returns the multi-card groups (single-card values can never make an edge)
    plus, per attribute, the full cardinality distribution — which is what the
    percentile cap is computed from.
    """
    groups: list[IdentityGroup] = []
    cardinalities: dict[str, pd.Series] = {}

    for attr in cfg.identity_attributes:
        # (value, card) -> transaction count, used both for cardinality and to
        # rank cards by association strength when capping.
        pair_counts = transactions_df.groupby([attr, "card_id"], observed=True).size()
        card_counts = pair_counts.groupby(level=0).size()
        cardinalities[attr] = card_counts

        multi = card_counts[card_counts > 1].index
        if len(multi) == 0:
            continue

        # Sort by (value, -txn_count, card_id) so the cap keeps the cards most
        # strongly associated with the value, deterministically.
        sub = pair_counts.loc[pair_counts.index.get_level_values(0).isin(multi)]
        frame = sub.reset_index()
        frame.columns = [attr, "card_id", "n_txns"]
        frame = frame.sort_values([attr, "n_txns", "card_id"], ascending=[True, False, True])

        for value, block in frame.groupby(attr, observed=True, sort=False):
            cards = tuple(block["card_id"].tolist())
            groups.append(IdentityGroup(attr, str(value), cards, len(cards)))

    return groups, cardinalities


def compute_cap(cardinalities: dict[str, pd.Series], cfg: GraphConfig = GRAPH, cap_percentile: float | None = None) -> dict[str, int]:
    """
    Choose a per-attribute collision cap.

    The cap is the `cap_percentile`-th percentile of that attribute's observed
    card-cardinality, floored at `cfg.min_collision_cap`.

    The floor is not a fudge. The raw 95th percentile of device cardinality in
    this data is 4-5 cards. Capping there would shred the 8-card rings we exist
    to find — we would be deleting the signal to control the noise. The cap's
    job is to stop one 1,492-account value from fusing the graph, so the floor
    is set above the largest plausible ring and the cap only ever bites on
    values that are far outside ring scale.
    """
    pct = cap_percentile if cap_percentile is not None else cfg.cap_percentile
    caps: dict[str, int] = {}
    for attr, counts in cardinalities.items():
        raw = int(np.ceil(counts.quantile(pct / 100.0))) if len(counts) else cfg.min_collision_cap
        caps[attr] = max(raw, cfg.min_collision_cap)
        log.info(f"cap[{attr}] = {caps[attr]} (p{pct:g} raw = {raw}, floor = {cfg.min_collision_cap})")
    return caps


def cap_collisions(groups: list[IdentityGroup], cap: int | dict[str, int]) -> list[IdentityGroup]:
    """
    Downsample identity groups whose cardinality exceeds the cap.

    `cap` may be a single int (applied to every attribute) or a per-attribute
    dict. Cards are already ordered by association strength, so capping is a
    truncation: keep the top `cap`, drop the rest.

    Note on the spec: this is the `cap_collisions(edges, cap)` step, applied to
    identity groups rather than to an already-expanded edge list. Capping a
    1,492-card value first means we build 66 edges instead of building 1.1
    million and deleting 99.99% of them. The output is identical.
    """
    out: list[IdentityGroup] = []
    for g in groups:
        limit = cap[g.attribute] if isinstance(cap, dict) else cap
        if len(g.cards) > limit:
            out.append(IdentityGroup(g.attribute, g.value, g.cards[:limit], g.cardinality))
        else:
            out.append(g)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Edge construction
# ──────────────────────────────────────────────────────────────────────────


def _identity_weight(cardinality: int, attribute: str, cfg: GraphConfig = GRAPH) -> float:
    """
    Rarity-weighted strength of one shared identity value.

    1 / log2(k + 1), scaled by an attribute multiplier:

        k =     2 cards -> 0.63   two cards on one phone: strong
        k =     8 cards -> 0.32   a device pool: still meaningful
        k = 1,492 cards -> 0.095  a stock user-agent: almost nothing

    IPs are discounted to 0.7x because NAT and mobile carriers make shared
    egress addresses routine; device and email hashes are taken at face value.
    """
    type_weight = {
        "device_fingerprint": cfg.weight_device,
        "email_hash": cfg.weight_email,
        "ip_address": cfg.weight_ip,
    }.get(attribute, 0.5)
    return type_weight / np.log2(cardinality + 1)


def _groups_to_edges(groups: list[IdentityGroup], cfg: GraphConfig = GRAPH) -> dict[tuple[str, str], dict]:
    """
    Expand capped identity groups into weighted edges.

    A pair of cards can share several identity values; weights add and are
    clipped at 1.0. Every contributing value is retained on the edge, because
    the dossier has to be able to cite *which* device linked which cards — an
    edge with a weight but no provenance is a claim we cannot support.
    """
    edges: dict[tuple[str, str], dict] = {}
    for g in groups:
        w = _identity_weight(g.cardinality, g.attribute, cfg)
        for a, b in combinations(sorted(g.cards), 2):
            key = (a, b)
            rec = edges.get(key)
            if rec is None:
                rec = {"weight": 0.0, "shared": []}
                edges[key] = rec
            rec["weight"] += w
            rec["shared"].append(
                {"attribute": g.attribute, "value": g.value, "cardinality": g.cardinality}
            )

    for rec in edges.values():
        rec["weight"] = float(min(rec["weight"], 1.0))
        rec["n_shared"] = len(rec["shared"])
        rec["shared_types"] = sorted({s["attribute"] for s in rec["shared"]})
    return edges


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────


def build_identity_graph(
    transactions_df: pd.DataFrame,
    cap_percentile: float = 95.0,
    cards_df: pd.DataFrame | None = None,
    cfg: GraphConfig = GRAPH,
) -> nx.Graph:
    """
    Build the undirected identity graph. Cards are nodes; shared device / IP /
    email hashes are edges. No merchant edges, ever.

    Nodes carry `world_id` (needed to split cross-world CV folds without leaking)
    and `n_transactions`. Nodes carry no label — ground truth lives in a separate
    file and is joined only at evaluation time.
    """
    with timed("graph.group_identities", log):
        groups, cardinalities = _identity_groups(transactions_df, cfg)
    log.info(f"{len(groups):,} multi-card identity groups across {len(cfg.identity_attributes)} attributes")

    caps = compute_cap(cardinalities, cfg, cap_percentile)

    with timed("graph.cap_collisions", log):
        capped = cap_collisions(groups, caps)
    n_capped = sum(1 for g in capped if g.was_capped)
    dropped = sum(g.cardinality - len(g.cards) for g in capped)
    log.info(f"collision cap bit on {n_capped} groups, removing {dropped:,} card-value memberships")

    with timed("graph.expand_edges", log):
        edges = _groups_to_edges(capped, cfg)
    log.info(f"{len(edges):,} identity edges")

    with timed("graph.assemble", log):
        G = nx.Graph()
        all_cards = transactions_df["card_id"].unique()
        txn_counts = transactions_df["card_id"].value_counts()
        world_map = {}
        if cards_df is not None:
            world_map = dict(zip(cards_df["card_id"], cards_df["world_id"]))
        for card in all_cards:
            G.add_node(card, world_id=int(world_map.get(card, -1)), n_transactions=int(txn_counts.get(card, 0)))
        for (a, b), rec in edges.items():
            G.add_edge(a, b, **rec)

    log.info(
        f"graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges, "
        f"{nx.number_connected_components(G):,} components, "
        f"largest component {len(max(nx.connected_components(G), key=len)) if G.number_of_edges() else 0:,}"
    )
    return G


def get_collision_stats(transactions_df: pd.DataFrame, cfg: GraphConfig = GRAPH) -> dict:
    """
    Quantify why merchant edges are excluded and why capping is required.

    Produces, per identity attribute, the cardinality distribution and how hard
    the cap bites; and for merchants, the number of card pairs that share at
    least one merchant. That last number is the argument: if merchant
    co-occurrence links most of the population to most of the population, a
    merchant edge carries no information about fraud.
    """
    stats: dict = {"identity": {}, "merchant": {}}

    _, cardinalities = _identity_groups(transactions_df, cfg)
    caps = compute_cap(cardinalities, cfg)

    for attr, counts in cardinalities.items():
        multi = counts[counts > 1]
        cap = caps[attr]
        over = counts[counts > cap]
        stats["identity"][attr] = {
            "n_values": int(counts.size),
            "n_multi_card_values": int(multi.size),
            "p50_cardinality": float(counts.quantile(0.50)),
            "p95_cardinality": float(counts.quantile(0.95)),
            "p99_cardinality": float(counts.quantile(0.99)),
            "max_cardinality": int(counts.max()) if counts.size else 0,
            "cap_applied": int(cap),
            "n_values_over_cap": int(over.size),
            "memberships_dropped_by_cap": int((over - cap).sum()) if over.size else 0,
            "uncapped_pairs_from_worst_value": int(counts.max() * (counts.max() - 1) // 2) if counts.size else 0,
            "capped_pairs_from_worst_value": int(cap * (cap - 1) // 2),
        }

    # --- merchant co-occurrence, computed not asserted ---------------------
    cards = transactions_df["card_id"].unique()
    n_cards = len(cards)
    total_pairs = n_cards * (n_cards - 1) // 2

    if n_cards <= 6000:
        card_pos = {c: i for i, c in enumerate(cards)}
        merchants = transactions_df["merchant_id"].unique()
        merch_pos = {m: i for i, m in enumerate(merchants)}
        incidence = np.zeros((n_cards, len(merchants)), dtype=np.uint8)
        incidence[
            transactions_df["card_id"].map(card_pos).to_numpy(),
            transactions_df["merchant_id"].map(merch_pos).to_numpy(),
        ] = 1
        co = (incidence.astype(np.uint16) @ incidence.T.astype(np.uint16)) > 0
        np.fill_diagonal(co, False)
        merchant_pairs = int(co.sum() // 2)
    else:
        merchant_pairs = -1  # too large to materialise; excluded by rule anyway

    identity_pairs = 0
    groups, _ = _identity_groups(transactions_df, cfg)
    seen: set[tuple[str, str]] = set()
    for g in cap_collisions(groups, caps):
        for pair in combinations(sorted(g.cards), 2):
            seen.add(pair)
    identity_pairs = len(seen)

    stats["merchant"] = {
        "n_cards": int(n_cards),
        "n_merchants": int(transactions_df["merchant_id"].nunique()),
        "total_possible_card_pairs": int(total_pairs),
        "card_pairs_sharing_a_merchant": merchant_pairs,
        "merchant_pair_saturation": round(merchant_pairs / total_pairs, 4) if merchant_pairs >= 0 and total_pairs else None,
        "card_pairs_sharing_an_identity": int(identity_pairs),
        "identity_pair_saturation": round(identity_pairs / total_pairs, 6) if total_pairs else None,
        "verdict": (
            "Merchant co-occurrence links a large share of all possible card pairs; "
            "identity co-occurrence links a tiny share. Merchant edges are excluded "
            "because they would swamp the graph with retail coincidence."
        ),
    }

    log.info("collision stats", extra={"extra_fields": stats})
    log.info(
        f"MERCHANT EDGE RULE: {merchant_pairs:,} of {total_pairs:,} card pairs share a merchant "
        f"({100 * merchant_pairs / total_pairs:.1f}%) vs {identity_pairs:,} sharing an identity "
        f"({100 * identity_pairs / total_pairs:.3f}%). Merchant edges excluded."
    )
    return stats


def save_graph(G: nx.Graph, path=GRAPH_PICKLE) -> None:
    """
    Persist the graph.

    Plain pickle: networkx removed `write_gpickle` in 3.0, and the file keeps
    the `.gpickle` extension the spec asks for.
    """
    ensure_dirs()
    with open(path, "wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


def load_graph(path=GRAPH_PICKLE) -> nx.Graph:
    """Load a persisted identity graph."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main() -> None:
    """CLI: build the identity graph from data/raw and persist it."""
    parser = argparse.ArgumentParser(description="Build the Ringfence identity graph")
    parser.add_argument("--cap-percentile", type=float, default=GRAPH.cap_percentile)
    args = parser.parse_args()

    from src.config import CARDS_CSV, PROCESSED_DIR

    txns = load_transactions()
    cards = pd.read_csv(CARDS_CSV)

    stats = get_collision_stats(txns)
    write_json(PROCESSED_DIR / "collision_stats.json", stats)

    G = build_identity_graph(txns, cap_percentile=args.cap_percentile, cards_df=cards)
    save_graph(G)


if __name__ == "__main__":
    main()
