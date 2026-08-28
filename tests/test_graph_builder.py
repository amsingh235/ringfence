"""
Graph builder tests: collision capping, the no-merchant-edge rule, and edge
weighting.

The capping tests are the important ones. Without a cap, one stock user-agent
fuses the graph into a single component and every downstream stage returns
"everybody"; these assertions are what stop that regressing silently.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.config import GRAPH
from src.graph_builder import (
    IdentityGroup,
    _identity_groups,
    _identity_weight,
    build_identity_graph,
    cap_collisions,
    compute_cap,
    get_collision_stats,
)


def test_graph_has_expected_shape(graph, cards_df):
    """Every card is a node; the graph is undirected and non-empty."""
    assert isinstance(graph, nx.Graph)
    assert not graph.is_directed()
    assert graph.number_of_nodes() == cards_df["card_id"].nunique()
    assert graph.number_of_edges() > 0


def test_nodes_carry_world_id_but_no_label(graph):
    """
    Nodes carry `world_id` (needed to split CV folds) and no fraud label.

    If a label ever leaked onto the graph, every feature computed from the graph
    would be contaminated and the cross-world metrics would be meaningless.
    """
    sample = next(iter(graph.nodes))
    attrs = graph.nodes[sample]
    assert "world_id" in attrs
    assert attrs["world_id"] >= 0
    for forbidden in ("is_ring_member", "ring_id", "is_fraud", "label", "y"):
        assert forbidden not in attrs, f"label {forbidden!r} leaked onto graph nodes"


def test_no_merchant_edges(graph, transactions):
    """
    No edge exists that is justified only by a shared merchant.

    Every edge must cite at least one identity attribute, and merchant is never
    among them.
    """
    for _, _, data in graph.edges(data=True):
        shared = data.get("shared", [])
        assert shared, "edge with no provenance"
        for s in shared:
            assert s["attribute"] in GRAPH.identity_attributes
            assert "merchant" not in s["attribute"]


def test_merchant_collisions_dwarf_identity_collisions(transactions):
    """
    The measured justification for excluding merchant edges.

    Merchant co-occurrence should link a large majority of all card pairs while
    identity co-occurrence links a tiny fraction. This is the statistic the
    module logs and the README quotes.
    """
    stats = get_collision_stats(transactions)
    m = stats["merchant"]
    assert m["card_pairs_sharing_a_merchant"] > m["card_pairs_sharing_an_identity"] * 20
    assert m["merchant_pair_saturation"] > 0.5
    assert m["identity_pair_saturation"] < 0.2


def test_collision_cap_floor_exceeds_max_ring_size(transactions):
    """
    The cap must never be tight enough to shred a ring.

    The raw 95th percentile of device cardinality is 3-5 cards; capping there
    would delete the 8-card rings we exist to find. The configured floor has to
    sit above the maximum ring size.
    """
    _, cardinalities = _identity_groups(transactions)
    caps = compute_cap(cardinalities)
    from src.config import GENERATION

    for attr, cap in caps.items():
        assert cap >= GRAPH.min_collision_cap
        assert cap > GENERATION.ring_size_max, f"cap for {attr} would truncate a maximum-size ring"


def test_generic_device_is_capped(transactions, gen_config):
    """
    The stock user-agent shared by ~1,492 accounts must be capped.

    Verifies the cap actually bites on the pathological value, and that the
    resulting clique is bounded rather than quadratic in the original size.
    """
    groups, cardinalities = _identity_groups(transactions)
    caps = compute_cap(cardinalities)

    generic = [g for g in groups if g.value == gen_config.generic_device_string]
    assert generic, "generic device string not present in the generated data"
    g = generic[0]
    assert g.cardinality > 50, "generic device is not colliding widely enough to test capping"

    capped = cap_collisions(groups, caps)
    capped_generic = next(c for c in capped if c.value == gen_config.generic_device_string)
    assert capped_generic.was_capped
    assert len(capped_generic.cards) == caps["device_fingerprint"]
    assert capped_generic.cardinality == g.cardinality, "true cardinality must survive capping"


def test_cap_prevents_graph_fusion(graph, cards_df):
    """
    No single component may swallow the graph.

    Without capping, the generic device alone would merge a large majority of
    cards into one component and community detection would be meaningless.
    """
    largest = len(max(nx.connected_components(graph), key=len))
    assert largest < 0.85 * graph.number_of_nodes(), "identity graph fused into one blob"


def test_cap_collisions_is_a_truncation():
    """`cap_collisions` keeps the first `cap` cards and preserves true cardinality."""
    group = IdentityGroup("device_fingerprint", "DEV_X", tuple(f"CARD_{i}" for i in range(30)), 30)
    capped = cap_collisions([group], 12)[0]
    assert len(capped.cards) == 12
    assert capped.cards == group.cards[:12]
    assert capped.cardinality == 30
    assert capped.was_capped


def test_cap_collisions_leaves_small_groups_alone():
    """A group under the cap passes through untouched."""
    group = IdentityGroup("ip_address", "10.0.0.1", ("CARD_A", "CARD_B"), 2)
    out = cap_collisions([group], 12)[0]
    assert out.cards == group.cards
    assert not out.was_capped


def test_cap_collisions_accepts_per_attribute_dict():
    """Per-attribute caps are honoured independently."""
    groups = [
        IdentityGroup("device_fingerprint", "D", tuple(f"C{i}" for i in range(20)), 20),
        IdentityGroup("ip_address", "I", tuple(f"C{i}" for i in range(20)), 20),
    ]
    out = cap_collisions(groups, {"device_fingerprint": 5, "ip_address": 15})
    assert len(out[0].cards) == 5
    assert len(out[1].cards) == 15


def test_edge_weight_falls_with_cardinality():
    """
    Rarity weighting: a value shared by two cards is strong evidence, the same
    value shared by 1,492 is nearly none.
    """
    strong = _identity_weight(2, "device_fingerprint")
    medium = _identity_weight(8, "device_fingerprint")
    noise = _identity_weight(1492, "device_fingerprint")
    assert strong > medium > noise
    assert strong > 0.6
    assert noise < 0.15


def test_ip_edges_are_discounted_against_device_edges():
    """IPs are noisier evidence than devices and must weigh less at equal cardinality."""
    assert _identity_weight(4, "ip_address") < _identity_weight(4, "device_fingerprint")


def test_edge_weights_are_bounded_and_cited(graph):
    """Every edge weight is in (0, 1] and carries at least one citation."""
    for u, v, data in graph.edges(data=True):
        assert 0.0 < data["weight"] <= 1.0
        assert data["n_shared"] >= 1
        assert len(data["shared"]) == data["n_shared"]
        assert set(data["shared_types"]).issubset(set(GRAPH.identity_attributes))


def test_ring_members_are_connected(graph, rings_df):
    """
    Most planted rings must be connected in the identity graph.

    Not all: a ring whose members all defected to their own devices creates no
    identity edge and is legitimately invisible to the graph. That is the honest
    ceiling, so the assertion is a majority, not a totality.
    """
    connected = 0
    for ring in rings_df.itertuples(index=False):
        members = [c for c in ring.card_id_list if c in graph]
        sub = graph.subgraph(members)
        if sub.number_of_edges() > 0:
            connected += 1
    assert connected / len(rings_df) >= 0.8


def test_build_is_deterministic(transactions, cards_df):
    """Same input, same graph — the reproducibility bar."""
    a = build_identity_graph(transactions, cards_df=cards_df)
    b = build_identity_graph(transactions, cards_df=cards_df)
    assert a.number_of_nodes() == b.number_of_nodes()
    assert a.number_of_edges() == b.number_of_edges()
    assert set(map(frozenset, a.edges())) == set(map(frozenset, b.edges()))
