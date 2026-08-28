"""
Candidate generator tests: ring recall, ego coverage, and deduplication.

The recall assertion is the load-bearing one. If candidate generation drops
below the bar, nothing downstream can recover — a ring never proposed is a ring
never caught, no matter how good the model is.
"""

from __future__ import annotations

import pytest

from src.candidate_generator import (
    Candidate,
    _ego_candidates,
    deduplicate,
    evaluate_ring_recall,
    generate_candidates,
    identity_ego_sets,
    label_candidates,
)
from src.config import CANDIDATES
from src.utils import jaccard, overlap_fraction


def test_ring_recall_meets_bar(candidates, rings_df):
    """
    Ring recall must be at least 0.90.

    "Covered" means some candidate contains more than half of a planted ring's
    cards. Padding is allowed — a candidate that finds all 5 ring cards plus 40
    innocents has found the ring; trimming it is the scorer's job.

    Why 0.90 and not 1.000
    ----------------------
    Recall has a hard ceiling below 1.0 in this dataset, by construction. 15% of
    ring members are *defectors* who run their own fresh device and connection
    instead of the ring pool. They create no identity edge at all, so no
    identity-based generator can propose them — and when enough of a small ring
    defects, its coverage drops under 50% and it is genuinely unfindable on the
    graph.

    The measured figures are 0.942 at full scale (113/120 rings, 737k
    transactions) and ~0.96 at this test scale. The bar sits at 0.90 so it holds
    at both scales rather than passing here and failing on the real run.
    Removing the defectors would push recall to 1.000 and would be tuning the
    data to flatter the metric.
    """
    metrics = evaluate_ring_recall(candidates, rings_df)
    assert metrics["ring_recall"] >= 0.90, (
        f"ring recall {metrics['ring_recall']:.3f} below the 0.90 bar; "
        f"missed {metrics['rings_missed']}"
    )


def test_ego_generator_carries_recall(graph, rings_df):
    """
    The identity-ego generator alone should reach most of the recall.

    This is the claim the architecture rests on: a ring is defined by its shared
    device pool, so the ego set of a ring device *is* the ring. If this drops,
    the ensemble is papering over a broken ego generator.
    """
    ego_only = _ego_candidates(graph)
    metrics = evaluate_ring_recall(ego_only, rings_df)
    assert metrics["ring_recall"] >= 0.90


def test_ensemble_beats_every_single_generator(candidates, rings_df):
    """The ensembled recall is at least as good as any generator alone."""
    metrics = evaluate_ring_recall(candidates, rings_df)
    for generator, recall in metrics["recall_by_generator_alone"].items():
        assert metrics["ring_recall"] >= recall, f"{generator} alone beat the ensemble"


def test_all_three_generators_contribute(candidates):
    """All three generators survive into the final candidate set."""
    sources = {c.source for c in candidates}
    assert sources == {"ego", "cc", "louvain"}, f"missing generators: {{'ego','cc','louvain'}} - {sources}"


def test_ego_sets_respect_the_collision_cap(graph):
    """
    Ego sets are read off capped edges, so they inherit the cap.

    Reading them from raw transactions instead would re-introduce the
    1,492-card blob that the graph builder exists to prevent.
    """
    from src.config import GRAPH

    ego = identity_ego_sets(graph)
    assert ego, "no ego sets derived from the graph"
    for (attr, value), cards in ego.items():
        assert len(cards) <= max(GRAPH.min_collision_cap, 60), (
            f"ego set for {attr}={value} has {len(cards)} cards; the cap was not applied"
        )


def test_candidates_respect_size_bounds(candidates):
    """Every candidate sits inside the configured size window."""
    for c in candidates:
        assert CANDIDATES.min_candidate_size <= c.size <= CANDIDATES.max_candidate_size


def test_candidates_are_unique_and_sorted(candidates):
    """Card tuples are sorted and no exact duplicate survives deduplication."""
    seen = set()
    for c in candidates:
        assert list(c.cards) == sorted(c.cards)
        key = tuple(c.cards)
        assert key not in seen, f"duplicate candidate {c.candidate_id}"
        seen.add(key)


def test_deduplication_removes_near_duplicates(candidates):
    """No two surviving candidates exceed the Jaccard threshold."""
    from collections import defaultdict

    index = defaultdict(list)
    for i, c in enumerate(candidates):
        for card in c.cards:
            index[card].append(i)

    for i, c in enumerate(candidates):
        rivals = {j for card in c.cards for j in index[card] if j != i}
        for j in rivals:
            sim = jaccard(c.cards, candidates[j].cards)
            assert sim <= CANDIDATES.jaccard_dedup_threshold, (
                f"{c.candidate_id} and {candidates[j].candidate_id} overlap at Jaccard {sim:.2f}"
            )


def test_dedup_keeps_the_larger_of_two_overlapping_sets():
    """
    When two candidates are near-duplicates, the larger survives.

    Regression guard. Keeping the smaller one silently halved a 6-card ring's
    coverage to 0.50 — below the recall threshold — and cost real recall.
    """
    big = Candidate("BIG", tuple(f"C{i}" for i in range(6)), "ego", "device=D")
    small = Candidate("SMALL", tuple(f"C{i}" for i in range(4)), "ego", "device=D2")
    kept = deduplicate([small, big])
    assert len(kept) == 1
    assert kept[0].candidate_id == "BIG"
    assert "SMALL" in kept[0].merged_from


def test_dedup_keeps_genuinely_distinct_sets():
    """Two disjoint candidates both survive."""
    a = Candidate("A", ("C1", "C2", "C3"), "ego", "device=D1")
    b = Candidate("B", ("C4", "C5", "C6"), "ego", "device=D2")
    assert len(deduplicate([a, b])) == 2


def test_merged_candidates_are_recorded(candidates):
    """Deduplication is auditable: absorbed candidates are named on their survivor."""
    assert any(c.merged_from for c in candidates), "no merge provenance recorded"


def test_labels_match_the_overlap_rule(candidates, labels, rings_df):
    """
    A candidate is positive exactly when it covers >50% of some planted ring.

    Re-derived here independently of `label_candidates` so a bug in the labeller
    cannot make the model look good against its own mistake.
    """
    ring_cards = {r.ring_id: set(r.card_id_list) for r in rings_df.itertuples(index=False)}
    by_id = {c.candidate_id: c for c in candidates}

    for row in labels.itertuples(index=False):
        cand = by_id[row.candidate_id]
        best = max((overlap_fraction(cand.cards, cs) for cs in ring_cards.values()), default=0.0)
        expected = int(best > CANDIDATES.ring_overlap_threshold)
        assert row.is_fraud_ring == expected, f"{row.candidate_id} mislabelled (overlap {best:.2f})"


def test_positive_rate_is_neither_degenerate_nor_trivial(labels):
    """Both classes are present in usable quantity."""
    rate = labels["is_fraud_ring"].mean()
    assert 0.01 < rate < 0.95, f"degenerate positive rate {rate:.3f}"


def test_candidate_worlds_are_assigned(candidates):
    """Every candidate carries a world, which the CV splitter groups on."""
    for c in candidates:
        assert c.world_id >= 0


def test_generation_is_deterministic(graph):
    """Same graph, same candidates."""
    a = generate_candidates(graph)
    b = generate_candidates(graph)
    assert [tuple(c.cards) for c in a] == [tuple(c.cards) for c in b]
