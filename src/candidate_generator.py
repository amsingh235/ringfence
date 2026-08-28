"""
Candidate generation — recall-first, unapologetically.

The job here is to *propose every ring*. Precision is not this module's
problem; it is the GBM's. That division is the second of the three design
decisions this system rests on, and it is deliberate: a ring the generator
never proposes is a ring the model can never catch, while a bad candidate the
generator proposes costs one row of inference.

Three generators are ensembled because each fails in a different direction:

- **Louvain, six resolutions.** Finds cohesive blocks. At gamma=0.5 it smears
  loosely-connected clusters together; at gamma=3.0 it shatters real rings.
  Running the sweep and keeping everything means we never have to pick.
- **Connected components, four weight floors.** Finds infrastructure islands.
  At weight >= 0.3 a NAT pool merges half the graph; at >= 0.9 only near-clone
  identities survive. Again: run the sweep, keep everything.
- **Identity-ego sets.** For every device / IP / email touched by more than one
  card, emit exactly the set of cards that touched it. This is the one that
  matters. A ring is *defined* by its shared device pool, so the ego set of a
  ring device is the ring, exactly, with no smearing. Louvain and CC alone
  leave recall in the low tenths; the ego generator is what lifts it to ~1.0.

Deduplication is greedy, not agglomerative. Candidates are visited in priority
order — ego first (most precise), then CC, then Louvain — and a candidate is
dropped if it overlaps an already-kept one at Jaccard > 0.5. Union-merging
instead would chain overlapping sets into one blob, which is the exact failure
we spent the graph builder avoiding.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

from src.config import CANDIDATES, CandidateConfig, PROCESSED_DIR, RANDOM_SEED
from src.data_generator import load_ground_truth
from src.graph_builder import load_graph
from src.utils import get_logger, jaccard, overlap_fraction, timed, write_json

log = get_logger("ringfence.candidate_generator")

# Lower number = visited first during deduplication = kept in preference.
_SOURCE_PRIORITY = {"ego": 0, "cc": 1, "louvain": 2}


@dataclass
class Candidate:
    """A proposed set of cards that might be one ring."""

    candidate_id: str
    cards: tuple[str, ...]
    source: str            # "ego" | "cc" | "louvain"
    source_detail: str     # e.g. "device_fingerprint=DEV_003412" or "gamma=1.5"
    world_id: int = -1
    merged_from: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Number of cards in the candidate."""
        return len(self.cards)

    def to_dict(self) -> dict:
        """JSON-serialisable form, used for the on-disk candidate store."""
        return {
            "candidate_id": self.candidate_id,
            "cards": list(self.cards),
            "source": self.source,
            "source_detail": self.source_detail,
            "world_id": self.world_id,
            "size": self.size,
            "merged_from": self.merged_from,
        }


# ──────────────────────────────────────────────────────────────────────────
# Generator 1: Louvain communities
# ──────────────────────────────────────────────────────────────────────────


def _louvain_candidates(G: nx.Graph, resolutions, seed: int = RANDOM_SEED) -> list[Candidate]:
    """
    Run Louvain at several resolutions and emit every community as a candidate.

    Falls back to networkx's own Louvain implementation if `python-louvain` is
    not installed, so the pipeline never hard-fails on an optional dependency.
    """
    try:
        import community as community_louvain  # python-louvain

        def partition(g, gamma):
            mapping = community_louvain.best_partition(g, resolution=gamma, random_state=seed, weight="weight")
            buckets = defaultdict(list)
            for node, comm in mapping.items():
                buckets[comm].append(node)
            return list(buckets.values())

        backend = "python-louvain"
    except ImportError:  # pragma: no cover - exercised only without the extra
        log.warning("python-louvain not installed; falling back to networkx.louvain_communities")

        def partition(g, gamma):
            return [list(c) for c in nx.community.louvain_communities(g, resolution=gamma, seed=seed, weight="weight")]

        backend = "networkx"

    out: list[Candidate] = []
    for gamma in resolutions:
        communities = partition(G, gamma)
        kept = 0
        for i, comm in enumerate(communities):
            if len(comm) < 2:
                continue  # singletons are not candidates
            out.append(
                Candidate(
                    candidate_id=f"LOUV_g{gamma}_{i}",
                    cards=tuple(sorted(comm)),
                    source="louvain",
                    source_detail=f"gamma={gamma}",
                )
            )
            kept += 1
        log.info(f"  louvain[{backend}] gamma={gamma}: {len(communities)} communities, {kept} with >=2 cards")
    return out


# ──────────────────────────────────────────────────────────────────────────
# Generator 2: connected components at edge-weight thresholds
# ──────────────────────────────────────────────────────────────────────────


def _connected_component_candidates(G: nx.Graph, thresholds) -> list[Candidate]:
    """
    Emit connected components of the subgraph filtered to edges of at least
    each weight threshold.

    Sweeping the threshold is how we trade fusion against fragmentation without
    committing to a single answer: low floors merge weakly-linked populations,
    high floors isolate only cards sharing rare identity values.
    """
    out: list[Candidate] = []
    for t in thresholds:
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True) if d.get("weight", 0.0) >= t)
        comps = [c for c in nx.connected_components(H) if len(c) >= 2]
        for i, comp in enumerate(comps):
            out.append(
                Candidate(
                    candidate_id=f"CC_w{t}_{i}",
                    cards=tuple(sorted(comp)),
                    source="cc",
                    source_detail=f"min_edge_weight={t}",
                )
            )
        log.info(f"  connected components @ weight>={t}: {len(comps)} components with >=2 cards")
    return out


# ──────────────────────────────────────────────────────────────────────────
# Generator 3: identity-ego sets  (the one that carries recall)
# ──────────────────────────────────────────────────────────────────────────


def identity_ego_sets(G: nx.Graph) -> dict[tuple[str, str], set[str]]:
    """
    Reconstruct, from the graph's edge provenance, the set of cards sharing
    each identity value.

    Reading these back off the edges rather than off the raw transactions is
    intentional: the edges are already collision-capped, so the ego sets
    inherit the cap for free and we cannot accidentally re-introduce the
    1,492-card blob that capping exists to prevent.
    """
    ego: dict[tuple[str, str], set[str]] = defaultdict(set)
    for u, v, data in G.edges(data=True):
        for shared in data.get("shared", []):
            key = (shared["attribute"], shared["value"])
            ego[key].add(u)
            ego[key].add(v)
    return dict(ego)


def _ego_candidates(G: nx.Graph) -> list[Candidate]:
    """Emit one candidate per identity value shared by more than one card."""
    ego = identity_ego_sets(G)
    out = []
    for i, ((attr, value), cards) in enumerate(sorted(ego.items())):
        if len(cards) < 2:
            continue
        out.append(
            Candidate(
                candidate_id=f"EGO_{i}",
                cards=tuple(sorted(cards)),
                source="ego",
                source_detail=f"{attr}={value}",
            )
        )
    by_attr: dict[str, int] = defaultdict(int)
    for (attr, _), cards in ego.items():
        if len(cards) >= 2:
            by_attr[attr] += 1
    log.info(f"  identity-ego sets: {len(out)} " + ", ".join(f"{k}={v}" for k, v in sorted(by_attr.items())))
    return out


# ──────────────────────────────────────────────────────────────────────────
# Deduplication
# ──────────────────────────────────────────────────────────────────────────


def deduplicate(candidates: list[Candidate], cfg: CandidateConfig = CANDIDATES) -> list[Candidate]:
    """
    Greedy near-duplicate removal at Jaccard > threshold.

    Implemented with an inverted card -> kept-candidate index so we only
    compare against candidates that actually share a card. The naive all-pairs
    version is O(n^2) and, at ~10k candidates, is the slowest thing in the
    pipeline by an order of magnitude.

    Priority: ego sets first (they are the precise proposals), then connected
    components, then Louvain — and within a source, **larger first**. That last
    detail is not cosmetic. Visiting smaller sets first meant a 3-card subset
    of a 6-card ring got kept, the full 6-card ego set was absorbed into it as
    a near-duplicate, and the ring's coverage silently fell from 1.00 to 0.50 —
    below the recall threshold. When two candidates are near-duplicates, the
    larger one is the one that still contains the ring.

    A dropped candidate is recorded on its survivor's `merged_from`, so the
    audit trail can answer "why is there no candidate for X?" with "it was
    folded into Y".
    """
    ordered = sorted(candidates, key=lambda c: (_SOURCE_PRIORITY.get(c.source, 9), -c.size, c.candidate_id))

    kept: list[Candidate] = []
    card_index: dict[str, list[int]] = defaultdict(list)

    for cand in ordered:
        rivals = {i for card in cand.cards for i in card_index.get(card, ())}
        absorbed_by = None
        for i in rivals:
            if jaccard(cand.cards, kept[i].cards) > cfg.jaccard_dedup_threshold:
                absorbed_by = i
                break
        if absorbed_by is not None:
            kept[absorbed_by].merged_from.append(cand.candidate_id)
            continue
        idx = len(kept)
        kept.append(cand)
        for card in cand.cards:
            card_index[card].append(idx)

    log.info(f"deduplication: {len(candidates):,} -> {len(kept):,} unique candidates "
             f"(Jaccard > {cfg.jaccard_dedup_threshold})")
    return kept


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────


def generate_candidates(G: nx.Graph, cfg: CandidateConfig = CANDIDATES) -> list[Candidate]:
    """
    Run all three generators, filter by size, deduplicate, and assign worlds.

    Size filtering is a cost control, not a detection decision: a "candidate"
    of 400 cards is a population, and scoring it as a ring would be a category
    error. The bound is in config, and the count of candidates dropped by it is
    logged so the trade-off stays visible.
    """
    log.info("generating candidates (recall-first ensemble)")
    raw: list[Candidate] = []

    with timed("candidates.ego", log):
        raw += _ego_candidates(G)
    with timed("candidates.cc", log):
        raw += _connected_component_candidates(G, cfg.cc_weight_thresholds)
    with timed("candidates.louvain", log):
        raw += _louvain_candidates(G, cfg.louvain_resolutions)

    log.info(f"{len(raw):,} raw candidates before size filter and dedup")

    sized = [c for c in raw if cfg.min_candidate_size <= c.size <= cfg.max_candidate_size]
    log.info(f"size filter [{cfg.min_candidate_size}, {cfg.max_candidate_size}]: "
             f"dropped {len(raw) - len(sized):,}, kept {len(sized):,}")

    with timed("candidates.dedup", log):
        deduped = deduplicate(sized, cfg)

    # World assignment: majority world of member cards. Cross-world candidates
    # exist because the capped generic-device clique can span worlds; they are
    # counted and logged because they are the one place cross-world CV could
    # bleed, and a number we can see is better than an assumption we cannot.
    cross_world = 0
    for i, cand in enumerate(deduped):
        worlds = [G.nodes[c].get("world_id", -1) for c in cand.cards]
        counts: dict[int, int] = defaultdict(int)
        for w in worlds:
            counts[w] += 1
        majority, n_major = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
        cand.world_id = int(majority)
        if n_major < len(worlds):
            cross_world += 1
        cand.candidate_id = f"CAND_{i:06d}"

    log.info(f"{len(deduped):,} final candidates; {cross_world:,} span more than one world "
             f"({100 * cross_world / max(len(deduped), 1):.1f}%)")

    sizes = pd.Series([c.size for c in deduped])
    log.info(f"candidate size distribution: min={sizes.min()} p50={sizes.quantile(0.5):.0f} "
             f"p95={sizes.quantile(0.95):.0f} max={sizes.max()} mean={sizes.mean():.1f}")
    return deduped


def evaluate_ring_recall(
    candidates: list[Candidate], rings_df: pd.DataFrame, cfg: CandidateConfig = CANDIDATES
) -> dict:
    """
    Ring recall: what fraction of planted rings are covered by at least one
    candidate, where "covered" means the candidate contains more than 50% of
    the ring's cards.

    Also reports recall attributable to each generator in isolation, which is
    the honest way to show that the ego generator is doing the work rather than
    asserting it.
    """
    ring_cards = {r.ring_id: set(r.card_ids.split("|")) for r in rings_df.itertuples(index=False)}

    def recall_for(subset: list[Candidate]) -> tuple[float, list[str]]:
        missed = []
        hit = 0
        for ring_id, cards in ring_cards.items():
            best = max((overlap_fraction(c.cards, cards) for c in subset), default=0.0)
            if best > cfg.ring_overlap_threshold:
                hit += 1
            else:
                missed.append(ring_id)
        return hit / max(len(ring_cards), 1), missed

    overall, missed = recall_for(candidates)
    per_source = {}
    for source in ("ego", "cc", "louvain"):
        subset = [c for c in candidates if c.source == source]
        per_source[source] = round(recall_for(subset)[0], 4) if subset else 0.0

    sizes = pd.Series([c.size for c in candidates])
    result = {
        "n_candidates": len(candidates),
        "n_rings": len(ring_cards),
        "ring_recall": round(overall, 4),
        "rings_missed": missed,
        "recall_by_generator_alone": per_source,
        "candidate_size": {
            "min": int(sizes.min()) if len(sizes) else 0,
            "p50": float(sizes.quantile(0.50)) if len(sizes) else 0.0,
            "p95": float(sizes.quantile(0.95)) if len(sizes) else 0.0,
            "max": int(sizes.max()) if len(sizes) else 0,
            "mean": round(float(sizes.mean()), 2) if len(sizes) else 0.0,
        },
    }
    log.info(f"RING RECALL = {overall:.4f} ({len(ring_cards) - len(missed)}/{len(ring_cards)} rings covered)")
    log.info(f"  recall if we used only one generator: {per_source}")
    if missed:
        log.info(f"  missed rings: {missed[:10]}{' ...' if len(missed) > 10 else ''}")
    return result


def label_candidates(
    candidates: list[Candidate], rings_df: pd.DataFrame, cfg: CandidateConfig = CANDIDATES
) -> pd.DataFrame:
    """
    Attach the training label: `is_fraud_ring` = 1 when the candidate covers
    more than 50% of some planted ring.

    Returns one row per candidate with its label, best overlap, matched ring and
    world — the world column is what the cross-world CV splitter groups on.
    """
    ring_cards = {r.ring_id: set(r.card_ids.split("|")) for r in rings_df.itertuples(index=False)}

    rows = []
    for cand in candidates:
        best_ring, best_overlap = "", 0.0
        for ring_id, cards in ring_cards.items():
            ov = overlap_fraction(cand.cards, cards)
            if ov > best_overlap:
                best_ring, best_overlap = ring_id, ov
        rows.append(
            {
                "candidate_id": cand.candidate_id,
                "world_id": cand.world_id,
                "size": cand.size,
                "source": cand.source,
                "best_ring_id": best_ring if best_overlap > cfg.ring_overlap_threshold else "",
                "best_overlap": round(best_overlap, 4),
                "is_fraud_ring": int(best_overlap > cfg.ring_overlap_threshold),
            }
        )
    df = pd.DataFrame(rows)
    pos = int(df["is_fraud_ring"].sum())
    log.info(f"labels: {pos:,} positive / {len(df):,} candidates "
             f"({100 * pos / max(len(df), 1):.2f}% positive rate)")
    return df


def save_candidates(candidates: list[Candidate], path=None) -> None:
    """Persist candidates as JSON so downstream stages need not re-derive them."""
    path = path or (PROCESSED_DIR / "candidates.json")
    write_json(path, [c.to_dict() for c in candidates])
    log.info(f"wrote {len(candidates):,} candidates to {path}")


def load_candidates(path=None) -> list[Candidate]:
    """Load persisted candidates."""
    import json
    from pathlib import Path

    path = Path(path or (PROCESSED_DIR / "candidates.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Candidate(
            candidate_id=c["candidate_id"],
            cards=tuple(c["cards"]),
            source=c["source"],
            source_detail=c["source_detail"],
            world_id=c["world_id"],
            merged_from=c.get("merged_from", []),
        )
        for c in payload
    ]


def main() -> None:
    """CLI: generate candidates from the persisted graph and report recall."""
    parser = argparse.ArgumentParser(description="Generate ring candidates")
    parser.parse_args()

    G = load_graph()
    rings = load_ground_truth()

    candidates = generate_candidates(G)
    metrics = evaluate_ring_recall(candidates, rings)
    labels = label_candidates(candidates, rings)

    save_candidates(candidates)
    labels.to_csv(PROCESSED_DIR / "candidate_labels.csv", index=False)
    write_json(PROCESSED_DIR / "candidate_metrics.json", metrics)
    log.info(f"wrote {PROCESSED_DIR / 'candidate_labels.csv'} and candidate_metrics.json")


if __name__ == "__main__":
    main()
