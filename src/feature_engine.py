"""
Feature engine — 31 features per candidate, under 50ms at p99.

Three families, and the split is the argument of the whole system:

  Structural (10)   — what the identity graph says about the shape of the set
  Identity   (8)    — how much infrastructure the set actually shares
  Behavioural (13)  — what the set *did*: probe, escalate, burst, cash out

Structure alone cannot separate a fraud ring from a family sharing a tablet;
both are small dense cliques on one device. Behaviour alone cannot separate a
ring from twelve unrelated people who each happened to buy a recharge. You need
both, and the model needs to be free to weight them. Hence: identity links the
graph, behaviour ranks the candidate.

Which transactions get measured
-------------------------------
Behavioural features are computed over the candidate's **shared-infrastructure
slice** — the transactions of candidate cards that ran on a device, IP or email
that at least two candidate cards used — not over those cards' entire histories.

This matters enormously. A ring member makes ~300 normal transactions and ~4
ring transactions. Averaged over everything, `probe_fraction` for a real ring
is 0.013 and completely invisible. Restricted to the shared infrastructure it
is 0.6 and obvious. We are scoring what the cluster did *together*, which is
the only thing a cluster-level detector should be scoring. The slice is derived
purely from the candidate definition — no labels, no ground truth, nothing that
would not exist at inference time.

Vulcan scores are deliberately NOT features. Ringfence's score is computed
without them, so that composing the two signals downstream is genuinely
combining two independent opinions rather than counting one twice.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from src.config import FEATURES_CSV, MERCHANTS_CSV, PROCESSED_DIR
from src.utils import get_logger, safe_div, shannon_entropy, timed

log = get_logger("ringfence.feature_engine")

# Thresholds that define the behavioural vocabulary, in INR.
PROBE_AMOUNT_MAX = 10.0
CASHOUT_AMOUNT_MIN = 10_000.0
SMALL_MERCHANT_AVG_TICKET = 50.0
BURST_WINDOW_S = 300  # five minutes

FEATURE_NAMES: tuple[str, ...] = (
    # --- A. Structural: graph geometry (10) ---
    "cand_size",
    "cand_density",
    "cand_avg_degree",
    "cand_clustering_coeff",
    "cand_num_components",
    "cand_expansion",
    "cand_cut_ratio",
    "cand_modularity_contribution",
    "cand_ego_overlap_ratio",
    "cand_diameter",
    # --- B. Identity: raw sharing counts (8) ---
    "n_unique_devices",
    "n_unique_ips",
    "n_unique_emails",
    "max_device_reuse",
    "max_ip_reuse",
    "max_email_reuse",
    "device_to_card_ratio",
    "ip_to_card_ratio",
    # --- C. Behavioural: probe & cashout signatures (13) ---
    "probe_fraction",
    "amount_escalation",
    "burst_compression",
    "small_merchant_concentration",
    "time_span_hours",
    "velocity_txn_per_hour",
    "cashout_fraction",
    "probe_to_cashout_gap_hours",
    "merchant_diversity",
    "geo_spread",
    "amount_entropy",
    "night_fraction",
    "weekend_fraction",
)

assert len(FEATURE_NAMES) == 31, f"expected 31 features, found {len(FEATURE_NAMES)}"

FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "structural": FEATURE_NAMES[:10],
    "identity": FEATURE_NAMES[10:18],
    "behavioural": FEATURE_NAMES[18:],
}


# ──────────────────────────────────────────────────────────────────────────
# Transaction index
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class TransactionIndex:
    """
    Columnar, per-card transaction index built once and reused for every
    candidate.

    Everything is integer-coded numpy: strings are factorised to int32 codes so
    membership tests become `np.isin` over a contiguous block rather than a
    Python set lookup per row. Transactions are sorted by card so a candidate's
    rows are a handful of contiguous slices instead of a scattered gather.

    This is where the 50ms budget is actually won. Re-deriving a candidate's
    transactions from the DataFrame per call costs ~400ms; this costs ~2ms.
    """

    card_to_slice: dict[str, tuple[int, int]]
    amount: np.ndarray
    ts: np.ndarray            # int64 unix seconds
    hour: np.ndarray          # int8
    weekday: np.ndarray       # int8
    merchant_code: np.ndarray
    device_code: np.ndarray
    ip_code: np.ndarray
    email_code: np.ndarray
    subnet_code: np.ndarray
    vulcan: np.ndarray
    txn_id: np.ndarray        # object array of transaction ids, for citations
    merchant_id: np.ndarray   # object array of merchant ids, for citations
    device_name: np.ndarray   # code -> original device string
    ip_name: np.ndarray
    email_name: np.ndarray
    merchant_name: np.ndarray
    merchant_is_small: np.ndarray   # by merchant code
    card_devices: dict[str, np.ndarray]
    card_ips: dict[str, np.ndarray]
    card_emails: dict[str, np.ndarray]

    @property
    def n_transactions(self) -> int:
        """Total transactions held in the index."""
        return int(self.amount.size)


def build_index(transactions_df: pd.DataFrame, merchants_df: pd.DataFrame | None = None) -> TransactionIndex:
    """
    Build the columnar transaction index.

    `merchants_df` supplies each merchant's average ticket, which defines the
    "small merchant" flag. If it is not supplied we derive the average ticket
    empirically from the transactions themselves — the feature stays available
    either way, which matters because the API has to score candidates without a
    merchant reference table loaded.
    """
    with timed("features.build_index", log):
        df = transactions_df.sort_values("card_id", kind="stable").reset_index(drop=True)

        cards = df["card_id"].to_numpy()
        boundaries: dict[str, tuple[int, int]] = {}
        change = np.flatnonzero(cards[1:] != cards[:-1]) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [len(cards)]])
        for s, e in zip(starts, ends):
            boundaries[cards[s]] = (int(s), int(e))

        device_code, device_name = pd.factorize(df["device_fingerprint"])
        ip_code, ip_name = pd.factorize(df["ip_address"])
        email_code, email_name = pd.factorize(df["email_hash"])
        merchant_code, merchant_name = pd.factorize(df["merchant_id"])

        # /24 subnet, for geographic spread.
        subnets = df["ip_address"].astype(str).str.rsplit(".", n=1).str[0]
        subnet_code, _ = pd.factorize(subnets)

        ts = df["timestamp"].astype("int64") // 10**9
        dt = pd.to_datetime(df["timestamp"])

        amount = df["amount"].to_numpy(dtype=np.float64)

        if merchants_df is not None:
            ticket = merchants_df.set_index("merchant_id")["avg_ticket"]
            avg_by_code = np.array([float(ticket.get(m, np.nan)) for m in merchant_name])
        else:
            avg_by_code = (
                pd.Series(amount).groupby(merchant_code).mean().reindex(range(len(merchant_name))).to_numpy()
            )
        merchant_is_small = np.nan_to_num(avg_by_code, nan=np.inf) < SMALL_MERCHANT_AVG_TICKET

        idx = TransactionIndex(
            card_to_slice=boundaries,
            amount=amount,
            ts=ts.to_numpy(dtype=np.int64),
            hour=dt.dt.hour.to_numpy(dtype=np.int8),
            weekday=dt.dt.weekday.to_numpy(dtype=np.int8),
            merchant_code=np.asarray(merchant_code, dtype=np.int32),
            device_code=np.asarray(device_code, dtype=np.int32),
            ip_code=np.asarray(ip_code, dtype=np.int32),
            email_code=np.asarray(email_code, dtype=np.int32),
            subnet_code=np.asarray(subnet_code, dtype=np.int32),
            vulcan=df["vulcan_score"].to_numpy(dtype=np.float64),
            txn_id=df["transaction_id"].to_numpy(dtype=object),
            merchant_id=df["merchant_id"].to_numpy(dtype=object),
            device_name=np.asarray(device_name, dtype=object),
            ip_name=np.asarray(ip_name, dtype=object),
            email_name=np.asarray(email_name, dtype=object),
            merchant_name=np.asarray(merchant_name, dtype=object),
            merchant_is_small=merchant_is_small,
            card_devices={}, card_ips={}, card_emails={},
        )

        # Per-card distinct identity codes — tiny arrays, used to detect which
        # identity values are shared *within* a candidate.
        for card, (s, e) in boundaries.items():
            idx.card_devices[card] = np.unique(idx.device_code[s:e])
            idx.card_ips[card] = np.unique(idx.ip_code[s:e])
            idx.card_emails[card] = np.unique(idx.email_code[s:e])

    log.info(f"indexed {idx.n_transactions:,} transactions across {len(boundaries):,} cards")
    return idx


# ──────────────────────────────────────────────────────────────────────────
# Feature computation
# ──────────────────────────────────────────────────────────────────────────


def _shared_identity_codes(cards, per_card: dict[str, np.ndarray]) -> np.ndarray:
    """
    Identity codes used by two or more cards in the candidate.

    This is the definition of "shared infrastructure" that the behavioural
    slice keys off. A value one card used alone is that card's private life,
    not the cluster's joint activity.
    """
    counts: dict[int, int] = {}
    for card in cards:
        for code in per_card.get(card, ()):
            counts[int(code)] = counts.get(int(code), 0) + 1
    shared = [c for c, n in counts.items() if n >= 2]
    return np.asarray(shared, dtype=np.int32)


def _gather_rows(cards, index: TransactionIndex) -> np.ndarray:
    """Row indices for every transaction made by any card in the candidate."""
    slices = [index.card_to_slice[c] for c in cards if c in index.card_to_slice]
    if not slices:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in slices])


def _structural_features(cards: tuple[str, ...], G: nx.Graph, ego_ratio: float) -> dict[str, float]:
    """
    Graph-geometry features for the candidate's induced subgraph.

    `cand_diameter` is -1 when the subgraph is disconnected or too large to
    measure cheaply. -1 is a real value here, not a missing marker: LightGBM
    splits on it happily, and "this candidate is not even connected" is
    genuinely informative — rings are connected, sweeps of a shared NAT pool
    often are not.
    """
    n = len(cards)
    sub = G.subgraph(cards)
    m_in = sub.number_of_edges()
    possible = n * (n - 1) / 2

    degrees = dict(G.degree(cards))
    total_degree = sum(degrees.values())
    boundary = total_degree - 2 * m_in  # edges leaving the candidate

    m_total = G.number_of_edges()
    n_total = G.number_of_nodes()

    if n <= 50 and m_in > 0 and nx.is_connected(sub):
        diameter = float(nx.diameter(sub))
    else:
        diameter = -1.0

    modularity = (
        (m_in / m_total) - (total_degree / (2 * m_total)) ** 2 if m_total > 0 else 0.0
    )

    return {
        "cand_size": float(n),
        "cand_density": safe_div(m_in, possible),
        "cand_avg_degree": safe_div(2 * m_in, n),
        "cand_clustering_coeff": float(nx.average_clustering(sub)) if n >= 3 else 0.0,
        "cand_num_components": float(nx.number_connected_components(sub)),
        "cand_expansion": safe_div(boundary, n),
        "cand_cut_ratio": safe_div(boundary, n * max(n_total - n, 1)),
        "cand_modularity_contribution": float(modularity),
        "cand_ego_overlap_ratio": float(ego_ratio),
        "cand_diameter": diameter,
    }


def _identity_features(cards, rows: np.ndarray, index: TransactionIndex) -> dict[str, float]:
    """
    Raw sharing counts over the candidate's transactions.

    `max_*_reuse` counts CARDS per value, not transactions per value: one card
    hammering one device a thousand times is not evidence of a ring, three
    cards touching it once each is.
    """
    n = max(len(cards), 1)
    dev = index.device_code[rows]
    ip = index.ip_code[rows]
    email = index.email_code[rows]

    def max_card_reuse(per_card: dict[str, np.ndarray]) -> float:
        counts: dict[int, int] = {}
        for card in cards:
            for code in per_card.get(card, ()):
                counts[int(code)] = counts.get(int(code), 0) + 1
        return float(max(counts.values())) if counts else 0.0

    n_dev = float(np.unique(dev).size)
    n_ip = float(np.unique(ip).size)

    return {
        "n_unique_devices": n_dev,
        "n_unique_ips": n_ip,
        "n_unique_emails": float(np.unique(email).size),
        "max_device_reuse": max_card_reuse(index.card_devices),
        "max_ip_reuse": max_card_reuse(index.card_ips),
        "max_email_reuse": max_card_reuse(index.card_emails),
        "device_to_card_ratio": safe_div(n_dev, n),
        "ip_to_card_ratio": safe_div(n_ip, n),
    }


def _behavioural_features(rows: np.ndarray, index: TransactionIndex) -> dict[str, float]:
    """
    Probe-and-cashout signature over the shared-infrastructure slice.

    The four that carry the pattern:
      probe_fraction              — how much of this is INR <10 reconnaissance
      amount_escalation           — max/min; a ring runs ~5,000x, a family ~100x
      burst_compression           — how much of it landed inside five minutes
      probe_to_cashout_gap_hours  — the delay between the two phases
    """
    if rows.size == 0:
        return {name: 0.0 for name in FEATURE_FAMILIES["behavioural"]}

    amount = index.amount[rows]
    ts = np.sort(index.ts[rows])
    n = float(rows.size)

    span_s = float(ts[-1] - ts[0])
    span_h = span_s / 3600.0

    # Burst compression: densest five-minute window, as a share of the whole.
    # searchsorted over the sorted timestamps makes this O(n log n) with no
    # Python loop — a rolling window in pandas costs ~40x more here.
    right = np.searchsorted(ts, ts + BURST_WINDOW_S, side="right")
    max_in_window = float((right - np.arange(ts.size)).max())

    is_probe = amount < PROBE_AMOUNT_MAX
    is_cashout = amount > CASHOUT_AMOUNT_MIN

    # Gap measured from the first probe to the first cashout that follows it.
    gap_h = -1.0
    if is_probe.any() and is_cashout.any():
        order = np.argsort(index.ts[rows])
        ts_sorted = index.ts[rows][order]
        probe_sorted = is_probe[order]
        cash_sorted = is_cashout[order]
        first_probe_t = ts_sorted[probe_sorted][0]
        later = ts_sorted[cash_sorted & (ts_sorted > first_probe_t)]
        if later.size:
            gap_h = float((later[0] - first_probe_t) / 3600.0)

    hour = index.hour[rows]
    weekday = index.weekday[rows]

    return {
        "probe_fraction": float(is_probe.mean()),
        "amount_escalation": safe_div(float(amount.max()), float(amount.min()), default=1.0),
        "burst_compression": safe_div(max_in_window, n),
        "small_merchant_concentration": float(index.merchant_is_small[index.merchant_code[rows]].mean()),
        "time_span_hours": span_h,
        "velocity_txn_per_hour": safe_div(n, max(span_h, 1e-6), default=n),
        "cashout_fraction": float(is_cashout.mean()),
        "probe_to_cashout_gap_hours": gap_h,
        "merchant_diversity": safe_div(float(np.unique(index.merchant_code[rows]).size), n),
        "geo_spread": safe_div(float(np.unique(index.subnet_code[rows]).size), n),
        "amount_entropy": shannon_entropy(amount),
        "night_fraction": float(((hour >= 0) & (hour < 6)).mean()),
        "weekend_fraction": float((weekday >= 5).mean()),
    }


def compute_features(
    cards: tuple[str, ...] | list[str], G: nx.Graph, index: TransactionIndex
) -> dict[str, float]:
    """
    Compute all 31 features for one candidate.

    Returns a dict keyed exactly by FEATURE_NAMES, in order. Never raises on a
    degenerate candidate: an empty or unknown card set yields a zero vector, so
    a malformed API request produces a low score and a REVIEW rather than a 500.
    """
    cards = tuple(c for c in cards if c in G)
    if not cards:
        return {name: 0.0 for name in FEATURE_NAMES}

    all_rows = _gather_rows(cards, index)

    shared_dev = _shared_identity_codes(cards, index.card_devices)
    shared_ip = _shared_identity_codes(cards, index.card_ips)
    shared_email = _shared_identity_codes(cards, index.card_emails)

    if all_rows.size:
        mask = (
            np.isin(index.device_code[all_rows], shared_dev)
            | np.isin(index.ip_code[all_rows], shared_ip)
            | np.isin(index.email_code[all_rows], shared_email)
        )
        shared_rows = all_rows[mask]
    else:
        shared_rows = all_rows

    # Fall back to the full history when nothing is shared. This happens for
    # Louvain communities glued together by transitive edges rather than one
    # common value; scoring them on their whole history is the conservative
    # reading and keeps the feature vector defined.
    behaviour_rows = shared_rows if shared_rows.size else all_rows

    # Ego overlap: largest share of the candidate explained by one identity
    # value. 1.0 means "this candidate IS a device's user set".
    ego_ratio = 0.0
    for per_card, shared in (
        (index.card_devices, shared_dev),
        (index.card_ips, shared_ip),
        (index.card_emails, shared_email),
    ):
        for code in shared:
            covered = sum(1 for c in cards if code in per_card.get(c, ()))
            ego_ratio = max(ego_ratio, covered / len(cards))

    features: dict[str, float] = {}
    features.update(_structural_features(cards, G, ego_ratio))
    features.update(_identity_features(cards, all_rows if all_rows.size else behaviour_rows, index))
    features.update(_behavioural_features(behaviour_rows, index))

    return {name: float(features[name]) for name in FEATURE_NAMES}


def compute_candidate_meta(
    cards: tuple[str, ...] | list[str], index: TransactionIndex
) -> dict:
    """
    Non-feature context for the dossier: Vulcan aggregates, transaction counts,
    and the specific transaction IDs that evidence the probe and the cashout.

    Kept out of the feature vector on purpose — see the module docstring on why
    Vulcan is not an input to the Ringfence score.
    """
    cards = [c for c in cards if c in index.card_to_slice]
    rows = _gather_rows(cards, index)
    if rows.size == 0:
        return {
            "n_transactions": 0, "vulcan_mean": 0.0, "vulcan_max": 0.0,
            "probe_txn_ids": [], "cashout_txn_ids": [], "shared_devices": [],
            "min_amount": 0.0, "max_amount": 0.0, "window_start": None, "window_end": None,
        }

    shared_dev = _shared_identity_codes(cards, index.card_devices)
    mask = np.isin(index.device_code[rows], shared_dev)
    focus = rows[mask] if mask.any() else rows

    amount = index.amount[focus]
    probes = focus[amount < PROBE_AMOUNT_MAX]
    cashouts = focus[amount > CASHOUT_AMOUNT_MIN]

    return {
        "n_transactions": int(focus.size),
        "vulcan_mean": float(index.vulcan[focus].mean()),
        "vulcan_max": float(index.vulcan[focus].max()),
        "vulcan_cashout_mean": float(index.vulcan[cashouts].mean()) if cashouts.size else 0.0,
        "probe_txn_ids": [str(t) for t in index.txn_id[probes[:5]]],
        "cashout_txn_ids": [str(t) for t in index.txn_id[cashouts[:5]]],
        "probe_merchants": sorted({str(m) for m in index.merchant_id[probes[:20]]}),
        "cashout_merchants": sorted({str(m) for m in index.merchant_id[cashouts[:20]]}),
        "shared_devices": [str(index.device_name[c]) for c in shared_dev[:10]],
        "min_amount": float(amount.min()),
        "max_amount": float(amount.max()),
        "window_start": str(pd.to_datetime(index.ts[focus].min(), unit="s")),
        "window_end": str(pd.to_datetime(index.ts[focus].max(), unit="s")),
    }


def compute_features_batch(candidates, G: nx.Graph, index: TransactionIndex) -> pd.DataFrame:
    """
    Compute features for many candidates and record the per-candidate latency
    distribution.

    Returns a DataFrame with `candidate_id` plus the 31 feature columns. The
    latency percentiles are logged, not assumed — the <50ms p99 claim in the
    README is this measurement, and `tests/test_scorer.py` asserts against it.
    """
    from src.utils import LATENCY_RECORDER

    LATENCY_RECORDER.samples.pop("feature.per_candidate", None)
    rows = []
    for cand in candidates:
        with LATENCY_RECORDER.time("feature.per_candidate"):
            feats = compute_features(cand.cards, G, index)
        rows.append({"candidate_id": cand.candidate_id, **feats})

    df = pd.DataFrame(rows)
    pct = LATENCY_RECORDER.percentiles("feature.per_candidate")
    log.info(
        f"computed 31 features for {len(df):,} candidates — "
        f"p50={pct['p50']:.2f}ms p95={pct['p95']:.2f}ms p99={pct['p99']:.2f}ms max={pct['max']:.2f}ms"
    )
    return df


def main() -> None:
    """CLI: compute the feature store for all persisted candidates."""
    argparse.ArgumentParser(description="Compute Ringfence candidate features").parse_args()

    from src.candidate_generator import load_candidates
    from src.data_generator import load_transactions
    from src.graph_builder import load_graph
    from src.utils import LATENCY_RECORDER, write_json

    txns = load_transactions()
    merchants = pd.read_csv(MERCHANTS_CSV)
    G = load_graph()
    candidates = load_candidates()

    index = build_index(txns, merchants)
    features = compute_features_batch(candidates, G, index)
    features.to_csv(FEATURES_CSV, index=False)
    write_json(PROCESSED_DIR / "feature_latency.json", LATENCY_RECORDER.percentiles("feature.per_candidate"))
    log.info(f"wrote {FEATURES_CSV}")


if __name__ == "__main__":
    main()
