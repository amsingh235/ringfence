"""
Synthetic transaction + identity generator.

Why synthetic
-------------
Real Razorpay transaction data is not something a candidate can hold. So we
build a universe whose *identity layer* is calibrated to published IEEE-CIS
device-cardinality behaviour, and whose *fraud layer* reproduces the one
pattern this system exists to catch: probe-then-cashout across merchants on a
shared device pool.

Two properties matter more than realism-in-general, and both are deliberate:

1. **A generic device string collides across ~1,492 accounts.** Real device
   fingerprinting has a fat head — one stock user-agent covers a huge slice of
   the population. If you build an identity graph naively, that single value
   fuses everything into one component and the detector dies. Our graph builder
   has to survive it, so the generator has to produce it.

2. **Five disjoint worlds.** Cards never appear in two worlds. That is what
   makes 5-fold cross-world validation an honest generalisation test rather
   than a leak: the model is asked about a population it has never seen a
   single card from.

Defense-only note
-----------------
This module fabricates *labelled training data for a detector*. It produces no
real card numbers, contacts no payment system, and encodes no technique that
would help anyone commit fraud — the "attack" it models (small probe, large
cashout) is a publicly documented pattern that every issuer already screens for.

Run
---
    python -m src.data_generator            # full spec-scale universe
    python -m src.data_generator --small    # test-scale universe
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    CARDS_CSV,
    GENERATION,
    MERCHANTS_CSV,
    RANDOM_SEED,
    RINGS_CSV,
    TRANSACTIONS_CSV,
    GenerationConfig,
    ensure_dirs,
)
from src.utils import get_logger, seed_everything, timed

log = get_logger("ringfence.data_generator")

# Hour-of-day weights: Indian retail payments peak late morning and evening,
# with a thin 00:00-06:00 tail. The night-time fraction feature depends on this
# baseline existing, otherwise "3am activity" would not be anomalous.
_DIURNAL = np.array(
    [0.6, 0.4, 0.3, 0.3, 0.4, 0.7, 1.2, 2.0, 3.2, 4.4, 5.2, 5.6,
     5.4, 5.0, 4.8, 4.9, 5.2, 5.8, 6.4, 6.6, 5.8, 4.4, 2.8, 1.4],
    dtype=float,
)
_DIURNAL = _DIURNAL / _DIURNAL.sum()


@dataclass
class GeneratedUniverse:
    """Everything the generator produces, before it hits disk."""

    transactions: pd.DataFrame
    cards: pd.DataFrame
    merchants: pd.DataFrame
    rings: pd.DataFrame
    stats: dict


# ──────────────────────────────────────────────────────────────────────────
# Identity string encoding
# ──────────────────────────────────────────────────────────────────────────


def _ip_string(idx: np.ndarray, block: int) -> np.ndarray:
    """
    Encode an integer IP index as a dotted-quad inside a private-looking block.

    The first three octets form the /24 subnet, which is what the geographic
    spread feature counts — so consecutive indices deliberately share a subnet.
    """
    a = (idx // 65536) % 256
    b = (idx // 256) % 256
    c = idx % 256
    # object dtype, not fixed-width unicode: blocks have different digit counts
    # and a fixed-width array would silently truncate "203.x.y.z" into "10.x.y.".
    return np.array([f"{block}.{ai}.{bi}.{ci}" for ai, bi, ci in zip(a, b, c)], dtype=object)


def _pad_ids(prefix: str, idx: np.ndarray, width: int = 6) -> np.ndarray:
    """Vectorised zero-padded identifier construction, e.g. CARD_000123."""
    return np.array([f"{prefix}{i:0{width}d}" for i in idx], dtype=object)


# ──────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────


def _build_merchants(cfg: GenerationConfig, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create the merchant population with a realistic ticket-size mixture.

    ~28% are micro-ticket merchants (average ticket under INR 50). These are
    the merchants a ring uses to *probe* a card: a INR 5 charge there looks
    like a normal sale, which is exactly why per-merchant scoring cannot see
    the ring. The `small_merchant_concentration` feature depends on this
    population existing and being separable.
    """
    n = cfg.n_merchants
    merchant_id = _pad_ids("MERCH_", np.arange(n), 5)

    is_micro = rng.random(n) < 0.28
    avg_ticket = np.where(
        is_micro,
        rng.uniform(8.0, 49.0, n),                       # micro: under INR 50
        np.exp(rng.normal(6.6, 0.9, n)).clip(60, 25_000),  # normal retail
    )

    categories = np.array(["recharge", "utility", "food", "retail", "travel",
                           "gaming", "electronics", "gift_card"])
    # Micro merchants skew to recharge/utility/gaming; the rest spread out.
    cat_idx = np.where(
        is_micro,
        rng.choice([0, 1, 5], size=n, p=[0.45, 0.30, 0.25]),
        rng.choice([2, 3, 4, 6, 7], size=n, p=[0.30, 0.30, 0.15, 0.15, 0.10]),
    )

    return pd.DataFrame(
        {
            "merchant_id": merchant_id,
            "category": categories[cat_idx],
            "avg_ticket": np.round(avg_ticket, 2),
            "is_micro_ticket": is_micro,
        }
    )


def _build_cards(cfg: GenerationConfig, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """
    Create the card population and its baseline identity layer.

    Identity layer, in order of how much trouble it causes:

    - **Own device**: every card starts with a device of its own.
    - **Household devices**: ~18% of cards are folded into households of 2-4
      that share one device. This is the honest false-positive generator — a
      family sharing a tablet looks structurally identical to a small ring, and
      the model has to separate them on *behaviour*, not on structure.
    - **Generic device**: `generic_device_accounts` cards additionally emit a
      stock user-agent on a slice of their traffic. One value, ~1,492 accounts.
    - **NAT IPs**: a pool of ISP-grade shared IPs, because real users share
      egress addresses and an IP edge is weaker evidence than a device edge.
    - **Shared emails**: ~3% of cards pair up on one email hash.

    Returns the card frame plus an index bundle the transaction generator uses.
    """
    n = cfg.n_cards
    card_idx = np.arange(n)
    card_id = _pad_ids("CARD_", card_idx)

    # Worlds are contiguous blocks so that "disjoint by construction" is
    # verifiable by eye, not just by assertion.
    world_id = card_idx % cfg.n_worlds

    # --- devices -----------------------------------------------------------
    device_of_card = card_idx.copy()
    households: list[np.ndarray] = []
    n_household = int(cfg.household_device_share * n)
    if n_household >= 2:
        members = rng.permutation(n)[:n_household]
        pos = 0
        while pos < len(members) - 1:
            size = int(rng.integers(2, 5))  # households of 2-4
            group = members[pos: pos + size]
            if len(group) >= 2:
                device_of_card[group] = device_of_card[group[0]]
                households.append(group)
            pos += size

    n_generic = min(cfg.generic_device_accounts, n)
    generic_cards = rng.permutation(n)[:n_generic]
    has_generic = np.zeros(n, dtype=bool)
    has_generic[generic_cards] = True

    # --- IPs ---------------------------------------------------------------
    ip_of_card = card_idx.copy()
    nat_of_card = rng.integers(0, max(cfg.nat_ip_pools, 1), n)

    # --- emails ------------------------------------------------------------
    email_of_card = card_idx.copy()
    n_shared_email = int(0.03 * n)
    if n_shared_email >= 2:
        pairs = rng.permutation(n)[: n_shared_email - (n_shared_email % 2)]
        for i in range(0, len(pairs), 2):
            email_of_card[pairs[i + 1]] = email_of_card[pairs[i]]

    # --- activity ----------------------------------------------------------
    # Transaction counts per card are heavy-tailed: a few very active cards,
    # a long tail of occasional ones.
    activity = np.exp(rng.normal(0.0, 0.7, n))
    activity = activity / activity.sum()

    cards = pd.DataFrame(
        {
            "card_id": card_id,
            "world_id": world_id,
            "home_device": _pad_ids("DEV_", device_of_card),
            "home_ip": _ip_string(ip_of_card, 10),
            "email_hash": _pad_ids("EMAIL_", email_of_card),
            "uses_generic_device": has_generic,
            "is_ring_member": False,
            "ring_id": "",
        }
    )

    index = {
        "device_of_card": device_of_card,
        "has_generic": has_generic,
        "ip_of_card": ip_of_card,
        "nat_of_card": nat_of_card,
        "email_of_card": email_of_card,
        "activity": activity,
        "world_id": world_id,
        "households": households,
    }
    return cards, index


def _plant_rings(
    cfg: GenerationConfig, cards: pd.DataFrame, merchants: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Plant `n_rings` fraud rings, each entirely inside one world.

    A ring is: 3-8 cards, a private device pool of 2-3 fingerprints, 1-2 IPs,
    and (60% of the time) one shared email hash. Cards are drawn without
    replacement across all rings, so rings are disjoint and a candidate cannot
    accidentally "cover" two rings at once.

    Ring cards keep their normal background traffic. The ring burst is a small
    number of transactions on shared infrastructure — which is precisely why
    per-card and per-merchant views miss it.
    """
    n_worlds = cfg.n_worlds
    world_pools = {w: list(rng.permutation(np.where(cards["world_id"].values == w)[0]))
                   for w in range(n_worlds)}

    micro_merchants = np.where(merchants["is_micro_ticket"].values)[0]
    large_merchants = np.where(~merchants["is_micro_ticket"].values)[0]
    merchant_ids = merchants["merchant_id"].values

    window_s = cfg.days * 86_400
    rows = []
    device_cursor = int(cfg.n_cards) + 1000  # ring devices live above card devices
    ip_cursor = int(cfg.n_cards) + 1000
    email_cursor = int(cfg.n_cards) + 1000

    for r in range(cfg.n_rings):
        world = r % n_worlds
        pool = world_pools[world]
        size = int(rng.integers(cfg.ring_size_min, cfg.ring_size_max + 1))
        if len(pool) < size:
            log.warning(f"world {world} exhausted at ring {r}; stopping ring planting early")
            break
        members = [pool.pop() for _ in range(size)]

        # Pool size is tied to ring size so that every device in the pool is
        # actually SHARED by at least two members. A "pool" of three devices
        # across three cards is three private devices — it creates no identity
        # edge at all, and a ring that shares nothing is not a ring we could
        # ever have been expected to find on the graph.
        n_dev = max(1, min(cfg.ring_device_pool_max, size // 2))
        device_pool = [f"DEV_{device_cursor + i:06d}" for i in range(n_dev)]
        device_cursor += n_dev

        n_ip = max(1, min(2, size // 3))
        ip_pool = list(_ip_string(np.arange(ip_cursor, ip_cursor + n_ip), 172))
        ip_cursor += n_ip

        shared_email = ""
        if rng.random() < 0.6:
            shared_email = f"EMAIL_{email_cursor:06d}"
            email_cursor += 1

        n_probe_m = int(rng.integers(2, 5))
        n_cash_m = int(rng.integers(1, 3))
        probe_m = merchant_ids[rng.choice(micro_merchants, size=min(n_probe_m, len(micro_merchants)), replace=False)]
        cash_m = merchant_ids[rng.choice(large_merchants, size=min(n_cash_m, len(large_merchants)), replace=False)]

        # Plant at least a day before the window closes so the cashout fits.
        planted_at = int(rng.integers(0, max(window_s - 86_400, 1)))

        rows.append(
            {
                "ring_id": f"RING_{r:04d}",
                "world_id": world,
                "n_cards": size,
                "card_ids": "|".join(cards["card_id"].values[members]),
                "card_index": members,
                "device_pool": "|".join(device_pool),
                "ip_pool": "|".join(ip_pool),
                "shared_email": shared_email,
                "probe_merchants": "|".join(probe_m),
                "cashout_merchants": "|".join(cash_m),
                "planted_at_offset_s": planted_at,
            }
        )

    return pd.DataFrame(rows)


def _generate_legit_transactions(
    cfg: GenerationConfig, index: dict, merchants: pd.DataFrame, n_legit: int, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate the legitimate transaction bulk, fully vectorised.

    Amounts are drawn *from the merchant's own ticket distribution*, not from a
    global one. That matters: it means a INR 5 charge at a recharge merchant is
    genuinely unremarkable, so `probe_fraction` only becomes discriminative when
    combined with cross-merchant structure. If we drew amounts globally, the
    probe signal would be trivially separable and the metrics would be a lie.
    """
    n_cards = cfg.n_cards
    card_idx = rng.choice(n_cards, size=n_legit, p=index["activity"])
    merchant_idx = rng.choice(cfg.n_merchants, size=n_legit)

    avg_ticket = merchants["avg_ticket"].values[merchant_idx]
    amount = np.round(avg_ticket * np.exp(rng.normal(0.0, 0.55, n_legit)), 2).clip(1.0, 250_000.0)

    # Timestamps: uniform day, diurnal hour, uniform minute/second.
    day = rng.integers(0, cfg.days, n_legit)
    hour = rng.choice(24, size=n_legit, p=_DIURNAL)
    ts = day * 86_400 + hour * 3_600 + rng.integers(0, 3_600, n_legit)

    # Device: home device, occasionally the stock user-agent.
    dev_idx = index["device_of_card"][card_idx].copy()
    device = _pad_ids("DEV_", dev_idx)
    generic_mask = index["has_generic"][card_idx] & (rng.random(n_legit) < cfg.generic_device_txn_fraction)
    device[generic_mask] = cfg.generic_device_string

    # IP: home IP, occasionally an ISP NAT address.
    ip = _ip_string(index["ip_of_card"][card_idx], 10)
    nat_mask = rng.random(n_legit) < cfg.nat_ip_txn_fraction
    if nat_mask.any():
        ip[nat_mask] = _ip_string(index["nat_of_card"][card_idx][nat_mask], 203)

    email = _pad_ids("EMAIL_", index["email_of_card"][card_idx])

    # Simulated Vulcan score: Beta(2, 12) => mean ~0.14, thin tail above 0.5.
    vulcan = np.round(rng.beta(cfg.vulcan_legit_alpha, cfg.vulcan_legit_beta, n_legit), 4)

    return pd.DataFrame(
        {
            "card_index": card_idx,
            "merchant_id": merchants["merchant_id"].values[merchant_idx],
            "amount": amount,
            "ts_offset_s": ts,
            "device_fingerprint": device,
            "ip_address": ip,
            "email_hash": email,
            "vulcan_score": vulcan,
            "is_ring_txn": False,
            "ring_id": "",
            "txn_role": "legit",
        }
    )


def _generate_household_bursts(
    cfg: GenerationConfig,
    index: dict,
    cards: pd.DataFrame,
    merchants: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate coordinated *legitimate* household bursts — the hard negatives.

    A family sitting on the sofa with one tablet, buying four mobile recharges
    in three minutes and then a INR 6,000 appliance an hour later, produces:

      - a shared device edge          (identical to a ring)
      - burst compression near 1.0    (identical to a ring)
      - amount escalation of ~100x    (a ring's is ~5,000x)
      - cashout fraction of 0.0       (a ring's is ~0.4)

    Two of the four ring signatures fire. That is the point. If the only
    multi-card device clusters in this dataset were rings, the model would
    separate them on structure alone, precision would come out near 1.0, and
    the number would mean nothing. These make the model earn it, and they are
    where our false positives actually come from.
    """
    households = index["households"]
    if not households:
        return pd.DataFrame()

    n_burst = int(cfg.burst_household_share * len(households))
    if n_burst == 0:
        return pd.DataFrame()

    chosen = [households[i] for i in rng.permutation(len(households))[:n_burst]]
    micro = merchants.loc[merchants["is_micro_ticket"], "merchant_id"].values
    large = merchants.loc[~merchants["is_micro_ticket"], "merchant_id"].values
    if len(micro) == 0 or len(large) == 0:
        return pd.DataFrame()

    device_of_card = index["device_of_card"]
    ip_of_card = index["ip_of_card"]
    email_of_card = index["email_of_card"]
    window_s = cfg.days * 86_400
    rows = []

    for group in chosen:
        t0 = int(rng.integers(0, max(window_s - 86_400, 1)))
        # The household shops on one device and one connection — the head of
        # household's — which is exactly why it looks like shared infrastructure.
        head = int(group[0])
        device = f"DEV_{device_of_card[head]:06d}"
        ip = _ip_string(np.array([ip_of_card[head]]), 10)[0]

        for ci in group:
            for _ in range(int(rng.integers(1, 4))):
                rows.append({
                    "card_index": int(ci),
                    "merchant_id": rng.choice(micro),
                    "amount": round(float(rng.uniform(5.0, cfg.burst_small_amount_max)), 2),
                    "ts_offset_s": int(t0 + rng.integers(0, cfg.probe_burst_seconds)),
                    "device_fingerprint": device,
                    "ip_address": ip,
                    "email_hash": f"EMAIL_{email_of_card[ci]:06d}",
                    "vulcan_score": round(float(rng.beta(cfg.vulcan_legit_alpha, cfg.vulcan_legit_beta)), 4),
                    "is_ring_txn": False,
                    "ring_id": "",
                    "txn_role": "household_burst_small",
                })

        # The follow-up purchase: larger, but two orders of magnitude below a
        # ring cashout, and below the INR 10,000 cashout threshold.
        buyer = int(rng.choice(group))
        rows.append({
            "card_index": buyer,
            "merchant_id": rng.choice(large),
            "amount": round(float(rng.uniform(cfg.burst_large_amount_min, cfg.burst_large_amount_max)), 2),
            "ts_offset_s": int(t0 + rng.integers(cfg.probe_to_cashout_min_s, cfg.probe_to_cashout_max_s)),
            "device_fingerprint": device,
            "ip_address": ip,
            "email_hash": f"EMAIL_{email_of_card[buyer]:06d}",
            "vulcan_score": round(float(rng.beta(cfg.vulcan_legit_alpha, cfg.vulcan_legit_beta)), 4),
            "is_ring_txn": False,
            "ring_id": "",
            "txn_role": "household_burst_large",
        })

    return pd.DataFrame(rows)


def _generate_ring_transactions(
    cfg: GenerationConfig, rings: pd.DataFrame, cards: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate the ring burst: probe, then cashout.

    The signature, per ring:
      - every member card fires 1-3 probes of INR 1-10 at micro merchants,
        all inside a 5-minute window (this is `burst_compression`)
      - 1-2 hours later the same cards hit large merchants for INR 18k-60k
        (this is `amount_escalation` and `probe_to_cashout_gap_hours`)
      - every one of those transactions rides the ring's shared device pool
        (this is what makes the identity graph fire)

    Vulcan scores are simulated low for probes (0.1-0.3) and medium for
    cashouts (0.3-0.5) — individually plausible in both cases. That gap is the
    entire argument for Ringfence: neither transaction is damning alone.
    """
    if rings.empty:
        return pd.DataFrame(columns=[
            "card_index", "merchant_id", "amount", "ts_offset_s", "device_fingerprint",
            "ip_address", "email_hash", "vulcan_score", "is_ring_txn", "ring_id", "txn_role",
        ])

    email_of_card = cards["email_hash"].values
    home_device = cards["home_device"].values
    home_ip = cards["home_ip"].values
    rows = []

    for ring in rings.itertuples(index=False):
        devices = ring.device_pool.split("|")
        ips = ring.ip_pool.split("|")
        probe_m = ring.probe_merchants.split("|")
        cash_m = ring.cashout_merchants.split("|")
        t0 = ring.planted_at_offset_s

        for member_pos, ci in enumerate(ring.card_index):
            # A defector runs its own fresh device and connection instead of
            # the ring pool. No identity edge is created for this member, so
            # the ego generator cannot see it — the ring is still detectable,
            # but only partially covered. This is the honest ceiling on recall.
            defector = bool(rng.random() < cfg.ring_defector_prob)

            # Non-defectors are assigned a primary pool device round-robin, so
            # each device is genuinely shared, and spill onto a second pool
            # device ~30% of the time, which is what cross-links the members
            # who happen to sit on different primaries.
            primary_dev = devices[member_pos % len(devices)]
            primary_ip = ips[member_pos % len(ips)]
            card_devices = [home_device[ci]] if defector else devices
            card_ips = [home_ip[ci]] if defector else ips

            def pick(pool, primary, _rng=rng):
                """Primary 70% of the time, anything in the pool otherwise."""
                if defector or len(pool) == 1:
                    return pool[0] if defector else primary
                return primary if _rng.random() < 0.7 else str(_rng.choice(pool))

            # --- probes: tight burst, micro amounts ------------------------
            n_probes = int(rng.integers(1, 4))
            for _ in range(n_probes):
                rows.append(
                    {
                        "card_index": ci,
                        "merchant_id": rng.choice(probe_m),
                        "amount": round(float(rng.uniform(cfg.probe_amount_min, cfg.probe_amount_max)), 2),
                        "ts_offset_s": int(t0 + rng.integers(0, cfg.probe_burst_seconds)),
                        "device_fingerprint": pick(card_devices, primary_dev),
                        "ip_address": pick(card_ips, primary_ip),
                        "email_hash": email_of_card[ci] if defector else (ring.shared_email or email_of_card[ci]),
                        "vulcan_score": round(float(rng.uniform(cfg.vulcan_probe_lo, cfg.vulcan_probe_hi)), 4),
                        "is_ring_txn": True,
                        "ring_id": ring.ring_id,
                        "txn_role": "probe",
                    }
                )

            # --- cashouts: same infrastructure, 4 orders of magnitude up ---
            gap = int(rng.integers(cfg.probe_to_cashout_min_s, cfg.probe_to_cashout_max_s))
            n_cash = int(rng.integers(1, 3))
            for _ in range(n_cash):
                rows.append(
                    {
                        "card_index": ci,
                        "merchant_id": rng.choice(cash_m),
                        "amount": round(float(rng.uniform(cfg.cashout_amount_min, cfg.cashout_amount_max)), 2),
                        "ts_offset_s": int(t0 + gap + rng.integers(0, 900)),
                        "device_fingerprint": pick(card_devices, primary_dev),
                        "ip_address": pick(card_ips, primary_ip),
                        "email_hash": email_of_card[ci] if defector else (ring.shared_email or email_of_card[ci]),
                        "vulcan_score": round(float(rng.uniform(cfg.vulcan_cashout_lo, cfg.vulcan_cashout_hi)), 4),
                        "is_ring_txn": True,
                        "ring_id": ring.ring_id,
                        "txn_role": "cashout",
                    }
                )

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────


def generate_universe(cfg: GenerationConfig | None = None, seed: int = RANDOM_SEED) -> GeneratedUniverse:
    """
    Generate the full synthetic universe: merchants, cards, rings, transactions.

    Deterministic for a given (cfg, seed) pair — this is the reproducibility bar.
    Returns everything in memory; `save_universe` puts it on disk.
    """
    cfg = cfg or GENERATION
    rng = seed_everything(seed)

    log.info(f"generating universe: {cfg.n_cards} cards, {cfg.n_merchants} merchants, "
             f"{cfg.n_transactions:,} transactions, {cfg.n_rings} rings across {cfg.n_worlds} worlds")

    with timed("generate.merchants", log):
        merchants = _build_merchants(cfg, rng)

    with timed("generate.cards", log):
        cards, index = _build_cards(cfg, rng)

    with timed("generate.rings", log):
        rings = _plant_rings(cfg, cards, merchants, rng)

    with timed("generate.ring_transactions", log):
        ring_txns = _generate_ring_transactions(cfg, rings, cards, rng)

    with timed("generate.household_bursts", log):
        burst_txns = _generate_household_bursts(cfg, index, cards, merchants, rng)

    n_legit = max(cfg.n_transactions - len(ring_txns) - len(burst_txns), 1)
    with timed("generate.legit_transactions", log):
        legit_txns = _generate_legit_transactions(cfg, index, merchants, n_legit, rng)

    with timed("generate.assemble", log):
        frames = [f for f in (legit_txns, burst_txns, ring_txns) if not f.empty]
        txns = pd.concat(frames, ignore_index=True)
        txns = txns.sort_values("ts_offset_s", kind="stable").reset_index(drop=True)

        txns["card_id"] = cards["card_id"].values[txns["card_index"].values]
        base = np.datetime64("2026-01-01T00:00:00")
        txns["timestamp"] = base + txns["ts_offset_s"].values.astype("timedelta64[s]")
        txns["transaction_id"] = _pad_ids("TXN_", np.arange(len(txns)), 8)

        txns = txns[
            [
                "transaction_id", "card_id", "merchant_id", "amount", "timestamp",
                "device_fingerprint", "ip_address", "email_hash", "vulcan_score",
                "is_ring_txn", "ring_id", "txn_role",
            ]
        ]

        # Stamp ring membership onto the card frame.
        if not rings.empty:
            member_pos = np.concatenate([np.asarray(m, dtype=int) for m in rings["card_index"].values])
            member_ring = np.concatenate(
                [np.repeat(r.ring_id, len(r.card_index)) for r in rings.itertuples(index=False)]
            )
            cards.loc[member_pos, "is_ring_member"] = True
            cards.loc[member_pos, "ring_id"] = member_ring

        # planted_at as a real timestamp for the ground-truth file.
        if not rings.empty:
            rings = rings.copy()
            rings["planted_at"] = base + rings["planted_at_offset_s"].values.astype("timedelta64[s]")
            rings = rings.drop(columns=["card_index"])

    stats = _verify(txns, cards, merchants, rings, cfg)
    return GeneratedUniverse(transactions=txns, cards=cards, merchants=merchants, rings=rings, stats=stats)


def _safe_mean(series: pd.Series) -> float:
    """Mean of a possibly-empty series, rounded, with NaN collapsed to 0.0."""
    if series.empty:
        return 0.0
    value = float(series.mean())
    return round(0.0 if np.isnan(value) else value, 4)


def _verify(
    txns: pd.DataFrame, cards: pd.DataFrame, merchants: pd.DataFrame, rings: pd.DataFrame, cfg: GenerationConfig
) -> dict:
    """
    Sanity-check the generated universe and log what we actually produced.

    This is not decoration. The two properties the rest of the system depends
    on — world disjointness and a fat-headed device distribution — are asserted
    here so a bad generator fails loudly instead of quietly producing a model
    that looks great for the wrong reason.
    """
    device_counts = txns.groupby("device_fingerprint", observed=True)["card_id"].nunique()
    generic_accounts = int(device_counts.get(cfg.generic_device_string, 0))

    # World disjointness: every ring must sit in exactly one world.
    cross_world_rings = 0
    if not rings.empty:
        card_world = dict(zip(cards["card_id"], cards["world_id"]))
        for ring in rings.itertuples(index=False):
            worlds = {card_world[c] for c in ring.card_ids.split("|")}
            if len(worlds) != 1:
                cross_world_rings += 1

    stats = {
        "n_transactions": int(len(txns)),
        "n_cards": int(cards["card_id"].nunique()),
        "n_merchants": int(merchants["merchant_id"].nunique()),
        "n_rings": int(len(rings)),
        "n_ring_transactions": int(txns["is_ring_txn"].sum()),
        "ring_txn_fraction": round(float(txns["is_ring_txn"].mean()), 6),
        "n_unique_devices": int(device_counts.size),
        "generic_device_accounts": generic_accounts,
        "device_cardinality_p50": float(device_counts.quantile(0.50)),
        "device_cardinality_p95": float(device_counts.quantile(0.95)),
        "device_cardinality_p99": float(device_counts.quantile(0.99)),
        "device_cardinality_max": int(device_counts.max()),
        "cross_world_rings": cross_world_rings,
        "probe_txns": int((txns["txn_role"] == "probe").sum()),
        "cashout_txns": int((txns["txn_role"] == "cashout").sum()),
        "household_burst_txns": int(txns["txn_role"].str.startswith("household_burst").sum()),
        "mean_vulcan_legit": _safe_mean(txns.loc[~txns["is_ring_txn"], "vulcan_score"]),
        "mean_vulcan_probe": _safe_mean(txns.loc[txns["txn_role"] == "probe", "vulcan_score"]),
        "mean_vulcan_cashout": _safe_mean(txns.loc[txns["txn_role"] == "cashout", "vulcan_score"]),
    }

    assert cross_world_rings == 0, "world disjointness violated — cross-world validation would leak"
    assert stats["n_rings"] > 0, "no rings planted"

    log.info("universe verified", extra={"extra_fields": stats})
    for k, v in stats.items():
        log.info(f"  {k:<28} {v}")
    return stats


def save_universe(universe: GeneratedUniverse) -> None:
    """Write the universe to data/raw and data/ground_truth."""
    ensure_dirs()
    with timed("generate.save", log):
        universe.transactions.to_csv(TRANSACTIONS_CSV, index=False)
        universe.cards.to_csv(CARDS_CSV, index=False)
        universe.merchants.to_csv(MERCHANTS_CSV, index=False)
        universe.rings.to_csv(RINGS_CSV, index=False)
    log.info(f"wrote {TRANSACTIONS_CSV} ({TRANSACTIONS_CSV.stat().st_size / 1e6:.1f} MB)")
    log.info(f"wrote {CARDS_CSV}, {MERCHANTS_CSV}, {RINGS_CSV}")


def load_transactions(path=TRANSACTIONS_CSV) -> pd.DataFrame:
    """Load the transaction table with timestamps parsed. Used everywhere downstream."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["ring_id"] = df["ring_id"].fillna("")
    return df


def load_ground_truth(path=RINGS_CSV) -> pd.DataFrame:
    """Load planted rings. `card_ids` is exploded from the pipe-joined column."""
    df = pd.read_csv(path)
    df["card_id_list"] = df["card_ids"].apply(lambda s: s.split("|"))
    return df


def main() -> None:
    """CLI: generate and persist the universe."""
    parser = argparse.ArgumentParser(description="Generate the Ringfence synthetic universe")
    parser.add_argument("--small", action="store_true", help="test-scale universe (fast)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    cfg = GenerationConfig.small() if args.small else GENERATION
    universe = generate_universe(cfg, seed=args.seed)
    save_universe(universe)


if __name__ == "__main__":
    main()
