# Ringfence — System Design

Track 02 · Abuse-Ring Sentinel. This document is the system-design walkthrough: what the
system must do, how it is put together, what it trades away, and what it would take to run
it at Razorpay scale.

---

## 1. Requirements

### Functional

| # | Requirement |
|---|---|
| F1 | Ingest a transaction stream and maintain a card-to-card identity graph from shared device / IP / email evidence |
| F2 | Propose candidate card clusters that might be abuse rings, optimising for **recall** |
| F3 | Score each candidate as a ring, optimising for **cost**, not F1 |
| F4 | Compose that score with a Vulcan per-transaction score into a gated recommendation |
| F5 | Emit a fully-cited dossier with every alert |
| F6 | Persist analyst dispositions — both confirmations and rejections — and use them |
| F7 | Expose all of the above over an API and a demo dashboard |

### Non-functional

| # | Requirement | Target | Measured |
|---|---|---|---|
| N1 | Feature computation | < 50 ms p99 | **32.9 ms** |
| N2 | Model inference | < 10 ms p99 | ~1 ms |
| N3 | Score-only API response | < 100 ms p99 | **29.4 ms** (p50 7.5, p95 10.5) |
| N4 | Full dossier response | < 300 ms p99 | **92.9 ms** (p50 12.5, p95 16.6) |
| N5 | Reproducibility | bit-identical for a given seed | `random_seed = 42` throughout |
| N6 | Observability | structured logs, Prometheus metrics, request tracing | JSON logs + `/metrics` + `X-Request-ID` |
| N7 | Configurability | no hardcoded thresholds | all in `src/config.py` |
| N8 | Portability | one command | `docker compose up` |
| N9 | Safety | no offense capability, no money movement | asserted in `tests/test_api.py` |
| N10 | Degradation | never fail open | forced REVIEW on any degraded path |

---

## 2. Scale Estimation

**Demo (this repo).** 737,000 transactions · 2,396 cards · 690 merchants · 120 planted
rings · 5 disjoint worlds. Full pipeline: **~90 seconds**. Graph: 2,396 nodes, 4,357 edges.
Candidates: 557.

**Production shape (Razorpay-scale, estimated).**

| Quantity | Estimate | Consequence |
|---|---|---|
| Transactions | ~10K TPS peak | Graph cannot be rebuilt per transaction |
| Cards in a 90-day window | ~10^8 | Graph does not fit one machine |
| Identity edges | ~10^9 after capping | Needs a distributed store |
| Candidates per rebuild | ~10^6 | Feature computation must be a batch job with an online cache |
| Alerts per day | ~10^3–10^4 | Analyst capacity is the real constraint, and is exactly why the threshold is cost-tuned |

The load-bearing observation: **the graph is a batch artefact, the score is an online
lookup**. This is a two-speed system, and the split is what makes the scale estimate
tractable — see §8.

---

## 3. Core APIs

```
POST /ingest/transaction              → {status, transaction_id, buffered}
POST /detect/candidate                → {ringfence_score, composite{…}, dossier{…}, degraded}
GET  /alerts?limit&offset&recommendation&disposition
                                      → {total, alerts[]}
GET  /alerts/{id}/dossier             → {dossier{evidence[]…}, quality{uncited, refs_per_claim}}
POST /alerts/{id}/disposition         → {stored, effect_on_future_scoring, templates_consolidated}
GET  /alerts/{id}/failure-recovery    → {before_case_memory, after_case_memory, score_delta, audit_trail}
GET  /health                          → {status, components{}, model_version, defense_only}
GET  /metrics                         → Prometheus text
GET  /dossier-quality                 → {uncited_claims, refs_per_claim}
```

Contract notes:

- `/detect/candidate` is **synchronous** for features, scoring, case-memory lookup,
  composition and the deterministic dossier. The **LLM narrative and durable write are
  background tasks** — a slow language model must never be able to slow down a fraud
  decision.
- `/ingest/transaction` **buffers and acknowledges**. It does not update identity edges
  inline, and the response says so. Pretending otherwise would misrepresent the
  architecture.
- Every endpoint degrades rather than 500s. `degraded: true` plus a forced REVIEW is a
  first-class response state, not an error path.

Full OpenAPI schema at `/docs` when the service is running.

---

## 4. Data Model

```
                        ┌────────────────────┐
                        │      MERCHANT      │
                        │ merchant_id  PK    │
                        │ category           │
                        │ avg_ticket         │──┐  defines "small merchant"
                        │ is_micro_ticket    │  │  (probe target)
                        └────────────────────┘  │
                                                │
┌──────────────┐        ┌────────────────────┐  │      ┌─────────────────────┐
│    CARD      │        │    TRANSACTION     │◀─┘      │   GROUND TRUTH      │
│ card_id  PK  │◀───────│ transaction_id PK  │         │   ring_id  PK       │
│ world_id     │   1:N  │ card_id       FK   │         │   world_id          │
│ home_device  │        │ merchant_id   FK   │         │   card_ids[]        │
│ home_ip      │        │ amount             │         │   device_pool[]     │
│ email_hash   │        │ timestamp          │         │   probe_merchants[] │
│ is_ring_mbr  │        │ device_fingerprint │───┐     │   cashout_merch[]   │
└──────┬───────┘        │ ip_address         │───┼──┐  │   planted_at        │
       │                │ email_hash         │───┼──┼─▶└─────────────────────┘
       │                │ vulcan_score       │   │  │   (evaluation only —
       │                └────────────────────┘   │  │    never a model input)
       │                                          │  │
       │  ┌───────────────────────────────────────┘  │
       │  │  identity attributes, grouped and CAPPED │
       ▼  ▼                                          │
┌──────────────────────┐                             │
│    IDENTITY EDGE     │◀────────────────────────────┘
│ card_a, card_b   PK  │
│ weight  (0,1]        │   1/log2(k+1) × type multiplier
│ shared[]             │   {attribute, value, cardinality}  ← dossier citations
└──────────┬───────────┘
           ▼
┌──────────────────────┐        ┌──────────────────────┐
│      CANDIDATE       │───────▶│   FEATURE VECTOR     │
│ candidate_id  PK     │        │ 31 floats            │
│ cards[]              │        │ structural(10)       │
│ source, world_id     │        │ identity(8)          │
└──────────┬───────────┘        │ behavioural(13)      │
           │                    └──────────┬───────────┘
           ▼                               ▼
┌──────────────────────┐        ┌──────────────────────┐
│       ALERT          │───────▶│       DOSSIER        │
│ alert_id      PK     │        │ evidence[] (cited)   │
│ scores, recommendation│       │ feature_snapshot     │
│ disposition          │        │ threshold_used       │
└──────────┬───────────┘        └──────────────────────┘
           ▼
┌──────────────────────┐        ┌──────────────────────┐
│        CASE          │───────▶│    RING TEMPLATE     │
│ alert_id      PK     │  ≥5    │ template_id     PK   │
│ disposition          │        │ n_confirmations      │
│ signature (31-dim)   │        │ centroid (31-dim)    │
│ notes, analyst       │        │ auto_block_eligible  │  (>10 confirmations)
└──────────────────────┘        └──────────────────────┘
           │
           ▼
┌──────────────────────┐
│      AUDIT LOG       │   append-only: who decided what, when
└──────────────────────┘
```

**Two invariants worth naming.** Ground truth is joined only at evaluation time and never
reaches a feature. Identity edges carry their `shared[]` provenance, because an edge with a
weight but no provenance is a claim the dossier cannot support.

---

## 5. High-Level Architecture

### Stage 1 — Identity graph (`graph_builder.py`)

Cards are nodes; shared device / IP / email hashes are edges. **No merchant edges** — the
measured reason is in §6. Each identity value is capped at
`max(p95_cardinality, min_collision_cap=12)` cards before edges are expanded, which turns a
1,488-card stock user-agent from 1.1M edges into 66. Edge weight is
`1/log2(k+1) × type_multiplier`, computed from the **true** pre-cap cardinality so capping
cannot launder a weak signal into a strong one.

### Stage 2 — Candidate generation (`candidate_generator.py`)

Recall-first ensemble of Louvain (6 resolutions), connected components (4 weight floors)
and identity-ego sets. Greedy Jaccard>0.5 deduplication with an inverted index, keeping the
**larger** of two near-duplicates. Measured recall 0.942; ego sets alone reach 0.908.

### Stage 3 — Feature engine (`feature_engine.py`)

31 features over a columnar, integer-coded, per-card transaction index. Behavioural
features are computed over the candidate's **shared-infrastructure slice** — transactions
that ran on a device/IP/email used by two or more candidate cards — not the members' whole
histories. Averaged over everything, a real ring's `probe_fraction` is 0.013 and invisible;
restricted to shared infrastructure it is 0.6 and obvious. p99: 32.9 ms.

### Stage 4 — Scorer (`scorer.py`)

LightGBM, `scale_pos_weight` for imbalance, leave-one-world-out CV, threshold chosen by
minimising `FP×₹2.3L + FN×₹8.5L` over the out-of-fold predictions.

### Stage 5 — Dossier (`dossier_builder.py`)

Deterministic evidence first, narrative second. `Dossier` cannot be constructed with an
uncited claim.

### Stage 6 — Case memory (`case_memory.py`)

SQLite metadata + vectors, numpy cosine index by default. Precedent boost +15% above 0.85
similarity; false-positive damp −20% above 0.80.

### Stage 7 — Vulcan composition (`vulcan_integration.py`)

`composite = 1 − (1 − vulcan)(1 − ringfence)`, then bounded case-memory adjustments, then
gating.

---

## 6. Bottlenecks & Trade-offs

### Why no merchant edges

Measured on the generated data, logged on every run:

| Linking rule | Card pairs linked | Share of all 2,869,210 pairs |
|---|---:|---:|
| Shared merchant | 2,868,910 | 100.0% |
| Shared identity | 4,357 | 0.152% |

Merchant co-occurrence links essentially everyone to everyone. It is a statement about
retail, not fraud.

### Why collision capping needs a floor

The raw 95th percentile of device cardinality is **4 cards**. Capping there would shred the
8-card rings we exist to find — we would be deleting the signal to control the noise. The
cap's job is to stop one 1,488-account value from fusing the graph, so the floor is set at
12, above the maximum ring size. It bites on 41 groups out of thousands.

### Why LightGBM and not a GNN

- **Supervision density.** 120 rings and 557 candidates. A GNN needs dense supervision to
  learn representations that hand-designed features already encode.
- **Division of labour.** The graph is already doing the relational reasoning — that is
  what candidate generation *is*. Asking a GNN to re-learn "these cards share a device" from
  message passing is paying for something we have exactly.
- **Attribution.** The dossier needs per-feature attribution with a citation. Gain-based
  importance over 31 named features gives that; a learned embedding does not.
- **Latency.** 1 ms inference on CPU, no accelerator in the serving path.

If ring supervision reached tens of thousands and the pattern space got genuinely richer,
a GNN over the identity graph becomes the right call. It is not the right call for this
data, and choosing it would be modelling for the demo rather than for the problem.

### Why a deterministic dossier with the LLM on top

Reversed — narrative first, citations hunted afterwards — is how you get a confident
sentence nobody can defend. Building evidence first and constraining the model to restate
it means the failure mode of a hallucinating LLM is *bad prose*, not *a fabricated claim*.
It also means a missing API key degrades the polish, not the product.

### Why signatures are population-standardised (a bug we measured and fixed)

Case memory compares candidates by cosine similarity over their 31-dimensional
behavioural signature. The first implementation log-compressed the features and
L2-normalised them. It looked right and was badly wrong.

**Every one of the 31 features is non-negative.** So every compressed vector lived in a
single orthant, and pairwise cosine similarity across the whole candidate population came
out at a **median of 0.964**. One analyst rejection damped **all 557 candidates** — case
memory was matching "is a candidate", not "is this pattern". Precedent-boost precision was
exactly the base rate, which is the signature of a feature carrying zero information.

The fix is to standardise each dimension against population mean and standard deviation
before normalising, which recentres the cloud on the origin so vectors spread across the
sphere. Measured after:

| | Before | After |
|---|---:|---:|
| Median pairwise cosine | 0.964 | **−0.070** |
| Pairs above the 0.80 damp threshold | 100% | **4.45%** |
| Precedent-boost precision (base rate 0.34) | 0.34 | **0.77** |

The scaler is fitted by the trainer on the same candidate population the model sees —
fitting it at query time from whatever is in the case store would make similarity depend on
how many cases you had stored. It is persisted to `data/processed/signature_scaler.json`
with a fingerprint, and `CaseMemory` warns loudly if the store holds signatures from a
different scaler. `tests/test_case_memory.py::test_signatures_are_not_saturated` guards it.

### Why SQLite

Zero setup, durable, inspectable with any sqlite client, and correct for a demo's write
volume. It is the wrong answer at 10K TPS — see §8.

### Why the empirical cost distribution, not `ring_incidence = 0.001`

Applied literally, a 0.001 base rate makes misses arithmetically free and the cost optimum
becomes "never alert". The candidate population is already filtered by the graph, so its
positive rate (34%) is orders of magnitude above the population rate. We default to
empirical and expose the prior shift as an explicit override.

### Accepted costs

| Trade | Cost | Why we took it |
|---|---|---|
| Recall-first generation | 557 candidates for 120 rings | A ring never proposed can never be caught |
| Cost-optimal threshold | Precision 0.869 instead of 0.923 | Misses cost 3.7× more; +17% cheaper |
| Batch graph | Ring detected on rebuild, not in-flight | Probe→cashout gaps are hours; a nightly rebuild is inside the window |
| Behavioural slice | Depends on shared infra existing | Falls back to full history when nothing is shared |
| Synthetic data | Precision likely optimistic | No access to real transaction data; flagged, not hidden |

---

## 7. Vulcan Integration

Vulcan scores 3,000 signals per transaction and answers *"is this transaction odd?"*.
Ringfence answers *"is this cluster of cards a ring?"*. Different questions, different
evidence, different time horizons.

**Independence is mechanical, not rhetorical.** Vulcan's score is deliberately **not** a
feature of the Ringfence model. Compose two opinions only when they are actually two
opinions; otherwise the composite is one signal counted twice.

**Noisy-OR:** `composite = 1 − (1 − v)(1 − r)`

| v | r | mean | max | **noisy-OR** |
|---:|---:|---:|---:|---:|
| 0.34 | 0.91 | 0.63 → REVIEW | 0.91 | **0.94 → BLOCK** |
| 0.10 | 0.12 | 0.11 | 0.12 | **0.21 → APPROVE** |
| 0.85 | 0.05 | 0.45 → BLOCK-adjacent | 0.85 | **0.86 → BLOCK** |

Either signal alone being high is enough; neither being high leaves it low; and two
individually unremarkable signals compose into something worth reviewing. A mean would have
buried row 1 in the REVIEW pile; a max would have discarded Vulcan's contribution entirely.

**Production integration point.** `VulcanComposer.compute_composite` is the seam. In
production, Ringfence posts a ring score for a card cluster against Razorpay's test-mode
API and receives per-transaction scores back. The composition maths and the gating bands
are unchanged by that swap — which is precisely why they live in their own class with no
dependency on how the Vulcan number arrived.

---

## 8. Evolution Path

The current repo is the **research kernel**: correct, measured, and honest about its scale.
Here is what changes at 10K TPS, in dependency order.

### Phase 1 — Streaming ingest
`POST /ingest/transaction` becomes a **Kafka** producer. Partition by `card_id` so a card's
history is locally ordered.

### Phase 2 — Windowed features
**Flink** maintains rolling windows per card and per identity value: 5-minute burst counts,
1-hour velocity, 24-hour amount escalation. Behavioural features become window reads
instead of full-history scans, and the p99 stops depending on how much history a card has.

### Phase 3 — Graph cache
Identity edges move to **Redis** (adjacency + weights) with the full graph in a property
store. Full rebuild nightly; incremental edge insertion on the stream, with collision caps
re-evaluated on the batch pass — capping is a global-percentile decision and cannot be made
per-event.

Sharding: **merchant-ego subgraphs**. Cards are partitioned by their dominant merchant
cluster, and cross-shard edges are resolved on the batch pass. *Designed, not stress-tested
— stated in Honest Limitations.*

### Phase 4 — Storage
SQLite → **PostgreSQL** for alerts, cases and the audit log. The numpy cosine index →
**pgvector** or a managed vector store; the `CaseMemory` interface already abstracts this
(ChromaDB is wired as an alternate backend today).

### Phase 5 — Serving
FastAPI scorer horizontally scaled behind a load balancer, stateless, reading the Redis
graph cache. The model artifact ships as an immutable versioned blob; `/health` already
reports `model_version` so a rollout can be verified per instance.

### Phase 6 — Model lifecycle
Sliding-window retraining with a **temporal** holdout alongside the cross-world one — the
gap named in Honest Limitations. Champion/challenger on live traffic, with the cost function
as the promotion criterion rather than AUC.

```
Kafka ──▶ Flink ──▶ Redis graph cache ──▶ FastAPI scorer ──▶ Alert queue
  │       (windowed      (adjacency +        (stateless,        │
  │        features)      weights)            horizontal)       ▼
  │                            ▲                         PostgreSQL + pgvector
  └──────▶ nightly batch ──────┘                         (alerts · cases · audit)
           graph rebuild
           + collision recap
```
