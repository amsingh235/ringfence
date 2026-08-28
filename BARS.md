# How Ringfence Hits Every Razorpay Bar

Each bar below is quoted from razorpay.com/buildathon, mapped to the implementation, and
pointed at the file that does the work and the test that proves it.

All numbers are from the full 737,000-transaction run, out-of-fold, leave-one-world-out.
Reproduce with `make all && make test`.

---

## Bar 1 — "Build a working detector… with measured precision and recall on a held-out test set"

| | |
|---|---|
| **Implementation** | `src/scorer.py` → `cross_world_validate` |
| **Test** | `tests/test_scorer.py::test_every_world_is_held_out_exactly_once`, `::test_no_card_appears_in_two_worlds`, `::test_rings_never_straddle_worlds` |

- **Leave-one-world-out validation.** Cards belong to exactly one of five disjoint
  populations. Folds are whole worlds: train on four, test on the fifth. Every test
  candidate comes from a population the model has never seen a single card from.
- **Zero leakage, asserted not assumed.** A random k-fold split would put two cards from
  the same ring — sharing a device, an IP and a behavioural signature — on both sides of
  the split. Three tests verify that cards and rings never straddle a world boundary.
- **Precision and recall per fold**, all five reported, no cherry-picking:

  | World | n | pos | Precision | Recall | F1 | PR-AUC |
  |---|---:|---:|---:|---:|---:|---:|
  | 0 | 148 | 43 | 0.909 | 0.930 | 0.920 | 0.979 |
  | 1 | 142 | 47 | 0.978 | 0.957 | 0.968 | 0.993 |
  | 2 |  95 | 31 | 1.000 | 0.806 | 0.893 | 0.970 |
  | 3 |  78 | 35 | 0.943 | 0.943 | 0.943 | 0.991 |
  | 4 |  94 | 36 | 0.818 | 1.000 | 0.900 | 0.989 |

- **Out-of-fold aggregate:** precision **0.869**, recall **0.964**, PR-AUC **0.978**,
  ROC-AUC **0.987** at the operating threshold.
- **Every metric in this repo is computed from the out-of-fold prediction vector**, which
  is persisted to `data/processed/oof_predictions.csv`. There is no separate "held-out set"
  that was quietly peeked at during threshold tuning.

---

## Bar 2 — "Honest metrics including false-positive cost"

| | |
|---|---|
| **Implementation** | `src/scorer.py` → `expected_cost`, `optimise_threshold`; `src/config.py` → `CostConfig` |
| **Test** | `tests/test_scorer.py::test_chosen_threshold_actually_minimises_cost`, `::test_cost_optimal_is_cheaper_than_f1_optimal`, `::test_cost_optimal_favours_recall_over_precision` |
| **Notebook** | `notebooks/03_cost_curve.ipynb` |

- **The cost function is the objective, not a post-hoc report.**
  `Cost = FP × ₹2.3L + FN × ₹8.5L`. A miss is worth **3.7 false positives**.
- **We chose the cost-optimal threshold and paid for it in precision:**

  | | Threshold | Precision | Recall | Expected cost |
  |---|---:|---:|---:|---:|
  | **Cost-optimal (we operate here)** | 0.041 | 0.869 | **0.964** | **₹1.24 Cr** |
  | F1-optimal | 0.461 | **0.923** | 0.932 | ₹1.45 Cr (**+17.0%**) |

  The F1 threshold has *better precision and better F1*. It is also **17% more expensive**,
  because it misses 13 rings instead of 7. That trade is the entire point.
- **A test independently re-sweeps the whole threshold grid** and asserts no cheaper point
  exists, so `optimise_threshold` cannot grade its own homework.
- **We flag a trap in the spec rather than walking into it.** Applying
  `ring_incidence = 0.001` naively makes misses arithmetically free and the optimum
  degenerates to "never alert". The candidate population is already heavily filtered by the
  graph, so its positive rate is orders of magnitude above the population rate. We default
  to the **empirical** out-of-fold distribution and expose `ring_incidence` as an explicit
  prior-shift override. Both paths are implemented and tested; the default is the honest
  one. See the module docstring in `src/scorer.py`.
- **We flag our own optimistic numbers.** Precision of 0.869 is higher than we expect
  against real traffic, because our hard negatives are synthetic. Stated in the README and
  in `demo/qna_prep.md`, not buried.

---

## Bar 3 — "Strictly defense-only: anything offense-capable is disqualified"

| | |
|---|---|
| **Implementation** | Whole codebase; `src/api.py` route table |
| **Test** | `tests/test_api.py::test_no_offensive_endpoints_exist`, `::test_no_endpoint_can_move_money` |

- **No offensive capability exists anywhere.** No card testing, no merchant probing, no
  credential handling, no traffic generation against any live system, no evasion tooling.
- **The system only observes and recommends.** Its outputs are a score, a dossier and one
  of three recommendation strings. There is no code path that declines, captures, refunds,
  or blocks anything.
- **Asserted against the live OpenAPI schema, not a README promise.** One test scans every
  registered route for offense-adjacent verbs; another asserts the set of mutating
  endpoints is exactly `{/ingest/transaction, /detect/candidate, /alerts/{id}/disposition}`
  — accept a record, score a set, record a human's verdict.
- **The synthetic data generator is a labelled-training-data tool, not an attack tool.** It
  fabricates no real card numbers, contacts no payment system, and models a probe-then-
  cashout pattern that every issuer already screens for publicly.
- `GET /` states the posture in the response body; `GET /health` returns
  `"defense_only": true`.

---

## Bar 4 — "Every money action explainable, bounded and gated"

| | |
|---|---|
| **Implementation** | `src/dossier_builder.py`, `src/vulcan_integration.py`, `src/config.py` → `GatingConfig` |
| **Test** | `tests/test_dossier_builder.py` (16 tests), `tests/test_api.py::test_novel_pattern_never_permits_automated_action`, `tests/test_case_memory.py::test_auto_block_requires_more_than_ten_precedents` |

**Explainable.**
- Every alert carries a Dossier whose every claim has a citation — a graph edge, a
  transaction ID, a feature value, a case-memory reference or a Vulcan score.
- `Dossier` runs a Pydantic model validator that raises
  `ValueError("Uncited claim detected")` if any evidence has an empty `citation_id`.
  **There is no bypass flag.** An uncited alert is unconstructible, not merely discouraged.
- Measured: **0 uncited claims**, **4.0 references per claim**.
- The composite response carries the composition formula, both input scores, the gate
  bands, the boost and damp applied, and the bounded-action rule — an analyst can
  reconstruct any decision without reading our source.

**Bounded.**
- Three actions only: APPROVE / REVIEW / BLOCK. No fourth option, no free-text action.
- Case-memory adjustments are **bounded constants from config**: +15% precedent boost,
  −20% false-positive damp. Not learned, not compounding. A test asserts that five matching
  false positives still damp by exactly 20%.
- The LLM cannot introduce an action: its prompt forbids suggesting anything beyond the
  bounded recommendation, and it is only ever shown already-cited evidence.

**Gated.**
- **Automated action requires a Ring Template with more than 10 analyst-confirmed
  precedents.** This is the only automated path in the system.
- Every novel-pattern BLOCK returns `auto_action_allowed: false` and
  `human_review_required: true`, regardless of how high it scores.
- Degraded state forces REVIEW — never APPROVE — via `GatingConfig.safe_default_recommendation`.

---

## Bar 5 — "Show the audit trail"

| | |
|---|---|
| **Implementation** | `src/dossier_builder.py`, `src/case_memory.py` → `audit_log`, `src/utils.py` → request tracing |
| **Test** | `tests/test_case_memory.py::test_every_disposition_is_audited`, `tests/test_dossier_builder.py::test_dossier_carries_graph_provenance` |
| **Demo** | Dashboard page 2 (Dossier Viewer), page 5 (Failure Recovery) |

The full provenance chain, end to end, for any decision:

```
transaction  →  identity edge  →  candidate  →  feature vector  →  score  →  composite  →  decision
   TXN_id        edge:A~B:device=D   CAND_id      31 values, snapshotted   0.91      0.94        BLOCK
```

- **Dossier** carries the evidence list, every citation ID, the contributing transaction
  IDs, the complete 31-feature snapshot, the threshold in force, and the composition
  breakdown — enough to replay the decision exactly.
- **Case memory** appends every disposition and every template consolidation to an
  immutable `audit_log` table with a timestamp and an analyst, readable via
  `GET /alerts/{id}/failure-recovery` or straight out of SQLite.
- **Request tracing**: every API call binds a UUID, returned as `X-Request-ID` and attached
  to every structured JSON log line it produces.
- **Observability**: `GET /metrics` exposes Prometheus-format counters, per-stage latency
  percentiles and case-memory statistics.

---

## Bar 6 — "One failure handled gracefully"

| | |
|---|---|
| **Implementation** | `src/case_memory.py`; `GET /alerts/{id}/failure-recovery` |
| **Test** | `tests/test_case_memory.py::test_false_positive_damp_lowers_the_composite_score`, `tests/test_api.py::test_disposition_then_failure_recovery_shows_the_damp` |
| **Demo** | **Dashboard page 5 — this page exists solely for this bar** |

**The failure, and the recovery, on real data:**

1. A household sharing one tablet fires an alert. Structurally it is identical to a small
   ring: dense clique, one shared device, compressed timing. It scores 0.94 → BLOCK.
   **This is a real false positive that costs ₹2.3L, and we own it.**
2. An analyst marks it *False Positive* with notes.
3. Case memory stores the 31-dimensional behavioural signature and the reasoning.
4. The same pattern re-scores at **0.75 — a −20% damp** — and its new dossier carries a
   cited claim naming the earlier case and the cosine similarity.
5. The audit trail shows the whole sequence, straight from SQLite.

Nothing on that page is staged: the button writes to the real database and step 4 is a
genuine second scoring pass. Clear `data/processed/case_memory.sqlite` and it re-arms.

**Failure handling beyond that one demo:**

| Failure | Behaviour |
|---|---|
| Model artifact missing | `degraded: true`, forced REVIEW, `/health` returns 503 with the reason |
| Card unknown to the graph | Degraded response naming how many cards were dropped — not a 500 |
| LLM key missing / API down / timeout | Deterministic template summary, `llm_status` records why, dossier still complete |
| ChromaDB unavailable | Falls back to the numpy cosine index; SQLite stays authoritative |
| python-louvain missing | Falls back to `networkx.louvain_communities` |
| Malformed request | Pydantic 422 with a field-level reason |

The design rule behind all of it: **a detector that cannot see must escalate to a human,
never quietly approve.** A fraud API that 500s is worse than useless — the caller times
out and the transaction goes through.

---

## Bar 7 — "Your code speaks louder than your resume"

| | |
|---|---|
| **Test** | `make test` — 114 tests |

- `make setup && make all && make test && make run` works from a clean clone.
  `docker compose up --build` brings up the API and the dashboard together.
- **114 tests**, covering every critical path: collision capping, the no-merchant-edge
  rule, ring recall, deduplication correctness, the citation guarantee, cost-optimal
  thresholding, cross-world leakage, the false-positive damp, template gating, API
  contracts, graceful degradation, and the defense-only posture.
- Every module and every public function has a docstring that explains **why**, not just
  what. Type hints throughout.
- **No hardcoded magic numbers.** Every threshold, cost, cap and band lives in
  `src/config.py` or an environment variable — greppable in one place, because the panel
  will ask "where does 0.7 come from?"
- No notebooks in `src/`. No dead code. No secrets in the repo — `.env` is gitignored and
  `.env.example` documents every variable.
- **Reproducible**: `random_seed = 42` everywhere; a clean clone reproduces every number in
  this document.

---

## Cross-Track Theme: "One cherry-picked match proves nothing"

We report:

- **All five folds**, not the best one.
- **Recall of 0.942, not 1.000**, with the structural reason it cannot reach 1.000 —
  15% of ring members defect to their own devices and create no identity edge at all. We
  could delete them from the generator and report a perfect score. We did not.
- **Precision of 0.869 flagged as optimistic**, because our hard negatives are synthetic.
- **Ablation deltas flagged as within run-to-run variance**, because 120 rings is not
  enough statistical power to rank individual features.
- **A trap in the spec's own cost formula**, called out and handled, rather than
  implemented literally into a degenerate optimum.
