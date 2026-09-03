# Razorpay AI Buildathon 2026 — Submission Summary

> One page. Everything a reviewer needs, with the file that proves each claim.

---

## Applicant

| Field | Value |
|---|---|
| Name | Amiya Manas Singh |
| College | SRM University |
| Track | **02 — AI Risk Manager** |
| Sub-direction | Abuse-ring sentinel |

## Project

| Field | Value |
|---|---|
| Name | **Ringfence** |
| Repo | https://github.com/amsingh235/ringfence |
| Video | _(YouTube link — added at upload)_ |
| One-liner | Network-layer fraud-ring detector that feeds Vulcan composite scores |
| Deploy | `docker compose up --build` → API `:8000`, dashboard `:8501` |
| Stack | Python 3.13, FastAPI, LightGBM, NetworkX, Pydantic, Streamlit, Docker |

---

## Headline Numbers

All out-of-fold, from **leave-one-world-out** cross validation over five disjoint card
populations with zero card overlap between train and test. Reproduce with `make all`.

| Metric | Value |
|---|---:|
| Ring recall (candidate generation) | **0.942** (113/120 rings) |
| Precision @ cost-optimal threshold | **0.869** |
| Recall @ cost-optimal threshold | **0.964** |
| PR-AUC / ROC-AUC, out-of-fold | 0.978 / 0.987 |
| Cost at cost-optimal vs F1-optimal threshold | ₹1.24 Cr vs ₹1.45 Cr (**+17.0%**) |
| Feature computation, p99 | **32.9 ms** (budget 50 ms) |
| Uncited claims across 133 dossiers | **0** (structurally impossible) |
| Tests | **114 passing** |

**Two numbers we flag ourselves.** Ring recall is 0.942 and *cannot* reach 1.000: 15% of
ring members are defectors running their own device, generating no identity edge, so no
graph-based generator can propose them. We could have deleted the defectors and reported
1.000. We did not. And precision of 0.869 is *optimistic* — our hard negatives are
synthetic and real device entropy is messier.

---

## How It Hits Every Bar

| Bar | Evidence in repo |
|---|---|
| Working detector + measured precision/recall on held-out data | [`src/scorer.py`](src/scorer.py), leave-one-world-out CV; per-fold table in [BARS.md § Bar 1](BARS.md); [`notebooks/03_cost_curve.ipynb`](notebooks/03_cost_curve.ipynb) |
| Honest metrics including false-positive cost | [`src/scorer.py`](src/scorer.py) `expected_cost()` — threshold chosen by minimising ₹, not F1; both thresholds reported side by side in [BARS.md § Bar 2](BARS.md) |
| Strictly defense-only | [`src/vulcan_integration.py`](src/vulcan_integration.py) emits a recommendation only — no code path can block a payment or move money; [BARS.md § Bar 3](BARS.md) |
| Every action explainable, bounded and gated | [`src/vulcan_integration.py`](src/vulcan_integration.py) APPROVE/REVIEW/BLOCK bands; `auto_action_allowed` requires >10 confirmed precedents, so every novel-pattern BLOCK needs an analyst; [BARS.md § Bar 4](BARS.md) |
| Audit trail | [`src/dossier_builder.py:117`](src/dossier_builder.py#L117) — a Pydantic `model_validator` **raises** on any uncited claim, so an unevidenced dossier is unconstructible, not merely discouraged; [BARS.md § Bar 5](BARS.md) |
| One failure handled gracefully | [`frontend/components/failure_demo.py`](frontend/components/failure_demo.py) shows a false positive being corrected live; analyst rejection damps similar future candidates via [`src/case_memory.py`](src/case_memory.py); [BARS.md § Bar 6](BARS.md) |
| Code speaks louder than resume | 114 tests, CI on every push ([`.github/workflows/`](.github/workflows/)), 3 executable notebooks, verified `docker compose up`; [BARS.md § Bar 7](BARS.md) |

Full bar-by-bar mapping with line references: **[BARS.md](BARS.md)**.

---

## The Bug Worth Writing Down

Case memory compared candidate behavioural signatures by cosine similarity over
log-compressed feature vectors. All 31 features are non-negative, so every vector sat in
a single orthant — **median pairwise similarity across the whole population was 0.964**.
One analyst rejection would have damped all 557 candidates. The feature passed its tests
and carried no information.

Fix: standardise each dimension against population statistics before normalising. Median
pairwise similarity dropped to **−0.070**; only **4.45%** of pairs now exceed the damp
threshold; precedent-boost precision rose from the 0.34 base rate to **0.77**. Locked in
by a regression test that fails if the space ever re-saturates:
[`tests/test_case_memory.py:91`](tests/test_case_memory.py#L91) `test_signatures_are_not_saturated`.

Written up in [README § A bug worth writing down](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Verify It Yourself

```bash
git clone https://github.com/amsingh235/ringfence.git
cd ringfence
docker compose up --build        # API :8000 + dashboard :8501
```

Or from source:

```bash
make setup && make all && make test && make demo
```

`make all` regenerates every number in this document from seed 42 in about two minutes.
