# Submission Form — Pre-Written Answers

Paste these. Do not improvise at the form.
Every number here is verified against the repo and reproducible with `make all`.

---

## Project name

```
Ringfence
```

## One-liner

```
Network-layer fraud-ring detector that feeds Vulcan composite scores — it finds the ring that lives between transactions, not inside one.
```

## Track / sub-direction

```
Track 02 — AI Risk Manager | Sub-direction: Abuse-ring sentinel
```

## Repo

```
https://github.com/amsingh235/ringfence
```

---

## What it solves

```
Razorpay Vulcan scores 3,000 signals per transaction, but fraud rings operate across
merchants through shared device pools and probe-then-cashout patterns that are invisible
to any per-transaction model. Three ₹5 probes at three merchants and a ₹47,500 cashout at
a fourth are each individually plausible — the highest Vulcan score in that sequence is
0.34, comfortably inside APPROVE — yet both cards ran on the same two device fingerprints,
a fact that does not exist inside any single transaction. Ringfence is an abuse-ring
sentinel that builds a cross-merchant identity graph from shared device, IP and email
evidence, generates candidate rings at 94.2% recall, scores them with a GBM whose
threshold is chosen by minimising rupee cost rather than F1 (0.869 precision, 0.964
recall out-of-fold), and ships every alert with a deterministic evidence-cited dossier
in which an uncited claim is structurally unconstructible. It does not replace Vulcan;
it feeds it a ring score that lifts that 0.34 cashout to a 0.94 composite. Strictly
defensive throughout — it observes and recommends, and cannot block a payment, decline
a card, or move money without human review.
```

## What broke, and how you got out

```
Case memory compares a candidate ring's behavioural signature against past analyst
decisions, so that rejecting one false positive damps similar future candidates. It used
cosine similarity over log-compressed feature vectors. All 31 features are non-negative,
so every vector sat in a single orthant of the space — the median pairwise similarity
across the whole population was 0.964. Everything looked like everything. One analyst
rejection would have damped all 557 candidates at once. The feature passed its unit tests
and carried no information whatsoever.

The fix was to standardise each dimension against population statistics before
normalising, so similarity measures deviation from typical rather than shared
non-negativity. Median pairwise similarity dropped from 0.964 to −0.070, only 4.45% of
pairs now exceed the damp threshold, and precedent-boost precision rose from the 0.34
base rate to 0.77. I added a regression test, test_signatures_are_not_saturated, that
fails if the signature space ever re-saturates, and documented the whole thing in the
README and ARCHITECTURE.md rather than quietly patching it.
```

---

## Metrics (if asked for numbers)

```
Ring recall (candidate generation): 0.942 (113/120 rings)
Precision @ cost-optimal threshold: 0.869
Recall @ cost-optimal threshold:    0.964
PR-AUC / ROC-AUC, out-of-fold:      0.978 / 0.987
Cost-optimal vs F1-optimal:         ₹1.24 Cr vs ₹1.45 Cr (+17.0%)
Feature computation p99:            32.9 ms (budget 50 ms)
Uncited claims across 133 dossiers: 0 (structurally impossible)
Tests:                              114 passing

Validation: leave-one-world-out CV across five disjoint card populations,
zero card overlap between train and test. Every number is out-of-fold.
```

## Known limitations (if asked — answer honestly, it is a trust signal)

```
Ring recall is 0.942 and cannot reach 1.000. 15% of ring members are defectors who run
their own device instead of the ring pool; they generate no identity edge, so no
graph-based generator can propose them, and when enough of a small ring defects it is
genuinely unfindable. We could have deleted the defectors from the dataset and reported
1.000 — we did not.

Precision of 0.869 is optimistic. Our hard negatives (households sharing a tablet, ISP
NAT pools, one stock user-agent across 1,488 accounts) are synthetic, and real device
entropy is messier than anything we generated.
```

---

## Video metadata

**Title**

```
Ringfence — Abuse-Ring Sentinel for Razorpay | AI Risk Manager Track | Razorpay AI Buildathon 2026
```

**Description**

```
Built by Amiya Manas Singh (SRM University) for the Razorpay AI Buildathon 2026, Track 02: AI Risk Manager.

Vulcan scores 3,000 signals per transaction. Fraud rings live between transactions. Ringfence builds a cross-merchant identity graph from shared devices, IPs and emails, generates candidate rings at 94.2% recall, scores them with a cost-optimal GBM, and ships every alert with a deterministic evidence-backed dossier where an uncited claim is structurally unconstructible. It does not replace Vulcan — it feeds it.

Repo: https://github.com/amsingh235/ringfence

Metrics: 0.942 ring recall | 0.869 precision | 0.964 recall | 0.978 PR-AUC | 32.9ms p99 | 114 tests
Validation: leave-one-world-out CV, five disjoint card populations, all numbers out-of-fold
Tech: Python 3.13, FastAPI, LightGBM, NetworkX, Pydantic, Streamlit, Docker

00:00 The problem — a ring hiding inside four plausible transactions
00:45 Identity graph — linking cards by device, IP, email, never merchant
01:30 Candidate generation — recall-first, 0.942
02:15 Cost-optimal scoring — why we minimise rupees, not F1
03:00 The dossier — uncited claims are a type error
03:45 Case memory — an analyst rejection, and the bug we found
04:30 Bars, tests, deploy

#razorpay #fintech #frauddetection #machinelearning #buildathon
```

---

## Recording rules (for me, at record time)

- Start speaking the instant recording starts. No "uhh, okay, so".
- Louder than feels natural — laptop mics compress quiet voices.
- No pause longer than 2 seconds.
- Stumble → keep going. Do not restart.
- End at 4:55–5:00. Under is fine. Over is disqualifying.
- Three takes, pick the best, do not chase perfection.

Full script and timings: [demo/pitch_script.md](demo/pitch_script.md) · Q&A prep: [demo/qna_prep.md](demo/qna_prep.md)
