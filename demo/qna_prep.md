# Panel Q&A Prep

Answers to what the panel will actually ask. Every number here is measured on the full
737,000-transaction run and reproducible with `make all`.

Rule for all of these: **lead with the number, then the reason, then the limitation.**
Never defend a weakness the panel hasn't found yet — volunteer it.

---

## Model & architecture

### "Why not a GNN?"

> Four reasons, in order of weight.
>
> **Supervision density.** 120 rings, 557 candidates. A GNN learns representations from
> dense supervision; we don't have it, and it would overfit before it generalised.
>
> **Division of labour.** The graph is already doing the relational reasoning — that's what
> candidate generation *is*. Asking a GNN to re-learn "these cards share a device" through
> message passing is paying to rediscover something I can compute exactly in one pass.
>
> **Attribution.** Every alert ships a dossier where each claim cites a feature value or a
> graph edge. Gain over 31 named features gives me that. A learned embedding doesn't, and
> I'd have to bolt an explainer on top and hope it's faithful.
>
> **Latency.** 1ms on CPU, no accelerator in the serving path.
>
> If ring supervision reached tens of thousands, a GNN over the identity graph becomes the
> right call. It isn't the right call for this data, and picking it would be modelling for
> the demo instead of the problem.

### "What if the ring uses fresh devices every time?"

> Then identity edges never fire and the graph can't propose them. That's a real failure
> mode and it's already in my numbers — 15% of my ring members are **defectors** who run
> their own device instead of the pool. They're the reason recall is 0.942 and not 1.000.
>
> What survives is behavioural: probe fraction, amount escalation, burst compression,
> probe-to-cashout gap — those are the top features after the identity ones and they don't
> need a shared device. Graceful degradation, not a cliff.
>
> The honest limit: with *zero* shared infrastructure, we're back to per-transaction
> scoring, which is Vulcan's job, not mine. My value is specifically the cross-merchant
> identity link. I'd rather say that than claim I catch everything.

### "Why 31 features and not 300?"

> Every one of them has to be explainable in a dossier sentence an analyst can act on.
> "Amounts escalated from ₹5 to ₹47,500 on the same shared infrastructure" is actionable.
> "Feature 247 was 0.83" isn't.
>
> There's a measurement reason too: at 120 rings, the ablation deltas between feature
> *families* are already close to run-to-run variance. Adding 270 more features would give
> me numbers I couldn't distinguish from noise. That's in Honest Limitations.

### "Why is Vulcan's score not one of your features?"

> Because then composing them would be counting one signal twice, and the composite would
> look more confident than the evidence justifies.
>
> Ringfence's score is computed with no knowledge of Vulcan. That's what makes the noisy-OR
> a combination of two genuinely independent opinions. It's a deliberate constraint in
> `feature_engine.py` and it's documented at the top of the module.

### "Why noisy-OR rather than a mean or a learned combiner?"

> Behaviour at the edges. Vulcan 0.34 with a ring score of 0.91: a mean gives 0.63 — buried
> in the review pile. A max gives 0.91 and throws Vulcan away entirely. Noisy-OR gives
> **0.94 — block**. Two individually unremarkable signals composing into something worth
> acting on is exactly the case this system exists for.
>
> A learned combiner needs labelled composite outcomes, which I don't have without real
> Vulcan scores. When I have them, that's the first thing I'd fit.

---

## Metrics

### "Your precision is 0.869 — is that good?"

> It's honest, and it's probably optimistic. Two separate points.
>
> **It's deliberately not maximised.** At the F1-optimal threshold I'd report 0.923.
> That threshold is 17% *more expensive*, because it misses 13 rings instead of 7 and a
> miss costs ₹8.5L against ₹2.3L for a false positive. I picked the cheaper point and paid
> for it in precision. The cost curve on page 4 is that argument in rupees.
>
> **It's higher than I'd expect on your data.** My hard negatives — households sharing a
> tablet, NAT pools, a stock user-agent across 1,488 accounts — are synthetic. Real device
> entropy is messier and will produce confusions I haven't modelled. That's the number I'd
> most want test-mode API access to check.

### "How do I know you're not overfitting?"

> Leave-one-world-out cross-validation. Cards belong to exactly one of five disjoint
> populations, and folds are whole worlds — every test candidate comes from a population
> the model has never seen a single card from.
>
> Three tests assert the premise rather than trusting it: no card in two worlds, no ring
> straddling a world, every world held out exactly once. A random k-fold would put two cards
> from the same ring on both sides of the split and I'd be measuring memorisation.
>
> Every metric in the repo comes from the out-of-fold prediction vector, including the
> threshold selection. There's no separate holdout I quietly peeked at while tuning.

### "You're showing me your best fold, aren't you?"

> All five are in the README and in BARS.md. The worst is world 4 at precision 0.818; the
> best is world 2 at 1.000 — with recall 0.806, which is the trade-off moving fold to fold.
> One cherry-picked fold proves nothing.

### "Why can't you get recall to 1.0?"

> Because of a property I deliberately built into the data. 15% of ring members run their
> own fresh device and connection instead of the ring pool. They generate no identity edge,
> so **no** graph-based generator can propose them — and when enough of a small ring
> defects, its coverage drops under 50% and it's genuinely unfindable on the graph.
>
> I could delete the defectors from the generator and report 1.000. That would be tuning the
> data to flatter the metric. Seven rings out of 120 are missed and I can tell you why each
> one is missed.

### "Where did ₹2.3L and ₹8.5L come from?"

> They're estimates, and they're in `config.py` as runtime config precisely because they're
> estimates. ₹2.3L is analyst review time plus merchant friction plus the reputational cost
> of a false decline on a good customer. ₹8.5L is the average ring cashout plus chargeback
> handling plus the merchant that churns afterwards.
>
> What matters isn't the absolute numbers — it's the **ratio**, 3.7, and the fact that the
> threshold is a function of it rather than a constant somebody typed. Change either number
> in config and the operating point moves. If Razorpay's real ratio is 2.0 or 6.0, the
> system re-tunes; nothing downstream is hardcoded.

---

## Engineering

### "How would you deploy this?"

> Kafka for ingest, partitioned by card. Flink for windowed features — 5-minute burst
> counts, 1-hour velocity, 24-hour escalation — so behavioural features become window reads
> instead of full-history scans. Redis as the graph cache for adjacency and weights.
> FastAPI scorer, stateless and horizontally scaled. PostgreSQL plus pgvector for alerts,
> cases and the audit log.
>
> Full graph rebuild nightly. One subtlety: **collision capping can't be done per-event** —
> it's a global-percentile decision, so caps are re-evaluated on the batch pass and
> incremental edges inherit the last batch's caps.
>
> This repo is the research kernel. It's correct and measured; it's not sharded. The
> sharding strategy — merchant-ego subgraphs — is designed in ARCHITECTURE.md and I've
> flagged it as not stress-tested.

### "2,396 cards is tiny. Does any of this survive at your scale?"

> The graph doesn't, unsharded. That's the honest answer and it's in Honest Limitations.
>
> What does survive is the **shape**: the two-speed split between a batch graph and an
> online score is exactly what makes 10K TPS tractable, because the expensive relational
> work happens once a night and the request path is a lookup plus 31 features plus a GBM.
> Feature computation is already O(candidate transactions), not O(graph), so it doesn't
> degrade with graph size — only candidate size, which is bounded at 60 by config.
>
> The part I'd need to prove is cross-shard edge resolution. I haven't.

### "Why SQLite? That's not production."

> Correct, and it's a demo choice, not a claim. Zero setup, durable, inspectable with any
> sqlite client, and right for this write volume. It's wrong at 10K TPS, which is why
> ARCHITECTURE.md §8 has the PostgreSQL migration.
>
> The interface is already abstracted — `CaseMemory` has a ChromaDB backend wired in behind
> a flag today, so swapping the vector store doesn't touch calling code.

### "What happens when something breaks?"

> Every degradation path returns a scored response with `degraded: true` and a forced
> **REVIEW** — never a 500, never an approve. Missing model, unknown card, LLM timeout,
> ChromaDB absent, python-louvain absent: all handled, all tested.
>
> The design rule is that **a fraud API that errors is worse than useless** — whatever calls
> it times out and the transaction goes through. Failing loud to an operator is fine.
> Failing open to a fraudster isn't.

### "Where's the AI?"

> Three places, and I'd argue the ordering matters.
>
> The **case memory** is embedding-based: 31-dimensional behavioural signatures, cosine
> similarity, precedent boost and false-positive damp. That's the part that makes the system
> get better with use.
>
> The **dossier summariser** is an LLM, grounded in deterministic evidence at temperature 0.
> It's never shown anything that doesn't already carry a citation, so its worst failure mode
> is bad prose, not a fabricated claim.
>
> The **threshold** is tuned against a cost function rather than a metric.
>
> The core detection is graph plus gradient boosting because **that's what generalises on
> 120 rings.** I'd rather show you a system that works than a model that's fashionable.

---

## Safety & scope

### "How do I know this is defense-only?"

> There's no offensive capability anywhere: no card testing, no merchant probing, no
> credential handling, no traffic generation against a live system, no evasion tooling.
>
> It's asserted, not promised. One test scans the live OpenAPI schema for offense-adjacent
> routes. Another asserts the set of mutating endpoints is exactly three: accept a record,
> score a set, record a human's verdict. Nothing declines, captures, refunds or blocks.
>
> The synthetic generator produces labelled training data. It fabricates no real card
> numbers, contacts no payment system, and the pattern it models — small probe, large
> cashout — is publicly documented and already screened for by every issuer.

### "Your system says BLOCK. Does it block?"

> No. It recommends. The only automated path in the entire codebase requires a **Ring
> Template with more than ten analyst-confirmed precedents** — meaning a human independently
> confirmed that exact pattern more than ten times. Every novel-pattern BLOCK returns
> `auto_action_allowed: false` and `human_review_required: true`, and there's a test that
> asserts a novel pattern can never unlock it.

### "What stops case memory from learning something wrong?"

> Bounds. The precedent boost is +15% and the damp is −20%, both **constants from config**,
> not learned weights. They don't compound — five matching false positives still damp by
> exactly 20%, and there's a test for that.
>
> So a bad memory can move a score by at most 20% and can never on its own flip a genuine
> ring to approve. And it's always visible: the adjustment appears in the dossier as a cited
> claim naming the case ID and the similarity, so an analyst can see it was applied and
> disagree.

---

## The close

### "Why should we hire you?"

> Because I built a system, not a model.
>
> I can tell you where it breaks — fresh-device rings, unsharded graph scale, precision
> that's probably optimistic on real data. I can tell you how it recovers — every
> degradation path returns a safe review, and an analyst's correction measurably changes the
> next score. And I can tell you exactly how it feeds Vulcan: independent signal, noisy-OR
> composition, bounded gating, one integration seam in one class.
>
> I also found a trap in my own spec's cost formula and wrote up why I didn't implement it
> literally, rather than shipping a threshold that never alerts. That's the part of the job
> I think you're actually hiring for.

### "What would you build next, with a month?"

> In order.
>
> **Temporal validation.** Cross-world proves it generalises across populations, not across
> time. Sliding-window retraining and a temporal holdout — that's the biggest gap in my
> evaluation and I know it.
>
> **Real device entropy.** Test-mode API access to check whether precision survives real
> fingerprint cardinality. That's the number most likely to be wrong.
>
> **Shard the graph.** Merchant-ego subgraphs, and actually stress-test the cross-shard edge
> resolution I've only designed.
>
> Not more features. At 120 rings I'd be fitting noise, and I'd rather have a smaller model
> I can explain than a bigger one I can't.
