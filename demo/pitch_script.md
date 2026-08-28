# Ringfence — 5-Minute Pitch Script

**Setup before you start:** `make all` has run, `make run` and `make demo` are up, browser
on the dashboard **page 1**, terminal visible for one curl. Clear case memory first
(`make clean-memory`) so the failure demo on page 5 is re-armed.

Timings are read-aloud tested. Total: **5:00**.

---

## 0:00 – 0:30 · The ₹5 probe

> At 14:23:17, a card charges Merchant A five rupees. Two seconds later, the same card
> charges Merchant B five rupees. Three seconds after that, a *different* card charges
> Merchant C five rupees.
>
> Ninety-one minutes later, both cards hit Merchant D for forty-seven thousand five hundred
> rupees each.
>
> Merchant A saw a five rupee sale. Merchant B saw a five rupee sale. Merchant D saw two
> large sales. Everyone is happy. Vulcan scored the highest of those transactions at
> **0.34** — comfortably inside approve, and *correctly so*, because not one of them is
> odd on its own.
>
> Razorpay just saw three fragments of one fraud ring, and had no way to assemble them.

*[Page 1 is on screen. Do not click anything yet.]*

---

## 0:30 – 1:15 · Where Ringfence sits

> Vulcan scores three thousand signals per transaction. It is very good at "is this
> transaction odd?". The ring pattern is not odd. It lives in **the space between
> merchants** — the shared device pool, the probe-then-cashout signature, the cross-merchant
> velocity that only an aggregator graph can see.
>
> Razorpay's own launch material says the industry problem is specialised models that don't
> talk to each other. **Ringfence is the specialist that talks to the foundation layer.**
> It doesn't replace Vulcan. It feeds it.
>
> Cards are nodes. Edges are shared device, IP, or email hash — **never merchant**. I'll
> come back to why that "never" is measured and not assumed. Three generators propose
> candidate clusters, thirty-one features describe each one, a gradient-boosted model ranks
> them, and every alert ships a case file.

*[Scroll the README architecture diagram, or gesture at page 1's queue.]*

---

## 1:15 – 2:00 · The dossier

*[Click "Open →" on the top BLOCK alert. Land on page 2.]*

> This is an alert. Ringfence score 0.91. Vulcan on the same activity: 0.34. Composite:
> **0.94 — block.**
>
> Here's the timeline — log scale, because the story is four orders of magnitude wide.
> Blue is probes under ten rupees. Red is cashouts over ten thousand. Same device pool,
> ninety minutes apart.
>
> And here is the part I want you to look at. **Every claim in this table has a citation.**
> "These four cards share device DEV-3412" cites the graph edge. "73% probe transactions"
> cites the feature value and four transaction IDs. "Amounts escalated 9,500x" cites the
> two specific transactions.
>
> **Uncited claims: zero.** Not because we were careful — because the `Dossier` constructor
> raises `ValueError: Uncited claim detected` if any claim lacks a citation. There is no
> bypass flag. An unexplainable alert is *unconstructible*.
>
> The LLM writes the summary at the top **from** this evidence at temperature zero. It is
> never shown anything that doesn't already carry a citation. Pull the API key out and the
> dossier still ships with a deterministic summary. **The evidence is the product.**

---

## 2:00 – 2:45 · Honest metrics

*[Page 4.]*

> Ring recall **0.942**. Precision at our operating point **0.869**.
>
> That precision is not the best we could report — and here's the chart that explains why
> we didn't. This is expected cost against threshold. A false positive costs us two point
> three lakh: analyst time, merchant friction, a declined good customer. A **miss** costs
> eight and a half lakh: the cashout, the chargebacks, the merchant that churns afterwards.
>
> **A miss is worth 3.7 false positives.** So we don't optimise F1. The star is where cost
> is minimised. The X is where F1 peaks — better precision, better F1, and **seventeen
> percent more expensive**, because it misses thirteen rings instead of seven.
>
> We'd rather review four extra cases than miss one ring, and the curve says so in rupees.
>
> These are all five cross-validation folds, not the best one. Folds are whole **worlds** —
> five disjoint card populations. Every test candidate comes from a population the model
> has never seen a single card from.
>
> Two things I'll flag before you ask. Recall is 0.942 and **it cannot reach 1.0**: fifteen
> percent of ring members in our data run their own device instead of the pool, so they
> create no identity edge at all. We could delete them and report a perfect score. We
> didn't. And precision of 0.869 is **optimistic** — our hard negatives are synthetic, and
> real device entropy is messier.

---

## 2:45 – 3:30 · One failure, handled

*[Page 5.]*

> Now the failure. This candidate scored 0.94 and blocked. Ground truth says it's **not a
> ring** — it's a household sharing one tablet. Four recharges in three minutes, then a
> six thousand rupee appliance an hour later. Same device, same compressed timing, same
> escalation direction. Two of our four ring signatures fire.
>
> **That's a real false positive and it cost us two point three lakh.**
>
> The analyst says no.

*[Type a note. Click "Mark as False Positive". Page reruns.]*

> Case memory stores the thirty-one dimensional behavioural signature and the reasoning.
> And now — same pattern, scored again: **0.94 becomes 0.75.** A twenty percent damp, and
> the new dossier carries a cited claim naming the earlier case and the similarity.
>
> That's the audit trail underneath, straight out of SQLite. Nothing here is staged — the
> button wrote to the real database and that second score is a real second scoring pass.
>
> The correction is **bounded**: twenty percent, from config, capped, never enough to
> silently flip a genuine ring to approve on its own.

---

## 3:30 – 4:15 · Bounded, gated, and it doesn't fall over

> Ringfence never moves money. It emits a recommendation: approve, review, or block.
> **Automated action requires a Ring Template with more than ten analyst-confirmed
> precedents** — that's the only automated path in the codebase, and there's a test that
> asserts a novel pattern can never unlock it.
>
> Ask for a card we've never seen.

*[Terminal:]*
```bash
curl -s -X POST localhost:8000/detect/candidate \
  -H 'content-type: application/json' -d '{"cards":["CARD_NOT_REAL"]}' | jq '.degraded, .composite.recommendation'
```

> `true`, `"REVIEW"`. Not a five hundred. **A fraud API that errors is worse than useless
> — the caller times out and the transaction goes through.** Missing model, unknown card,
> LLM down, ChromaDB missing: every one of those degrades to a safe review with the reason
> attached. A detector that cannot see escalates to a human. It never quietly approves.
>
> And the "never merchant edges" claim from earlier — that's measured. Merchant
> co-occurrence links **2.87 million of 2.87 million** possible card pairs. A hundred
> percent. Identity links **four thousand**. Zero point one five percent. Merchant edges
> would be a statement about retail, not about fraud.

---

## 4:15 – 5:00 · The ask

> One hundred and ten tests. `make setup`, `make all`, `make test`, `make run` from a clean
> clone. Every threshold, cost and band in one config file, because you were going to ask
> where 0.7 comes from.
>
> What I want next is **Razorpay test-mode API access**. Two reasons. Our identity layer is
> calibrated to published IEEE-CIS device quantiles, but real Razorpay fingerprints have
> higher cardinality and messier collisions — that's the single biggest threat to the
> precision number I just showed you, and I'd rather find out than defend it. And the
> Vulcan integration is simulated: the composition maths and the gating are real and
> tested, but the score itself is generated.
>
> Ringfence is the abuse-ring sentinel that sits beside the foundation model and hands it
> the one thing a per-transaction view structurally cannot have: **what happened between
> the transactions.**
>
> It doesn't replace Vulcan. It completes it.

---

## Delivery notes

- **Do not read the metrics table aloud.** Point at the chart, say the two numbers that
  matter, move on.
- **Volunteer the weaknesses at 2:45.** Saying "precision is optimistic and here's why"
  before a panel finds it is worth more than the extra decimal place.
- **The failure demo is the differentiator.** Most submissions show a model. Almost none
  show being wrong and recovering. Give it the full 45 seconds and let the score change on
  screen.
- **If the demo breaks:** the dashboard reads local artifacts when the API is down, so kill
  the API and keep going. That's a feature, and saying so out loud is a better moment than
  the one you lost.
- **If asked to skip ahead:** cut 0:30–1:15 (architecture) and 3:30–4:15 (degradation).
  Never cut the dossier or the failure recovery.
