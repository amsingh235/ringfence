# Ringfence — Recording Runbook

The operator's half of the pitch. [pitch_script.md](pitch_script.md) is what you *say*;
this is what you *open, click and press*, second by second.

Every number quoted here was read off the running app, not off a doc. Where the old
script and the live app disagreed, **the app wins** — see [Corrected numbers](#6--corrected-numbers)
before you record.

---

## 1 · Pre-flight (T−30 minutes)

Run these in order. Do not skip the clean-memory step — the failure demo only lands if
case memory is empty.

```bash
cd c:/Users/Amiya/ringfence

make clean-memory     # re-arms the failure demo AND regenerates the alert queue
make test             # expect: 114 passed
make run              # terminal 1 — API on :8000, leave it running
make demo             # terminal 2 — dashboard on :8501, leave it running
```

Then, in the browser, walk all five pages once. You are checking three things:

- [ ] Sidebar reads **API online · ok** (green). If it says offline, the demo still works
      off local artifacts — but you lose the "133 alerts from the API" line.
- [ ] Page 4 shows **0.942 / 0.869 / 0.964 / 0.978 / +17%**.
- [ ] Page 5 step 3 says **"Click the button to write this verdict into case memory"** —
      i.e. it has *not* already been clicked. If it shows a damp already applied, run
      `make clean-memory` again and restart `make demo`.

> **`make clean-memory` deletes `alerts.sqlite`, and alert IDs are random UUIDs**
> ([src/utils.py:276](../src/utils.py#L276)). Every `RF-…` id changes after a reset.
> **Never memorise an alert id.** Use the filter rule in §4.

---

## 2 · Window layout

Exactly **three** windows. Close everything else — notifications, chat, extra tabs.

| Slot | Window | State |
|---|---|---|
| **Win+1** | Browser — dashboard, **one tab only**, `localhost:8501` | F11 fullscreen |
| **Win+2** | Terminal — where you run `demo/show_degraded.py` | Maximised, large font |
| **Win+3** | Terminal — `make run` API logs (optional B-roll) | Maximised |

**Pin all three to the taskbar in that order and switch with `Win+1` / `Win+2` / `Win+3`.**
Do not rely on Alt+Tab: with three windows it cycles most-recent-first, so the second
press lands somewhere different depending on where you just were. `Win+<n>` is
deterministic every time. Alt+Tab is fine only as a straight toggle between two windows.

**Before you hit record:**

- [ ] Browser zoom **100%**, F11 fullscreen (hides tabs, URL bar, bookmarks)
- [ ] Terminal font bumped to ~18pt — panellists may watch on a laptop
- [ ] `python demo/show_degraded.py` **already typed at the prompt, not yet run.** You press
      Enter on camera. Never type a command live.
- [ ] Windows notifications off (Win+A → Do not disturb)
- [ ] Screen recorder capturing **the whole screen**, not a single window — window capture
      goes black when you `Win+<n>`

---

## 3 · Switch map

You are in the browser for 4 of the 5 minutes. There is exactly **one** round trip out.

```
0:00 ──────────────────────────────────────────  browser, page 1
0:30 ──────────────────────────────────────────  browser, page 1  (still)
1:15 ── click Open → ──────────────────────────  browser, page 2
2:00 ── sidebar 4 ─────────────────────────────  browser, page 4
2:45 ── sidebar 5 ─────────────────────────────  browser, page 5
3:30 ── Win+2 ─────────────────────────────────  TERMINAL   ← the only switch out
3:50 ── Win+1 ─────────────────────────────────  browser, page 4
4:15 ──────────────────────────────────────────  browser (talk to camera, page 4 behind you)
```

Say the first words of the next section *while* the switch is happening. Never switch in
silence — dead air is what makes a reviewer close the tab.

---

## 4 · Which card to choose

**Do not hunt for a good example on camera.** Two exhibits, both selected by a rule, not
by an id.

### Page 2 — the BLOCK exhibit

1. On page 1, set **Recommendation → BLOCK** (left dropdown).
2. Click **Open →** on the **top row**.
3. Sidebar → **2 · Dossier Viewer**.

That row is a genuine planted ring, and the rule survives any `make clean-memory`. Rehearse
this exact click path — it is the only multi-step interaction in the video.

What you will see (a representative BLOCK ring, 6 cards):

| Field | Value |
|---|---|
| Ringfence score | **1.000** |
| Vulcan (per-txn) | **0.383** |
| Composite | **1.000 → BLOCK** |
| Cards in cluster | 6 |
| Cited claims | **14** |
| Shared identity | 3 device fingerprints + 1 email hash |

The exact figures shift a little between BLOCK rows. **Read them off the screen, don't
recite them from memory** — say "Ringfence one point oh, Vulcan point three-eight,
composite blocks it" only if that is what is showing.

> Under the old script this click was broken: **Open →** set an `RF-…` id that the dossier
> picker (which listed `CAND_…` ids) silently discarded, dumping you on `CAND_000000` — a
> 12-card **APPROVE**. Fixed in [dossier_viewer.py:117](../frontend/components/dossier_viewer.py#L117).
> If you are recording from an older checkout, pull first.

### Page 5 — the false-positive exhibit

**Nothing to choose.** The page picks it itself: the highest-scoring true negative in the
out-of-fold predictions ([failure_demo.py:28](../frontend/components/failure_demo.py#L28)).
Chosen by the labels, not written in by hand — say that out loud, it is the whole point.

You will get **CAND_000132** — 4 cards, Ringfence **0.019**, composite **0.403 · REVIEW**.

---

## 5 · The segments

### 0:00 – 0:30 · The ₹5 probe

**Window:** browser, page 1. **Action:** none — hands off the mouse.

Read the opening of [pitch_script.md](pitch_script.md) verbatim. Start talking on frame one;
no "okay, so". The queue sitting still behind you is the point: 133 alerts, and not one of
them is a transaction.

### 0:30 – 1:15 · Where Ringfence sits

**Window:** browser, page 1. **Action:** slow scroll down the queue, then back to top.

Land the "never merchant edges" promise here so you can pay it off at 3:50.

### 1:15 – 2:00 · The dossier

**Action:** filter **BLOCK** → **Open →** top row → sidebar **2**.

Scroll: summary → cards → timeline → **evidence table**. Park on the evidence table and stay
there. That table is the strongest thing in the submission.

Land this line exactly: *"Uncited claims: zero — not because we were careful, but because
the `Dossier` constructor raises if a claim has no citation. There is no bypass flag."*
Enforced at [dossier_builder.py:117](../src/dossier_builder.py#L117).

### 2:00 – 2:45 · Honest metrics

**Action:** sidebar **4**. Scroll to the cost curve. Point at the ⭐ and the ✕.

Say only **0.942**, **0.869**, and **+17%**. Do not read the table aloud. The ⭐/✕ gap is the
whole argument — a miss costs ₹8.5L, a false positive ₹2.3L, so a miss is worth 3.7 false
positives and F1 is the wrong objective.

Then volunteer both weaknesses (recall can't reach 1.0 because of defectors; precision is
optimistic because the hard negatives are synthetic). Saying it before they ask is worth
more than the decimal place.

### 2:45 – 3:30 · One failure, handled

**Action:** sidebar **5**. Scroll to step 2. The analyst note is pre-filled — **do not
retype it**. Click **🚫 Mark as False Positive**. Wait for the rerun. Scroll to step 3.

**Say the real numbers:**

> This is the highest-scoring thing the model got wrong — picked by the labels, not by me.
> Four cards, a household sharing one tablet. Ringfence gave it **0.019**; composited with
> Vulcan's 0.391 it still lands at **0.403 — review**. A human has to open it, and at ₹2.3
> lakh a false positive, that is a real cost we own.
>
> *[click]*
>
> Case memory stores the 31-dimensional signature and the reasoning. Same pattern, scored
> again: **0.403 becomes 0.322.** A 20% damp — it moves out of review into approve, and the
> new dossier carries a cited claim naming the earlier case. That wrote to real SQLite. The
> damp is bounded at 20% from config, capped, and never enough on its own to flip a genuine
> ring.

**Do not say "0.94" or "blocked" anywhere in this segment.** Those numbers are not in the
app and an expert panel will catch it. See §6.

### 3:30 – 4:15 · Bounded, gated, doesn't fall over

**Action:** `Win+2` → press **Enter** on the pre-typed command.

```bash
python demo/show_degraded.py
```

```
HTTP status     : 200          <- not a 500
degraded        : True
recommendation  : REVIEW
auto_action     : False
```

> A fraud API that errors is worse than useless — the caller times out and the transaction
> goes through anyway. Unknown card, missing model, LLM down: every one degrades to a safe
> review with the reason attached. Automated action needs a template with **more than ten
> analyst-confirmed precedents**, and there's a test asserting a novel pattern can never
> unlock it.

`Win+1` back. Pay off the merchant-edge promise:

> Merchant co-occurrence links **2,868,910 of 2,869,210** possible card pairs — **99.99%**.
> Identity links **4,357**. **0.15%.** A merchant edge would be a statement about retail,
> not about fraud. That's measured, not assumed.

### 4:15 – 5:00 · The ask

**Action:** none. Talk to the camera, page 4 behind you.

**114 tests** (not 110). Then the ask: **Razorpay test-mode API access** — the identity layer
is calibrated to published IEEE-CIS device quantiles, and real fingerprints have higher
cardinality and messier collisions, which is the single biggest threat to the precision
number you just showed. Close on: *"It doesn't replace Vulcan. It completes it."*

---

## 6 · Corrected numbers

Claims in the original pitch script that do **not** match the running app. Recite the
right-hand column.

| Segment | Old script says | The app actually shows |
|---|---|---|
| 2:45 failure demo | "scored **0.94** and blocked" | Ringfence **0.019**, composite **0.403 · REVIEW** |
| 2:45 failure demo | "**0.94 becomes 0.75**" | **0.403 becomes 0.322** (REVIEW → APPROVE) |
| 4:15 the ask | "**110** tests" | **114** tests |
| 3:30 degraded call | `curl … \| jq …` | `jq` is **not installed** — use `python demo/show_degraded.py` |
| 1:15 dossier | "Ringfence 0.91, composite 0.94" | Read the live values; a typical BLOCK row is **1.000 / 0.383 / 1.000** |

---

## 7 · If it breaks

| Symptom | Do this, out loud |
|---|---|
| Sidebar says **API offline** | Keep going. "The dashboard falls back to local artifacts when the API is down — that's the degradation story I'm about to show you." Turning the fault into the feature beats the moment you lost. |
| **Open →** lands on the wrong dossier | You're on an old checkout. Stop, `git pull`, restart `make demo`. |
| Page 5 shows a damp already applied | Case memory wasn't cleared. `make clean-memory`, restart `make demo`. |
| `show_degraded.py` says unreachable | `make run` died. Skip to the merchant-edge numbers; they need no API. |
| You stumble | **Keep going.** Never restart mid-take. |

---

## 8 · Recording rules

- Start speaking on frame one. No "uhh, okay, so".
- Louder than feels natural — laptop mics compress quiet voices.
- No pause longer than 2 seconds.
- **End at 4:55–5:00.** Under is fine. Over is disqualifying.
- Three takes, pick the best. Do not chase perfection.
- If you must cut: drop 0:30–1:15 (architecture) and 3:30–4:15 (degradation).
  **Never cut the dossier or the failure recovery** — those are the two things almost
  nobody else will have.

---

## 9 · After recording

- [ ] Watch the first 15 seconds only. If the hook doesn't land, that take is dead.
- [ ] Upload; title and description are in [../FORM_ANSWERS.md](../FORM_ANSWERS.md).
- [ ] Paste the link into [../SUBMISSION.md](../SUBMISSION.md) (Video row).
- [ ] `git add -A && git commit -m "submission: video link" && git push`
- [ ] Fill the form from [../FORM_ANSWERS.md](../FORM_ANSWERS.md). Paste — do not improvise.
