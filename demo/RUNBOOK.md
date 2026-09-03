# Ringfence — Recording Runbook

The operator's half of the pitch. [pitch_script.md](pitch_script.md) is what you *say*;
this is what you *open, click and press*. The same thing formatted to read from while
recording is [Ringfence_Video_Script.docx](Ringfence_Video_Script.docx).

Every number here was read off the running app, not off a doc. Where the old script and
the live app disagreed, **the app wins** — see [§6](#6--corrected-numbers) before you record.

---

## 1 · Pre-flight (T−30 minutes)

**`make` is a Unix tool and is not installed on Windows.** Use `run.ps1` from the repo
root — same target names, same behaviour.

**Terminal 1** — reset, test, and start the API:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

.\run.ps1 clean-memory     # re-arms the failure demo - also EMPTIES the alert queue
.\run.ps1 test             # expect: 114 passed
.\run.ps1 run              # API on :8000 - leave this terminal alone from here
```

**Terminal 2** — refill the queue, then start the dashboard:

```powershell
.\.venv\Scripts\Activate.ps1

python demo\seed_alerts.py --limit 150   # ~5 min. Drop --limit for all 557 (~20 min)
.\run.ps1 demo                           # dashboard on :8501 - leave this running too
```

> **Why the seed step exists.** `clean-memory` deletes `alerts.sqlite` along with case
> memory — that is the only way to re-arm the failure demo — and nothing refills it,
> because an alert row is only written when something is actually scored. Skip the seed
> and page 1 falls back to scoring 40 candidates in-process, labelled *"scored locally"*,
> which is honest but looks broken next to a green **API online** badge. Seeding at about
> two seconds a candidate, 150 is plenty for a convincing queue; the full 557 takes around
> twenty minutes and buys you nothing on camera.

Then walk all five pages once in the browser. You are checking four things:

- [ ] Sidebar reads **API online · ok** (green)
- [ ] Page 1 says **"… alerts from the API"** — *not* "scored locally". If it says scored
      locally, the seed did not run or the API is down.
- [ ] Page 4 shows **0.942 / 0.869 / 0.964 / 0.978 / +17%**
- [ ] Page 5 step 3 still says **"Click the button to write this verdict into case
      memory"** — if a damp is already applied, run `.\run.ps1 clean-memory` again and
      restart the dashboard

> **Alert IDs are random UUIDs** ([src/utils.py:276](../src/utils.py#L276)) and they all
> change on every reset. **Never memorise an alert id.** Use the filter rule in §4.

---

## 2 · Window layout — exactly two

Two windows, because that is what makes **Alt+Tab** reliable: with two it is a clean
toggle that lands correctly every time. A third window makes it cycle most-recent-first
and it will drop you somewhere you did not expect, on camera.

| Window | What | Setup |
|---|---|---|
| **1 · Browser** | dashboard, `localhost:8501` | One tab only. **F11** for fullscreen. Zoom 100%. |
| **2 · Terminal** | where you press Enter once, at 3:35 | Font ~18pt. Command pre-typed, **not** run. |

The terminal running the API and the one running the dashboard stay **minimised** — they
are not part of the recording. Terminal 2 above becomes your on-camera terminal once the
dashboard is up; open a third console for it if you prefer, but keep only two windows
*visible*.

Pre-type this and leave the cursor sitting on it — the only command in the whole video:

```powershell
python demo\show_degraded.py
```

**Before you hit record:**

- [ ] Everything else closed; Do Not Disturb on (Win+A)
- [ ] Recorder capturing the **whole screen**, not a single window — window capture goes
      black the moment you Alt+Tab

---

## 3 · Switch map — two Alt+Tabs, total

```
0:00 – 3:35  ─────────────────────────────  BROWSER
3:35  Alt+Tab ────────────────────────────  TERMINAL   (press Enter, read 4 lines)
3:55  Alt+Tab ────────────────────────────  BROWSER    (stay here to the end)
```

Within the browser you move by clicking the sidebar: page 1 → 2 at 1:10, → 4 at 2:00,
→ 5 at 2:45, back to 4 at 3:55.

Start saying the next line *while* you switch. Never switch in silence.

---

## 4 · Which exhibit to choose

**Do not hunt for a good example on camera.** Both exhibits are picked by a rule, not an id.

### Page 2 — the BLOCK ring

1. Page 1 → set **Recommendation → BLOCK** (left dropdown)
2. Click **Open →** on the **top row**
3. Sidebar → **2 · Dossier Viewer**

Survives any reset. Rehearse this click path — it is the only multi-step interaction in
the video.

A representative BLOCK ring looks like this:

| Field | Value |
|---|---|
| Ringfence score | **1.000** |
| Vulcan (per-txn) | **0.383** |
| Composite | **1.000 → BLOCK** |
| Cards in cluster | 6 |
| Cited claims | **14** |

Figures shift between rows. **Read them off the screen; never recite from memory.**

> This click used to be broken: **Open →** set an `RF-…` id that the picker (listing
> `CAND_…` ids) silently discarded, dumping you on a 12-card **APPROVE**. Fixed in
> [dossier_viewer.py:117](../frontend/components/dossier_viewer.py#L117). Pull if you are
> on an older checkout.

### Page 5 — the false positive

**Nothing to choose.** The page picks the highest-scoring true negative in the
out-of-fold predictions ([failure_demo.py:28](../frontend/components/failure_demo.py#L28)).
Chosen by the labels, not by hand — say that out loud, it is the point.

You get **CAND_000132** — 4 cards, Ringfence **0.019**, composite **0.403 · REVIEW**,
dropping to **0.322 · APPROVE** after the damp.

---

## 5 · The segments

Full spoken text: [Ringfence_Video_Script.docx](Ringfence_Video_Script.docx) or
[pitch_script.md](pitch_script.md). Actions only, here.

| Time | Window | Do |
|---|---|---|
| 0:00 – 0:30 | Browser p1 | Nothing. Hands off the mouse. Talk from frame one. |
| 0:30 – 1:10 | Browser p1 | Slow scroll down the queue, back to top. |
| 1:10 – 2:00 | Browser p2 | Filter BLOCK → Open → top row → sidebar 2. Scroll to the evidence table and **stop there**. |
| 2:00 – 2:45 | Browser p4 | Sidebar 4. Scroll to the cost curve. Point at the ⭐, then the ✕. |
| 2:45 – 3:35 | Browser p5 | Sidebar 5. Scroll to step 2. Note is pre-filled — **do not retype**. Click **Mark as False Positive**. Wait for reload. Scroll to step 3. |
| 3:35 – 3:55 | **Terminal** | Alt+Tab. Press Enter. Point at the first two lines. |
| 3:55 – 4:15 | Browser p4 | Alt+Tab back. Merchant-edge numbers. |
| 4:15 – 5:00 | Browser p4 | Stop scrolling. Talk to the camera. |

Two lines that must land exactly:

- **1:10** — *"Uncited claims: zero. Not because we were careful — because the dossier
  refuses to be built if a claim has no citation. There is no bypass flag."*
  Enforced at [dossier_builder.py:117](../src/dossier_builder.py#L117).
- **3:55** — merchant co-occurrence links **2,868,910 of 2,869,210** card pairs
  (**99.99%**); identity links **4,357** (**0.15%**).

---

## 6 · Corrected numbers

Claims in the original pitch script that do **not** match the running app.

| Where | Old script said | The app shows |
|---|---|---|
| 2:45 failure demo | "scored **0.94** and blocked" | Ringfence **0.019**, composite **0.403 · REVIEW** |
| 2:45 failure demo | "**0.94** becomes **0.75**" | **0.403 → 0.322** (REVIEW → APPROVE) |
| 4:15 the ask | "**110** tests" | **114** tests |
| 3:35 degraded call | `curl … \| jq …` | `jq` is not installed — `python demo\show_degraded.py` |
| 1:10 dossier | "Ringfence 0.91, composite 0.94" | Read live values; typical BLOCK row is **1.000 / 0.383 / 1.000** |
| every command | `make …` | `make` is not on Windows — `.\run.ps1 …` |

---

## 7 · If it breaks

| Symptom | Do this, out loud |
|---|---|
| Page 1 says "scored locally" | The seed did not run. Not fatal — the queue is real either way. Keep going; nobody but you knows. |
| Sidebar says **API offline** | Keep going. "The dashboard falls back to local files when the API is down — that's the degradation story I'm about to show you." The fault becomes the feature. |
| **Open →** opens the wrong case file | Old checkout. Stop, `git pull`, restart the dashboard. |
| Page 5 already shows a damp | Case memory not cleared. `.\run.ps1 clean-memory`, reseed, restart. |
| `show_degraded.py` unreachable | API died. Skip to the merchant-edge numbers — they need nothing running. |
| You stumble | **Keep going.** Never restart mid-take. |

---

## 8 · Recording rules

- Start speaking on frame one. No "uhh, okay, so".
- Louder than feels natural — laptop mics compress quiet voices.
- No pause longer than two seconds.
- **End at 4:55–5:00.** Under is fine. Over is disqualifying.
- Three takes, pick the best first fifteen seconds.
- If you must cut: drop 0:30–1:10 and 3:35–4:15. **Never cut the dossier or the failure
  recovery** — those two are why this submission is different.

---

## 9 · After recording

- [ ] Watch only the first fifteen seconds. If the hook does not land, that take is dead.
- [ ] Upload; title and description are in [../FORM_ANSWERS.md](../FORM_ANSWERS.md).
- [ ] Paste the link into the Video row of [../SUBMISSION.md](../SUBMISSION.md).
- [ ] `git add -A && git commit -m "submission: video link" && git push`
- [ ] Fill the form from [../FORM_ANSWERS.md](../FORM_ANSWERS.md). Paste — do not improvise.
