"""
Generate the short teleprompter — the words and nothing else.

The full document (build_script_docx.py) carries pre-flight, window setup and
break-glass tables, which matter before you record and are noise while you are
recording. This one is what you actually read from: two pages, seven segments,
one stage line each.

Spoken text is identical to the full version, word for word.
"""
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUT = r"c:\Users\Amiya\ringfence\demo\Ringfence_Video_Script_Short.docx"

INK = RGBColor(0x1A, 0x1A, 0x1A)
BLUE = RGBColor(0x1F, 0x4E, 0x79)
RED = RGBColor(0xB3, 0x1B, 0x1B)

doc = Document()
for s in doc.sections:
    s.top_margin = s.bottom_margin = Inches(0.55)
    s.left_margin = s.right_margin = Inches(0.7)

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)
normal.font.color.rgb = INK


def shade(par, fill):
    """Paint a paragraph background, for the stage strips."""
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), fill)
    par._p.get_or_add_pPr().append(el)


def timing(clock, title):
    """Segment header: clock in blue, title in black, kept with what follows."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(clock + "   ")
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = BLUE
    r2 = p.add_run(title)
    r2.font.size = Pt(15)
    r2.font.bold = True


def stage(text):
    """One grey line: where you are and what you click. Never read aloud."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(7)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.size = Pt(10)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    shade(p, "EDEDED")


def say(chunks):
    """The words, at reading-from-a-distance size."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.3
    for text, bold in chunks:
        r = p.add_run(text)
        r.font.size = Pt(12.5)
        r.font.bold = bold


# ── header ────────────────────────────────────────────────────────────────
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
r = p.add_run("Ringfence — 5-Minute Script")
r.font.size = Pt(19)
r.font.bold = True

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(4)
r = p.add_run("Grey = do, never read.   Alt+Tab twice: out at 3:35, back at 3:55.   "
              "795 words, 4:49.   End by 5:00.")
r.font.size = Pt(10)
r.font.italic = True
r.font.color.rgb = RED

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(4)
r = p.add_run("BETWEEN TAKES: .\\run.ps1 reset-demo  then restart the dashboard. "
              "The 2:45 click writes to a real database — without a reset the next take "
              "shows the after-numbers before you press anything.")
r.font.size = Pt(10)
r.font.bold = True
r.font.color.rgb = RED

# ── 0:00 ──────────────────────────────────────────────────────────────────
timing("0:00", "The five-rupee probe")
stage("BROWSER, page 1. Hands off the mouse. Talk from frame one.")
say([
    ("At twenty-three past two, a card charges Merchant A five rupees. Two seconds later, "
     "the same card charges Merchant B. Three seconds after that, a ", False),
    ("different", True),
    (" card charges Merchant C. Ninety-one minutes later, both cards hit Merchant D for "
     "forty-seven thousand five hundred rupees each.", False),
])
say([
    ("Every one of those is individually plausible. Vulcan scored the highest at ", False),
    ("zero point three four", True),
    (" — comfortably inside approve. Razorpay just saw three fragments of one fraud ring, "
     "with no way to assemble them.", False),
])

# ── 0:30 ──────────────────────────────────────────────────────────────────
timing("0:30", "Where Ringfence sits")
stage("BROWSER, page 1. Scroll slowly down the queue, then back to top.")
say([
    ("Vulcan scores three thousand signals per transaction. It is very good at “is this "
     "transaction odd?”. The ring pattern is not odd. It lives in ", False),
    ("the space between merchants", True),
    (" — the shared device pool, the probe-then-cashout signature, the cross-merchant "
     "velocity only an aggregator can see.", False),
])
say([
    ("So Ringfence does not replace Vulcan. ", False),
    ("It feeds it.", True),
    (" Cards are nodes; edges are a shared device, IP or email — ", False),
    ("never a merchant", True),
    (". I will come back to why that “never” is measured. Three generators propose clusters, "
     "thirty-one features describe each one, a model ranks them, and every alert ships a "
     "case file.", False),
])

# ── 1:10 ──────────────────────────────────────────────────────────────────
timing("1:10", "The case file")
stage("Filter Recommendation → BLOCK. Open → top row. Sidebar 2. Scroll to the evidence "
      "table and STOP.   The three scores below: READ THEM OFF THE SCREEN, they vary by row.")
say([
    ("This is one alert. Ringfence scores the cluster at one point oh. Vulcan, on the same "
     "activity, point three eight. Composite: ", False),
    ("block.", True),
])
say([
    ("Here is the timeline, log scale, because the story is four orders of magnitude wide. "
     "Blue is probes under ten rupees, red is cashouts over ten thousand — same device pool, "
     "ninety minutes apart.", False),
])
say([
    ("And here is the part I want you to look at. ", False),
    ("Every claim in this table carries a citation.", True),
    (" “These cards share a device” cites the graph edge. “Seventy-three percent probes” "
     "cites the transaction IDs.", False),
])
say([
    ("Uncited claims: ", False),
    ("zero", True),
    (". Not because we were careful — because the dossier refuses to be built if a claim has "
     "no citation. There is no bypass flag. ", False),
    ("The evidence is the product.", True),
])

# ── 2:00 ──────────────────────────────────────────────────────────────────
timing("2:00", "Honest metrics")
stage("Sidebar 4. Scroll to the cost curve. Point at the star, then the cross.")
say([
    ("Ring recall ", False), ("nought point nine four two", True),
    (". Precision ", False), ("nought point eight six nine", True),
    (". That is not the best we could report — and this chart is why we did not chase it.", False),
])
say([
    ("A false positive costs two point three lakh. A ", False), ("miss", True),
    (" costs eight and a half — so a miss is worth three point seven false positives, and F1 "
     "is the wrong objective. The star is where cost is lowest. The cross is where F1 peaks, "
     "and it is ", False),
    ("seventeen percent more expensive.", True),
])
say([
    ("Two things before you ask. Recall ", False), ("cannot reach one", True),
    (" — fifteen percent of ring members use their own device and create no link. We could "
     "have deleted them and reported a perfect score. We did not. And that precision is ", False),
    ("optimistic", True),
    (" — our hard negatives are synthetic.", False),
])

# ── 2:45 ──────────────────────────────────────────────────────────────────
timing("2:45", "One failure, handled live")
stage("Sidebar 5. Scroll to step 2. Note is pre-filled — do NOT retype.")
say([
    ("Now the failure. This is the highest-scoring thing the model got wrong — picked by the "
     "labels, not chosen by me. Four cards: a household sharing one tablet. Ringfence gave "
     "it ", False),
    ("nought point oh one nine", True),
    (", but composited with Vulcan it still lands at ", False),
    ("nought point four zero three — review", True),
    (". Two point three lakh — a cost we own.", False),
])
stage("CLICK “Mark as False Positive”. Wait for reload. Scroll to step 3.")
say([
    ("The analyst says no. Case memory stores the signature and the reasoning. Same pattern, "
     "scored again: ", False),
    ("point four zero three becomes point three two two.", True),
    (" A twenty percent damp — out of review, into approve, with a cited claim naming the "
     "earlier case.", False),
])
say([
    ("Nothing there was staged, and the correction is ", False), ("bounded", True),
    (" — twenty percent, capped, never enough on its own to wave through a genuine ring.", False),
])

# ── 3:35 ──────────────────────────────────────────────────────────────────
timing("3:35", "It does not fall over")
stage("ALT+TAB → TERMINAL. Press Enter. That is the only key you press.")
say([
    ("Ringfence never moves money. It recommends: approve, review, or block. Anything "
     "automatic needs more than ", False),
    ("ten analyst-confirmed precedents", True),
    (", and a test says a new pattern can never unlock it. Now let me ask it about a card "
     "that does not exist.", False),
])
say([
    ("Two hundred, not a five hundred. Degraded: true. Recommendation: ", False),
    ("review", True),
    (". A fraud API that throws an error is worse than useless — the caller times out and "
     "the payment goes through anyway. ", False),
    ("A detector that cannot see escalates to a human. It never quietly approves.", True),
])
stage("ALT+TAB → BROWSER. Back on page 4, stay there.")
say([
    ("And that “never a merchant edge” claim — that is measured. Sharing a merchant links ", False),
    ("ninety-nine point nine nine percent", True),
    (" of all card pairs. Sharing an identity links ", False),
    ("nought point one five percent", True),
    (".", False),
])

# ── 4:15 ──────────────────────────────────────────────────────────────────
timing("4:15", "The ask")
stage("Stop scrolling. Look at the camera.")
say([
    ("A hundred and fourteen tests. Clone it, run four commands, and you get every number I "
     "just showed you.", False),
])
say([
    ("What I want next is ", False),
    ("Razorpay test-mode API access", True),
    (", for two honest reasons. Our identity layer is calibrated to published data, and real "
     "Razorpay fingerprints will be messier — that is the biggest threat to the precision "
     "number I just showed you, and I would rather find out than defend it. And the Vulcan "
     "score is simulated: the composition and gating are real and tested; the score is "
     "generated.", False),
])
say([
    ("Ringfence hands the foundation model the one thing a per-transaction view structurally "
     "cannot have — ", False),
    ("what happened between the transactions.", True),
])
say([
    ("It does not replace Vulcan. ", False),
    ("It completes it.", True),
])

doc.save(OUT)
print("saved:", OUT)
