"""Generate the Ringfence teleprompter script as a Word document."""
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUT = r"c:\Users\Amiya\ringfence\demo\Ringfence_Video_Script.docx"

INK = RGBColor(0x1A, 0x1A, 0x1A)
GREY = RGBColor(0x66, 0x66, 0x66)
RED = RGBColor(0xB3, 0x1B, 0x1B)
BLUE = RGBColor(0x1F, 0x4E, 0x79)

doc = Document()

# ── page + base styles ────────────────────────────────────────────────────
for s in doc.sections:
    s.top_margin = s.bottom_margin = Inches(0.7)
    s.left_margin = s.right_margin = Inches(0.85)

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)
normal.font.color.rgb = INK


def shade(par, hexfill):
    """Paint a paragraph background — used for the stage-direction strips."""
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), hexfill)
    par._p.get_or_add_pPr().append(el)


def spacer(pts=8):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(pts)
    p.paragraph_format.space_before = Pt(0)
    return p


def h1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text)
    r.font.size = Pt(24)
    r.font.bold = True
    r.font.color.rgb = INK
    return p


def h2(text, colour=INK):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = colour
    return p


def timing(clock, title):
    """Segment header: big clock + title."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(20)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(clock + "   ")
    r.font.size = Pt(17)
    r.font.bold = True
    r.font.color.rgb = BLUE
    r2 = p.add_run(title)
    r2.font.size = Pt(17)
    r2.font.bold = True
    r2.font.color.rgb = INK
    return p


def stage(text):
    """Grey strip: what to do with your hands and windows. Never read aloud."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.left_indent = Inches(0.05)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.size = Pt(10.5)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    shade(p, "EDEDED")
    return p


def say(text):
    """The words. Large, generous line spacing, readable from a distance."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.line_spacing = 1.45
    p.paragraph_format.left_indent = Inches(0.12)
    for chunk, bold in text:
        r = p.add_run(chunk)
        r.font.size = Pt(13.5)
        r.font.bold = bold
        r.font.color.rgb = INK
    return p


def note(text, colour=GREY):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.left_indent = Inches(0.12)
    r = p.add_run(text)
    r.font.size = Pt(10)
    r.font.italic = True
    r.font.color.rgb = colour
    return p


def bullets(items, bold_lead=True):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.left_indent = Inches(0.3)
        if isinstance(it, tuple):
            r = p.add_run(it[0])
            r.font.size = Pt(11)
            r.font.bold = bold_lead
            r2 = p.add_run(it[1])
            r2.font.size = Pt(11)
        else:
            r = p.add_run(it)
            r.font.size = Pt(11)


def mono(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.left_indent = Inches(0.25)
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(12)
    r.font.bold = True
    shade(p, "F2F2F2")
    return p


def simple_table(rows, widths, header=True):
    t = doc.add_table(rows=0, cols=len(widths))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci, val in enumerate(row):
            cells[ci].width = Inches(widths[ci])
            par = cells[ci].paragraphs[0]
            par.paragraph_format.space_after = Pt(2)
            r = par.add_run(val)
            r.font.size = Pt(10.5)
            if header and ri == 0:
                r.font.bold = True
        if header and ri == 0:
            for c in cells:
                shade(c.paragraphs[0], "E8E8E8")
    return t


# ══════════════════════════════════════════════════════════════════════════
# COVER
# ══════════════════════════════════════════════════════════════════════════
h1("Ringfence — 5-Minute Video Script")
p = doc.add_paragraph()
r = p.add_run("Razorpay AI Buildathon 2026  ·  Track 02, AI Risk Manager  ·  Amiya Manas Singh")
r.font.size = Pt(11.5)
r.font.color.rgb = GREY

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(10)
r = p.add_run("Grey strips are stage directions — never read them aloud. "
              "Large text is what you say, word for word.")
r.font.size = Pt(11)
r.font.italic = True
r.font.color.rgb = RED

# ── Before you record ─────────────────────────────────────────────────────
h2("Before you press record")

note("Run these four lines, then leave both terminals alone for the rest of the session.")
mono("make clean-memory        make test        make run        make demo")

note("make clean-memory is not optional — the failure demo at 2:45 only works if case "
     "memory is empty. Then open the dashboard and click through all five pages once.")

bullets([
    ("Sidebar says API online · ok", "  (green box, top left)"),
    ("Page 4 shows", "  0.942 / 0.869 / 0.964 / 0.978 / +17%"),
    ("Page 5 still says", "  “Click the button to write this verdict into case memory” — "
     "if it already shows a damp, run make clean-memory again"),
])

# ── Two windows ───────────────────────────────────────────────────────────
h2("Open exactly two windows")

note("Two windows is the whole trick. With only two, Alt+Tab is a clean toggle — it lands "
     "in the right place every single time. Add a third window and Alt+Tab starts cycling "
     "most-recent-first and will drop you somewhere you did not expect, on camera.")

simple_table([
    ["Window", "What it is", "How to set it up"],
    ["1  Browser", "The dashboard, localhost:8501", "One tab only. Press F11 for fullscreen "
     "so tabs, address bar and bookmarks disappear. Zoom 100%."],
    ["2  Terminal", "Where you press Enter once, at 3:35", "Font bumped to about 18pt. The "
     "command already typed at the prompt — do NOT press Enter until the video."],
], [1.0, 2.0, 3.6])

spacer(6)
note("This is the one line to type into the terminal beforehand. Type it, then leave the "
     "cursor sitting there. It is the only command in the whole video.")
mono("python demo/show_degraded.py")

note("Close everything else. Turn on Do Not Disturb (Win + A). Record the WHOLE SCREEN, "
     "not a single window — window capture goes black the moment you Alt+Tab.")

# ── Alt+Tab plan ──────────────────────────────────────────────────────────
h2("Your Alt+Tab plan — you press it twice, total")

simple_table([
    ["When", "Press", "You land on"],
    ["0:00 – 3:35", "nothing", "Browser. You stay here for the first three and a half minutes."],
    ["3:35", "Alt + Tab", "Terminal. Press Enter on the command that is already typed."],
    ["3:55", "Alt + Tab", "Browser again, for the rest of the video."],
], [1.4, 1.4, 3.8])

spacer(6)
note("Start saying the next line WHILE you Alt+Tab. Never switch in silence — a two-second "
     "gap is where a reviewer clicks away.")

# ── Which alert ───────────────────────────────────────────────────────────
h2("Which alert to click — read this twice")

note("There is only one place in the whole video where you choose something, and it happens "
     "at 1:10. Do not go hunting for a good example while recording.")

bullets([
    ("On page 1, set the left dropdown ", "Recommendation → BLOCK"),
    ("Click ", "Open →  on the very top row"),
    ("In the sidebar, click ", "2 · Dossier Viewer"),
])

spacer(4)
note("Never memorise an alert ID. The IDs are random and they all change every time you run "
     "make clean-memory — which you do right before recording. Filtering to BLOCK and taking "
     "the top row always works. Rehearse this click path five times before your first take.")

spacer(4)
note("Page 5 needs no choice at all. It picks its own example — the highest-scoring case the "
     "model got wrong. Say that out loud; it is the point.")

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════
# THE SCRIPT
# ══════════════════════════════════════════════════════════════════════════
h1("The Script")
note("Everything in large text is spoken word for word. 796 words — that is 4:49 at a "
     "steady pitch pace, leaving you about ten seconds of slack for the two clicks. "
     "Time your first take. If it runs past 5:00, cut the 0:30–1:10 segment, not words "
     "from the rest.", RED)

# ── 0:00 ──────────────────────────────────────────────────────────────────
timing("0:00 – 0:30", "The five-rupee probe")
stage("BROWSER, page 1 — Alert Queue.  Hands off the mouse. Do not click anything. "
      "Start talking on frame one.")
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
timing("0:30 – 1:10", "Where Ringfence sits")
stage("BROWSER, page 1.  Scroll slowly down the queue, then back to the top.")
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
timing("1:10 – 2:00", "The case file")
stage("BROWSER — filter Recommendation → BLOCK.  Click Open → on the top row.  "
      "Sidebar → 2 · Dossier Viewer.  Then scroll down to the evidence table and STOP there.")
say([
    ("This is one alert. Ringfence scores the cluster at one point oh. Vulcan, on the same "
     "activity, point three eight. Composite: ", False),
    ("block.", True),
])
note("Read those three numbers off your screen as you say them — they shift slightly "
     "between rows. Never recite them from memory.")
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
timing("2:00 – 2:45", "Honest metrics")
stage("BROWSER — sidebar → 4 · Metrics.  Scroll to the cost curve.  "
      "Point your cursor at the star, then at the cross.")
say([
    ("Ring recall ", False), ("nought point nine four two", True),
    (". Precision ", False), ("nought point eight six nine", True),
    (". That is not the best we could report — and this chart is why we did not chase it.", False),
])
say([
    ("A false positive costs two point three lakh. A ", False), ("miss", True),
    (" costs eight and a half — so a miss is worth three point seven false positives, and "
     "F1 is the wrong objective. The star is where cost is lowest. The cross is where F1 "
     "peaks, and it is ", False),
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
timing("2:45 – 3:35", "One failure, handled live")
stage("BROWSER — sidebar → 5 · Failure Recovery.  Scroll to step 2.  "
      "The analyst note is already filled in — do NOT retype it.")
say([
    ("Now the failure. This is the highest-scoring thing the model got wrong — picked by the "
     "labels, not chosen by me. Four cards: a household sharing one tablet. Ringfence gave "
     "it ", False),
    ("nought point oh one nine", True),
    (", but composited with Vulcan it still lands at ", False),
    ("nought point four zero three — review", True),
    (". Two point three lakh — a cost we own.", False),
])
stage("CLICK the red “Mark as False Positive” button.  Wait for the page to reload.  "
      "Scroll down to step 3 and let the two bars land on screen.")
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
timing("3:35 – 4:15", "It does not fall over")
stage("ALT+TAB → TERMINAL.  Press Enter on the command that is already typed. That is the "
      "only key you press. Say the first line while you are switching.")
say([
    ("Ringfence never moves money. It recommends: approve, review, or block. Anything "
     "automatic needs more than ", False),
    ("ten analyst-confirmed precedents", True),
    (", and a test says a new pattern can never unlock it. Now let me ask it about a card "
     "that does not exist.", False),
])
note("Four lines appear. Point at the first two.")
say([
    ("Two hundred, not a five hundred. Degraded: true. Recommendation: ", False),
    ("review", True),
    (". A fraud API that throws an error is worse than useless — the caller times out and "
     "the payment goes through anyway. ", False),
    ("A detector that cannot see escalates to a human. It never quietly approves.", True),
])
stage("ALT+TAB → BROWSER.  You are back on page 4 and you stay there.")
say([
    ("And that “never a merchant edge” claim — that is measured. Sharing a merchant links ", False),
    ("ninety-nine point nine nine percent", True),
    (" of all card pairs. Sharing an identity links ", False),
    ("nought point one five percent", True),
    (".", False),
])

# ── 4:15 ──────────────────────────────────────────────────────────────────
timing("4:15 – 5:00", "The ask")
stage("BROWSER, page 4 behind you.  Stop scrolling. Look at the camera and talk.")
say([
    ("A hundred and fourteen tests. Clone it, run four make commands, and you get every "
     "number I just showed you.", False),
])
say([
    ("What I want next is ", False),
    ("Razorpay test-mode API access", True),
    (", for two honest reasons. Our identity layer is calibrated to published data, and "
     "real Razorpay fingerprints will be messier — that is the biggest threat to the "
     "precision number I just showed you, and I would rather find out than defend it. And "
     "the Vulcan score is simulated: the composition and gating are real and tested; the "
     "score is generated.", False),
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

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════
# BACK PAGE
# ══════════════════════════════════════════════════════════════════════════
h1("If something goes wrong")

simple_table([
    ["What happens", "What you do — out loud, without stopping"],
    ["Sidebar says API offline",
     "Keep going. Say: “the dashboard falls back to local files when the API is down — "
     "that is the degradation story I am about to show you.” Turning the fault into the "
     "feature is a better moment than the one you lost."],
    ["Open → opens the wrong case file",
     "You are on an old copy of the code. Stop the take, run git pull, restart the dashboard."],
    ["Page 5 already shows a damp",
     "Case memory was not cleared. Run make clean-memory, restart the dashboard, start again."],
    ["The terminal command fails",
     "The API stopped. Skip straight to the merchant-edge numbers — they need nothing running."],
    ["You stumble over a word",
     "Keep going. Never restart in the middle of a take. You will fix it in the next one."],
], [1.7, 4.9])

h2("Five rules for the take")
bullets([
    ("Start talking on frame one. ", "No “okay, so…”. The first fifteen seconds decide "
     "whether anyone watches the rest."),
    ("Speak louder than feels natural. ", "Laptop microphones flatten a quiet voice."),
    ("Never pause longer than two seconds. ", "Dead air is what makes people click away."),
    ("Finish between 4:55 and 5:00. ", "Under is fine. Over is disqualifying."),
    ("Record three takes and pick one. ", "Do not chase a perfect take — pick the one with "
     "the best first fifteen seconds."),
])

h2("If you have to cut something")
note("Cut 0:30–1:10 (where Ringfence sits) and 3:35–4:15 (it does not fall over). "
     "Never cut the case file at 1:10 or the failure recovery at 2:45 — those two are the "
     "reason this submission is different from the other twelve thousand.")

h2("The moment you stop recording")
bullets([
    ("Watch only the first fifteen seconds. ", "If the hook does not land, that take is dead."),
    ("Upload it. ", "Title and description are ready to paste in FORM_ANSWERS.md."),
    ("Paste the link into SUBMISSION.md, ", "then commit and push."),
    ("Fill the form from FORM_ANSWERS.md. ", "Paste it. Do not improvise at the form."),
])

doc.save(OUT)
print("saved:", OUT)
