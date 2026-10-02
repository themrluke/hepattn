"""Assemble the TrackML-baseline deck from the generated figures.

    .venv/bin/python build_deck.py

Slides are declared as data in build(), so re-ordering or re-wording is a one-line edit.
Figures come from figures/; regenerate them with figs_*.py first. The slide helpers are the
same as the OR-amplification deck's, so the two talks look like a series.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.lang import MSO_LANGUAGE_ID
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).parent
FIGS = HERE / "figures"
# Writes a separate file: TrackML_baseline.pptx has been hand-edited by Luke and must not be overwritten.
OUT = HERE / "TrackML_baseline_generated.pptx"

W, H = Inches(13.333), Inches(7.5)          # 16:9
BG = RGBColor(0x0F, 0x16, 0x20)
INK = RGBColor(0xE8, 0xED, 0xF4)
MUTED = RGBColor(0x8F, 0xA3, 0xBC)
ACCENT = RGBColor(0xFF, 0xB4, 0x54)
GOOD = RGBColor(0x6B, 0xCB, 0x77)
WARN = RGBColor(0xFF, 0x6B, 0x6B)
BASE = RGBColor(0x5A, 0xC8, 0xFA)
FONT = "Trebuchet MS"


def _bg(slide):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = BG


def _text(slide, left, top, width, height, runs, align=PP_ALIGN.LEFT, spacing=1.0):
    """runs: list of (text, size, colour, bold)."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, (text, size, colour, bold) in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(6)
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.color.rgb = colour
        r.font.bold = bold
        r.font.name = FONT
        # Without an explicit language PowerPoint can fall back to line-breaking rules that allow
        # a break *inside* a word ("precis / ion"). Tagging the run as English restores normal
        # word-boundary wrapping. This sets an attribute only - it does not touch layout.
        r.font.language_id = MSO_LANGUAGE_ID.ENGLISH_UK
    return box


def _rule(slide, top, colour=ACCENT, left=Inches(0.7), width=Inches(2.0)):
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Pt(4))
    s.fill.solid(); s.fill.fore_color.rgb = colour
    s.line.fill.background()
    s.shadow.inherit = False


def _number(slide, n):
    """Slide number, bottom-right. Skipped on the title slide."""
    box = slide.shapes.add_textbox(Inches(12.2), Inches(6.85), Inches(0.9), Inches(0.4))
    p = box.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    r = p.add_run()
    r.text = str(n)
    r.font.size = Pt(12)
    r.font.color.rgb = MUTED
    r.font.name = FONT
    r.font.language_id = MSO_LANGUAGE_ID.ENGLISH_UK


def slide_title(prs, title, subtitle, footer):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _text(s, Inches(1.0), Inches(2.3), Inches(11.3), Inches(1.6),
          [(title, 44, INK, True)])
    _rule(s, Inches(3.95), left=Inches(1.05), width=Inches(2.6))
    _text(s, Inches(1.0), Inches(4.25), Inches(11.3), Inches(1.4),
          [(subtitle, 21, MUTED, False)])
    _text(s, Inches(1.0), Inches(6.35), Inches(11.3), Inches(0.6),
          [(footer, 14, ACCENT, False)])
    return s


def slide_section(prs, kicker, title):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _text(s, Inches(1.0), Inches(2.9), Inches(11.3), Inches(0.5),
          [(kicker.upper(), 15, ACCENT, True)])
    _text(s, Inches(1.0), Inches(3.35), Inches(11.3), Inches(1.4),
          [(title, 36, INK, True)])
    _rule(s, Inches(4.9), left=Inches(1.05), width=Inches(2.0))
    return s


def slide_figure(prs, title, figure, caption=None, bullets=None):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _text(s, Inches(0.7), Inches(0.35), Inches(12.0), Inches(0.8), [(title, 26, INK, True)])
    _rule(s, Inches(1.12))

    img = FIGS / f"{figure}.png"
    if img.exists():
        from PIL import Image
        iw, ih = Image.open(img).size
        # Leave room underneath for the caption or bullets, or they fall off the slide.
        avail_w = Inches(11.9)
        avail_h = Inches(4.15) if bullets else (Inches(4.9) if caption else Inches(5.6))
        scale = min(avail_w / iw, avail_h / ih)
        w, h = int(iw * scale), int(ih * scale)
        s.shapes.add_picture(str(img), int((W - w) / 2), Inches(1.45), w, h)
        bottom = Inches(1.45) + Emu(h)
    else:
        bottom = Inches(3.0)

    if bullets:
        _text(s, Inches(0.8), bottom + Inches(0.14), Inches(11.7), Inches(1.0),
              [(f"•  {b}", 14.5, c, bold) for b, c, bold in bullets], spacing=1.2)
    elif caption:
        _text(s, Inches(0.8), bottom + Inches(0.15), Inches(11.7), Inches(0.8),
              [(caption, 16, ACCENT, True)], align=PP_ALIGN.CENTER)
    return s


def slide_bullets(prs, title, bullets, note=None):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _text(s, Inches(0.7), Inches(0.35), Inches(12.0), Inches(0.8), [(title, 26, INK, True)])
    _rule(s, Inches(1.12))
    runs = []
    for text, size, colour, bold in bullets:
        runs.append((text, size, colour, bold))
    _text(s, Inches(0.85), Inches(1.6), Inches(11.6), Inches(4.8), runs, spacing=1.3)
    if note:
        _text(s, Inches(0.85), Inches(6.35), Inches(11.6), Inches(0.7),
              [(note, 14, MUTED, False)])
    return s


def slide_two_col(prs, title, left_head, left_items, right_head, right_items, note=None):
    s = prs.slides.add_slide(prs.slide_layouts[6]); _bg(s)
    _text(s, Inches(0.7), Inches(0.35), Inches(12.0), Inches(0.8), [(title, 26, INK, True)])
    _rule(s, Inches(1.12))
    for x, head, items, colour in ((Inches(0.8), left_head, left_items, WARN),
                                   (Inches(7.0), right_head, right_items, GOOD)):
        _text(s, x, Inches(1.6), Inches(5.5), Inches(0.6), [(head, 22, colour, True)])
        _text(s, x, Inches(2.3), Inches(5.5), Inches(4.0),
              [(f"•  {t}", 15, INK if not em else colour, em) for t, em in items], spacing=1.3)
    if note:
        _text(s, Inches(0.8), Inches(6.4), Inches(11.7), Inches(0.7), [(note, 14, MUTED, False)])
    return s


VIOLET = RGBColor(0xC7, 0x92, 0xEA)


# What we know about the first filter run's failure. Update as the retrain passes the learning-rate peak.
FAILURE_BULLETS = [
    ("Healthy for 3 epochs, then the loss climbed for ~0.8 epoch and went NaN at step 34,199", WARN, True),
    ("It looked like the learning-rate warm-up — but the retrain, with the identical schedule, passed the same "
     "peak cleanly", MUTED, False),
    ("The real cause was six corrupt pixels in the TrackML data (next slides)", ACCENT, True),
]

CORRUPT_BULLETS = [
    ("Kaggle's raw files: one pixel per event with charge up to 716,689, where every other value is ≤ 1", ACCENT, True),
    ("Re-downloaded and checksum-verified; not in Kaggle's blacklist; not documented anywhere we found", MUTED, False),
    ("No hepattn dataloader — ours or any upstream branch — cleans them: cast to fp16, they become inf", WARN, True),
]


def build() -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    slide_title(
        prs,
        "From OR amplification to a baseline we can trust",
        "Fixing the ColliderML setup, choosing what to reproduce on TrackML, "
        "and what broke on the way",
        "TrackML · ColliderML · hepattn  ·  work since the OR-amplification talk",
    )

    slide_bullets(prs, "Where we left off", [
        ("OR amplification is built, tested and benchmarked: about 1.44x per training step.", 19, INK, False),
        ("But the A/B comparison said nothing — both arms trained equally badly.", 19, WARN, True),
        (" ", 10, INK, False),
        ("You cannot judge an encoder on a baseline that doesn't learn.", 21, ACCENT, True),
        (" ", 10, INK, False),
        ("Plan:  fix the ColliderML baseline at zero pile-up,", 18, INK, False),
        ("           and in parallel reproduce a published TrackML baseline to test OR against.", 18, INK, False),
        ("Goal:  a short paper — the OR encoder with UCL's MaskFormer decoders.", 18, GOOD, True),
    ])

    slide_figure(prs, "ColliderML: the selection was the problem, not the model", "01_colliderml", bullets=[
        ("Pixel barrel with |eta| < 1 left 4.2 hits per particle — barely more than the 3 needed to be a target",
         MUTED, False),
        ("Widened to the whole tracker out to |eta| < 2.5 at zero pile-up: 11.8 hits per particle", ACCENT, True),
        ("Overfit on 10 events reaches 98% efficiency — targets, masks and matching all work", GOOD, True),
    ])

    slide_figure(prs, "TrackML: what we have to reproduce", "04_targets", bullets=[
        ("UCL's June paper (arXiv 2606.17631) compares three decoders on one filter + encoder", MUTED, False),
        ("Three of the five decoders we want for our paper, with every training setting stated", ACCENT, True),
        ("DQ+MA at 98.1% is the baseline OR has to beat", GOOD, True),
    ])

    slide_figure(prs, "Our TrackML copy matches the paper before any training", "03_table3", bullets=[
        ("All 8,850 Kaggle training events, prepped; 100 val / 100 test as in the paper", MUTED, False),
        ("Pixel hits and particle counts agree with the paper's Table 3", GOOD, True),
        ("So any difference later comes from the models, not the data", ACCENT, True),
    ])

    slide_figure(prs, "Getting the code the paper actually used", "06_code_audit", bullets=[
        ("The paper code was on an unmerged branch (hepattn-clean), not on upstream main", ACCENT, True),
        ("As pushed it crashes on every decoder forward pass (NameError) — fixed; 409 / 409 tests pass", WARN, True),
        ("All OR-amplification tests still pass after the merge", GOOD, False),
    ])

    slide_figure(prs, "The experiment: four arms", "07_four_arms", bullets=[
        ("A vs D: OR inside UCL's own pipeline — same input, only the encoder differs", GOOD, True),
        ("B vs C: OR with no filter, ~57k hits per event, where a 512-hit window misses the most", ACCENT, False),
    ])

    slide_figure(prs, "First hit-filter run: diverged after 3 epochs", "08_nan_failure",
                 bullets=FAILURE_BULLETS)

    slide_figure(prs, "Six corrupt pixels in the TrackML data", "05_corrupt_charge",
                 bullets=CORRUPT_BULLETS)

    slide_figure(prs, "Why the filter broke: the fp16 gradient scale hit zero", "08b_grad_scale",
                 bullets=[
        ("fp16 skips each corrupt step, but halves the gradient scale — which only regrows after 2,000 clean steps",
         MUTED, False),
        ("One corrupt event every ~1,400 steps, so the scale only fell: 4,096 → 256 → 16 → 0", WARN, True),
        ("Small gradients underflowed to zero while Lion kept taking full steps → loss climbed; scale 0 → NaN",
         ACCENT, True),
    ])

    slide_figure(prs, "In bf16 they would silently freeze training instead", "05b_freeze", bullets=[
        ("bf16 (every tracking config) has no scaler: Lion's momentum turns NaN, and sign(NaN) = 0 — weights never "
         "move again", WARN, True),
        ("No NaN anywhere and a plausible loss: only a flat curve would give it away", MUTED, False),
        ("Removed from training (6 of 8,650 events) — worth reporting to UCL", ACCENT, False),
    ])

    slide_figure(prs, "Filter retrain: in progress", "09_retrain", bullets=[
        ("Same settings as the first run, minus the six corrupt events; ~16 h for 80 epochs", MUTED, False),
        ("Gradient scale steady at 32,768; passed the learning-rate peak cleanly; still improving every epoch",
         GOOD, True),
        ("Then: run it over train / val / test and check 98.6% particle retention", ACCENT, True),
    ])

    slide_bullets(prs, "Next, and what we need from UCL", [
        ("Next", 21, ACCENT, True),
        ("1.  Check the filter against the paper: 98.6% retention, 9.7k hits per event", 17, INK, False),
        ("2.  Train DQ+MA (arm A) to 98.1%, then the OR arms", 17, INK, False),
        ("3.  Full ColliderML pu0 run", 17, INK, False),
        (" ", 10, INK, False),
        ("Questions for UCL", 21, ACCENT, True),
        ("•  Which branch produced the paper: hepattn-clean or hepattn-dq-gcp?", 16, INK, False),
        ("•  Should the dq-cleanup matcher fix be in?   •  Where is sparse k-max?", 16, INK, False),
        ("•  DQ+MA config says 10 epochs, the paper 30   •  Filter learning rate / precision: config or tuned?",
         16, INK, False),
        ("•  Did you meet the six corrupt TrackML pixels? Any bf16 runs affected?", 16, INK, False),
    ])
    return prs


if __name__ == "__main__":
    deck = build()
    # Number every slide except the title, once the deck is complete so the numbering always
    # matches the final order however the content above is edited.
    for i, slide in enumerate(deck.slides, start=1):
        if i > 1:
            _number(slide, i)
    deck.save(OUT)
    print(f"  wrote {OUT.name}  ({len(deck.slides)} slides)")
