"""Assemble the OR-amplification deck from the generated figures.

    .venv/bin/python build_deck.py

Slides are declared as data in DECK below, so re-ordering or re-wording is a one-line edit.
Figures come from figures/; regenerate them with figs_*.py first.
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
OUT = HERE / "OR_amplification.pptx"

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


def build() -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # ============================================================ 1. framing
    slide_title(
        prs,
        "Per-head orderings and OR amplification",
        "Letting every attention head see a different arrangement of the same event — "
        "and why the way you implement it is worth 11x",
        "ColliderML tracking  ·  hepattn  ·  OR-amplification branch",
    )

    slide_bullets(prs, "The problem in one slide", [
        ("A pixel event has ~100,000 hits. Full attention is N² ≈ 10¹⁰ per layer.", 19, INK, False),
        ("So we let each hit attend only to a window of ~512 neighbours — a ~200x saving.", 19, INK, False),
        (" ", 10, INK, False),
        ("But attention operates on a 1-D list, and \"nearby in the list\" is not \"nearby in the detector\".",
         19, ACCENT, True),
        (" ", 10, INK, False),
        ("Ordering (serialization) is what makes those two things mean the same thing.", 19, INK, False),
        ("And no single ordering can keep every track intact.", 19, WARN, True),
    ], note="Everything in this talk is downstream of that last line.")

    slide_figure(prs, "Why we cannot afford full attention", "01_cost_scaling",
                 caption="Local attention is not an optimisation — it is the only way this runs at all")

    slide_figure(prs, "Serialization: flattening the detector onto a line", "02_serialization",
                 caption="Sorting is what gives \"my neighbours in the list\" any physical meaning")

    slide_figure(prs, "The catch: every ordering cuts somewhere", "04_split_track", bullets=[
        ("2-D geometry cannot be flattened to 1-D without cuts — this is structural, not a tuning problem",
         MUTED, False),
        ("A track split by one ordering is usually kept whole by a different one", GOOD, True),
        ("So give each head several orderings and combine them. That is OR amplification.", ACCENT, True),
    ])

    # ============================================================ 2. attention mechanisms
    slide_section(prs, "background", "Which local-attention pattern, and why it matters")

    slide_figure(prs, "Two ways to define \"the window of neighbours\"", "03_window_vs_block", bullets=[
        ("Sliding window: symmetric, overlapping, no boundaries — but windows do not tile, so it needs a mask",
         BASE, False),
        ("Block-sparse: fixed blocks, dense inside each, no mask at all — but it introduces seams",
         VIOLET, False),
        ("HEPTv2 is block-sparse — code-confirmed: no mask argument anywhere in its encoder", MUTED, False),
    ])

    slide_figure(prs, "\"BlockMask\" is not \"block-sparse\"", "03a_blockmask_vs_blocksparse", bullets=[
        ("BlockMask is a PyTorch data structure: which 128x128 tiles hold any admitted pair", MUTED, False),
        ("Block-sparse is an attention PATTERN: fixed disjoint blocks, dense inside each", VIOLET, False),
        ("We store a sliding window in a BlockMask — the structure, not the pattern", GOOD, True),
    ])

    slide_two_col(
        prs, "Sliding window vs block-sparse: which for us?",
        "Block-sparse", [
            ("Hard seams every block_size tokens", False),
            ("A track straddling a seam loses its other half at that layer", False),
            ("Edge hits see companions on one side only", False),
            ("But: gather → reshape → dense, no mask. Perfect kernel utilisation.", True),
        ],
        "Sliding window", [
            ("No seams — every hit gets a symmetric neighbourhood", False),
            ("Preserves more track locality per unit of compute", False),
            ("Requires a real mask (flex mask_mod + BlockMask)", False),
            ("Chosen: it is what our encoder and benchmarks are already built around", True),
        ],
        note="Key insight: block-sparse is not better in the abstract — it becomes SAFE once OR "
             "amplification exists to repair its seams. HEPTv2's two choices are a matched pair, not "
             "independent.")

    # ============================================================ 3. the mechanism
    slide_section(prs, "the mechanism", "How several orderings become one answer")

    slide_figure(prs, "OR amplification: the log-sum-exp merge", "08_lse_merge", bullets=[
        ("You cannot just average — a neighbour found by two orderings must count more than one found by one",
         WARN, True),
        ("lse is the attention mass each neighbourhood gathered: exactly the right weight", MUTED, False),
        ("The result is mathematically EXACT attention over the union of all the neighbourhoods", GOOD, True),
    ])

    slide_bullets(prs, "Two axes that are easy to confuse", [
        ("c = n_hashes       different LSH orderings   →   merged by LSE, inside each head", 17, ACCENT, True),
        ("                            buys:  OR amplification, i.e. recall", 15, MUTED, False),
        (" ", 8, INK, False),
        ("h = num_heads    different learned subspaces   →   concat + out_proj", 17, BASE, True),
        ("                            buys:  ordinary multi-head capacity", 15, MUTED, False),
        (" ", 8, INK, False),
        ("Head-mixing is NOT OR. OR is the LSE merge.", 22, WARN, True),
        (" ", 8, INK, False),
        ("You cannot fake this with a cleverer global sort: one list gives each hit exactly one set of",
         16, INK, False),
        ("neighbours. Multi-ordering must live INSIDE the attention layer.", 16, INK, False),
    ])

    slide_figure(prs, "How one ordering is actually built", "05a_lsh_construction", bullets=[
        ("Equal-OCCUPANCY quantile bins in eta and phi — each bin holds the same number of hits",
         ACCENT, False),
        ("Plus a random projection direction, which breaks ties INSIDE a bin", WARN, False),
        ("Bin terms step by D, the projection can only vary by D — so bins are coarse, projection fine",
         ACCENT, True),
    ])

    slide_figure(prs, "Why two cells disagree", "05b_lsh_two_cells", bullets=[
        ("Each cell draws its own bin counts, its own bin edges, and its own projection direction",
         MUTED, False),
        ("The same six hits end up spanning very different stretches of the sequence", ACCENT, True),
        ("That disagreement is not a defect — it is exactly what OR amplification exploits", GOOD, True),
    ])

    slide_figure(prs, "Where the diversity comes from", "05_or_grid", bullets=[
        ("Each (layer, head, hash) cell draws its own random projection and quantile grid", MUTED, False),
        ("Frozen at construction — the same event always serialises the same way, so inference is reproducible",
         GOOD, True),
        ("The RULE is frozen, not the order: different events still give different permutations", MUTED, False),
    ])

    # ============================================================ 4. implementation
    slide_section(prs, "implementation", "Two ways to build it — and they are not equal")

    slide_figure(prs, "masked vs sorted: same pairs, different question", "06_masked_vs_sorted", bullets=[
        ("masked — leave the hits, put the ordering in a lookup table. Every head needs its OWN mask.",
         WARN, True),
        ("sorted — rearrange the hits, and the rule stops mentioning any ordering. ONE mask for all 24 cells.",
         GOOD, True),
        ("Both compute bit-for-bit identical output. A test asserts it.", MUTED, False),
    ])

    slide_figure(prs, "Background: how tiled attention actually runs", "07a_tile_mechanics", bullets=[
        ("Flash and flex never build the full N x N matrix — they walk it in tiles sized for on-chip SRAM",
         MUTED, False),
        ("The mask is applied to the scores AFTER the tile's matmul: masking discards, it does not save",
         WARN, True),
        ("So a tile is all-or-nothing — skipped entirely, or paid for in full", ACCENT, True),
    ])

    slide_figure(prs, "Why that matters: the GPU skips tiles, not pairs", "07_tiles", bullets=[
        ("Same number of admitted pairs in both panels — only their POSITION relative to the tile grid differs",
         ACCENT, True),
        ("Schematic at 96 tokens; on real events the measured figures are 4.8% and 23.2% of tiles",
         MUTED, False),
    ])

    slide_figure(prs, "How much of that computed work is wasted", "07b_waste", bullets=[
        ("Only 3.9% of the score matrix is genuinely wanted — and that is identical for every ordering",
         MUTED, False),
        ("sorted computes 1.2x more than needed;  masked computes 5.9x more", ACCENT, True),
    ])

    slide_figure(prs, "Where it plugs into the encoder", "09_pipeline", bullets=[
        ("Coordinates must be permuted alongside the tokens — the subtlest bug in the whole feature",
         WARN, False),
        ("Padding sorts to the tail of every ordering, so one shared mask can still exclude it", MUTED, False),
    ])

    # ============================================================ 5. results
    slide_section(prs, "results", "What we measured")

    slide_figure(prs, "Step 4a — the go/no-go: is the mask actually sparse?", "10_density", bullets=[
        ("Worry: 24 different orderings over one layout could make every tile non-empty (density 1.0)",
         MUTED, False),
        ("It did not — LSH orderings sort the same geometry, so they stay correlated", GOOD, True),
        ("How you lay the sequence out matters as much as the orderings do", ACCENT, True),
    ])

    slide_figure(prs, "Where the time actually goes", "11_cost_breakdown", bullets=[
        ("masked must rebuild a 24-row mask every layer of every event — its rule depends on the event",
         WARN, False),
        ("sorted's window is position-only: one row, built once, reused across layers and events", GOOD, True),
    ])

    slide_figure(prs, "The headline numbers", "12_end_to_end", bullets=[
        ("sorted is 11x cheaper than masked, for identical output — this decided the design", GOOD, True),
        ("vs today's sliding window: 5.6x for the encoder, but only 1.44x per training step", ACCENT, True),
        ("(attention is a smaller share of the full model than the isolated benchmark suggests)",
         MUTED, False),
    ])

    slide_figure(prs, "Tiling cannot rescue the masked implementation", "13_block_sweep",
                 caption="The gap is a property of how scattered the ranks are, not of the tile size")

    slide_figure(prs, "A methodology warning worth your time", "14_compile_trap", bullets=[
        ("This flipped a conclusion: mask-building went from \"<4% of the cost\" to \"92% of the cost\"",
         WARN, True),
        ("Nothing about the mask changed — only what it was being divided by", ACCENT, True),
    ])

    # ============================================================ 6. correctness
    slide_section(prs, "correctness", "How we know it is right")

    slide_bullets(prs, "Testing strategy: degenerate cases with known answers", [
        ("1.  One hash + identity ordering MUST reproduce the plain sliding window, exactly.", 17, GOOD, True),
        ("     Collapses the new path onto an old, trusted one.", 15, MUTED, False),
        (" ", 8, INK, False),
        ("2.  Give ONLY head 0 the identity ordering — catches head-major vs hash-major flattening,", 17, GOOD, True),
        ("     which the single-hash test structurally cannot see.", 15, MUTED, False),
        (" ", 8, INK, False),
        ("3.  sorted and masked must agree — this is what makes changing the default safe.", 17, GOOD, True),
        (" ", 8, INK, False),
        ("4.  Padding must not change the answer for real hits.", 17, GOOD, True),
        (" ", 8, INK, False),
        ("Every test was checked by reintroducing the bug and confirming it goes red.", 17, ACCENT, True),
    ], note="Bugs found this way: a fp16 dtype mismatch that would have crashed the first real forward "
            "pass, and a cached mask reusing the first event's padding.")

    slide_bullets(prs, "Bugs found along the way — all silent", [
        ("Coordinates not permuted with their tokens  →  every ordering describes hits that have moved",
         16, WARN, True),
        ("OR merge returned float32 into a half-precision layer  →  would have crashed the first real step",
         16, WARN, True),
        ("Cached mask captured the first event's padding  →  every later event masked wrongly", 16, WARN, True),
        ("set_backend silently dropped the sliding window  →  benchmarks against an unwindowed baseline",
         16, WARN, True),
        ("flex attention ran uncompiled everywhere  →  27x slower, and it inverted a design conclusion",
         16, WARN, True),
        (" ", 10, INK, False),
        ("None of these raise. All were found by testing against a known answer, or by running at "
         "realistic size and precision.", 17, ACCENT, True),
    ])

    # ============================================================ 7. status
    slide_section(prs, "status", "Where we are, and the open question")

    slide_figure(prs, "Training comparison — in flight", "15_training", bullets=[
        ("Both arms are indistinguishable, so this says nothing about OR amplification YET", ACCENT, True),
        ("Metrics are poor for both: the config is pixel-barrel only, ~4.4 hits per particle", WARN, False),
        ("Next: fix the baseline setup before the A/B can mean anything", MUTED, False),
    ])

    slide_bullets(prs, "Status and what is next", [
        ("Done:  frozen ordering grid · OR axis + LSE merge · per-head mask · encoder and MaskFormer wiring",
         17, GOOD, False),
        ("           padding · 452 tests passing · benchmarked · audited", 17, GOOD, False),
        (" ", 8, INK, False),
        ("Default is or_impl=\"sorted\"; masked stays as the reference implementation for benchmarking.",
         17, INK, False),
        (" ", 8, INK, False),
        ("The open question is not engineering:", 20, ACCENT, True),
        ("Does OR amplification earn its cost?  That needs a training run on a config whose", 18, INK, False),
        ("baseline actually reconstructs tracks — and ideally two seeds per arm, because a", 18, INK, False),
        ("single-seed A/B cannot separate a real effect from noise.", 18, INK, False),
    ], note="Nothing measured so far shows the feature HELPS — only that it is correct and what it costs.")

    return prs


if __name__ == "__main__":
    deck = build()
    # Number every slide except the title, once the deck is complete so the numbering always
    # matches the final order however the content above is edited.
    for i, slide in enumerate(deck.slides, start=1):
        if i > 1:
            _number(slide, i)
    deck.save(OUT)
    print(f"  wrote {OUT.name}  ({len(deck.slides.__iter__.__self__._sldIdLst)} slides)")
