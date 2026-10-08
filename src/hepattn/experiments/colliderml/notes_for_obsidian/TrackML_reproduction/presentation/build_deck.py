"""Assemble TrackML_reproduction.pptx (tracking meeting, 8 Oct 2026) from the generated figures.

    .venv/bin/python build_deck.py

Slides are declared as data in build(), so re-ordering or re-wording is a one-line edit.
Regenerate figures with figs_*.py first. Colours carry meaning:
BASE (blue) = the reference/baseline, ACCENT (amber) = our new work / OR, WARN (red) = broke/failed,
GOOD (green) = confirmed/passing, MUTED = context.
"""

from __future__ import annotations

from pathlib import Path

from deckkit import (ACCENT, BASE, GOOD, INK, MUTED, WARN, new_presentation, save_deck, slide_bullets,
                     slide_figure, slide_section, slide_title)

OUT = Path(__file__).parent / "TrackML_reproduction_generated.pptx"

# The window sweep is still training: these bullets say what is known now. Refresh before the meeting
# (README: extract_data.py comet, figs_or.py, build_deck.py) and edit the text together with the script.
LIVE_BULLETS = [
    ("Seed 42, test set: OR is 0.4 below the baseline on DM and 1.3 below on perfect efficiency", MUTED, False),
    ("Halving the window costs the baseline nothing: 97.9 / 95.1 / 0.3, as good as the flash reproduction", ACCENT, True),
    ("One seed; the run that leaves the plateau first wins (also at 512). Seed 43 and window 128 running", INK, False),
]
LIVE_CURVE_BULLETS = [
    ("x = epochs since each run left the plateau, so runs are compared at equal real training", MUTED, False),
    ("At window 256 OR stays slightly above its baseline until they converge ≈ 16 epochs in", ACCENT, True),
    ("OR w256 diverged in its last epoch (loss 0.46 → 660,829 in one step); its epoch-28 checkpoint was evaluated", WARN, False),
]
STATUS_LINES = [
    ("Paper reproduction, A′, D and both window-256 runs: finished and evaluated (pT > 1 GeV)", 18, GOOD, False),
    ("Window 128, seed 42: training, done ≈ 8 Oct 23:00 and 9 Oct 03:00; seed 43 ≈ 10–12 Oct", 18, INK, False),
    ("Each sweep run is evaluated automatically as it finishes (same evaluator, same cuts)", 18, INK, False),
    (" ", 8, INK, False),
    ("The notebook session has a 7-day lifetime: it expires 12 Oct 10:08", 18, WARN, True),
    ("That is what killed A′ and D on 5 Oct (not a reboot). Checkpoints survive on /shared, so the", 16, MUTED, False),
    ("last sweep runs will be resumed in the next session, losing at most one epoch", 16, MUTED, False),
]


def build():
    prs = new_presentation()

    slide_title(prs, "Reproducing UCL's Pix0.6 tracker",
                "Training 2–3× faster, an inconsistency in the paper, and the first OR-vs-baseline results",
                "Luke Johnson  ·  Tracking meeting  ·  8 Oct 2026")

    slide_bullets(prs, "Where we left off last week", [
        ("Hit-filter run diverged: six corrupt pixels + fp16 gradient scale hitting zero", 18, MUTED, False),
        ("Plan: confirm the filter, train DQ+MA to the paper's number, then the OR arms", 18, MUTED, False),
        (" ", 10, INK, False),
        ("This week", 21, ACCENT, True),
        ("•  Switched to UCL's own data and 600 MeV filter: Pix0.6", 18, INK, False),
        ("•  Made training 2–3× faster, with identical results", 18, INK, False),
        ("•  Reproduced the paper, and found Table 4 mixes two pT thresholds", 18, INK, False),
        ("•  First OR vs baseline comparison; window sweep running", 18, INK, False),
    ])

    slide_bullets(prs, "We now use UCL's data and their 600 MeV hit filter", [
        ("Data: the CodaLab release UCL used, 8,543 train / 100 val / 100 test events", 18, INK, False),
        ("56.7k pixel hits and 2607 ± 329 particles per event: Table 3 says 56.7k and 2600 ± 340", 17, GOOD, False),
        (" ", 8, INK, False),
        ("Filter: Pippa's 600 MeV filter outputs, threshold 0.1", 18, INK, False),
        ("99.13% particle retention, 20.3k hits per event: Table 3 says 99.1% and 20.3k", 17, GOOD, False),
        (" ", 8, INK, False),
        ("So we reproduce Pix0.6: trained on pT ≥ 0.6 GeV, evaluated on pT > 1 GeV", 19, ACCENT, True),
        ("Configs set to the paper's values: 3,900 queries, focal weight 25, LSCA window 64", 16, MUTED, False),
    ])

    slide_section(prs, "Part 1", "Making training 2–3× faster")

    slide_figure(prs, "Training was waiting on the Hungarian matcher", "01_step_shares", bullets=[
        ("Matching pairs ~3,900 predicted tracks with ~2,500 particles, 4× per step, on the CPU", MUTED, False),
        ("≈ 35% of every step was just copying the 243 MB cost table around", ACCENT, True),
    ])

    slide_figure(prs, "Fix: do the reshuffling on the GPU, copy once", "02_copies_diagram", bullets=[
        ("Toggle 1 (prepare_on_device): same numbers reach the same solver → identical matching", GOOD, True),
        ("Toggle 2 (trim_padded_queries): also drops the 18–39% padded query columns → smaller solve", ACCENT, False),
        ("Toggle 2 changes which leftover queries get the 'no object' loss (0.16% early, 0% late)", MUTED, False),
    ])

    slide_figure(prs, "Measured: 1.9–2.6× faster, 3.1× with both toggles", "03_speedup", bullets=[
        ("Same config, seed and events; timed over steps 150–600, after compile", MUTED, False),
        ("Toggle 1 cuts a 30-epoch run from ≈ 53–70 h to ≈ 20–36 h, with identical results", GOOD, True),
    ])

    slide_figure(prs, "Only the matcher got faster, and it's robust", "04_late_breakdown", bullets=[
        ("Faster on 100% of steps; repeats agree within 1%; other users' CPU load didn't move it", MUTED, False),
        ("Left in the matcher (≈ 114 ms): 69 ms waiting for the GPU forward, 38 ms of actual solves", ACCENT, False),
        ("Next gain: overlap the solves with the forward pass (up to ≈ 1.2×)", INK, False),
    ])

    slide_section(prs, "Part 2", "Reproducing the paper")

    slide_figure(prs, "Every run plateaus, then its masks snap into place", "05_val_curves", bullets=[
        ("Plateau: track masks are poor (dice ≈ 0.3); first-hit finding is already ≈ 96% efficient", MUTED, False),
        ("A′ (flex) only escaped at epoch 24: same model, much less real training", ACCENT, True),
        ("How long a run stays on the plateau is very sensitive → the sweep uses two seeds", INK, False),
    ])

    slide_figure(prs, "A′ isn't worse because of flex attention", "06_flash_flex", bullets=[
        ("Paper-run weights through both backends: outputs differ by less than bf16 rounding, loss equal", MUTED, False),
        ("The flash/flex gap in A′ is the plateau, not the attention kernel", GOOD, True),
    ])

    slide_figure(prs, "Test set: the reproduction is close to the paper", "07_test_results", bullets=[
        ("UCL's evaluator, pT > 1 GeV, track valid > 0.5, IoU > 0.5, 100 test events", MUTED, False),
        ("DM 97.7% vs 98.0%, perfect 94.7% vs 95.3%, fake rate 0.3% vs 0.3%", ACCENT, True),
        ("Best val loss 0.289 vs UCL's own run 0.285, both at epoch 28", INK, False),
    ])

    slide_figure(prs, "Table 4's perfect efficiency is a pT ≥ 0.6 number", "08_table4_vs_fig7", bullets=[
        ("Integrating the paper's own Figure 7 (values extracted from the PDF) over each pT threshold", MUTED, False),
        ("DM column = pT > 1, perfect column = pT ≥ 0.6, for both decoders: 93.2 → 95.3, 91.5 → 93.9", ACCENT, True),
        ("The updated paper (23 Sep) still has it, and repeats 93.2% in the conclusion", WARN, False),
    ])

    slide_figure(prs, "Why the threshold matters so much", "09_pt_counts", bullets=[
        ("57.5% of trained particles are below 1 GeV, where efficiency is lowest (≈ 93% and 97%)", MUTED, False),
        ("Including them pulls efficiency down ≈ 2 points: Table 4's 98.0% couldn't be a pT ≥ 0.6 value", ACCENT, True),
    ])

    slide_figure(prs, "Per pT bin we track the paper's Figure 7", "10_eff_pt", bullets=[
        ("DM within ≈ 0.5 points of the paper up to 6 GeV; perfect ≈ 0.5–0.8 below at 1–3 GeV", ACCENT, True),
        ("The remaining gap is in complete hit assignment, not in finding tracks", INK, False),
    ])

    slide_figure(prs, "Efficiency vs η matches the paper's shape", "11_eff_eta", bullets=[
        ("Evaluated particles (pT > 1 GeV); the central dip at |η| < 1 is in the paper's Figure 7b too", MUTED, False),
        ("The paper reproduction has the best perfect efficiency almost everywhere", BASE, False),
    ])

    slide_figure(prs, "Operating points: the IoU cut is the lever", "12_cutscan", bullets=[
        ("Track-validity cut barely matters below 0.7; the predicted-IoU cut trades efficiency for fakes", MUTED, False),
        ("At the paper's fake rate: 97.7 / 94.7; loosest cuts: 98.3 / 95.0 at 3× the fakes", ACCENT, True),
    ])

    slide_section(prs, "Part 3", "OR amplification")

    slide_figure(prs, "OR vs its fair baseline at window 512", "13_or_minus_baseline", bullets=[
        ("Fair pair: D (OR) vs A′ (flex baseline), same window, same seed; only OR differs", MUTED, False),
        ("DM equal (−0.07 ± 0.07); perfect +0.44 ± 0.11 for OR", ACCENT, True),
        ("At 512 the plain window already sees enough; OR should matter at smaller windows", INK, False),
    ])

    slide_figure(prs, "OR's one weakness: the φ seam", "14_seam", bullets=[
        ("OR's hash orderings treat φ as linear, so every table has a seam at ±π (baselines wrap)", MUTED, False),
        ("−1.5 points within 0.05 rad of the seam, but only ≈ 0.02 points overall", ACCENT, True),
        ("Possible fix: rotate φ randomly per hash table so the OR merge repairs the seam", INK, False),
    ])

    slide_figure(prs, "Window 256: the baseline holds up, OR doesn't help (yet)", "16_sweep_results", bullets=LIVE_BULLETS)

    slide_figure(prs, "Compared at equal training after the plateau", "15_sweep_live", bullets=LIVE_CURVE_BULLETS)

    slide_bullets(prs, "Status", STATUS_LINES)

    slide_bullets(prs, "Questions for UCL", [
        ("1.  Table 4, Pix0.6: is the perfect efficiency (93.2%, 91.5%) a pT ≥ 0.6 value?", 18, ACCENT, True),
        ("     Figure 7 integrated over pT > 1 gives 95.3% and 93.9%. Could we get your __test.h5?", 16, MUTED, False),
        ("2.  Which track-validity and IoU cuts are behind Table 4?", 18, INK, False),
        ("3.  Did you see the training plateau? Anything that avoids it (warm-up, seeding)?", 18, INK, False),
        ("4.  Two upstream issues we found:", 18, INK, False),
        ("     •  the loss masks padded queries with an unpermuted query_mask after matching", 16, MUTED, False),
        ("     •  attn_type: torch gives garbage (loss 6.8 with good weights); lap1015_early is broken", 16, MUTED, False),
        ("5.  Interested in the matcher toggles? Toggle 1 is a drop-in, identical-results speed-up", 18, INK, False),
    ])

    slide_bullets(prs, "Next", [
        ("Finish the window sweep and evaluate every run (seed 42 first)", 19, INK, False),
        ("Seed 43 and window 128: is the window-256 result a plateau effect or real?", 19, ACCENT, True),
        ("Find out why OR w256 diverged at epoch 29.6 (numerical? which event?)", 19, INK, False),
        ("DQ+LSCA reproduction (target: 97.6 / 93.9 / 0.4)", 19, INK, False),
        ("Periodic-φ OR to remove the seam", 19, INK, False),
        ("Overlap the matcher with the forward pass for the next speed-up", 19, INK, False),
    ])

    save_deck(prs, OUT)


if __name__ == "__main__":
    build()
