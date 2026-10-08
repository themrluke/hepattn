# TrackML reproduction talk (tracking meeting, 8 Oct 2026)

About 15 minutes, 25 slides: switching to UCL's data and filter, the Hungarian-matcher speed-up, reproducing the
paper's Pix0.6 DQ+MA (and the Table 4 threshold inconsistency), the first OR-vs-baseline results, and the window-256 sweep results.

## Output

- `TrackML_reproduction.pptx`: **the deck as presented** on 8 Oct, 22 slides, hand-edited by Luke on top of the generated version
  (copied from the vault, `_attachments/08.10.2026 Powerpoint.pptx`)
- `SCRIPT.md`: the speaker script as presented (copied from the vault, `UCL/Meetings/Thurs Tracking Meetings/08.10.2026 Script.md`)
- `TrackML_reproduction_generated.pptx`: the last version built by `build_deck.py` (25 slides). `build_deck.py` writes here,
  so rebuilding never overwrites the presented deck

## Regenerating

From this directory:

```bash
# 1. snapshot the numbers (PROJECT python: pandas, pyarrow, comet_ml)
/shared/projects/hepattn/.pixi/envs/default/bin/python extract_data.py          # everything
/shared/projects/hepattn/.pixi/envs/default/bin/python extract_data.py comet    # just the live training curves

# 2. figures and deck (presentation venv: python-pptx, matplotlib)
.venv/bin/python figs_speed.py    # 01-04 matcher speed-up
.venv/bin/python figs_repro.py    # 05-12 reproduction, Table 4, efficiencies, cut scan
.venv/bin/python figs_or.py       # 13-16 OR vs baseline, phi seam, aligned sweep curves, sweep results table
.venv/bin/python build_deck.py
```

### Before the meeting: the live slides

Slides 21 (sweep test-set results, figure 16) and 22 (validation loss aligned at the plateau drop, figure 15) change as sweep runs finish.
Finished runs' results are read from `trackml/eval_results/*-s42_iou0.5.txt` (written by `eval_results/auto_eval_sweep.sh`). To refresh:

```bash
/shared/projects/hepattn/.pixi/envs/default/bin/python extract_data.py constants eval comet
.venv/bin/python figs_or.py && .venv/bin/python build_deck.py
```

then update `LIVE_BULLETS` (slide 21), `LIVE_CURVE_BULLETS` (slide 22) and `STATUS_LINES` (slide 23) in `build_deck.py`,
and the matching script sections (21–23) together. New runs need their plateau-drop epoch added to
`plateau_drop_epoch` in `extract_data.py` constants. Progress: `eval_results/auto_eval.log` in the trackml experiment.

## Where the numbers come from

Every figure reads only `data/`; each CSV has a `.source` file next to it.

| file | source |
|---|---|
| `extract_data.py` | everything below, written to `data/` |
| `val_curves.csv` | Comet, project `trackml-tracking` (keys in `extract_data.RUNS`) |
| `test_results.csv`, `cutscan_pt1.csv` | UCL's evaluator at pT > 1 GeV, `trackml/eval_results/*.txt` on `trackml-pix1p0` |
| `eff_*_bins.csv`, `seam.csv`, `or_minus_baseline.csv`, `pt_counts.csv` | per-particle cache `trackml/eval_results/cache/` (not in git; rebuilt by `eval_results/plot_eff.py` / `eff_vs_phi.py`) |
| `fig7_*.csv` | the paper's Figure 7(a), extracted exactly from the PDF (`eval_results/paper_fig7a_extracted.json`) |
| `timing_runs.csv` | `../../TrackML/analysis/matcher_timing_2026-10-05/*.json` |
| `constants.json` | numbers typed in from the paper and the notes, each with its source |

## Layout of the code

| file | what it holds |
|---|---|
| `theme.py`, `deckkit.py` | palette and slide helpers, shared with the earlier decks |
| `figs_speed.py`, `figs_repro.py`, `figs_or.py` | one function per figure, saved as `figures/NN_name.png` |
| `build_deck.py` | the slides as data; `LIVE_BULLETS` / `LIVE_CURVE_BULLETS` / `STATUS_LINES` hold the in-progress text |
