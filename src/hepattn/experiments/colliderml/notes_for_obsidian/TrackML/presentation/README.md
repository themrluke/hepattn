# TrackML-baseline talk

Slides for the group meeting after OR amplification: about 6 minutes on fixing the ColliderML baseline, choosing and preparing the TrackML baseline to reproduce, and the hit-filter failure and corrupt-data finding.

## Output

- `TrackML_baseline.pptx`: **the deck to present.** 12 slides, hand-edited by Luke on top of the generated version, with the corrected failure diagnosis folded in (2026-10-01)
- `TrackML_baseline_generated.pptx`: what `build_deck.py` produces. Never written over the hand-edited deck
- `backup/`: Luke's 12:41 versions of the deck and script, before the fold-in
- `SCRIPT.md`: speaker script, one section per slide, plus anticipated questions

## Regenerating

Everything is generated from data, so it can be refreshed without hand-editing PowerPoint. From this directory:

```bash
# 1. snapshot the numbers (needs the PROJECT python: pandas, pyarrow, comet_ml)
../../../../../../.pixi/envs/default/bin/python extract_data.py          # everything
../../../../../../.pixi/envs/default/bin/python extract_data.py comet    # just refresh the Comet curves

# 2. figures and deck (presentation venv: python-pptx, matplotlib)
.venv/bin/python figs_data.py       # 01 ColliderML, 03 Table 3, 05 corrupt charge, 05b freeze
.venv/bin/python figs_plan.py       # 04 targets, 06 code audit, 07 four arms
.venv/bin/python figs_training.py   # 08 NaN failure, 09 retrain
.venv/bin/python build_deck.py      # assembles TrackML_baseline_generated.pptx (never the hand-edited deck)
```

### Before the meeting

`TrackML_baseline.pptx` is now hand-edited, so it is **not** rebuilt by `build_deck.py`. To refresh a figure in it, regenerate the PNG (`extract_data.py comet`, then the figure script) and replace the picture in PowerPoint, or ask Claude to swap the image in place with python-pptx, which keeps the formatting.

## Layout of the code

| file | what it holds |
|---|---|
| `theme.py` | palette, fonts, figure sizing (same palette as the OR deck) |
| `extract_data.py` | pulls every number into `data/` from logs, the EOS dataset and Comet |
| `figs_data.py` | figures from the datasets and the momentum test |
| `figs_plan.py` | paper targets, four-arm design, code-audit diagram |
| `figs_training.py` | Comet training curves |
| `build_deck.py` | slide content declared as data in `build()`; helpers copied from the OR deck |
| `data/` | the snapshot the figures read; `stepcheck.json` holds the fp16/bf16 momentum test |

## Colour convention

- **blue**: the reference result (the paper, the baseline)
- **amber**: our work, OR amplification (also the deck's accent)
- **red**: failures, corrupt data, things that broke
- **green**: confirmed, matches the paper, passing

`.venv/` holds `python-pptx` and is not committed; recreate with `python -m venv .venv && .venv/bin/pip install python-pptx matplotlib pillow`.
