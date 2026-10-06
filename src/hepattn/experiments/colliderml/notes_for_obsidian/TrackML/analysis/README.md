# TrackML analysis snapshots

Raw data and scripts behind results in the Obsidian notes, kept so the numbers can be rebuilt for slides.
`<scratch>` in the scripts stands for the temporary working directory they originally ran in.

## `matcher_timing_2026-10-05/`

Idle-GPU timing of the Hungarian-matcher toggles (note: *Hungarian matching overhead*).

- `*_original_*.json`, `*_prep_*.json`, `*_trim_*.json`: per-step timestamps and per-call matcher times for
  the original matcher, toggle 1 (`prepare_on_device`) and toggles 1+2 (`trim_padded_queries`), from the
  paper-reproduction weights at epoch 0 (`early_*`) and epoch 29 (`late_*`); 600 steps per run
- `summarise.py` makes the steps/s tables (steps 150–600); `deep.py` checks stability, outliers and paired
  step-by-step savings; `breakdown.py` splits one matcher call into GPU prep, copy, solves and pool overhead
- `syslog.txt`, `runs.log`: host load and GPU memory every 5 s during the second batch of runs, with run
  start/end times
- `original.yaml`, `prep.yaml`, `trim.yaml`, `run.py`, `sequence*.sh`: configs and run scripts;
  `capture.py`, `check_real.py`, `make_cfg.py`: cost capture and the real-event identity check;
  `wt_boot/sitecustomize.py`: made the runs import the worktree's code

Result: toggle 1 is 1.9x (early) to 2.6x (late) faster with identical matching; toggles 1+2 2.2x to 3.1x.

## `flex_vs_flash_2026-10-05/`

Same weights (paper reproduction, epoch 29), same events, run through flash, flex, fp32 flex and the torch
backend (note: *Flex vs flash and the training plateau*).

- `compare.py`: encoder-output and loss comparison between attention backends; `compare_*.log` its outputs
- `flash.yaml`, `flex.yaml`, `torch.yaml`: the configs compared

Result: flash and flex differ by 6% in encoder output, less than bf16 rounding alone (11%), with equal loss;
the plain torch backend is broken (loss 6.8).
