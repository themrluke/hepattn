Training the Pix1.0 hit filter (`HF-900MeV-eta4`) that arms A and D of the [[Plan]] depend on. Target, from [[Retraining TrackML model]]: keep **98.6%** of reconstructable particles, leaving ≈ **9.7k** hits per event.

# Setup

- Config `filtering.yaml` on branch `trackml-pix1p0`: 8 encoder layers, window 1024, pT > 0.9 GeV, |η| < 4, pixel volumes 7, 8, 9
- **Lion**, one-cycle schedule 1e-5 → **5e-5** → 1e-6, `pct_start: 0.05` (peak at step ≈ 34,600), **fp16** (`16-mixed`), **80 epochs**, gradient clip 0.1
- These come from UCL's config. **Neither paper documents how the filter was trained**: the new paper's Lion 1e-5 → 5e-5 → 1e-5 in bf16 is for the *tracking* model, and the older paper only says 30 epochs, batch size 1, ≈ 10 h on an A100

## Smoke test

300 steps, no logger: data loads under the merged `data.py`, ≈ 16 steps/s once compiled, 11 GB of GPU memory, 4.4M parameters. → ≈ 12 min per epoch, ≈ 16 h for 80 epochs.

# Run 1: diverged

Comet: `trackml-filtering/d6fe946db8274229a3c14362b6bcf00f`, folder `logs/FAILED-nan-HF-900MeV-eta4_20260930-T163553`

- Healthy for 3 epochs: validation loss 0.781 → 0.364 → 0.329
- Then the training loss **climbed for ≈ 0.8 epoch** (0.35 → 0.6 → 2.2 → 34), turned `inf` at **step 34,199**, and was NaN for the remaining 76 epochs (≈ 13 h wasted)
- Afterwards the model marks every hit as "keep" (recall 1.0, precision 0.16). Only the epoch-2 checkpoint is usable

## Why: the fp16 gradient scale ratcheted to zero

> [!success] Cause found: the [[Corrupt TrackML events]], through the fp16 gradient scaler
> The scaler's scale, read from each epoch's checkpoint:
>
> | End of epoch | Run 1 (corrupt events in) | Run 2 (corrupt events out) |
> |---|---|---|
> | 0 | 4,096 | 8,192 |
> | 1 | **256** | 32,768 |
> | 2 | **16** | 32,768 |
> | 3 | **0** → NaN | 32,768 |
> | 4–6 | 0 | 32,768 |

The chain:
1. A corrupt event makes every gradient NaN. The fp16 gradient scaler sees this and **skips the step**, which on its own is harmless (tested)
2. But every skip also **halves the scale**, and the scale only doubles back after **2,000 consecutive clean steps** (`growth_interval: 2000`); a skip resets that count
3. Six corrupt events per ≈ 8,650-step epoch is one every ≈ 1,400 steps, so the count almost never reaches 2,000. **The scale can only fall**: ≈ 16× per epoch
4. As the scale shrinks, more small gradients round to exactly zero in fp16. Lion keeps taking full-size steps (`sign` of the momentum) on an increasingly incomplete gradient, while the warm-up raises the learning rate. **That is the slow climb in the loss**
5. When the scale reaches 0, gradients are divided by zero: **`inf`, then NaN** at step 34,199

The learning-rate peak only coincided with the end of the collapse, which is why it looked guilty.

> [!warning] Two earlier diagnoses were wrong
> 1. "The corrupt events turn into `inf` and break the step." True of one step, but the scaler skips it; on its own that can't make a slow climb.
> 2. "The learning-rate peak, or an unexplained fp16 instability." Ruled out: run 2 used the identical schedule and passed the same peak cleanly.
>
> My step test (6 steps) was too short to see a ratchet that acts over thousands of steps. Reading the scaler state from the checkpoints settled it.

# Run 2: retrain (finished, no longer needed)

Comet: `trackml-filtering/c466da9add12497e92a01a2bb76153fd`, folder `logs/HF-900MeV-eta4_20261001-T100156`, started 2026-10-01 10:01. Same settings, minus the six corrupt events (8,644 events).

| Steps | Run 1 mean loss | Run 2 mean loss |
|---|---|---|
| 24,000–25,999 | 0.333 | 0.272 |
| 26,000–27,999 | 0.379 | 0.261 |
| 28,000–29,999 | 0.547 | 0.254 |
| 30,000–31,999 | 1.086 | 0.244 |
| 32,000–33,999 | **14.7** | 0.242 |
| 34,000–35,999 | NaN | **0.239** |

> [!success] Run 2 passed the learning-rate peak cleanly
> At step 35,900 (rate exactly 5e-5) the loss is 0.239 and still falling, with no NaN or `inf`.

Validation loss (run 1 for comparison: 0.781 → 0.364 → 0.329 → NaN):

| Epoch | 0 | 1 | 2 | 3 | 6 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| Val loss | 0.642 | 0.337 | 0.264 | 0.225 | 0.170 | 0.127 | 0.096 | 0.080 | 0.073 | **0.068** |

Gradient scale over all 31 epochs so far: between 8,192 and 65,536 (the normal up-and-down of the scaler), 32,768 at epoch 30. *As of 2026-10-01 15:30, epoch 31 of 80; ETA ≈ 03:00 on 2 October.*

## Outcome

- Finished all 80 epochs overnight. **Best validation loss 0.054 at epoch 54**; it then overfitted, rising to 0.083 by epoch 79. (I earlier said the minimum was around epoch 30 from looking at too few epochs; that was wrong.)
- Only the epoch-54 checkpoint is kept, plus the config and log. The failed run keeps epochs 0–3 (the gradient-scale evidence)
- An overnight evaluation found that **hepattn-clean's `PredictionWriter` cannot write filter outputs**; `logs/write_filter_eval.py` writes them in the layout `data.py` reads. Kept for if we ever need our own filter outputs

> [!success] Superseded by UCL's filter (2026-10-02)
> We now use UCL's 600 MeV filter outputs on UCL's data, which reproduce the paper's Pix0.6 numbers (99.13% retention, 20.3k hits/event). See [[TrackML UCL data and cuts]]. This 900 MeV filter was trained on the Kaggle data, which has since been deleted.

# After training (not needed now)

1. Run the best checkpoint over train, val and test → three `hit_eval` h5 files (each run writes `<ckpt>__test.h5`, so rename after each)
2. Check with `logs/check_filter.py`: retention at threshold 0.314 against **98.6%**, post-filter hits against **9.7k**
