Six events in the published TrackML training data each contain one pixel with an absurd charge. hepattn does nothing to clean it, and it breaks training in both precisions: in fp16 it drives the gradient scale to zero (this is what made the first [[Hit filter training|hit-filter run]] go NaN), and in bf16 it silently freezes the model.

# What is wrong

Each hit carries cluster features computed from its cells. One of them is `charge_frac`, the total charge in the cluster divided by the number of cells:

```python
# cluster_features.py:248
cells_agg["charge_frac"] = cells_agg.charge_sum / cells_agg.channel_counts
```

Across the whole dataset that value is ≤ 1 (median ≈ 0.065). In six training events one hit has a value of **hundreds of thousands**:

| event | hit_id | volume | layer | module | ch0 | ch1 | raw `value` |
|---|---|---|---|---|---|---|---|
| 1266 | 26335 | 8 | 4 | 55 | 0 | 960 | 556,485 |
| 1532 | 25379 | 8 | 4 | 63 | 335 | 960 | 716,689 |
| 5375 | 22246 | 8 | 4 | 153 | 0 | 960 | 236,495 |
| 5611 | 33083 | 8 | 4 | 432 | 0 | 320 | 636,264 |
| 6032 | 24745 | 8 | 4 | 155 | 335 | 960 | 121,193 |
| 7545 | 39208 | 8 | 8 | 1066 | 0 | 320 | 301,119 |

- Every one is a **single-cell cluster**, so `charge_frac` *is* the raw cell value
- All in the **pixel barrel** (volume 8), five of six in layer 4
- All on **sensor or chip edges**: `ch0` = 0 or 335 (first/last row), `ch1` = 320 or 960 (chip boundaries). Looks like a simulation fault on edge pixels
- Val and test contain none

# Is it really in the data, not something we did?

> [!success] Yes. It is in Kaggle's original files
> 1. **Our processing can't create it:** for a single cell, `prep.py` divides the raw value by 1
> 2. **The download is intact:** re-downloaded the three zips holding these events; sizes match Kaggle's listing byte for byte and the zips' CRC checksums pass
> 3. **The raw `cells.csv` shows the same values**, before any of our code runs, while every other cell in those events is ≤ 1.00

- **Not in Kaggle's `blacklist_training`** (≈ 1.3M hits listed, none of these six or their particles)
- **Not mentioned** in either UCL paper or anywhere I could find
- Scale: **6 pixels out of ≈ 490 million pixel hits**, which is why nobody would notice unless inputs are stored in a limited-range format

# Does hepattn handle it?

> [!failure] No, nowhere
> Checked three ways:
> 1. **Code:** `data.py` builds every input with `torch.from_numpy(...).half()` and applies no clipping, normalisation or finiteness check (the only transform is `x, y, z × 0.01`). `InputNet` concatenates the raw fields straight into its first `Dense` layer. **Every upstream branch**, including `main` and both paper branches, does exactly the same.
> 2. **The dataset's output:** for event 1532 the returned `hit_charge_frac` tensor is fp16 and contains `inf` (fp16 tops out at 65,504).
> 3. **A real training step:** on event 1532 the loss is NaN and **every one of the 4,384,385 gradient entries is NaN** (a normal event: none).

# What it does to training

Tested with real Lightning training steps from the last healthy checkpoint, on the sequence normal → normal → **corrupt** → normal → normal → normal:

| Precision | Corrupt step | Lion momentum | Afterwards |
|---|---|---|---|
| **fp16** (`16-mixed`, the filter) | **Skipped** by the gradient scaler, but the scale **halves** (64 → 32) | Stays finite | Trains normally *for now* (see below) |
| **bf16** (`bf16-mixed`, every tracking config) | Applied, with zero effect | **100% NaN, permanently** | **Silently frozen**: every later weight change is exactly 0 |

**bf16:** Lion updates each weight by `sign(momentum)`, and `torch.sign(NaN)` is **0**. So the weights never become NaN, the loss on normal events still looks plausible, and the model simply never changes again. You would only see it as a flat loss curve.

**fp16:** one skipped step is harmless, but the skips add up. Each one halves the scale, and the scale only regrows after **2,000 consecutive clean steps** (each skip resets the count). With six corrupt events per epoch, one every ≈ 1,400 steps, the count almost never reaches 2,000, so **the scale ratchets down to zero**. That is exactly what killed the first filter run: scale 4,096 → 256 → 16 → 0 over epochs 0–3, read from its checkpoints, against a steady 32,768 once the events were removed. Full chain in [[Hit filter training]].

> [!important] Consequences
> - These events **caused** the first filter run to diverge, through the fp16 gradient-scale ratchet
> - They **would** silently freeze any bf16 tracker run: all three Pix1.0 tracking configs use `bf16-mixed`
> - So they break hepattn training in **both** precisions, in two different silent ways. Worth reporting to UCL, and asking whether any of their runs met them
>
> (An earlier version of this note said they did *not* cause the filter failure, based on a 6-step test that was too short to show the ratchet.)

# UCL's copy doesn't have them

UCL trained on the **CodaLab** release of TrackML, not Kaggle's: different events (IDs 21000–29999). Scanning all of it: no value above 65,504, only 6 milder outliers (largest 188.7) that fit in fp16. So **this only affects people using the Kaggle release**, which is why UCL never hit it. See [[TrackML UCL data and cuts]].

# What I did

Moved the six events (`-hits` and `-parts`) to `/eos/user/l/ljohnson/datasets/TrackML/prepped/excluded_train/`, with a `README.txt` recording the evidence. Nothing deleted; training is now 8,644 events.

A more general fix would be to clip or `log`-transform `charge_frac` in `data.py`, or to check inputs for non-finite values. Not done, to keep the input pipeline identical to UCL's.
