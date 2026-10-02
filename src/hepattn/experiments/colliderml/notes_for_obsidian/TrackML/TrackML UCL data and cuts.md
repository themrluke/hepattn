The TrackML data, hit-filter outputs and selection cuts used for the four-arm comparison (see [[Plan]]). **Every arm must use exactly these values** so that the only difference between arms is the encoder (and the filter, for B vs A). Replaces our own Kaggle copy, which was deleted on 2026-10-02 (see [[Getting Started]]).

# The data

| | |
|---|---|
| **Location** | `/eos/user/l/ljohnson/datasets/TrackML_UCL/` (copied 2026-10-02 from Mungo's `/eos/user/t/tmangat/trackml-data/`) |
| **Release** | TrackML **CodaLab throughput phase** (particle files carry `particle_type`), *not* the Kaggle release we downloaded |
| **Format** | Same `prep.py` output as ours: `event*-hits.parquet` (27 columns incl. cluster features), `-parts.parquet`, plus `-truth.parquet` |
| **train** | 8,543 events, IDs 21000–29799 (3 dropped vs his raw `train_csv/`: 22500, 23161, 23450; reason unknown) |
| **test** | 100 events, IDs 29800–29899 |
| **val** | 100 events, IDs 29900–29999 (note: val is *above* test, the reverse of our old split) |

> [!success] This is the paper's data
> Pixel hits per event **56.7k** and reconstructable particles **2607 ± 329** (test) match Table 3 of arXiv 2606.17631 exactly (56.7k, 2600 ± 340). Our Kaggle copy gave 56.5k.

> [!success] Clean for training
> No `charge_frac` above fp16's 65,504 anywhere (≈ 496M pixel hits scanned). The same edge-pixel fault exists but mildly: 6 single-pixel hits above 1, the largest 188.7, all of which fit in fp16. None of the [[Corrupt TrackML events]] are in this release.

# The hit filter

| | |
|---|---|
| **Files** | `TrackML_UCL/filter_evals/eta4pt600/epoch=070-val_loss=0.20153_{train,val,test}_eval.h5` |
| **What it is** | Pippa's `HF-pix-600MeV-eta4` filter, epoch 70: the exact filter the upstream Pix0.6 configs reference |
| **Its training cuts** (read from its saved targets) | pT > 0.6 GeV, \|η\| < 4, ≥ 3 pixel hits, at most 3,000 particles per event |
| **Coverage** | One entry per event, matching the three splits exactly. Only works with *this* data |
| **Threshold** | **0.1** |

Measured with `logs/check_filter_600.py` at threshold 0.1:

| | Paper (Pix0.6) | Test | Val |
|---|---|---|---|
| Particle retention (pT > 0.6, \|η\| < 4, ≥ 3 hits; kept if ≥ 3 hits survive) | 99.1% | **99.13%** | 98.33% |
| Hits per event after filter | 20.3k | **20.3k** | 21.0k |
| Hit efficiency / purity | not stated | 99.4% / 87.3% | 98.2% / 83.5% |

There are **no 900 MeV filter outputs**, so the comparison is done at **600 MeV (Pix0.6)**.

# Cuts and settings for every tracker arm

Source: `tracking-eta4-pt600-epochs.yaml` (DQ+MA) and `tracking-lca-eta4-600.yaml` (DQ+LSCA) on branch `trackml-pix1p0`, against Table 2 and Section 5.1 of the paper.

| Setting | Paper, Pix0.6 | DQ+MA config | DQ+LSCA config | Match? |
|---|---|---|---|---|
| Hit volumes | pixel only | `[7, 8, 9]` | `[7, 8, 9]` | ✅ |
| Particle pT | ≥ 0.6 GeV | `particle_min_pt: 0.6` | same | ✅ |
| Particle \|η\| | ≤ 4 | `particle_max_abs_eta: 4` | same | ✅ |
| Min. hits per particle | 3 pixel | `particle_min_num_hits: 3` | same | ✅ |
| Filter threshold | (99.1% retention) | `0.1` | `0.1` | ✅ (measured) |
| Encoder | 8 layers, window 512, φ-sorted, wrapped | 8, 512, `phi`, flash + wrap | same | ✅ |
| Decoder layers | 3 | 3 | 3 | ✅ |
| First-hit (query-init) threshold | 0.3 | 0.3 | 0.3 | ✅ |
| **Max. queries / particles** | **3,900** | **4,000** | **3,700** | ❌ |
| Loss weights | 0.1 valid + 2 dice + **25 focal** + quality + DQ | 0.1 / 2 / **100** | 0.1 / 2 / **100** | ❌ focal |
| **LSCA window** | **64** | — | **128** | ❌ |
| Optimiser | Lion, 1e-5 → 5e-5 → 1e-5, wd 1e-5 | same | same | ✅ |
| Warm-up fraction | not stated | `pct_start: 0.02` | `0.02` | ? |
| Precision | bf16 | `bf16-mixed` | `bf16-mixed` | ✅ |
| Epochs × events | 30 × 8,500 | 30 × all (8,543) | same | ✅ (≈) |
| Gradient clip | not stated | 0.1 | 0.1 | ? |
| **Targets** (DQ+MA / DQ+LSCA) | eff 98.0% / 97.6%, perfect 93.2% / 91.5%, fakes 0.3% / 0.4% | | | |

Both config names hint the mismatches are deliberate UCL choices, not typos: `TRK-v8-eta4-lca-3700-0p3-lw100-epochs30` = 3,700 queries, threshold 0.3, loss weight 100.

> [!question] Ask UCL before the long runs
> 1. Focal weight 25 (paper) or 100 (both configs)?
> 2. Max queries 3,900 (paper) or 4,000 / 3,700? With 3,700, the busiest events (up to ~3,600–3,900 particles) can be truncated
> 3. LSCA window 64 (paper) or 128 (config)?
> 4. Why were events 22500, 23161 and 23450 dropped from `train/`?

> [!note] Harmless quirk
> Both configs list `s` twice in `data.inputs.hit`. That only loads the same field twice; the model's own field list has 17 distinct entries matching `input_size: 17`. A 5-step test of the DQ+MA config on this data ran cleanly.

# Rules for the four arms

Keep identical across A, B, C, D (and across seeds):
- `train_dir` / `val_dir` / `test_dir` = `TrackML_UCL/{train,val,test}/`
- `hit_volume_ids: [7, 8, 9]`, `particle_min_pt: 0.6`, `particle_max_abs_eta: 4`, `particle_min_num_hits: 3`
- the same `event_max_num_particles` / `num_queries` (whatever UCL confirms)
- the same loss weights, optimiser, schedule, precision, epochs
- **Filter arms (A, D):** `hit_eval_*` = the three `eta4pt600` files, `hit_filter_threshold: 0.1`
- **No-filter arms (B, C):** remove the `hit_eval_*` lines (≈ 56.7k hits per event; check memory first)
- **Encoder:** `attn_type: flex`, `window_wrap: false` in *every* arm, with OR settings only in B and D. Optionally one extra run of A as configured (flash + wrap) to confirm the switch changes nothing
- Quote results the paper's way: double-majority efficiency for pT > 1 GeV? (Table 4 caption says "efficiency range derived from truth particle pT and η"; check before comparing)
