Setting up the new machine: where everything lives, how the data was downloaded, and the checks that showed it was usable. For what we are doing with it, see [[Plan]].

# Storage layout

| What | Where | Notes |
|---|---|---|
| Datasets ([[ColliderML vs TrackML\|TrackML and ColliderML]]) | `/eos/user/l/ljohnson/datasets/{TrackML,ColliderML}` | EOS home. Same place as `/eos/home-i00/l/ljohnson` and `$HOME` |
| [hepattn](https://github.com/themrluke/hepattn) fork | `/shared/projects/hepattn/` | 500 GB NFS, fast for code and environments |
| Notes branch (this vault) | `/shared/projects/hepattn-notes/` | A `git worktree` of `notes-and-presentation`, so the notes never land on a working branch |
| Scratch | `/scratch/ljohnson/` | 3.5 TB local NVMe, probably wiped between sessions. Package caches and temporary unzips only |

> [!tip] Don't switch branches in `/shared/projects/hepattn` while a training run is going
> The run reads its code and configs from that checkout. Use a second worktree instead.

# Environment

- **pixi** installed at `~/.pixi/bin/pixi`. The project environment is `/shared/projects/hepattn/.pixi/envs/default` (torch 2.9.1+cu128, flash-attn 2.8.3), with its package cache on `/scratch`
- `pixi run` can hang re-checking the environment, so I call the environment's Python directly: `.pixi/envs/default/bin/python`
- **GPU:** one NVIDIA H100 NVL (94 GB). 46 cores, ~1.5 TB RAM, no Slurm
- **Comet:** `~/.comet.config` (workspace `themrluke`). Projects: `colliderml`, `trackml-filtering`, `trackml-tracking`
- **GitHub:** SSH key `~/.ssh/id_ed25519` added to my account; `origin` = my fork, `upstream` = `samvanstroud/hepattn`

> [!warning] Rotate the Comet API key and the Kaggle token
> Both were pasted into a chat session at some point.

# Downloading data

## ColliderML pu0

Future [[ColliderML vs TrackML#ColliderML|ColliderML]] studies start from 0 pile-up, so the model has an easier problem to learn before we make it harder.

- Downloaded 102 shards from Hugging Face (`CERN/ColliderML-Release-1`) to `/eos/user/l/ljohnson/datasets/ColliderML/v1/pu0/`
- **100 train shards (100k events), 2 val shards (2k events), 34 GB**
- All 306 files are readable, each holding 1,000 events in one row group
- **Read speed, `EOS` vs `/shared`:**
	- A full shard loaded in *0.06–0.18 s* on `EOS` against *0.05–0.11 s* on `/shared`
	- The `EOS` files **may have been cached**, since they had just been written. Either way the data stays on `EOS`: one shard serves minutes of GPU time

### Overfit check

`pu0.yaml` trained on 10 events for 3,000 steps, validating on the same 10 events.

![[Pasted image 20260928113546.png]]

| step | val loss | eff (≥50% hits) | eff (all hits) | purity |
|---|---|---|---|---|
| 300 | 11.6 | 0.00 | 0.00 | 0.00 |
| 1,500 | 0.23 | 0.93 | 0.90 | 0.96 |
| 3,000 | 0.047 | **0.982** | **0.975** | **0.998** |

- Targets, masks and particle–hit matching all work
- So the old poor results most likely came from the **narrow event selection** (pixel barrel, |η| < 1, ≈ 4.2 hits per particle), not from a bug
- About 1.2 of 73.5 particles per event were still missed. Not investigated

> [!important] Where the extra hits actually come from: the strips, not η
> `pu0.yaml` sets `sihit_volume_ids: null`, i.e. **every tracker volume: pixel, short strip and long strip**. Measured on 200 pu0 events (pT > 0.9 GeV, ≥ 3 hits):
>
> | Selection | Hits / particle | Particles / event | Hits / event |
> |---|---|---|---|
> | Old: pixel barrel only, \|η\| < 1 | 4.3 | 20 | 258 |
> | Pixel barrel only, \|η\| < 2.5 | 4.7 | 50 | 695 |
> | All pixel (barrel + endcaps), \|η\| < 2.5 | 4.8 | 53 | 772 |
> | All tracker (pixel + strips), \|η\| < 1 | 11.4 | 30 | 1,271 |
> | **New `pu0.yaml`: all tracker, \|η\| < 2.5** | **11.8** | **66** | **3,433** |
>
> In the new selection a particle has on average **3.9 pixel + 4.4 short-strip + 3.5 long-strip** hits. So:
> - **Pixel-only stays at ≈ 4–5 hits per particle whatever the η range**: a track crosses a fixed number of pixel layers
> - **Widening η adds particles** (20 → 53 per event), not hits per particle
> - **Adding the strips is what takes it to ≈ 12**
>
> ColliderML volumes: pixel 16 / **17 (barrel)** / 18 (r 32–174 mm); short strip 23 / 24 / 25 (r 240–702 mm); long strip 28 / 29 / 30 (r 811–1031 mm). The deck's "widened to the whole tracker out to |η| < 2.5" is literally true but credits the wrong change.

## TrackML

> [!warning] Superseded on 2026-10-02
> We now use UCL's copy of TrackML (the CodaLab release the paper used) in `/eos/user/l/ljohnson/datasets/TrackML_UCL/`. The Kaggle data below was **deleted**, apart from the six corrupt events kept as evidence in `TrackML/prepped/excluded_train/`. See [[TrackML UCL data and cuts]].

Downloaded all five `train_N.zip` files from Kaggle (≈ 80 GB, byte-exact), gzipped the CSVs, and ran `prep.py` as 36 parallel processes: 8,850 events in ≈ 15 min, 98 GB of parquet.

```
/eos/user/l/ljohnson/datasets/TrackML/
    detectors.csv
    prepped/train            8,644 events   (event 1000 – 9799)
    prepped/val                100 events   (event 9800 – 9899)
    prepped/test               100 events   (event 9900 – 9999)
    prepped/excluded_train       6 events   (corrupt, see README.txt)
```

- **Contents:**
	- Pixel *and* strip hits are kept; the studies select pixels only (volumes 7, 8, 9) in the config
	- Every particle is charged (checked)
	- Pixel hits are ≈ ½ of each event
- **Split:** 100 val and 100 test events, as in the UCL paper. The paper does not say *which* events, so I used the 200 highest event IDs
- **6 training events were moved out** because each contains one corrupt pixel. See [[Corrupt TrackML events]]
- The 155 GB of raw `.zip` and `.csv` files were deleted. Re-downloading takes ≈ 5 min if `prep.py` ever changes
