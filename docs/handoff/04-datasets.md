# 04 — Downloading the datasets on the new machine

The data does not travel with the repo. Target locations on the new host:

| What | Where |
|---|---|
| TrackML | `/eos/user/l/ljohnson/datasets/TrackML/` |
| ColliderML | `/eos/user/l/ljohnson/datasets/ColliderML/` |
| Repo / working area | `/shared/projects/hepattn/` |

**A word on EOS before you start.** EOS is fine for archival storage but is a networked
filesystem, and training does many small random reads. If epochs turn out to be I/O-bound, keep
the master copy on EOS and stage the shards actually being trained on to local or project
storage under `/shared/projects/hepattn/data/`. Check the EOS quota before downloading
(`eos quota /eos/user/l/ljohnson/` or `df -h`): the ColliderML pu0 set below is about 34 GB and
the pu200 set already held on the old machine was about 141 GB.

---

## ColliderML

Hosted on Hugging Face as `CERN/ColliderML-Release-1` (dataset repo). Layout:

```
data/<process>_pu<pileup>_<collection>/train-XXXXX-of-01000.parquet
```

- `<process>`: `ttbar`, `dihiggs`, `ggf`, `zprime`, `hidden_valley`, `higgs_portal`,
  `pileup_only` (pu0 for all of those; `ttbar` also exists at pu200).
- `<collection>`: `particles`, `tracker_hits`, `tracks`, `calo_hits`.
- **Events per shard differs by pile-up**: pu0 shards hold **1000 events**, pu200 shards hold
  **100 events**. There are 1000 shards of each.

Approximate size per shard:

| Collection | pu0 | pu200 |
|---|---|---|
| `particles` | 227 MB | 747 MB |
| `tracker_hits` | 129 MB | 676 MB |
| `tracks` | 3.3 MB | 14 MB |

So 100 pu0 shards (100k events) is about 34 GB; 100 pu200 shards (10k events) is about 141 GB.

### Use the Python API, not the `hf` CLI

On the old cluster the `hf` launcher had a broken shebang (`/usr/bin/python`, which did not
exist) and failed with `bad interpreter`. An earlier `hf download` run also silently fetched only
4 of 6 requested files. `huggingface_hub.snapshot_download` was reliable and fast (about
150 MB/s). A ready-made script is committed next to this document:

```bash
python docs/handoff/download_colliderml.py \
    --pileup 0 --shards 102 \
    --dest /eos/user/l/ljohnson/datasets/ColliderML/v1
```

That downloads shards 00000–00101 of `particles`, `tracker_hits` and `tracks`, then splits them
into the layout the dataset class expects:

```
/eos/user/l/ljohnson/datasets/ColliderML/v1/pu0/
    train/ttbar_pu0_particles/      shards 00002-00101   (100k events)
    train/ttbar_pu0_tracker_hits/
    train/ttbar_pu0_tracks/
    val/ttbar_pu0_particles/        shards 00000-00001   (2k events)
    val/ttbar_pu0_tracker_hits/
    val/ttbar_pu0_tracks/
```

Pass `--collections particles tracker_hits tracks calo_hits` if calorimeter hits are needed;
the tracking configs set `return_calohits: false`, so they are not needed for this work.

Equivalent by hand, if the script is unavailable:

```python
from huggingface_hub import snapshot_download
pats = [f"data/ttbar_pu0_{c}/train-{i:05d}-of-01000.parquet"
        for c in ("particles", "tracker_hits", "tracks") for i in range(102)]
snapshot_download("CERN/ColliderML-Release-1", repo_type="dataset",
                  local_dir="/eos/user/l/ljohnson/datasets/ColliderML/v1/pu0/_download",
                  allow_patterns=pats, max_workers=8)
```

Files land under `<local_dir>/data/ttbar_pu0_<collection>/`; move them into `train/` and `val/`
as above. The dataset class finds shards by globbing `*.parquet` in each collection directory and
keeps only shard names present in **every** required collection, so a partial download of one
collection silently shrinks the dataset — verify the counts match.

### Then update the configs

`src/hepattn/experiments/colliderml/configs/*.yaml` still point at the old cluster:

```yaml
train_dir: /share/rcif2/ljohnson/data/colliderml/v1/pu0/train     # -> /eos/user/l/ljohnson/datasets/ColliderML/v1/pu0/train
val_dir:   /share/rcif2/ljohnson/data/colliderml/v1/pu0/val
test_dir:  /share/rcif2/ljohnson/data/colliderml/v1/pu0/val
```

Affected files: `base.yaml`, `pu0.yaml` (on `colliderml-baseline`), `or-amp.yaml`,
`filtering-lite.yaml`, `sorter.yaml`, `sorter_nlsh.yaml`, and the paths hard-coded in
`scripts/repack_parquet.py`.

Note `pu0.yaml` also sets `pileup: 0`; that argument was added to `ColliderMLDataset` on the
`colliderml-baseline` branch and does not exist on `main` or `OR-amplification`.

---

## TrackML

TrackML was **not present on the old machine** — every path in the TrackML configs
(`/share/rcifdata/maxhart/data/trackml/prepped/`, `data/trackml/prepped/`, and the `HC-v3`
filter-evaluation h5 files) was missing. So this is a fresh download, and the instructions below
are from the experiment's README and `prep.py` rather than from a run we did.

### 1. Get the raw data

- **Training data**: the CodaLab throughput phase,
  <https://competitions.codalab.org/competitions/20112#participate-get_data>. The Kaggle
  competition page, <https://www.kaggle.com/competitions/trackml-particle-identification/data>,
  hosts the same events (`train_1.zip` … `train_5.zip`, plus `train_sample.zip` for a quick
  start) and needs a Kaggle account and the `kaggle` CLI.
- **`detectors.csv`**: already committed in this repo at `data/trackml/detectors.csv`, so it does
  not need downloading. `prep.py` looks for it at `Path(in_dir).parent / "detectors.csv"`.
- A four-event sample already sits in `data/trackml/raw/` for testing the pipeline.

Suggested layout:

```
/eos/user/l/ljohnson/datasets/TrackML/
    detectors.csv          # copy of the repo's copy
    raw/                   # event000001000-{hits,cells,particles,truth}.csv.gz
    prepped/train/  val/  test/
```

### 2. Preprocess to parquet

```bash
python src/hepattn/experiments/trackml/prep.py \
    -i /eos/user/l/ljohnson/datasets/TrackML/raw \
    -o /eos/user/l/ljohnson/datasets/TrackML/prepped/train
```

`prep.py` globs `event*-truth.csv.gz` and needs the matching `-particles`, `-hits` and `-cells`
files per event, all **gzipped**. The Kaggle/CodaLab zips contain plain `.csv`, so gzip them
first (`gzip raw/*.csv`) — the sample files in the repo are `.csv.gz`, which is where the
expected naming comes from. Split the prepped events across `train/`, `val/` and `test/`
directories yourself; the configs take three directories.

### 3. Point the configs at it

In `src/hepattn/experiments/trackml/configs/*.yaml` set `train_dir`, `val_dir` and `test_dir`
to the prepped directories.

`tracking.yaml` additionally needs `hit_eval_train` / `hit_eval_val` / `hit_eval_test` — h5 files
produced by evaluating a **trained hit filter** over each split. Those do not exist yet either.
Either train the filter first:

```bash
python run_filtering.py fit  --config configs/filtering.yaml
python run_filtering.py test --config <path to the run's config> --data.test_dir <split dir>
```

or start from `tracking-lite.yaml`, which is single-stage (pixel barrel, pT > 1 GeV) and needs no
filter output. For a first OR comparison, `tracking-lite.yaml` is the cheaper starting point —
see [`03-colliderml-baseline.md`](03-colliderml-baseline.md#where-to-run-the-or-comparison).

Useful selections, from the experiment README:

```yaml
hit_volume_ids: [7, 8, 9]   # pixel barrel and endcaps
particle_min_pt: 0.6
particle_max_abs_eta: 2.5
```
