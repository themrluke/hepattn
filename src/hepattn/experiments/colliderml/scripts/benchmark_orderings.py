"""Benchmark hit-ordering techniques for ColliderML tracking.

For each ordering (random / phi / hilbert / lsh) we sort every event's silicon
hits, then for each reconstructable truth particle measure how tightly its hits
sit together in the sorted sequence:

    spread(particle) = max_position - min_position   (over that particle's hits)

A small spread means the particle's hits land close together, so a local /
sliding-window attention of a given size can "see" them all. We report, pooled
over all particles in all events:

    mean, p50, p90, p99, max   -- the index-spread distribution
    c@128, c@256, c@512, c@1024 -- containment: fraction of particles whose hits
                                   fit within an attention window of that size

Containment threshold per window W depends on --window-metric:
    span   (default): contained iff spread <= W          (index span within W)
    block           : contained iff spread <= W - 1      (fits in W consecutive slots)
    mutual          : contained iff spread <= W // 2     (matches this repo's
                      sliding_window_mask(W), half-width W//2 each side)

The sorted sequence is always treated as circular (a ring), matching the
deployed model's `window_wrap: true`: a particle whose hits straddle the
phi = +/-pi seam is joined back up rather than being split to opposite ends of
the sequence. There is no flag to turn this off.

Example:
    python scripts/benchmark_orderings.py --num-events 100
    python scripts/benchmark_orderings.py --split val --num-events 200 \
        --window-metric mutual --csv orderings.csv
"""

from __future__ import annotations

import os

# Keep native math/IO libraries single-threaded per process, so that N parallel
# worker processes don't each spawn a full thread pool and thrash the CPUs
# (N_workers x N_cores threads all contending). Must be set before numpy / torch
# / pyarrow are imported to take effect.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import argparse
import inspect
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hepattn.experiments.colliderml.data import ColliderMLDataset
from hepattn.experiments.colliderml.scripts.orderings import SORTERS

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is normally available

    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "base.yaml"

SPLIT_TO_KEYS = {
    "train": ("train_dir", "num_train"),
    "val": ("val_dir", "num_val"),
    "test": ("test_dir", "num_test"),
}


# Cut-disabling overrides for --no-cuts: None disables each hit/particle
# selection in the dataset (no pt/eta/volume cut, no truncation), and including
# both charges plus empty hit_cuts keeps every particle the raw files provide.
NO_CUTS_OVERRIDES = {
    "particle_min_pt": None,
    "particle_max_abs_eta": None,
    "particle_hit_cuts": {},
    "particle_include_charged": True,
    "particle_include_neutral": True,
    "sihit_volume_ids": None,
    "sihit_max_abs_eta": None,
    "max_num_particles": None,
}


def build_dataset(config_path: Path, split: str, num_events: int | None, no_cuts: bool = False) -> ColliderMLDataset:
    """Instantiate the real tracking dataset straight from a training config.

    With ``no_cuts=True`` every hit- and particle-level selection from the config
    is disabled so the orderings are measured on the full raw dataset.
    """
    with config_path.open() as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]

    dir_key, num_key = SPLIT_TO_KEYS[split]
    dirpath = data_cfg[dir_key]
    if num_events is None:
        num_events = data_cfg.get(num_key, -1)

    allowed = set(inspect.signature(ColliderMLDataset.__init__).parameters)
    kwargs = {k: v for k, v in data_cfg.items() if k in allowed}
    # Tracking-mode data (full particle set + CSR association) but skip the dense
    # (num_particles x num_sihits) mask we never read - essential under --no-cuts
    # where that matrix would otherwise blow up memory.
    kwargs.update(
        dirpath=dirpath,
        num_events=num_events,
        hit_filter=False,
        build_dense_masks=False,
        return_calohits=False,
        return_tracks=False,
    )
    if no_cuts:
        kwargs.update(NO_CUTS_OVERRIDES)
    return ColliderMLDataset(**kwargs)


def particle_spreads(order: np.ndarray, indptr: np.ndarray, indices: np.ndarray, min_hits: int, num_hits: int) -> np.ndarray:
    """Circular index spread of each particle's hits under a given ordering.

    ``rank[h]`` is the position of hit ``h`` in the sorted sequence. The sequence
    is treated as a ring (matching the model's ``window_wrap: true``), so a
    particle's spread is the shortest arc covering all of its hits: the ring
    length minus the largest gap between its consecutive hits, where the seam gap
    (across the phi = +/-pi wrap) is included as one of the candidate gaps.
    Fully vectorised over particles.
    """
    rank = np.empty(num_hits, dtype=np.int64)
    rank[order] = np.arange(num_hits, dtype=np.int64)

    counts = np.diff(indptr)
    keep = counts >= min_hits
    if not keep.any():
        return np.zeros(0, dtype=np.int64)

    num_particles = counts.shape[0]
    pos = rank[indices]
    pid = np.repeat(np.arange(num_particles), counts)

    # Sort positions within each particle segment (primary key pid, then pos).
    seg_order = np.argsort(pid.astype(np.int64) * (num_hits + 1) + pos, kind="stable")
    pid_s = pid[seg_order]
    pos_s = pos[seg_order]

    # Largest gap between a particle's consecutive hits (interior gaps only).
    diffs = pos_s[1:] - pos_s[:-1]
    same_seg = pid_s[1:] == pid_s[:-1]
    max_gap = np.zeros(num_particles, dtype=np.int64)
    np.maximum.at(max_gap, pid_s[1:][same_seg], diffs[same_seg])

    pmin = np.full(num_particles, num_hits, dtype=np.int64)
    pmax = np.full(num_particles, -1, dtype=np.int64)
    np.minimum.at(pmin, pid, pos)
    np.maximum.at(pmax, pid, pos)
    seam_gap = num_hits - (pmax - pmin)  # gap across the wrap seam

    largest_gap = np.maximum(max_gap, seam_gap)
    circular_spread = num_hits - largest_gap
    return circular_spread[keep]


def threshold(window: int, metric: str) -> int:
    if metric == "span":
        return window
    if metric == "block":
        return window - 1
    if metric == "mutual":
        return window // 2
    raise ValueError(f"Unknown window-metric: {metric}")


def summarise(spreads: np.ndarray, windows: list[int], metric: str) -> dict[str, float]:
    row = {
        "mean": float(np.mean(spreads)),
        "p50": float(np.percentile(spreads, 50)),
        "p90": float(np.percentile(spreads, 90)),
        "p99": float(np.percentile(spreads, 99)),
        "max": float(np.max(spreads)),
    }
    for w in windows:
        row[f"c@{w}"] = float(np.mean(spreads <= threshold(w, metric)))
    return row


def available_cpus() -> int:
    """CPUs actually available to this process (respects cgroup/SLURM affinity)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on all platforms
        return os.cpu_count() or 1


# Per-process state. Built once per worker (or once in the main process for the
# serial path) rather than shipped across the pool, since the dataset is heavy.
_WORKER: dict = {}


def _init_worker(config_path: Path, split: str, num_events: int | None, sorter_names: list[str], min_hits: int, seed: int, no_cuts: bool) -> None:
    # Belt-and-braces: also cap the runtime-configurable thread pools per process.
    import torch

    torch.set_num_threads(1)
    try:
        import pyarrow as pa

        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
    except Exception:  # pyarrow thread API is best-effort
        pass

    _WORKER.update(
        dataset=build_dataset(config_path, split, num_events, no_cuts=no_cuts),
        sorters=sorter_names,
        min_hits=min_hits,
        seed=seed,
    )


def event_spreads(idx: int) -> dict[str, np.ndarray]:
    """Per-particle spread arrays for one event, keyed by sorter.

    Reads per-process state set by ``_init_worker``. The RNG is seeded per event
    as (seed, sample_id) so the random/lsh orderings are reproducible and
    independent of worker count or event ordering.
    """
    dataset = _WORKER["dataset"]
    sample_id = dataset.sample_ids[idx]
    inputs, targets = dataset.load_event(sample_id)
    eta = inputs["sihit_eta"].numpy()
    phi = inputs["sihit_phi"].numpy()
    num_hits = eta.shape[0]
    indptr = targets["particle_sihit_indptr"].numpy()
    indices = targets["particle_sihit_indices"].numpy()

    empty = np.zeros(0, dtype=np.int64)
    if num_hits == 0 or indices.size == 0:
        return {name: empty for name in _WORKER["sorters"]}

    rng = np.random.default_rng([_WORKER["seed"], int(sample_id)])
    out = {}
    for name in _WORKER["sorters"]:
        order = SORTERS[name](eta, phi, rng)
        out[name] = particle_spreads(order, indptr, indices, _WORKER["min_hits"], num_hits)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Training config to pull data settings from.")
    p.add_argument("--split", choices=list(SPLIT_TO_KEYS), default="val", help="Which data split to benchmark on.")
    p.add_argument("--num-events", type=int, default=100, help="Number of events (-1 for all; default 100).")
    p.add_argument("--sorters", nargs="+", default=list(SORTERS), choices=list(SORTERS), help="Subset of orderings to benchmark.")
    p.add_argument("--windows", nargs="+", type=int, default=[128, 256, 512, 1024], help="Attention window sizes for containment.")
    p.add_argument("--window-metric", choices=["span", "block", "mutual"], default="span", help="Containment definition (see module docstring).")
    p.add_argument("--min-hits", type=int, default=2, help="Ignore particles with fewer surviving sihits than this.")
    p.add_argument(
        "--no-cuts",
        action="store_true",
        help="Disable all config hit/particle selection cuts and benchmark on the full raw dataset (--min-hits still applies).",
    )
    p.add_argument("--seed", type=int, default=0, help="Seed for the random and lsh orderings.")
    p.add_argument("--workers", type=int, default=1, help="Parallel worker processes across events (default 1 = serial; -1 = all available CPUs).")
    p.add_argument("--csv", type=Path, default=None, help="Optional path to also write the table as CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    num_events = None if args.num_events == -1 else args.num_events
    init_args = (args.config, args.split, num_events, args.sorters, args.min_hits, args.seed, args.no_cuts)

    # Build once up front (also gives us the event count n).
    _init_worker(*init_args)
    n = len(_WORKER["dataset"])

    workers = available_cpus() if args.workers == -1 else max(1, args.workers)
    workers = min(workers, n)  # no point spawning more workers than events

    spreads: dict[str, list[np.ndarray]] = {name: [] for name in args.sorters}

    def accumulate(res: dict[str, np.ndarray]) -> None:
        for name, arr in res.items():
            if arr.size:
                spreads[name].append(arr)

    if workers > 1:
        print(f"Using {workers} worker processes (building per-worker datasets first)...", flush=True)
        # submit + as_completed gives smooth per-event progress (results pooled,
        # so completion order doesn't matter) rather than jumping a chunk at a time.
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=init_args) as ex:
            futures = [ex.submit(event_spreads, idx) for idx in range(n)]
            for fut in tqdm(as_completed(futures), total=n, desc="events"):
                accumulate(fut.result())
    else:
        for idx in tqdm(range(n), total=n, desc="events"):
            accumulate(event_spreads(idx))

    rows = {}
    n_particles = 0
    for name in args.sorters:
        pooled = np.concatenate(spreads[name]) if spreads[name] else np.zeros(0, dtype=np.int64)
        if pooled.size == 0:
            print(f"WARNING: no particles passed the cuts for sorter '{name}'")
            continue
        n_particles = pooled.size
        rows[name] = summarise(pooled, args.windows, args.window_metric)

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "sorter"

    spread_cols = ["mean", "p50", "p90", "p99", "max"]
    fmt = {c: "{:.1f}".format for c in spread_cols}
    fmt.update({f"c@{w}": "{:.3f}".format for w in args.windows})

    print()
    cuts_state = "OFF (full raw dataset)" if args.no_cuts else "ON (from config)"
    print(f"ColliderML hit-ordering benchmark  |  split={args.split}  events={n}  particles={n_particles:,}")
    print(f"spread = circular hit-position span  |  containment metric='{args.window_metric}'  min_hits={args.min_hits}  cuts={cuts_state}")
    print(table.to_string(formatters=fmt))

    if args.csv is not None:
        table.to_csv(args.csv)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
