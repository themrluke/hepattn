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
        --window-metric mutual --md orderings.md
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


# Custom cut overrides applied under --custom-cuts: edit this block to benchmark
# on a hand-picked selection instead of the training config's cuts. A value of
# None disables that selection (e.g. sihit_volume_ids=None keeps every volume,
# particle_max_abs_eta=None keeps every eta). particle_hit_cuts must be nested by
# particle class: {<class>: {<cut>: value}}.
CUSTOM_CUTS = {
    "particle_min_pt": 0.9,
    "particle_max_abs_eta": None,
    "particle_hit_cuts": {
        "charged_hadron": {"min_num_sihit": 3},
        "electron": {"min_num_sihit": 3},
        "muon": {"min_num_sihit": 3},
        "tau": {"min_num_sihit": 3},
    },
    "particle_include_charged": True,
    "particle_include_neutral": False,
    "sihit_volume_ids": [16, 18],
    "sihit_max_abs_eta": None,
    "max_num_particles": None,
}


def build_dataset(
    config_path: Path,
    split: str,
    num_events: int | None,
    custom_cuts: bool = False,
    sihit_volumes: list[int] | None = None,
) -> ColliderMLDataset:
    """Instantiate the real tracking dataset straight from a training config.

    With ``custom_cuts=True`` the config's hit- and particle-level selections are
    replaced by the hand-edited ``CUSTOM_CUTS`` block, so the orderings are measured
    on whatever selection you configure there. ``sihit_volumes`` overrides which
    detector volumes are kept (e.g. barrel-only vs endcap-only); it takes precedence
    over both the config and ``custom_cuts``.
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
    # (num_particles x num_sihits) mask we never read - essential when the custom
    # cuts are loose, where that matrix would otherwise blow up memory.
    kwargs.update(
        dirpath=dirpath,
        num_events=num_events,
        hit_filter=False,
        build_dense_masks=False,
        return_calohits=False,
        return_tracks=False,
    )
    if custom_cuts:
        kwargs.update(CUSTOM_CUTS)
    if sihit_volumes is not None:
        kwargs["sihit_volume_ids"] = sihit_volumes
    return ColliderMLDataset(**kwargs)


def particle_spreads(
    order: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    min_hits: int,
    num_hits: int,
    keep_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Circular index spread of each particle's hits under a given ordering.

    ``rank[h]`` is the position of hit ``h`` in the sorted sequence. The sequence
    is treated as a ring (matching the model's ``window_wrap: true``), so a
    particle's spread is the shortest arc covering all of its hits: the ring
    length minus the largest gap between its consecutive hits, where the seam gap
    (across the phi = +/-pi wrap) is included as one of the candidate gaps.
    ``keep_mask`` optionally restricts which particles are reported (e.g. a truth
    eta window). Fully vectorised over particles.
    """
    rank = np.empty(num_hits, dtype=np.int64)
    rank[order] = np.arange(num_hits, dtype=np.int64)

    counts = np.diff(indptr)
    keep = counts >= min_hits
    if keep_mask is not None:
        keep = keep & keep_mask
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


def to_markdown(table: pd.DataFrame, fmt: dict, header_lines: list[str]) -> str:
    """Render the results table as a Markdown document for pasting into notes.

    Containment (c@*) columns are shown as percentages to 1 dp (rather than
    fractions), the spread columns keep the console formatting, and the first
    column (sorter name) is bolded.
    """
    cols = list(table.columns)

    def render(col: str, val: float) -> str:
        if col.startswith("c@"):
            return f"{val * 100:.1f}%"
        return fmt[col](val) if col in fmt else str(val)

    head = "| sorter | " + " | ".join(cols) + " |"
    align = "|:---|" + "|".join(["---:"] * len(cols)) + "|"
    body = []
    for idx, row in table.iterrows():
        cells = [render(c, row[c]) for c in cols]
        body.append(f"| **{idx}** | " + " | ".join(cells) + " |")
    return "\n".join([*header_lines, "", head, align, *body, ""])


def available_cpus() -> int:
    """CPUs actually available to this process (respects cgroup/SLURM affinity)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on all platforms
        return os.cpu_count() or 1


# Per-process state. Built once per worker (or once in the main process for the
# serial path) rather than shipped across the pool, since the dataset is heavy.
_WORKER: dict = {}


def _init_worker(run_cfg: dict) -> None:
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
        dataset=build_dataset(
            run_cfg["config"],
            run_cfg["split"],
            run_cfg["num_events"],
            custom_cuts=run_cfg["custom_cuts"],
            sihit_volumes=run_cfg["sihit_volumes"],
        ),
        sorters=run_cfg["sorters"],
        min_hits=run_cfg["min_hits"],
        seed=run_cfg["seed"],
        eta_min=run_cfg["particle_eta_min"],
        eta_max=run_cfg["particle_eta_max"],
        windows=run_cfg["windows"],
        max_tracks=run_cfg["max_tracks"],
    )


def event_spreads(idx: int) -> dict[str, np.ndarray]:
    """Per-particle spread arrays for one event, keyed by sorter.

    Reads per-process state set by ``_init_worker``. The RNG is seeded per event
    as (seed, sample_id) so the random/lsh orderings are reproducible and
    independent of worker count or event ordering. An optional truth-eta window
    (on the particle's true |eta| = arctanh(pz/p)) restricts which particles are
    reported, without changing the hit sequence or the ordering.
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

    # Truth-eta particle selection (population filter; leaves hits/ordering intact).
    keep_mask = None
    eta_min, eta_max = _WORKER["eta_min"], _WORKER["eta_max"]
    if eta_min is not None or eta_max is not None:
        abs_true_eta = np.abs(targets["particle_eta"].numpy())
        keep_mask = np.ones(abs_true_eta.shape[0], dtype=bool)
        if eta_min is not None:
            keep_mask &= abs_true_eta >= eta_min
        if eta_max is not None:
            keep_mask &= abs_true_eta <= eta_max

    rng = np.random.default_rng([_WORKER["seed"], int(sample_id)])
    out = {}
    for name in _WORKER["sorters"]:
        order = SORTERS[name](eta, phi, rng)
        out[name] = particle_spreads(order, indptr, indices, _WORKER["min_hits"], num_hits, keep_mask)
    return out


# Reachability "banks" (ported from Max's sorter_benchmark, learned sorter omitted).
# Each method maps to a list of orderings whose per-window coverage is unioned;
# LSH is run as K independent hash tables (fixed seed per table, held constant
# across events), so a bank of them is a proper multi-table LSH.
REACH_BANKS: dict[str, list[str]] = {
    "phi": ["phi"],
    "hilbert": ["hilbert"],
    "lsh K=1": ["lsh_0"],
    "lsh K=3": ["lsh_0", "lsh_1", "lsh_2"],
    "lsh K=24": [f"lsh_{k}" for k in range(24)],
}
# Union of the distinct orderings any bank needs (computed once per event).
REACH_ORDERINGS: list[str] = list(dict.fromkeys(o for ords in REACH_BANKS.values() for o in ords))


def _resolve_order(name: str, eta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Compute a named ordering. ``lsh_k`` uses a fixed seed k (a hash table held
    constant across events); phi/hilbert ignore the rng so None is passed."""
    if name.startswith("lsh_"):
        return SORTERS["lsh"](eta, phi, np.random.default_rng(int(name[4:])))
    return SORTERS[name](eta, phi, None)


def max_in_window(ranks: np.ndarray, window: int, ring: int) -> int:
    """Largest number of the given ranks falling in any circular window of width W.

    Ported verbatim from Max's benchmark (two-pointer over the ranks duplicated by
    +ring to handle the wrap seam).
    """
    ranks = np.sort(ranks)
    ext = np.concatenate([ranks, ranks + ring])
    best, j = 1, 0
    for i in range(len(ranks)):
        j = max(j, i)
        while j < len(ext) and ext[j] - ranks[i] < window:
            j += 1
        best = max(best, j - i)
    return best


def event_reach(idx: int) -> dict:
    """Per-(bank, window) arrays of per-particle coverage for one event.

    Coverage = best fraction of a particle's hits captured by one circular window
    of width W, taken as the max (union) over the bank's K orderings. Particle
    selection (min_hits, truth-eta window, max_tracks sampling) mirrors
    ``event_spreads``; the hit sequence/ordering is identical to the span table.
    """
    dataset = _WORKER["dataset"]
    sample_id = dataset.sample_ids[idx]
    inputs, targets = dataset.load_event(sample_id)
    eta = inputs["sihit_eta"].numpy()
    phi = inputs["sihit_phi"].numpy()
    ring = eta.shape[0]
    indptr = targets["particle_sihit_indptr"].numpy()
    indices = targets["particle_sihit_indices"].numpy()
    windows = _WORKER["windows"]
    if ring == 0 or indices.size == 0:
        return {}

    # Reconstructable particles passing min_hits and the optional truth-eta window.
    counts = np.diff(indptr)
    keep = counts >= _WORKER["min_hits"]
    eta_min, eta_max = _WORKER["eta_min"], _WORKER["eta_max"]
    if eta_min is not None or eta_max is not None:
        abs_true_eta = np.abs(targets["particle_eta"].numpy())
        if eta_min is not None:
            keep &= abs_true_eta >= eta_min
        if eta_max is not None:
            keep &= abs_true_eta <= eta_max
    kept = np.nonzero(keep)[0]
    if kept.size == 0:
        return {}

    # Sample at most max_tracks particles per event (bounds the O(tracks*K*W) cost).
    rng = np.random.default_rng([_WORKER["seed"], int(sample_id)])
    if kept.size > _WORKER["max_tracks"]:
        kept = rng.choice(kept, _WORKER["max_tracks"], replace=False)
    groups = [indices[indptr[p] : indptr[p + 1]] for p in kept]

    # Rank of each hit (position in the 1D order) for every ordering any bank needs.
    ranks = {}
    for name in REACH_ORDERINGS:
        order = _resolve_order(name, eta, phi)
        r = np.empty(ring, dtype=np.int64)
        r[order] = np.arange(ring, dtype=np.int64)
        ranks[name] = r

    out = {}
    for bank_name, ords in REACH_BANKS.items():
        for w in windows:
            cov = np.array([max(max_in_window(ranks[o][h], w, ring) for o in ords) / len(h) for h in groups])
            out[(bank_name, w)] = cov
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
        "--custom-cuts",
        action="store_true",
        help="Replace the config's hit/particle cuts with the hand-edited CUSTOM_CUTS block in this file (--min-hits still applies).",
    )
    p.add_argument(
        "--sihit-volumes",
        nargs="+",
        type=int,
        default=None,
        help="Restrict hits to these detector volume IDs (barrel: 17 24 29; endcap: 16 18 23 25 28 30). Overrides config/--custom-cuts.",
    )
    p.add_argument("--particle-eta-min", type=float, default=None, help="Keep only particles with truth |eta| >= this (population filter).")
    p.add_argument("--particle-eta-max", type=float, default=None, help="Keep only particles with truth |eta| <= this (population filter).")
    p.add_argument("--seed", type=int, default=0, help="Seed for the random and lsh orderings.")
    p.add_argument("--workers", type=int, default=1, help="Parallel worker processes across events (default 1 = serial; -1 = all available CPUs).")
    p.add_argument(
        "--reachability",
        action="store_true",
        help="Also compute the reachability table (best hit-fraction in one window, union over K orderings incl. multi-table LSH).",
    )
    p.add_argument("--max-tracks", type=int, default=500, help="Particles sampled per event for the reachability table.")
    p.add_argument("--md", type=Path, default=None, help="Optional path to write the table as a Markdown file (for pasting into notes).")
    return p.parse_args()


def _reach_markdown(rtable: pd.DataFrame) -> str:
    """Render the reachability table as a Markdown section (appended after the span table)."""
    lines = [
        "",
        "## Reachability (best hit-fraction in one window, union over K orderings)",
        "",
        "- `K` = number of orderings (LSH run as K hash tables) · `budget = K*W` = compute",
        "- `meanCov` = mean best-window hit coverage (%) · `DM≥0.5` = % with ≥50% (double-majority) · `full` = % with 100%",
        "",
        "| sorter | K | W | budget | meanCov | DM≥0.5 | full |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in rtable.iterrows():
        lines.append(
            f"| **{r['sorter']}** | {int(r['K'])} | {int(r['W'])} | {int(r['budget'])} | "
            f"{r['meanCov']:.1f} | {r['DM>=0.5']:.1f} | {r['full']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    num_events = None if args.num_events == -1 else args.num_events
    run_cfg = {
        "config": args.config,
        "split": args.split,
        "num_events": num_events,
        "sorters": args.sorters,
        "min_hits": args.min_hits,
        "seed": args.seed,
        "custom_cuts": args.custom_cuts,
        "sihit_volumes": args.sihit_volumes,
        "particle_eta_min": args.particle_eta_min,
        "particle_eta_max": args.particle_eta_max,
        "windows": args.windows,
        "max_tracks": args.max_tracks,
    }

    # Build once up front (also gives us the event count n).
    _init_worker(run_cfg)
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
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(run_cfg,)) as ex:
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

    cuts_state = "custom (CUSTOM_CUTS block)" if args.custom_cuts else "ON (from config)"
    volumes = _WORKER["dataset"].sihit_volume_ids
    vol_desc = "all" if volumes is None else str(volumes)
    if args.particle_eta_min is not None or args.particle_eta_max is not None:
        lo = args.particle_eta_min if args.particle_eta_min is not None else 0.0
        hi = args.particle_eta_max if args.particle_eta_max is not None else float("inf")
        eta_desc = f"[{lo:g}, {hi:g}]"
    else:
        eta_desc = "all"

    print()
    print(f"ColliderML hit-ordering benchmark  |  split={args.split}  events={n}  particles={n_particles:,}")
    print(f"spread = circular hit-position span  |  containment metric='{args.window_metric}'  min_hits={args.min_hits}  cuts={cuts_state}")
    print(f"hit volumes={vol_desc}  |  particle truth|eta|={eta_desc}")
    print(table.to_string(formatters=fmt))

    md_content = None
    if args.md is not None:
        header_lines = [
            "# ColliderML hit-ordering benchmark",
            "",
            f"- **split:** {args.split}  |  **events:** {n}  |  **particles:** {n_particles:,}",
            f"- **spread:** circular hit-position span  |  **containment metric:** `{args.window_metric}`"
            f"  |  **min_hits:** {args.min_hits}  |  **cuts:** {cuts_state}",
            f"- **hit volumes:** {vol_desc}  |  **particle truth \\|eta\\|:** {eta_desc}",
            "- `c@W` = percentage of particles whose hits fit within an attention window of size W",
        ]
        md_content = to_markdown(table, fmt, header_lines)

    # Reachability table (union over K orderings; ported from Max's sorter_benchmark).
    if args.reachability:
        reach = {(b, w): [] for b in REACH_BANKS for w in args.windows}
        if workers > 1:
            print(f"\nReachability pass using {workers} worker processes...", flush=True)
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(run_cfg,)) as ex:
                futures = [ex.submit(event_reach, idx) for idx in range(n)]
                for fut in tqdm(as_completed(futures), total=n, desc="reach"):
                    for bw, arr in fut.result().items():
                        reach[bw].append(arr)
        else:
            for idx in tqdm(range(n), total=n, desc="reach"):
                for bw, arr in event_reach(idx).items():
                    reach[bw].append(arr)

        rrows = []
        for bank in REACH_BANKS:
            k = len(REACH_BANKS[bank])
            for w in args.windows:
                if not reach[(bank, w)]:
                    continue
                cov = np.concatenate(reach[(bank, w)])
                rrows.append(
                    {
                        "sorter": bank,
                        "K": k,
                        "W": w,
                        "budget": k * w,
                        "meanCov": cov.mean() * 100,
                        "DM>=0.5": (cov >= 0.5).mean() * 100,
                        "full": (cov >= 0.999).mean() * 100,
                    }
                )
        rtable = pd.DataFrame(rrows)
        rfmt = {c: "{:.1f}".format for c in ("meanCov", "DM>=0.5", "full")}
        print("\n== Reachability: best fraction of a track in one window, union over K orderings ==")
        print(rtable.to_string(index=False, formatters=rfmt))
        if md_content is not None:
            md_content += _reach_markdown(rtable)

    if md_content is not None:
        args.md.write_text(md_content)
        print(f"\nWrote {args.md}")


if __name__ == "__main__":
    main()
