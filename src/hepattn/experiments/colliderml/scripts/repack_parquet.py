"""Repack ColliderML parquet shards with a smaller row-group size.

Each source shard stores ~100 events in a single row-group, so reading one event
decodes the whole group. Shard-contiguous sampling (see data.ShardContiguousSampler)
already removes most of that cost, but for heavily-shuffled or random-access workloads
a smaller row-group makes each decode touch only a few events. This writes a *copy* of
the dataset (never mutating the canonical data) with `--rows-per-group` events per group.

Point a training `*_dir` at the output to use it. Only the collections you pass are
repacked; symlink or copy the rest so the directory layout stays complete.

Example (repack the heavy tracker-hit + particle collections of the train split):
    python scripts/repack_parquet.py \
        --src /share/rcif2/ljohnson/data/colliderml/v1/pu200/train \
        --dst /share/rcif2/ljohnson/data/colliderml/v1_repacked/pu200/train \
        --collections ttbar_pu200_tracker_hits ttbar_pu200_particles \
        --rows-per-group 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


def repack_file(src: Path, dst: Path, rows_per_group: int, compression: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow
    with pq.ParquetWriter(dst, schema, compression=compression) as writer:
        # Stream in small batches so we never hold a whole shard in memory, and each
        # written row-group is exactly `rows_per_group` rows.
        for batch in pf.iter_batches(batch_size=rows_per_group):
            writer.write_batch(batch, row_group_size=rows_per_group)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="Source split dir (contains <event>_<collection>/ subdirs).")
    p.add_argument("--dst", type=Path, required=True, help="Destination split dir (created).")
    p.add_argument("--collections", nargs="+", required=True, help="Collection subdir names to repack.")
    p.add_argument("--rows-per-group", type=int, default=5, help="Events per output row-group (default 5).")
    p.add_argument("--compression", type=str, default="zstd", help="Parquet compression codec (default zstd).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for collection in args.collections:
        src_dir = args.src / collection
        dst_dir = args.dst / collection
        shards = sorted(src_dir.glob("*.parquet"))
        if not shards:
            print(f"WARNING: no shards under {src_dir}, skipping")
            continue
        for shard in tqdm(shards, desc=collection):
            repack_file(shard, dst_dir / shard.name, args.rows_per_group, args.compression)
    print(f"\nDone. Point your training *_dir at {args.dst}")
    print("Remember to also provide the collections you did NOT repack (symlink or copy them).")


if __name__ == "__main__":
    main()
