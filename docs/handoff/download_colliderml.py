"""Download ColliderML shards from Hugging Face and lay them out for ColliderMLDataset.

The `hf` CLI proved unreliable on the old cluster (broken shebang, and one run silently
fetched a subset of the requested files), so this goes through `snapshot_download`.

Example, 102 pu0 shards (100k train + 2k val events, ~34 GB):

    python download_colliderml.py --pileup 0 --shards 102 \
        --dest /eos/user/l/ljohnson/datasets/ColliderML/v1

Resulting layout, which is what the dataset class expects:

    <dest>/pu0/train/ttbar_pu0_particles/train-000{02..101}-of-01000.parquet
    <dest>/pu0/val/ttbar_pu0_particles/train-000{00,01}-of-01000.parquet
    ... and the same for every other collection.

Shards hold 1000 events each at pu0 and 100 events each at pu200.
"""

from argparse import ArgumentParser
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "CERN/ColliderML-Release-1"
NUM_SHARDS_IN_REPO = 1000


def download(dest: Path, process: str, pileup: int, collections: list[str], shards: int, num_val_shards: int, workers: int) -> None:
    prefix = f"{process}_pu{pileup}"
    staging = dest / f"pu{pileup}" / "_download"

    patterns = [f"data/{prefix}_{c}/train-{i:05d}-of-{NUM_SHARDS_IN_REPO:05d}.parquet" for c in collections for i in range(shards)]
    print(f"Downloading {len(patterns)} files ({shards} shards x {len(collections)} collections) into {staging}")
    snapshot_download(REPO_ID, repo_type="dataset", local_dir=str(staging), allow_patterns=patterns, max_workers=workers)

    # The first `num_val_shards` shards become validation, the rest training, so that
    # extending the training set later is just a matter of downloading more shards.
    for collection in collections:
        source_dir = staging / "data" / f"{prefix}_{collection}"
        for split in ("train", "val"):
            (dest / f"pu{pileup}" / split / f"{prefix}_{collection}").mkdir(parents=True, exist_ok=True)

        for shard_path in sorted(source_dir.glob("*.parquet")):
            shard_idx = int(shard_path.name.split("-")[1])
            split = "val" if shard_idx < num_val_shards else "train"
            shard_path.rename(dest / f"pu{pileup}" / split / f"{prefix}_{collection}" / shard_path.name)

    for split in ("train", "val"):
        counts = [f"{c}={len(list((dest / f'pu{pileup}' / split / f'{prefix}_{c}').glob('*.parquet')))}" for c in collections]
        print(f"  {split}: {', '.join(counts)} shards")

    # Only remove the staging tree if everything was moved out of it.
    leftover = list(staging.rglob("*.parquet"))
    if leftover:
        print(f"WARNING: {len(leftover)} parquet files left in {staging}; not removing it.")
    else:
        print(f"Staging directory {staging} is empty of parquet files; remove it by hand once you are happy.")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True, help="Base directory, e.g. /eos/user/l/ljohnson/datasets/ColliderML/v1")
    parser.add_argument("--process", default="ttbar", help="ttbar, dihiggs, ggf, zprime, hidden_valley, higgs_portal, pileup_only")
    parser.add_argument("--pileup", type=int, default=0, help="0 or 200 (only ttbar exists at 200)")
    parser.add_argument("--collections", nargs="+", default=["particles", "tracker_hits", "tracks"], help="add calo_hits if needed")
    parser.add_argument("--shards", type=int, default=102, help="number of shards, from 00000 upwards")
    parser.add_argument("--num-val-shards", type=int, default=2, help="how many of the first shards go to val/")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    assert args.num_val_shards < args.shards, "num_val_shards must leave some shards for training"
    download(args.dest, args.process, args.pileup, args.collections, args.shards, args.num_val_shards, args.workers)


if __name__ == "__main__":
    main()
