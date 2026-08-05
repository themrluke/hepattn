"""Measure flex ``BlockMask`` block-level sparsity for Route B per-head LSH windows.

Route B expresses the sliding window as a constraint on each head's LSH *rank*
(``|rank[h, q] - rank[h, kv]| <= window // 2``) instead of on the token index.
Element-wise that admits exactly as many pairs as an ordinary sliding window --
the rank window is the same width, and a rank table is a permutation, so the
admitted-pair count is *identical* for every ordering.

The catch is that ``flex_attention`` does not skip work per element. It skips per
**block** (``BLOCK_SIZE`` x ``BLOCK_SIZE``, 128 by default): a block is computed
in full if it contains even one admitted pair. So the question that decides
whether Route B is viable is not "how many pairs are admitted" (fixed) but "how
many *blocks* do those pairs touch".

Two extremes:

* Tokens laid out in the head's own rank order -> the admitted pairs form a
  narrow band around the diagonal, touching ~``window / BLOCK_SIZE`` blocks per
  row. This is what today's sort-then-slide encoder gets.
* Tokens laid out in an order uncorrelated with the head's ranks -> each query's
  ``window`` admitted partners are scattered uniformly over the sequence, so
  almost every block contains at least one of them and *nothing* is skipped. The
  mask is dense, costing full N^2 with no gain over global attention.

The tokens can only be laid out in one order while every head wants its own, so
the worry was that Route B sits at the second extreme. Measured on real events it
does not: at N ~ 13k with window 512 and 24 cells it keeps ~0.24 of blocks (vs
0.05 banded, 0.04 ideal), because LSH orderings are not random permutations --
every cell sorts the same 2D geometry with a different random quantile grid, so
the cells stay correlated. Layout still matters a lot: sorting the sequence by one
cell's ordering roughly halves the kept fraction versus raw dataset order.

It does not call ``create_block_mask`` for the main measurement: at N ~ 2.5e5
that would materialise an N x N mask. Instead :func:`kept_block_fraction`
computes the identical quantity directly and chunked on the GPU. Pass
``--verify`` to cross-check it against a real ``BlockMask.sparsity()`` at a small
N, which is the correctness argument for the fast path.

Usage:
    python -m hepattn.experiments.colliderml.scripts.measure_block_sparsity \
        --num-events 3 --window-size 512 --num-heads 8 --n-hashes 3 --verify
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.nn.attention.flex_attention import create_block_mask

from hepattn.experiments.colliderml.scripts.benchmark_orderings import build_dataset
from hepattn.flex import per_head_window_mask_mod
from hepattn.models.ordering import E2LSHOrderingGrid, lsh_order_values

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "base.yaml"


def element_density(num_tokens: int, window_size: int) -> float:
    """Exact fraction of the N x N score matrix a rank window admits.

    Independent of the ordering: a rank table is a permutation of ``0..N-1``, so
    the admitted-pair count only depends on N and the window width. This is the
    floor that block granularity is measured against.
    """
    half = window_size // 2
    r = torch.arange(num_tokens, dtype=torch.int64)
    per_query = torch.minimum(r + half, torch.tensor(num_tokens - 1)) - torch.maximum(r - half, torch.tensor(0)) + 1
    return float(per_query.sum()) / float(num_tokens) ** 2


def kept_block_fraction(ranks: torch.Tensor, window_size: int, block_size: int = 128, q_blocks_per_chunk: int = 4) -> torch.Tensor:
    """Fraction of ``(q_block, kv_block)`` pairs flex would have to compute.

    A block pair is kept iff it contains at least one admitted pair, which is
    exactly the rule ``create_block_mask`` applies when deciding whether to emit
    a block index. Computed chunk-wise over query blocks so peak memory is
    ``q_blocks_per_chunk * block_size * N`` bools rather than N^2.

    Args:
        ranks: ``(H, N)`` integer rank tables (H = num_heads * n_hashes).
        window_size: Full window width; half either side.
        block_size: flex ``BLOCK_SIZE`` (128 is the default for both q and kv).
        q_blocks_per_chunk: Query blocks per chunk; trades memory for speed.

    Returns:
        ``(H,)`` float tensor, the kept-block fraction per head. 1.0 means the
        mask is fully dense and flex skips nothing.
    """
    num_heads, num_tokens = ranks.shape
    half = window_size // 2
    num_blocks = math.ceil(num_tokens / block_size)
    padded = num_blocks * block_size

    # Pad to a whole number of blocks. Padded slots are excluded via `valid`
    # rather than by a sentinel rank, so they can never admit anything (a
    # sentinel would admit against itself and invent kept blocks).
    valid = torch.zeros(padded, dtype=torch.bool, device=ranks.device)
    valid[:num_tokens] = True

    out = torch.empty(num_heads, dtype=torch.float64)
    for h in range(num_heads):
        # int32 halves the peak of the (chunk*B, padded) difference tensor; ranks
        # are bounded by N so they cannot overflow.
        r = torch.zeros(padded, dtype=torch.int32, device=ranks.device)
        r[:num_tokens] = ranks[h].to(torch.int32)
        kept = torch.zeros(num_blocks, num_blocks, dtype=torch.bool, device=ranks.device)

        for start in range(0, num_blocks, q_blocks_per_chunk):
            stop = min(start + q_blocks_per_chunk, num_blocks)
            q_slice = slice(start * block_size, stop * block_size)
            rq = r[q_slice]

            admitted = (rq[:, None] - r[None, :]).abs() <= half  # (chunk*B, padded)
            admitted &= valid[q_slice][:, None] & valid[None, :]
            # Any admitted pair inside a (q_block, kv_block) tile keeps that tile.
            admitted = admitted.view(-1, num_blocks, block_size).any(dim=-1)  # (chunk*B, nb)
            kept[start:stop] = admitted.view(-1, block_size, num_blocks).any(dim=1)

        out[h] = kept.sum().double() / float(num_blocks) ** 2
    return out


def verify_against_flex(ranks: torch.Tensor, window_size: int, block_size: int = 128) -> None:
    """Cross-check :func:`kept_block_fraction` against a real ``BlockMask``.

    ``BlockMask.sparsity()`` reports the percentage of blocks skipped, so
    ``1 - sparsity/100`` is the kept fraction our fast path predicts.
    """
    num_heads, num_tokens = ranks.shape
    mask = create_block_mask(
        per_head_window_mask_mod(ranks, window_size),
        B=None,
        H=num_heads,
        Q_LEN=num_tokens,
        KV_LEN=num_tokens,
        device=str(ranks.device),
        BLOCK_SIZE=block_size,
    )
    flex_kept = 1.0 - mask.sparsity() / 100.0
    ours = float(kept_block_fraction(ranks, window_size, block_size).mean())
    print(f"  verify N={num_tokens:>6}  flex={flex_kept:.6f}  ours={ours:.6f}  delta={abs(flex_kept - ours):.2e}")
    assert abs(flex_kept - ours) < 1e-6, "kept_block_fraction disagrees with flex BlockMask.sparsity()"


def load_event_coords(config: Path, split: str, index: int) -> torch.Tensor:
    """``(N, 2)`` float64 ``(eta, phi)`` for one real event."""
    dataset = build_dataset(config, split, num_events=index + 1)
    inputs, _ = dataset.load_event(dataset.sample_ids[index])
    return torch.stack([inputs["sihit_eta"], inputs["sihit_phi"]], dim=-1).to(torch.float64)


def reorder_tokens(coords: torch.Tensor, mode: str) -> torch.Tensor:
    """Permute hits into the order the encoder would feed to attention.

    Block structure is defined over *token index*, so the layout the encoder
    chooses (``Encoder.forward``'s ``x_sort_value``) changes the answer even
    though the ranks themselves do not.

    * ``raw``  -- dataset order, no sort (worst case, no locality at all).
    * ``phi``  -- sort by phi, a cheap global 1D ordering.
    * ``lsh0`` -- sort by the head-0/hash-0 LSH ordering, i.e. align the layout
      with one cell of the grid. That cell then gets a perfectly banded mask;
      the measurement is what every *other* cell gets.

    Raises:
        ValueError: If ``mode`` is not one of the three orders above.
    """
    if mode == "raw":
        return coords
    if mode == "phi":
        return coords[torch.argsort(coords[:, 1], stable=True)]
    if mode == "lsh0":
        grid = E2LSHOrderingGrid(num_heads=1, n_hashes=1, dtype=torch.float64)
        o = lsh_order_values(coords, grid.alpha.flatten(0, 1), grid.region_counts.flatten(0, 1))
        return coords[torch.argsort(o[0], stable=True)]
    raise ValueError(f"unknown token order: {mode}")


def report(coords: torch.Tensor, args: argparse.Namespace, device: str) -> None:
    num_tokens = coords.shape[0]
    grid = E2LSHOrderingGrid(num_heads=args.num_heads, n_hashes=args.n_hashes, dtype=torch.float32).to(device)
    ranks = grid(coords.to(device).float()).flatten(0, 1)  # (H, N)

    elem = element_density(num_tokens, args.window_size)
    lsh = kept_block_fraction(ranks, args.window_size, args.block_size, args.q_blocks_per_chunk)
    baseline = kept_block_fraction(
        torch.arange(num_tokens, device=device)[None],  # identity ranks = today's sort-then-slide
        args.window_size,
        args.block_size,
        args.q_blocks_per_chunk,
    )

    print(f"\n  N = {num_tokens}   window = {args.window_size}   block = {args.block_size}   cells = {ranks.shape[0]}")
    print(f"  element density (ideal, ordering-independent) : {elem:9.5f}")
    print(f"  block density,  banded baseline (identity)    : {float(baseline[0]):9.5f}   waste x{float(baseline[0]) / elem:7.1f}")
    print(
        f"  block density,  per-head LSH ranks            : {float(lsh.mean()):9.5f} "
        f"  waste x{float(lsh.mean()) / elem:7.1f}   [min {float(lsh.min()):.5f}, max {float(lsh.max()):.5f}]"
    )
    print(f"  speedup vs dense attention: baseline x{1.0 / float(baseline[0]):.1f}   Route B x{1.0 / float(lsh.mean()):.2f}")

    if args.verify:
        small = ranks[:, : args.verify_tokens].contiguous()
        # Re-rank the truncated slice so it is a valid permutation of 0..n-1.
        small = torch.argsort(torch.argsort(small, dim=-1, stable=True), dim=-1, stable=True)
        verify_against_flex(small, args.window_size, args.block_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-events", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--n-hashes", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--token-order", default="lsh0", choices=["raw", "phi", "lsh0"])
    parser.add_argument("--max-hits", type=int, default=None, help="Truncate each event to this many hits.")
    parser.add_argument("--verify", action="store_true", help="Cross-check against a real flex BlockMask.")
    parser.add_argument("--verify-tokens", type=int, default=1024)
    parser.add_argument("--q-blocks-per-chunk", type=int, default=4, help="Lower this if the GPU is short on memory.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i in range(args.num_events):
        coords = load_event_coords(args.config, args.split, i)
        if args.max_hits is not None:
            coords = coords[: args.max_hits]
        coords = reorder_tokens(coords, args.token_order)
        print(f"\n=== event {i}  (token order: {args.token_order}) ===")
        report(coords, args, device)


if __name__ == "__main__":
    main()
