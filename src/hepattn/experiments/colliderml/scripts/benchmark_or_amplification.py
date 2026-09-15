"""Time the three ways of running a per-head windowed attention against each other.

Step 4.12. The question this answers is which of two mathematically identical implementations
of OR amplification to keep, and what either costs against the sliding window we run today.

The three arms:

* ``flash``  -- today's encoder: one global ordering, one sliding window, fused flash kernel.
  The cheapest, and the thing OR amplification has to justify itself against.
* ``route_b`` -- leave the tokens where they are and express each head's window as a constraint
  on that head's LSH *rank* (``|rank[h,q] - rank[h,kv]| <= W//2``), via a flex ``mask_mod``.
  Nothing moves, but the admitted pairs are scattered across the score matrix, so flex has to
  compute far more 128x128 tiles than a banded mask would.
* ``sorted``  -- physically gather each head's q/k/v into that head's own ordering, apply one
  *shared plain banded* window to every head, then scatter the outputs back before the merge.
  Every head's admitted pairs sit tight against the diagonal, so the tile count collapses --
  at the price of materialising the permuted tensors and a gather/scatter per head.

``route_b`` and ``sorted`` admit **exactly the same pairs** and must therefore produce the same
numbers; the script asserts that rather than assuming it. Any difference is a bug, not a result.

Projections, norms and the residual stream are identical across the arms, so only the attention
operation itself is timed. Mask construction is reported separately because it is not comparable:
route_b rebuilds a mask per layer per event (its rule depends on the event's geometry) whereas
the banded mask a sorted run needs is position-only and can be built once for the whole job.

Usage:
    python -m hepattn.experiments.colliderml.scripts.benchmark_or_amplification \
        --num-events 3 --window-size 512 --n-hashes 3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from hepattn.experiments.colliderml.scripts.benchmark_orderings import build_dataset
from hepattn.flex import per_head_window_mask_mod
from hepattn.flex.sliding_window import sliding_window_mask
from hepattn.models.attention import expand_heads_for_or, or_merge_lse, permute_to_rank_order, restore_token_order
from hepattn.models.ordering import E2LSHOrderingGrid, lsh_order_values

try:
    from flash_attn import flash_attn_func
except ImportError:  # pragma: no cover - flash is optional and GPU-only
    flash_attn_func = None

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_DIR / "configs" / "base.yaml"

compiled_flex = torch.compile(flex_attention, dynamic=True)
compiled_block_mask = torch.compile(create_block_mask, dynamic=True)


def load_event_coords(config: Path, split: str, index: int) -> torch.Tensor:
    """``(N, 2)`` ``(eta, phi)`` for one real event."""
    dataset = build_dataset(config, split, num_events=index + 1)
    inputs, _ = dataset.load_event(dataset.sample_ids[index])
    return torch.stack([inputs["sihit_eta"], inputs["sihit_phi"]], dim=-1).double()


def sort_by_lsh(coords: torch.Tensor) -> torch.Tensor:
    """Lay the sequence out in one cell's LSH order, which is what the encoder should do.

    Block structure is defined over token *index*, so the layout decides how tightly each head's
    admitted pairs cluster. Measured at 4a: raw 0.49, phi 0.30, lsh 0.24.
    """
    grid = E2LSHOrderingGrid(num_heads=1, n_hashes=1, dtype=torch.float64)
    order_values = lsh_order_values(coords, grid.alpha.flatten(0, 1), grid.region_counts.flatten(0, 1))
    return coords[torch.argsort(order_values[0], stable=True)]


def run_route_b(q, k, v, mask, n_hashes):
    qe, ke, ve = (expand_heads_for_or(t, n_hashes) for t in (q, k, v))
    out, lse = compiled_flex(qe, ke, ve, block_mask=mask, return_lse=True)
    return or_merge_lse(out, lse, n_hashes)


def run_sorted(q, k, v, banded_mask, ranks, n_hashes):
    qp, kp, vp = (permute_to_rank_order(t, ranks, n_hashes) for t in (q, k, v))
    out, lse = compiled_flex(qp, kp, vp, block_mask=banded_mask, return_lse=True)
    return or_merge_lse(restore_token_order(out, ranks), restore_token_order(lse, ranks), n_hashes)


def run_flash(q, k, v, half_window):
    # flash wants (B, N, H, Dh); everything else here is (B, H, N, Dh).
    qs, ks, vs = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    return flash_attn_func(qs, ks, vs, window_size=(half_window, half_window)).transpose(1, 2)


def timed(fn, repeats: int) -> float:
    """Milliseconds per call, after a warm-up that absorbs compilation."""
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1e3


def peak_gb(fn) -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def report_event(coords: torch.Tensor, args: argparse.Namespace, device: str) -> dict[str, float]:
    num_tokens = coords.shape[0]
    half_window = args.window_size // 2
    head_dim = args.dim // args.num_heads

    grid = E2LSHOrderingGrid(args.num_heads, args.n_hashes).to(device)
    ranks = grid(coords.to(device).float()).flatten(0, 1)  # (H*C, N)

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, args.num_heads, num_tokens, head_dim, device=device, dtype=torch.float16) for _ in range(3))

    mask_ms = timed(
        lambda: compiled_block_mask(
            per_head_window_mask_mod(ranks, args.window_size), B=None, H=ranks.shape[0], Q_LEN=num_tokens, KV_LEN=num_tokens, device=device
        ),
        args.repeats,
    )
    banded_ms = timed(
        lambda: compiled_block_mask(sliding_window_mask(args.window_size), B=None, H=None, Q_LEN=num_tokens, KV_LEN=num_tokens, device=device),
        args.repeats,
    )
    route_b_mask = compiled_block_mask(
        per_head_window_mask_mod(ranks, args.window_size), B=None, H=ranks.shape[0], Q_LEN=num_tokens, KV_LEN=num_tokens, device=device
    )
    banded_mask = compiled_block_mask(sliding_window_mask(args.window_size), B=None, H=None, Q_LEN=num_tokens, KV_LEN=num_tokens, device=device)

    with torch.no_grad():
        out_b = run_route_b(q, k, v, route_b_mask, args.n_hashes)
        out_s = run_sorted(q, k, v, banded_mask, ranks, args.n_hashes)
        # The two admit identical pairs, so this is an equality check, not a tolerance study.
        torch.testing.assert_close(out_b, out_s, rtol=2e-3, atol=2e-3)

        results = {
            "route_b_ms": timed(lambda: run_route_b(q, k, v, route_b_mask, args.n_hashes), args.repeats),
            "sorted_ms": timed(lambda: run_sorted(q, k, v, banded_mask, ranks, args.n_hashes), args.repeats),
            "route_b_gb": peak_gb(lambda: run_route_b(q, k, v, route_b_mask, args.n_hashes)),
            "sorted_gb": peak_gb(lambda: run_sorted(q, k, v, banded_mask, ranks, args.n_hashes)),
            "mask_route_b_ms": mask_ms,
            "mask_banded_ms": banded_ms,
            "route_b_density": 1.0 - route_b_mask.sparsity() / 100.0,
            "banded_density": 1.0 - banded_mask.sparsity() / 100.0,
        }
        if flash_attn_func is not None:
            results["flash_ms"] = timed(lambda: run_flash(q, k, v, half_window), args.repeats)
            results["flash_gb"] = peak_gb(lambda: run_flash(q, k, v, half_window))

    print(f"\n  N = {num_tokens}  window = {args.window_size}  heads = {args.num_heads}  hashes = {args.n_hashes}")
    print(f"  block density      route_b {results['route_b_density']:.3f}   banded {results['banded_density']:.3f}")
    print(f"  {'arm':<10}{'attn ms':>10}{'peak GB':>10}{'vs flash':>10}")
    base = results.get("flash_ms")
    for arm in ("flash", "route_b", "sorted"):
        if f"{arm}_ms" not in results:
            continue
        ms, gb = results[f"{arm}_ms"], results[f"{arm}_gb"]
        rel = f"{ms / base:.1f}x" if base else "-"
        print(f"  {arm:<10}{ms:>10.2f}{gb:>10.3f}{rel:>10}")
    # This is the number that decides the design, and it is easy to miss: route_b's rule closes
    # over the event's ranks and each layer draws its own, so its mask is rebuilt every layer of
    # every event. The sorted arm's window is position-only -- same mask for every head and every
    # layer, and reusable across events of equal length.
    print(f"  mask build   route_b {mask_ms:7.2f} ms/layer/event   banded {banded_ms:7.2f} ms (shared, cacheable)")
    print(
        f"  per layer, mask + attention:  route_b {mask_ms + results['route_b_ms']:7.2f} ms"
        f"   sorted {banded_ms + results['sorted_ms']:7.2f} ms   flash {results.get('flash_ms', float('nan')):7.2f} ms"
    )
    print("  route_b and sorted agree numerically: yes")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-events", type=int, default=2)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--n-hashes", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--max-hits", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs a GPU: the flash arm has no CPU path.")
    device = "cuda"

    totals: dict[str, list[float]] = {}
    for i in range(args.num_events):
        coords = load_event_coords(args.config, args.split, i)
        if args.max_hits is not None:
            coords = coords[: args.max_hits]
        coords = sort_by_lsh(coords)
        print(f"\n=== event {i} ===")
        for key, value in report_event(coords, args, device).items():
            totals.setdefault(key, []).append(value)

    print("\n=== mean over events ===")
    for key, values in totals.items():
        print(f"  {key:<18}{sum(values) / len(values):>10.3f}")


if __name__ == "__main__":
    main()
