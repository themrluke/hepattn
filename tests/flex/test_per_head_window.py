"""These checks evaluate the ``mask_mod`` closure directly over an index grid, so they
run on CPU without flex/compile -- the closure is just tensor indexing. The GPU
``create_block_mask`` integration is exercised by the encoder wiring.
"""

import pytest
import torch

from hepattn.flex.per_head_window import per_head_window_mask_mod


def _eval_mask(ranks: torch.Tensor, window: int) -> torch.Tensor:
    """Materialise the (H, N, N) boolean mask the closure defines."""
    h_dim, n = ranks.shape
    mask_mod = per_head_window_mask_mod(ranks, window)
    h = torch.arange(h_dim).view(h_dim, 1, 1)
    q = torch.arange(n).view(1, n, 1)
    kv = torch.arange(n).view(1, 1, n)
    return mask_mod(None, h, q, kv)


def test_matches_abs_rank_difference():
    torch.manual_seed(0)
    h_dim, n, window = 4, 32, 8
    ranks = torch.stack([torch.randperm(n) for _ in range(h_dim)])

    got = _eval_mask(ranks, window)
    expected = (ranks[:, :, None] - ranks[:, None, :]).abs() <= window // 2
    assert torch.equal(got, expected)


def test_diagonal_always_admitted():
    # Every query keeps itself as a key (|Δrank| == 0), so no row is fully masked.
    torch.manual_seed(1)
    ranks = torch.stack([torch.randperm(16) for _ in range(3)])
    got = _eval_mask(ranks, window=4)
    diag = got[:, torch.arange(16), torch.arange(16)]
    assert diag.all()


def test_symmetric():
    # |rank[q] - rank[kv]| is symmetric, so the admitted set is too.
    torch.manual_seed(2)
    ranks = torch.stack([torch.randperm(24) for _ in range(5)])
    got = _eval_mask(ranks, window=10)
    assert torch.equal(got, got.transpose(-1, -2))


def test_per_head_independence():
    # Distinct orderings per head give distinct masks (the whole point of Route B).
    # NB a reversed ordering is *not* distinct -- |Δrank| is invariant under reversal --
    # so use a genuinely shuffled permutation against the identity.
    torch.manual_seed(3)
    ranks = torch.stack([torch.arange(10), torch.randperm(10)])
    got = _eval_mask(ranks, window=4)
    assert not torch.equal(got[0], got[1])


@pytest.mark.parametrize("window", [2, 6, 100])
def test_window_width(window):
    # Along the identity ordering, a query reaches exactly the tokens within window//2.
    n = 40
    ranks = torch.arange(n).unsqueeze(0)  # single head, rank == index
    got = _eval_mask(ranks, window)[0]
    counts = got.sum(-1)
    half = window // 2
    expected = torch.tensor([min(i, half) + min(n - 1 - i, half) + 1 for i in range(n)])
    assert torch.equal(counts, expected)
