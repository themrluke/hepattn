"""Encoder wiring for the OR-amplification per-head ordering (step 4).

The load-bearing test here is :func:`test_identity_ranks_match_plain_sliding_window`.
Everything else in the step-4 wiring only ever showed that shapes are right and outputs
are finite, which a completely wrong mask would also satisfy. Forcing a single hash whose
ordering is the identity collapses Route B to *exactly* the plain sliding window --
``|rank[q] - rank[kv]|`` becomes ``|q - kv|``, and both mask_mods use the same
``<= window_size // 2`` convention -- so the two paths must agree numerically.

Runs on CPU (flex attention has a CPU path), so it is not marked ``gpu`` and therefore
does run in CI. It is slow: nearly all the wall time is ``torch.compile`` warm-up.
"""

import torch
from torch import Tensor, nn

from hepattn.models import encoder as encoder_module
from hepattn.models.encoder import Encoder

DIM = 32
SEQ_LEN = 256
WINDOW = 64


class _IdentityRankGrid(nn.Module):
    """Stand-in for ``E2LSHOrderingGrid`` whose every cell orders tokens by index.

    Matches the real grid's contract: takes ``(N, 2)`` coords, returns
    ``(num_heads, n_hashes, N)`` ranks, which ``Encoder.forward`` flattens to ``(H, N)``.
    """

    def __init__(self, num_heads: int, n_hashes: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.n_hashes = n_hashes

    def forward(self, coords: Tensor) -> Tensor:
        num_tokens = coords.shape[-2]
        return torch.arange(num_tokens, device=coords.device).expand(self.num_heads, self.n_hashes, num_tokens)


def test_identity_ranks_match_plain_sliding_window():
    torch.manual_seed(0)

    route_b = Encoder(num_layers=1, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=1)
    plain = Encoder(num_layers=1, dim=DIM, attn_type="flex", window_size=WINDOW)

    # Same weights in both. The two configs need not consume the RNG identically, so copy
    # rather than reseed. The only extra state Route B carries is the frozen grid buffers.
    missing, unexpected = plain.load_state_dict(route_b.state_dict(), strict=False)
    assert not missing, f"plain encoder is missing weights: {missing}"
    assert all("ordering_grids" in key for key in unexpected), f"unexpected non-grid keys: {unexpected}"

    num_heads = route_b.layers[0].attn.fn.num_heads
    route_b.ordering_grids[0] = _IdentityRankGrid(num_heads, n_hashes=1)

    route_b.eval()
    plain.eval()
    x = torch.randn(1, SEQ_LEN, DIM)
    # Coords are ignored by the stub but the forward asserts they are present.
    coords = torch.randn(1, SEQ_LEN, 2)

    with torch.no_grad():
        got = route_b(x, x_coords=coords)
        expected = plain(x)

    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)


class _HeadZeroIdentityRankGrid(nn.Module):
    """Grid stub where head 0 orders by index and every other head is shuffled.

    Both of head 0's hashes get the identity, so their merge collapses to that single
    window (equal ``lse`` -> 0.5/0.5 weights -> ``0.5*out + 0.5*out == out``) and head 0
    must reproduce the plain sliding window exactly. The other heads get a fixed shuffle,
    drawn once here rather than per call so the test cannot flake on a permutation that
    happens to sit close to the identity.
    """

    def __init__(self, num_heads: int, n_hashes: int, num_tokens: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.n_hashes = n_hashes
        generator = torch.Generator().manual_seed(1234)
        rows = [torch.arange(num_tokens).expand(n_hashes, num_tokens)]
        rows += [torch.randperm(num_tokens, generator=generator).expand(n_hashes, num_tokens) for _ in range(num_heads - 1)]
        self.register_buffer("ranks", torch.stack(rows))  # (num_heads, n_hashes, N)

    def forward(self, coords: Tensor) -> Tensor:
        return self.ranks.to(coords.device)


def _capture_per_head_output(encoder: Encoder) -> tuple[list, object]:
    """Hook ``out_proj`` to grab its input, which is still laid out head-by-head.

    ``recombine_heads`` is ``transpose(-3, -2)`` then ``flatten(-2)``, so head ``h``
    occupies columns ``[h*head_dim : (h+1)*head_dim]``. That is the last point before
    ``out_proj`` mixes the heads together.
    """
    captured: list[Tensor] = []

    def pre_hook(module, args):  # noqa: ARG001
        captured.append(args[0].detach())

    handle = encoder.layers[0].attn.fn.out_proj.register_forward_pre_hook(pre_hook)
    return captured, handle


def _head_slice(packed: Tensor, head: int, head_dim: int) -> Tensor:
    return packed[..., head * head_dim : (head + 1) * head_dim]


def test_head_major_layout_pairs_each_head_with_its_own_ranks():
    # With n_hashes == 1 the head-major and hash-major flattenings coincide, so test 1
    # cannot tell them apart. Two hashes plus heads that are distinguishable can: under
    # hash-major, flattened row 1 would carry head 1's ordering while still receiving
    # head 0's q/k/v, and head 0 would stop matching the plain window.
    torch.manual_seed(0)
    n_hashes = 2

    route_b = Encoder(num_layers=1, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=n_hashes)
    plain = Encoder(num_layers=1, dim=DIM, attn_type="flex", window_size=WINDOW)

    missing, unexpected = plain.load_state_dict(route_b.state_dict(), strict=False)
    assert not missing, f"plain encoder is missing weights: {missing}"
    assert all("ordering_grids" in key for key in unexpected), f"unexpected non-grid keys: {unexpected}"

    num_heads = route_b.layers[0].attn.fn.num_heads
    head_dim = DIM // num_heads
    route_b.ordering_grids[0] = _HeadZeroIdentityRankGrid(num_heads, n_hashes, SEQ_LEN)

    route_b.eval()
    plain.eval()
    x = torch.randn(1, SEQ_LEN, DIM)
    coords = torch.randn(1, SEQ_LEN, 2)

    got_heads, got_handle = _capture_per_head_output(route_b)
    want_heads, want_handle = _capture_per_head_output(plain)
    try:
        with torch.no_grad():
            route_b(x, x_coords=coords)
            plain(x)
    finally:
        got_handle.remove()
        want_handle.remove()

    got, want = got_heads[0], want_heads[0]

    # Head 0 ordered by index in both hashes, so it is the plain sliding window.
    torch.testing.assert_close(_head_slice(got, 0, head_dim), _head_slice(want, 0, head_dim), rtol=1e-4, atol=1e-5)

    # Guard against passing vacuously: a mask that admitted everything for every head
    # would make head 0 match for the wrong reason, so at least one shuffled head must
    # visibly disagree with the plain window.
    differs = [
        head
        for head in range(1, num_heads)
        if not torch.allclose(_head_slice(got, head, head_dim), _head_slice(want, head, head_dim), rtol=1e-4, atol=1e-5)
    ]
    assert differs, "every head matched the plain window; the shuffled orderings had no effect"


def test_real_grids_build_one_distinct_mask_per_layer():
    # Smoke test with the real E2LSHOrderingGrid rather than a stub: the wiring must
    # survive real coordinates, and each layer must build its *own* BlockMask (the whole
    # point of one grid per layer). `x_sort_value` is supplied because a sliding window
    # over token index is only meaningful once the sequence carries geometric locality --
    # unsorted, the mask comes out fully dense and the feature does nothing.
    torch.manual_seed(0)
    num_layers, n_hashes = 3, 3

    encoder = Encoder(num_layers=num_layers, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=n_hashes).eval()
    num_heads = encoder.layers[0].attn.fn.num_heads

    x = torch.randn(1, SEQ_LEN, DIM)
    coords = torch.randn(1, SEQ_LEN, 2)

    # create_block_mask is called through the encoder module namespace, so patch it there
    # rather than hooking a module -- there is no nn.Module to hang a hook on.
    built = []
    real_create_block_mask = encoder_module.create_block_mask

    def spy(*args, **kwargs):
        mask = real_create_block_mask(*args, **kwargs)
        built.append(mask)
        return mask

    encoder_module.create_block_mask = spy
    try:
        with torch.no_grad():
            out = encoder(x, x_sort_value=coords[..., 1], x_coords=coords)
    finally:
        encoder_module.create_block_mask = real_create_block_mask

    assert out.shape == (1, SEQ_LEN, DIM)
    assert torch.isfinite(out).all(), "Route B produced non-finite activations"

    assert len(built) == num_layers, f"expected one BlockMask per layer, got {len(built)}"
    for mask in built:
        # Height is the flattened (head, hash) grid, not 1 -- H=None here would silently
        # collapse every cell onto a single shared mask.
        assert mask.shape[1] == num_heads * n_hashes, mask.shape

    # Each layer must draw its own orderings. Assert that on the rank tables themselves
    # rather than on the masks: a BlockMask only records which 128x128 tiles survive, so
    # at small N every mask is fully dense and three genuinely different orderings are
    # indistinguishable at block granularity. The ranks are where the independence lives.
    ranks = [encoder.ordering_grids[i](coords[0]).flatten(0, 1) for i in range(num_layers)]
    assert not any(torch.equal(ranks[0], other) for other in ranks[1:]), (
        "layers produced identical rank tables; the per-layer grids are not independent"
    )
