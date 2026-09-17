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

import pytest
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

    def forward(self, coords: Tensor, valid: Tensor | None = None) -> Tensor:
        # `valid` is accepted to match E2LSHOrderingGrid's signature and ignored: these tests
        # have no padding, and the identity ordering is the point.
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

    def forward(self, coords: Tensor, valid: Tensor | None = None) -> Tensor:
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


def test_masked_impl_builds_one_distinct_mask_per_layer():
    # Pinned to or_impl="masked": building a mask per layer is that implementation's defining
    # behaviour, and the reason it is ~12x more expensive. The sorted default shares one
    # position-only mask across every layer and cell, so these assertions do not apply to it.
    #
    # Otherwise a smoke test with the real E2LSHOrderingGrid rather than a stub: the wiring must
    # survive real coordinates. `x_sort_value` is supplied because a sliding window over token
    # index is only meaningful once the sequence carries geometric locality -- unsorted, the mask
    # comes out fully dense and the feature does nothing.
    torch.manual_seed(0)
    num_layers, n_hashes = 3, 3

    encoder = Encoder(num_layers=num_layers, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=n_hashes, or_impl="masked").eval()
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


def test_sorted_and_masked_implementations_agree():
    """The two OR implementations must be interchangeable, not merely similar.

    They admit exactly the same query/key pairs -- "within W//2 of each other in this cell's
    ordering" -- so they compute the same attention by two routes: one permutes the tokens into
    that ordering and applies a plain banded window, the other leaves them alone and encodes the
    ordering in the mask. Any disagreement is a bug.

    This is the test that makes the choice of default safe: `sorted` is the one that runs, and
    `masked` is the reference it is checked against.
    """
    torch.manual_seed(0)
    n_hashes = 3

    sorted_enc = Encoder(num_layers=2, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=n_hashes, or_impl="sorted")
    masked_enc = Encoder(num_layers=2, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=n_hashes, or_impl="masked")

    missing, unexpected = masked_enc.load_state_dict(sorted_enc.state_dict(), strict=False)
    assert not missing, f"masked encoder is missing weights: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"

    sorted_enc.eval()
    masked_enc.eval()
    x = torch.randn(1, SEQ_LEN, DIM)
    coords = torch.randn(1, SEQ_LEN, 2)

    with torch.no_grad():
        got = sorted_enc(x, x_sort_value=coords[..., 1], x_coords=coords)
        expected = masked_enc(x, x_sort_value=coords[..., 1], x_coords=coords)

    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)


def test_sorted_is_the_default_and_keeps_the_shared_mask_cache():
    # The default matters: sorted is ~12x cheaper end to end. It is also the only OR path that can
    # use the encoder's position-only mask cache, which is where most of that saving comes from --
    # masked must rebuild a taller mask every layer of every event.
    default_enc = Encoder(num_layers=1, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=1)
    assert default_enc.or_impl == "sorted"

    x = torch.randn(1, SEQ_LEN, DIM)
    coords = torch.randn(1, SEQ_LEN, 2)
    with torch.no_grad():
        default_enc(x, x_sort_value=coords[..., 1], x_coords=coords)
    assert default_enc.mask_mod is not None, "sorted should populate the shared banded mask cache"

    masked_enc = Encoder(num_layers=1, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=1, or_impl="masked")
    with torch.no_grad():
        masked_enc(x, x_sort_value=coords[..., 1], x_coords=coords)
    assert masked_enc.mask_mod is None, "masked must not use the shared cache"


def test_unknown_or_impl_is_rejected():
    with pytest.raises(AssertionError, match="or_impl"):
        Encoder(num_layers=1, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=1, or_impl="banana")


def _padded_inputs(num_real: int, num_pad: int):
    """A batch whose last ``num_pad`` slots are padding, with the coords a collate would leave."""
    total = num_real + num_pad
    x = torch.randn(1, total, DIM)
    coords = torch.randn(1, total, 2)
    coords[0, num_real:] = 0.0  # padded slots typically arrive as zeros
    kv_mask = torch.zeros(1, total, dtype=torch.bool)
    kv_mask[0, :num_real] = True
    return x, coords, kv_mask


@pytest.mark.parametrize("or_impl", ["sorted", "masked"])
def test_padding_does_not_change_the_answer_for_real_tokens(or_impl):
    """Padding a batch must leave the real hits' outputs untouched.

    This is the property that matters and the one that is easy to get subtly wrong. Padded
    coordinates leak into the ordering through two routes -- the projection's span, and the
    equal-occupancy quantile bins -- and either one shifts where the bin edges fall, which
    reorders *genuine* hits. Nothing downstream would notice: no error, no NaN, just a slightly
    worse model. So the test runs the same real hits with and without padding attached and
    demands the same numbers.
    """
    torch.manual_seed(0)
    num_real, num_pad = 200, 56
    x, coords, kv_mask = _padded_inputs(num_real, num_pad)

    encoder = Encoder(num_layers=1, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=2, or_impl=or_impl).eval()

    with torch.no_grad():
        padded = encoder(x, x_sort_value=coords[..., 1], kv_mask=kv_mask, x_coords=coords)
        unpadded = encoder(x[:, :num_real], x_sort_value=coords[:, :num_real, 1], x_coords=coords[:, :num_real])

    assert torch.isfinite(padded).all(), "padding produced non-finite activations"
    torch.testing.assert_close(padded[:, :num_real], unpadded, rtol=1e-4, atol=1e-5)


def test_padded_queries_keep_a_finite_softmax_row():
    # A padded query has no valid keys, so without admitting the diagonal its whole row would be
    # -inf: an empty softmax, a NaN output and a NaN lse that then spreads through the merge into
    # the *real* tokens. The mask_mods always keep q == kv for exactly this reason.
    torch.manual_seed(0)
    x, coords, kv_mask = _padded_inputs(num_real=64, num_pad=192)  # mostly padding, worst case

    encoder = Encoder(num_layers=2, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=2).eval()
    with torch.no_grad():
        out = encoder(x, x_sort_value=coords[..., 1], kv_mask=kv_mask, x_coords=coords)

    assert torch.isfinite(out).all(), "a padded query produced NaN and it spread"


@pytest.mark.parametrize("or_impl", ["sorted", "masked"])
def test_a_second_event_is_not_masked_with_the_first_event_s_padding(or_impl):
    """Consecutive events with different amounts of padding must each get their own mask.

    The encoder caches its position-only ``mask_mod`` and the sorted path's closure has to know
    how many tokens are real. If that count is captured by value when the closure is built, every
    event after the first is masked with the *first* one's padding -- attending to junk, or
    ignoring real hits, depending on which way the count moved. Nothing raises.

    Every other padding test in this file runs one event per encoder, so none of them can see it.
    """
    torch.manual_seed(0)
    encoder = Encoder(num_layers=1, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=2, or_impl=or_impl).eval()

    first = _padded_inputs(num_real=200, num_pad=56)
    second = _padded_inputs(num_real=60, num_pad=196)

    reference = Encoder(num_layers=1, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=2, or_impl=or_impl).eval()
    reference.load_state_dict(encoder.state_dict())

    with torch.no_grad():
        x, coords, kv_mask = first
        encoder(x, x_sort_value=coords[..., 1], kv_mask=kv_mask, x_coords=coords)

        x, coords, kv_mask = second
        after_first = encoder(x, x_sort_value=coords[..., 1], kv_mask=kv_mask, x_coords=coords)
        on_its_own = reference(x, x_sort_value=coords[..., 1], kv_mask=kv_mask, x_coords=coords)

    torch.testing.assert_close(after_first[:, :60], on_its_own[:, :60], rtol=1e-4, atol=1e-5)


def test_padding_appearing_only_on_a_later_event_is_still_excluded():
    # The mirror image: the first event has no padding at all, so if the validity cut is only
    # wired up when a kv_mask happens to be present the closure is built without one and every
    # later padded event silently attends its padding.
    torch.manual_seed(0)
    encoder = Encoder(num_layers=1, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=2).eval()

    unpadded_x = torch.randn(1, SEQ_LEN, DIM)
    unpadded_coords = torch.randn(1, SEQ_LEN, 2)
    x, coords, kv_mask = _padded_inputs(num_real=100, num_pad=SEQ_LEN - 100)

    reference = Encoder(num_layers=1, dim=DIM, attn_type="flex", window_size=WINDOW, or_n_hashes=2).eval()
    reference.load_state_dict(encoder.state_dict())

    with torch.no_grad():
        encoder(unpadded_x, x_sort_value=unpadded_coords[..., 1], x_coords=unpadded_coords)
        after_unpadded = encoder(x, x_sort_value=coords[..., 1], kv_mask=kv_mask, x_coords=coords)
        on_its_own = reference(x, x_sort_value=coords[..., 1], kv_mask=kv_mask, x_coords=coords)

    torch.testing.assert_close(after_unpadded[:, :100], on_its_own[:, :100], rtol=1e-4, atol=1e-5)
