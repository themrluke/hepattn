"""fast_permute and slice_valid_targets must give bit-identical outputs and gradients to the indexing they replace.

"Bit-identical" is checked on the raw bits (so -0.0 and +0.0 count as different), with signed zeros planted in the
inputs and in the upstream gradients on purpose.
"""

import numpy as np
import pytest
import torch

from hepattn.models.fast_indexing import RowPermutation, take_prefix_rows, valid_prefix_length
from hepattn.models.loss import mask_focal_cost
from hepattn.models.task import ObjectHitMaskTask

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
INT_OF_SIZE = {2: torch.int16, 4: torch.int32, 8: torch.int64}


def assert_same_bits(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    a, b = a.detach().contiguous(), b.detach().contiguous()
    if a.dtype.is_floating_point:
        a, b = a.view(INT_OF_SIZE[a.element_size()]), b.view(INT_OF_SIZE[b.element_size()])
    assert torch.equal(a, b)


def randn_with_signed_zeros(shape, gen, dtype, device):
    """Random values with ~10% exact -0.0 and ~5% +0.0, where accumulation-based backwards behave differently."""
    x = torch.randn(shape, generator=gen)
    u = torch.rand(shape, generator=gen)
    x[u < 0.10] = -0.0
    x[(u >= 0.10) & (u < 0.15)] = 0.0
    return x.to(device=device, dtype=dtype)


SHAPES = [(1, 50, 7), (3, 50, 7), (2, 40), (2, 30, 4, 3)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_row_permutation_is_bit_identical_to_advanced_indexing(device, dtype, shape):
    gen = torch.Generator().manual_seed(0)
    x0 = randn_with_signed_zeros(shape, gen, dtype, device)
    upstream = randn_with_signed_zeros(shape, gen, dtype, device)
    # As the matcher returns it: an int32 CPU tensor, one permutation per batch entry
    idx = torch.stack([torch.randperm(shape[1], generator=gen) for _ in range(shape[0])]).to(torch.int32)

    # What MaskFormer._permute_outputs did before
    xa = x0.clone().requires_grad_()
    ya = xa[torch.arange(idx.shape[0], device=idx.device).unsqueeze(1), idx]
    ya.backward(upstream)

    xb = x0.clone().requires_grad_()
    yb = RowPermutation(idx, torch.device(device))(xb)
    yb.backward(upstream)

    assert_same_bits(ya, yb)
    assert_same_bits(xa.grad, xb.grad)
    assert yb.is_contiguous()


@pytest.mark.parametrize("device", DEVICES)
def test_row_permutation_falls_back_when_not_a_permutation(device):
    gen = torch.Generator().manual_seed(1)
    x0 = torch.randn(2, 20, 3, generator=gen).to(device)
    idx = torch.randint(0, 20, (2, 20), generator=gen, dtype=torch.int32)  # repeats: not a permutation
    permutation = RowPermutation(idx, torch.device(device))
    assert not permutation.is_permutation
    xa, xb = x0.clone().requires_grad_(), x0.clone().requires_grad_()
    ya = xa[torch.arange(2).unsqueeze(1), idx]
    yb = permutation(xb)
    ya.sum().backward()
    yb.sum().backward()
    assert_same_bits(ya, yb)
    assert_same_bits(xa.grad, xb.grad)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num", [0, 1, 17, 50])
def test_take_prefix_rows_is_bit_identical_to_boolean_indexing(device, dtype, num):
    gen = torch.Generator().manual_seed(2)
    x0 = randn_with_signed_zeros((1, 50, 9), gen, dtype, device)
    upstream = randn_with_signed_zeros((num, 9), gen, dtype, device)
    mask = (torch.arange(50) < num).unsqueeze(0).to(device)

    xa = x0.clone().requires_grad_()
    ya = xa[mask]
    ya.backward(upstream)

    xb = x0.clone().requires_grad_()
    yb = take_prefix_rows(xb, num)
    yb.backward(upstream)

    assert_same_bits(ya, yb)
    assert_same_bits(xa.grad, xb.grad)
    # Without gradients it is a plain slice with the same values
    assert_same_bits(x0[mask], take_prefix_rows(x0, num))


@pytest.mark.parametrize("device", DEVICES)
def test_valid_prefix_length(device):
    def mask(*bits):
        return torch.tensor([bits], dtype=torch.bool, device=device)

    assert valid_prefix_length(mask(1, 1, 1, 0, 0)) == 3
    assert valid_prefix_length(mask(0, 0, 0)) == 0
    assert valid_prefix_length(mask(1, 1, 1)) == 3
    assert valid_prefix_length(mask(1, 0, 1)) is None  # a gap: not a prefix
    assert valid_prefix_length(mask(0, 1, 1)) is None
    assert valid_prefix_length(torch.ones(2, 4, dtype=torch.bool, device=device)) is None  # batch of two


def make_mask_task():
    # The Pix0.6 config's track_hit_valid losses
    return ObjectHitMaskTask(
        name="track_hit_valid",
        input_constituent="hit",
        input_object="query",
        output_object="track",
        target_object="particle",
        losses={"mask_dice": 2.0, "mask_focal": 50.0},
        costs={"mask_dice": 2.0, "mask_focal": 25.0},
        dim=16,
        null_weight=1.0,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("num_particles", "num_queries"), [(30, 45), (45, 30), (60, 60), (0, 40)])
def test_mask_task_loss_with_valid_prefix_is_bit_identical(device, num_particles, num_queries):
    """ObjectHitMaskTask.loss with the prefix MaskFormer(slice_valid_targets=True) passes, against boolean indexing."""
    gen = torch.Generator().manual_seed(3)
    slots, hits = 60, 200
    dtype = torch.bfloat16 if device == "cuda" else torch.float32  # the mask logits are bf16 under autocast in training
    logits0 = randn_with_signed_zeros((1, slots, hits), gen, dtype, device) * 4
    targets = {
        "particle_hit_valid": (torch.rand(1, slots, hits, generator=gen) < 0.05).to(device),
        "particle_valid": (torch.arange(slots) < num_particles).unsqueeze(0).to(device),
        "query_mask": (torch.arange(slots) < num_queries).unsqueeze(0).to(device),
        "hit_valid": (torch.rand(1, hits, generator=gen) < 0.9).to(device),
    }
    num_valid = valid_prefix_length(targets["particle_valid"] & targets["query_mask"])
    assert num_valid == min(num_particles, num_queries)
    task = make_mask_task().to(device)

    results = []
    for extra in ({}, {"_object_valid_prefix": num_valid}):
        logits = logits0.clone().requires_grad_()
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            losses = task.loss({"track_hit_logit": logits}, {**targets, **extra})
        (losses["mask_dice"] + losses["mask_focal"]).backward()
        results.append((losses, logits.grad))

    (losses_a, grad_a), (losses_b, grad_b) = results
    for name in losses_a:
        assert_same_bits(losses_a[name], losses_b[name])
    if num_valid:
        assert_same_bits(grad_a, grad_b)


@pytest.mark.parametrize("device", DEVICES)
def test_focal_cost_single_product_agrees_with_two_products(device):
    """Not bit-identical (different rounding), but the same cost to within the matmul precision, and no less accurate."""
    gen = torch.Generator().manual_seed(4)
    logits = (torch.randn(1, 80, 3000, generator=gen) * 3).to(device)
    targets = (torch.rand(1, 60, 3000, generator=gen) < 0.003).float().to(device)
    pad = (torch.rand(1, 3000, generator=gen) < 0.95).to(device)

    torch.backends.cuda.matmul.allow_tf32 = True  # as with matmul_precision "high" in training
    try:
        two = mask_focal_cost(logits, targets, input_pad_mask=pad)
        one = mask_focal_cost(logits, targets, input_pad_mask=pad, single_product=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = False
    exact = mask_focal_cost(logits.double(), targets.double(), input_pad_mask=pad)

    scale = exact.abs().max()
    err_two = ((two.double() - exact).abs().max() / scale).item()
    err_one = ((one.double() - exact).abs().max() / scale).item()
    assert err_two < 2e-3
    assert err_one < 2e-3
    assert err_one <= 2 * err_two + 1e-7
    # Same optimal matching on these (tie-free) costs
    from scipy.optimize import linear_sum_assignment

    for cost_a, cost_b in ((two, one), (two, exact)):
        rows_a = linear_sum_assignment(cost_a[0].T.double().cpu().numpy())[1]
        rows_b = linear_sum_assignment(cost_b[0].T.double().cpu().numpy())[1]
        assert np.array_equal(rows_a, rows_b)
