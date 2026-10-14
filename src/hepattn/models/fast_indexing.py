"""Faster replacements for two advanced-indexing patterns in the MaskFormer loss, giving bit-identical results.

Advanced (and boolean) indexing backpropagates by accumulating the gradient into zeros
(`index_put_(..., accumulate=True)`), which sorts the indices first, and boolean masks also need their number of
True entries on the CPU (a wait for the GPU) every time. Here the index structure is known in advance (a permutation,
or a prefix of rows), so the backward is a plain gather or slice. The gradient is still added to zeros
(`0.0 + g`), so even the sign of zero matches: -0.0 becomes +0.0 exactly as with accumulation.
"""

import numpy as np
import torch
from torch import Tensor


class _PermuteRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, flat_idx: Tensor, flat_inv: Tensor) -> Tensor:  # noqa: ARG004
        b, p = x.shape[:2]
        ctx.save_for_backward(flat_inv)
        ctx.shape = x.shape
        return x.reshape(b * p, *x.shape[2:]).index_select(0, flat_idx).view(x.shape)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None, None]:
        (flat_inv,) = ctx.saved_tensors
        b, p = ctx.shape[:2]
        grad_x = grad.reshape(b * p, *ctx.shape[2:]).index_select(0, flat_inv).view(ctx.shape)
        # Same bits as accumulating into zeros: 0.0 + g (turns -0.0 into +0.0)
        return grad_x.add_(0.0), None, None


class RowPermutation:
    """Gather rows along dim 1 by per-batch permutations, like `x[arange(B)[:, None], idx]`, but with the backward done
    by the inverse permutation instead of an accumulating scatter.

    Built once from the matcher's indices (shape [batch, num]) and applied to every output of a layer.
    """

    def __init__(self, idx: Tensor, device: torch.device):
        idx_np = idx.detach().cpu().numpy().astype(np.int64)
        batch, num = idx_np.shape
        rows = np.arange(num, dtype=np.int64)
        self.is_permutation = bool((np.sort(idx_np, axis=1) == rows).all())
        self.idx = idx
        if not self.is_permutation:
            return
        offsets = (np.arange(batch, dtype=np.int64) * num)[:, None]
        inv = np.empty_like(idx_np)
        np.put_along_axis(inv, idx_np, np.broadcast_to(rows, idx_np.shape), axis=1)
        self.flat_idx = torch.from_numpy((idx_np + offsets).ravel()).to(device)
        self.flat_inv = torch.from_numpy((inv + offsets).ravel()).to(device)

    def __call__(self, x: Tensor) -> Tensor:
        if not self.is_permutation:
            # Not a permutation (never from the Hungarian matcher): plain advanced indexing
            idx = self.idx.to(x.device)
            return x[torch.arange(idx.shape[0], device=x.device).unsqueeze(1), idx]
        return _PermuteRows.apply(x, self.flat_idx, self.flat_inv)


class _TakePrefixRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, num: int) -> Tensor:
        ctx.shape, ctx.num = x.shape, num
        return x[0, :num].clone()

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None]:
        grad_x = grad.new_zeros(ctx.shape)
        grad_x[0, : ctx.num].add_(grad)
        return grad_x, None


def take_prefix_rows(x: Tensor, num: int) -> Tensor:
    """`x[mask]` for a batch of one whose mask (over dim 1) is True for exactly the first `num` rows: the same values,
    shape [num, ...] and gradient, without the boolean-indexing kernels or the wait for the mask's count."""
    if not x.requires_grad:
        return x[0, :num]
    return _TakePrefixRows.apply(x, num)


def valid_prefix_length(mask: Tensor) -> int | None:
    """The number of leading True entries if `mask` (shape [1, num]) is True for exactly those, else None.

    One wait for the GPU, instead of one per boolean-mask indexing.
    """
    if mask.dim() != 2 or mask.shape[0] != 1:
        return None
    mask = mask[0].bool()
    count = mask.sum()
    is_prefix = (mask == (torch.arange(mask.shape[0], device=mask.device) < count)).all()
    count, is_prefix = torch.stack([count, is_prefix.long()]).tolist()
    return int(count) if is_prefix else None
