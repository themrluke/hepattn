from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature


def per_head_window_mask_mod(ranks: Tensor, window_size: int, kv_valid: Tensor | None = None) -> _mask_mod_signature:
    """Build a flex ``mask_mod`` for a per-head sliding window over LSH ranks.

    Args:
        ranks: ``(H, N)`` integer rank table, where ``H = num_heads * n_hashes`` is the
            flattened ``(head, hash)`` grid and ``ranks[h, t]`` is token ``t``'s position
            in cell ``h``'s ordering. Must live on the same device as the attention
            inputs (the closure indexes it under ``vmap`` during block-mask creation).
        window_size: Full window width; a query attends to keys within ``window_size //
            2`` rank positions either side (matching :func:`sliding_window_mask`).
        kv_valid: ``(N,)`` bool, False for padded slots, indexed by *token* because this
            implementation never moves the tokens. Sorting padding to the tail of every
            ordering is not enough on its own: the real tokens holding the highest ranks
            are still within half a window of it, so it has to be excluded here as well.

    Returns:
        A ``mask_mod(b, h, q_idx, kv_idx)`` closure suitable for ``create_block_mask``.
        The diagonal (``q == kv``, so ``|Δrank| == 0``) is always admitted, so every
        query keeps at least itself as a key and no row is fully masked.
    """
    half = window_size // 2

    def mask_mod(b, h, q_idx, kv_idx) -> Tensor:  # noqa: ARG001
        delta = ranks[h, q_idx] - ranks[h, kv_idx]
        admit = (delta <= half) & (-delta <= half)
        if kv_valid is None:
            return admit
        # A padded query keeps itself, so its softmax row is never empty and its lse stays
        # finite. Its output is meaningless but is discarded downstream; without this the
        # row would be all -inf and produce NaNs that spread through the merge.
        return admit & (kv_valid[kv_idx] | (q_idx == kv_idx))

    return mask_mod  # Returns tensor of Boolean values


def sorted_window_mask_mod(window_size: int, num_valid: Tensor | None = None) -> _mask_mod_signature:
    """Plain banded window over *position*, for the sorted OR implementation.

    Once each cell's tokens have been gathered into that cell's own ordering, position is
    rank, so the per-head rank window of :func:`per_head_window_mask_mod` collapses to an
    ordinary banded window that mentions no ordering at all -- the same rule for every
    cell, so one mask serves all of them.

    Args:
        window_size: Full window width; half either side.
        num_valid: Scalar tensor holding the number of real tokens, or None if there is no
            padding. Padded tokens take the last ranks of *every* cell (see
            :func:`hepattn.models.ordering.lsh_order_values`), so after the permutation
            they occupy the same trailing block of positions in all of them -- which is
            exactly why "not padding" can be written as ``kv_idx < num_valid`` here and
            still be shared across cells. Pass a tensor rather than an int so the closure
            can be reused as the sequence length changes without recompiling.

    Returns:
        A ``mask_mod(b, h, q_idx, kv_idx)`` closure for ``create_block_mask``, built with
        ``H=None``; flex broadcasts it across every head.
    """
    half = window_size // 2

    def mask_mod(b, h, q_idx, kv_idx) -> Tensor:  # noqa: ARG001
        admit = (q_idx - kv_idx <= half) & (kv_idx - q_idx <= half)
        if num_valid is None:
            return admit
        return admit & ((kv_idx < num_valid[0]) | (q_idx == kv_idx))

    return mask_mod
