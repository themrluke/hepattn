from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature


def per_head_window_mask_mod(ranks: Tensor, window_size: int) -> _mask_mod_signature:
    """Build a flex ``mask_mod`` for a per-head sliding window over LSH ranks.

    Args:
        ranks: ``(H, N)`` integer rank table, where ``H = num_heads * n_hashes`` is the
            flattened ``(head, hash)`` grid and ``ranks[h, t]`` is token ``t``'s position
            in cell ``h``'s ordering. Must live on the same device as the attention
            inputs (the closure indexes it under ``vmap`` during block-mask creation).
        window_size: Full window width; a query attends to keys within ``window_size //
            2`` rank positions either side (matching :func:`sliding_window_mask`).

    Returns:
        A ``mask_mod(b, h, q_idx, kv_idx)`` closure suitable for ``create_block_mask``.
        The diagonal (``q == kv``, so ``|Δrank| == 0``) is always admitted, so every
        query keeps at least itself as a key and no row is fully masked.
    """
    half = window_size // 2

    def mask_mod(b, h, q_idx, kv_idx) -> Tensor:  # noqa: ARG001
        delta = ranks[h, q_idx] - ranks[h, kv_idx]
        return (delta <= half) & (-delta <= half)

    return mask_mod  # Returns tensor of Boolean values
