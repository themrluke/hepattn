"""OR-of-orderings LSE merge and head replication (step 2 of the OR-amplification work).

The merge is validated against an explicit reference: softmax attention over the
*multiset* union of the per-hash windows. Everything here is pure tensor algebra, so
it runs on CPU without flex/compile.
"""

import torch

from hepattn.models.attention import expand_heads_for_or, or_merge_lse


def test_expand_heads_layout():
    # Head `head` must land in rows [head*C : (head+1)*C] (head-major), matching the
    # (num_heads, n_hashes) rank-grid flattening (h = head*C + hash).
    b, h, n, d, c = 1, 3, 5, 4, 2
    x = torch.arange(h, dtype=torch.float32).view(1, h, 1, 1).expand(b, h, n, d).clone()
    out = expand_heads_for_or(x, c)
    assert out.shape == (b, h * c, n, d)
    for head in range(h):
        for hash_i in range(c):
            assert torch.equal(out[:, head * c + hash_i], x[:, head])


def _reference_multiset_attention(scores, values, masks):
    """Softmax attention over the multiset union of per-hash admitted keys.

    scores: (N, N), values: (N, Dh), masks: (C, N, N) bool. Returns (N, Dh).
    """
    n = scores.shape[0]
    out = torch.empty(n, values.shape[1])
    for i in range(n):
        sel_scores = torch.cat([scores[i][masks[c, i]] for c in range(masks.shape[0])])
        sel_values = torch.cat([values[masks[c, i]] for c in range(masks.shape[0])])
        w = torch.softmax(sel_scores, dim=0)
        out[i] = w @ sel_values
    return out


def _per_hash_flex_outputs(scores, values, masks):
    """Emulate flex's per-hash (out, lse) for a single head, shape (1, C, N, Dh)/(1, C, N)."""
    c, n, _ = masks.shape
    d = values.shape[1]
    out = torch.zeros(c, n, d)
    lse = torch.full((c, n), float("-inf"))
    for ci in range(c):
        for i in range(n):
            sel = masks[ci, i]
            s = scores[i][sel]
            lse[ci, i] = torch.logsumexp(s, dim=0)
            out[ci, i] = torch.softmax(s, dim=0) @ values[sel]
    return out.unsqueeze(0), lse.unsqueeze(0)


def test_lse_merge_equals_multiset_union():
    torch.manual_seed(0)
    n, d, c, window = 30, 8, 3, 6

    q = torch.randn(n, d)
    k = torch.randn(n, d)
    values = torch.randn(n, d)
    scores = (q @ k.T) / d**0.5

    # Distinct random orderings -> distinct windows (with the diagonal always in).
    ranks = torch.stack([torch.randperm(n) for _ in range(c)])
    masks = (ranks[:, :, None] - ranks[:, None, :]).abs() <= window // 2

    out, lse = _per_hash_flex_outputs(scores, values, masks)
    merged = or_merge_lse(out, lse, c)  # (1, 1, N, Dh)

    expected = _reference_multiset_attention(scores, values, masks)
    assert torch.allclose(merged[0, 0], expected, atol=1e-6)


def test_lse_merge_identical_orderings_is_a_noop():
    # If every hash has the same window, the multiset union just triples every key
    # uniformly, so the merge reproduces a single-ordering attention.
    torch.manual_seed(1)
    n, d, c, window = 20, 8, 3, 6

    scores = torch.randn(n, n)
    values = torch.randn(n, d)
    ranks = torch.arange(n).unsqueeze(0).expand(c, n)
    masks = (ranks[:, :, None] - ranks[:, None, :]).abs() <= window // 2

    out, lse = _per_hash_flex_outputs(scores, values, masks)
    merged = or_merge_lse(out, lse, c)
    assert torch.allclose(merged[0, 0], out[0, 0], atol=1e-6)


def test_lse_merge_multi_head_batch_shapes():
    # Shape/independence check: heads and batch must not leak into the merge.
    torch.manual_seed(2)
    b, h, c, n, d = 2, 4, 3, 12, 8
    out = torch.randn(b, h * c, n, d)
    lse = torch.randn(b, h * c, n)
    merged = or_merge_lse(out, lse, c)
    assert merged.shape == (b, h, n, d)

    # Merging head `head` alone matches the corresponding slice of the full merge.
    head = 2
    sl = slice(head * c, (head + 1) * c)
    single = or_merge_lse(out[:, sl], lse[:, sl], c)
    assert torch.allclose(single[:, 0], merged[:, head], atol=1e-6)
