---
name: route-b-per-head-ordering
description: Per-head ordering (Route B) MaskFormer work lives on the OR-amplification branch
metadata: 
  node_type: memory
  type: project
  originSessionId: 194cdad1-1dc1-49da-8d60-11018d3c1bb3
  modified: 2026-08-06T09:08:13.318Z
---

The per-head ordering / OR-amplification MaskFormer feature work is done on the git branch **`OR-amplification`** (exists locally and on origin). The user asked (2026-07-30) to stay on this branch for the whole session — do not switch away.

Chosen design is **Route B**: keep the sliding window; mask in place with per-head rank tables via a `flex_attention` `mask_mod` (`|rank_h[q]-rank_h[kv]| < W`), never physically permuting tokens (so no un-sort needed). The alternative, Route A (gather into buckets + dense attention, block-sparse only), was rejected because it can't keep a sliding window.

Implementation steps: (1) E2LSH ordering grid `(num_heads × n_hashes)` frozen at construction + `rank_h` inverse-permutation tables; (2) OR axis + LSE merge via `return_lse=True` (without this the extra orderings buy nothing); (3) Route B `mask_mod`; (4) flex backend with H-dim `BlockMask`; (5) padding via the `inf`-hash trick, not a `kv_mask`. The frozen LSH must reproduce the benchmark's `lsh_k` seeding (`np.random.default_rng(k)` per table; `lsh K=3` = one head, 3 hashes) so the model matches the ordering study in `scripts/benchmark_orderings.py` / `scripts/orderings.py`.

**Progress (as of 2026-08-03):**
- Step 1 DONE — `src/hepattn/models/ordering.py` (`E2LSHOrderingGrid`, `draw_lsh_table`, `lsh_order_values`); benchmark `scripts/orderings.py` refactored to delegate.
- Step 3 DONE — `src/hepattn/flex/per_head_window.py::per_head_window_mask_mod(ranks, window_size)`: admits `|ranks[h,q]-ranks[h,kv]| <= window//2`; `ranks` is `(H=num_heads*n_hashes, N)`, head-major flatten (`h = head*n_hashes + hash`); no wrapped variant (LSH is non-periodic). Tests: `tests/flex/test_per_head_window.py`.
- Step 2 DONE — `src/hepattn/models/attention.py`: `expand_heads_for_or(x, n_hashes)` (head-major replicate to `H*C` via `expand`) + `or_merge_lse(out, lse, n_hashes)` (softmax(lse) blend = exact softmax attn over the MULTISET union of per-hash windows). New opt-in `Attention.forward(..., or_n_hashes=…)` flex branch: expand q/k/v → one `flex_attention(..., return_lse=True)` → merge. `or_n_hashes=None` default = byte-for-byte old path. Tests: `tests/models/test_or_merge.py`. All 11 new tests pass; ruff clean; no regression to torch-backend attention tests (4 flash failures are pre-existing pre-Ampere-GPU).
- Steps 1-3 committed as `090aaef` (2026-08-05). Write-up: `src/hepattn/experiments/colliderml/notes/per_head_ordering.md`.

**Progress (as of 2026-08-05):**
- Step 4a DONE — `scripts/measure_block_sparsity.py`. Verdict: **Route B is viable.** Real events N~13k, window 512, 24 cells: block density 0.24 (lsh0 layout) vs 0.05 banded baseline vs 0.04 element-ideal — i.e. ~4.3x faster than dense, ~5x more block-work than today's window. Not the feared 1.0/dense. Token layout matters: raw 0.49 > phi 0.30 > lsh0 0.24.
- **OPEN DESIGN FORK (2026-08-06): a third option, "sort each head's q/k/v into its own ordering + one shared banded window", currently looks ~4-5x cheaper than Route B and computes the identical result.** Not the rejected Route A (that was bucketing, which breaks the sliding window). Block-size sweep confirms the gap is 4.0-4.9x at every `BLOCK_SIZE`, so tiling cannot close it. Steps 4.1-4.6 are common to both; the fork is at 4.8. Full analysis in §0b of the plan note — do not let this drop.
**Progress (as of 2026-09-08):**
- Steps 4.1-4.7 DONE (uncommitted, in `encoder.py`): config args, four guard asserts, per-layer `E2LSHOrderingGrid` ModuleList, `x_coords` accepted in `forward`, coords gathered alongside `x`/`kv_mask` in the sort block, and the sliding-window mask cache bypassed via `if self.ordering_grids is None:`. **Next step is 4.8** (per-layer BlockMask in the layer loop) — the fork point with the sorted variant.
- Also fixed a pre-existing blocker: `Encoder(attn_type="flex", window_size=...)` used to raise, because the encoder forwarded `window_size` into `Attention`, which rejects it for non-flash. This was the `attention.py:275` TODO. Two `set_backend` bugs remain deferred (§2b of the plan note).
- Both notes refreshed 2026-09-08 with current line numbers (§0c of the plan note is the authority; inline numbers go stale every step).

- **Steps 4-5 are being done in TEACHING MODE at the user's request (2026-08-05): Luke writes all the code himself, Claude gives one small numbered step at a time with the reasoning and an observable check. Do NOT write the implementation for him.** The full reference answers + 12-step sequence live in `src/hepattn/experiments/colliderml/notes/step4-5-plan.md` — read that file before giving the next step.
