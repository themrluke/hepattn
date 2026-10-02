# Steps 4-5 — teaching plan + reference answers

> **This is Claude's working file**, written at Luke's request so the plan survives a long
> conversation. It began as a teaching plan — Luke wrote the code, this file held the reference
> answers — and is now mostly a **record of what was built, what was measured, and what went
> wrong**. The step-by-step sections below are kept as written; the ones marked DONE record what
> actually happened, which often differed from the plan.
>
> Written 2026-08-05, after step 4a (block-sparsity measurement) came back positive.
> **Last updated 2026-09-15 (post-audit): steps 4 AND 5 complete. The sorted variant is the default
> (`or_impl="sorted"`), and padding is handled on both paths. The feature is done; what is
> left is a training run to find out whether OR amplification is worth its cost.**
> Steps 1-3 are committed as `090aaef`. Branch: `OR-amplification` (stay on it).

> **New here? Read "The two implementations, in plain terms" (just below §0) first.** It
> explains what `sorted` and `masked` are, why one is faster, and collects every timing in
> one place.

> **Line numbers in this file drift as the lesson edits `encoder.py`.** The table in §0c
> holds the *current* ones and is the thing to trust. Inline references elsewhere were
> correct when written; re-grep before relying on one.

---

## 0. Where we are

| Step | What | Status |
|------|------|--------|
| 1 | `models/ordering.py` — frozen E2LSH grid, rank tables | done, committed |
| 2 | `models/attention.py` — `expand_heads_for_or`, `or_merge_lse`, `or_n_hashes` | done, committed |
| 3 | `flex/per_head_window.py` — `per_head_window_mask_mod` | done, committed |
| 4a | `scripts/measure_block_sparsity.py` — is the BlockMask actually sparse? | done, **verdict: viable** |
| 4.1-4.7 | Encoder wiring: config, asserts, grids, coords, coord permute, cache bypass | **done** |
| 4.4b | `flex` + `window_size` no longer raises (pre-existing blocker) | **done** |
| 4.8 | Per-layer BlockMask with `H = num_heads * n_hashes` in the layer loop | **done** |
| 4.9 | `or_n_hashes` threaded to `Attention` (fell out of 4.8) | **done** |
| 4.11 | `tests/models/test_encoder_or.py` — 3 tests, all passing ([[test_encoder_or]]) | **done** |
| 4.10 | Coords through `MaskFormer.forward` + `tests/models/test_maskformer_or_coords.py` ([[test_maskformer_or_coords]]) | **done** |
| 4.12 | Config + three-arm benchmark | **done** — sorted variant wins |
| — | Sorted variant implemented, `or_impl="sorted"` is the default | **done**, 6.8x end to end |
| 5 | Padding, on both implementations | **done** |

**The feature is finished.** Everything above is committed except the step 5 changes, which are
still in the working tree, and this `notes/` directory, which is deliberately untracked.

Suite: **452 passed, 69 skipped** on an A100. Two `tests/flex` OOM failures appear when
`tests/models/` and `tests/flex/` run in one process on a 40 GB card; `tests/flex/` alone passes
48/48 and the same two fail identically on unmodified HEAD. Not ours.

**The one thing left is not engineering:** does OR amplification earn its **5.6x** cost over
today's sliding window (§0f)? That needs a training run with `configs/or-amp.yaml` against `configs/base.yaml`,
compared on physics metrics. Timings cannot answer it, and neither can comparing the two OR
implementations against each other — they are numerically identical by construction.

### Step 4a result (the go/no-go)

Real ColliderML val events (2 events, `base.yaml`, val split), N ≈ 13k hits, window 512,
block 128, 8 heads x 3 hashes = 24 cells.

| token layout | Route B block density | per-cell [min, max] | vs dense | waste vs ideal |
|---|---|---|---|---|
| `lsh0` (sorted by head-0/hash-0's ordering) | **0.237 / 0.222** | [0.049, 0.323] | **4.2-4.5x** | 5.9x |
| `phi` (sorted by phi) | 0.305 / 0.303 | [0.184, 0.492] | 3.3x | 7.7x |
| `raw` (dataset order, no sort) | 0.495 / 0.470 | [0.344, 0.694] | 2.0x | 12.4x |

Reference rows (identical for all layouts, since a rank table is a permutation):
element-level ideal density **0.039**; banded baseline (today's sort-then-slide) **0.049**,
i.e. 20x faster than dense at 1.2x waste.

**The token layout matters as much as the orderings do.** Block structure is defined over token
*index*, so how the encoder lays the hits out changes the answer even though the ranks do not.
Sorting the sequence by one cell's LSH ordering gives that cell a perfectly banded mask (0.049,
the `min` above) and drags every other cell along with it. This is a free 2x over `raw` — the
encoder should keep sorting, and ideally sort by an LSH ordering rather than phi.

**Verdict: Route B works, at roughly 4-5x the attention cost of the current banded window,
still ~4x cheaper than dense.** It did *not* collapse to density 1.0. The reason is that LSH
orderings are not random permutations — every cell is sorting the same 2D geometry, just with
a different random quantile grid, so the cells' ranks stay strongly correlated with each other.

The fast path in `measure_block_sparsity.py` was verified to agree with a real
`BlockMask.sparsity()` to 0 error at N=1024, so these numbers are trustworthy.

**Consequence for step 4:** proceed with the in-place masking design, but see §0b — a third
option surfaced while teaching and it currently looks better than Route B. Wire Route B anyway;
it is the honest baseline and almost all of the wiring is shared. The 4-5x is a
mask-construction-free estimate; the real cost also includes `n_hashes` x the q/k/v memory and
`num_layers` block-mask builds.

## The two implementations, in plain terms

> Read this before §0a-§0d, which assume it. It explains what `or_impl="sorted"` and
> `or_impl="masked"` actually do and why one is several times faster.

### What they have in common

Both do exactly the same job, and produce exactly the same numbers.

Each attention head is given several different orderings of the hits (3 by default, so 8 heads x
3 = 24 orderings per layer). A hit attends to its ~512 nearest neighbours **in each of those
orderings**, and the results are merged. The point is that no single ordering keeps every group
of nearby hits together, so several orderings catch what one misses.

The two implementations differ in **where the ordering is expressed** — and nothing else.

### `masked`: leave the hits alone, put the ordering in the mask

The hits stay exactly where they are in memory. Each head is handed a lookup table saying *"in my
ordering, hit `t` sits at position `p`"*. The rule for whether two hits may attend becomes:

> are their positions **in my ordering** within 256 of each other?

Every head has a different ordering, so **every head needs its own mask**.

### `sorted`: physically reorder the hits, and the mask becomes trivial

Each head gets its own rearranged copy of the hits, shuffled into that head's ordering. Now
position-in-memory *is* position-in-the-ordering, so the rule becomes:

> are they within 256 **slots** of each other?

That sentence mentions no ordering at all. It is the **same rule for all 24 cells**, so they
share **one mask**. Afterwards the outputs are shuffled back into the original order before the
merge, so the rest of the model never notices.

### A five-hit example

Say head 1's ordering puts the hits in the order C, E, A, D, B.

```
masked   memory:  A  B  C  D  E          <- untouched
         table :  A is 3rd, B is 5th, C is 1st, D is 4th, E is 2nd
         rule  :  "within 2 places of each other IN MY TABLE"   <- different per head

sorted   memory:  C  E  A  D  B          <- head 1's own copy, rearranged
         table :  not needed
         rule  :  "within 2 slots of each other"                <- same for every head
```

Both admit the same pairs. `sorted` paid a shuffle to make the question simpler.

### Why `sorted` is faster — three reasons, all about the mask

The mask (`BlockMask`) is the object that tells the GPU which 128x128 tiles of the score matrix
it can skip. Building it is not free: it means evaluating the rule over the whole N x N grid.

1. **It is ~19x cheaper to build**, because it has one row instead of 24. One shared rule versus
   one rule per cell. *(40.3 ms vs 1.9 ms at N~13k.)*
2. **It is built once instead of every layer of every event.** `masked`'s rule depends on *this
   event's* ordering tables, and every layer draws its own orderings, so its mask cannot be
   reused for anything. `sorted`'s rule — "within 256 slots" — depends only on the sequence
   length, so the encoder's existing mask cache handles it. **This is the big one.**
3. **Fewer tiles actually get computed.** After sorting, each hit's 512 partners sit right next
   to it, so the admitted pairs form a narrow band down the diagonal and most tiles are empty.
   Under `masked` those same partners are scattered across the row, so most tiles contain at
   least one and must be computed in full. *(4.8% of tiles vs 23.2%.)*

Reason 3 makes the attention itself ~1.9x faster. Reasons 1 and 2 are what make the whole thing
~7x faster, because mask building dominates once the attention is compiled.

### What `sorted` pays for it

A gather to rearrange q/k/v, and a gather to put the outputs back. That is `O(N x dim)` work
against attention's `O(N x window x dim)` — a linear cost to avoid a much larger one, ~2 ms
against the tens of ms it saves. Memory is a wash (0.21 GB vs 0.19 GB): `masked` already
materialises a full replicated copy of q/k/v anyway.

### All the timings in one place

Real ColliderML val events, A100, N ~ 13k, 8 heads x 3 hashes, fp16, LSH-sorted layout.

**Per layer** (`scripts/benchmark_or_amplification.py`, mean over 3 events):

| arm | mask build | attention | total | vs flash |
|---|---|---|---|---|
| flash + window (today) | — | 0.26 ms | **0.26 ms** | 1x |
| `masked` | 40.3 ms | 3.70 ms | **44.0 ms** | 169x |
| `sorted` | 1.9 ms *(shared, cacheable)* | 1.94 ms | **3.8 ms** | 15x |

**Whole encoder**, which is the figure to quote (4 layers, dim 256, N=13000, same weights):

| `or_impl` | ms/event |
|---|---|
| `sorted` | **23.2** |
| `masked` | 158.1 |

**6.8x.** Lower than the per-layer 12x because a full forward also runs projections, norms and
the dense blocks, which are identical either way. *(Measured again post-audit in steady state:
13.9 ms for `sorted` against 2.5 ms for flash, i.e. **5.6x vs today's window** — see §0f. The
per-layer "15x vs flash" in the table above is attention only and overstates the real cost.)*

**Block density** — the fraction of tiles that must be computed. Ordering-independent floor is
0.039; today's single-ordering window is 0.047.

| | density |
|---|---|
| banded (today, and `sorted`) | 0.048 |
| `masked` | 0.232 |

**Compiling flex attention** (§2c item 3) — not an OR result, but it changed which of the above
numbers matter, so keep it here. N=4096, 24 mask rows, same outputs:

| | time | peak memory |
|---|---|---|
| uncompiled | 44.5 ms | 8.53 GB |
| compiled | 11.2 ms | 0.11 GB |

> [!warning] Why that last table belongs in this note
> Before flex was compiled, attention took 271 ms/layer and mask building looked like <4% of the
> cost — which is exactly what §2c item 1 originally concluded, and why I told Luke to stop
> worrying about mask construction. Fixing the compilation cut attention ~27x and the *same*
> mask cost became 92% of the layer. **The measurement never changed; the thing it was divided by
> did.** Anyone re-running these benchmarks needs to confirm flex is compiled first, or they will
> reach the same wrong conclusion.

---

### 0a — block size cannot rescue Route B

Same event, `lsh0` layout, sweeping flex's `BLOCK_SIZE`:

| block size | Route B density | banded baseline | Route B waste | **Route B / baseline** |
|---|---|---|---|---|
| 32 | 0.170 | 0.042 | 4.3x | **4.0x** |
| 64 | 0.205 | 0.045 | 5.1x | **4.6x** |
| 128 (default) | 0.237 | 0.049 | 5.9x | **4.8x** |
| 256 | 0.288 | 0.059 | 7.2x | **4.9x** |

Shrinking tiles 4x (128 -> 32) improves Route B only 1.4x. The last column is the point: the
Route-B-vs-banded gap sits at 4-4.9x *at every block size*. It is a property of how scattered
the ranks are, not of the tiling, so `BLOCK_SIZE` cannot close it. Practically only 64 is a
usable alternative (32 is generally not a valid flex kernel config), worth ~13% on paper and
probably less in wall-clock given worse tensor-core utilisation. Try it at benchmark time; do
not design around it.

Note the baseline barely moves (0.049 -> 0.042) — the signature of something already optimal.

### 0b — the sorted variant: the original argument for it (HISTORICAL)

> Superseded by "The two implementations, in plain terms" above and by the measurements in
> §0d. Kept because the reasoning about *why the orderings disagree* at the end is still the
> conceptual core — and as a reminder that the FLOP estimates below were the wrong thing to
> reason about: they ignore mask construction, which is what actually decided it.

**The idea.** Instead of leaving tokens in place and giving each head its own *mask*, physically
permute each head's copy of q/k/v into that head's own ordering, then apply **one shared plain
banded sliding window** to every head, and un-permute the outputs before the LSE merge.

**It computes exactly the same thing.** "Within `W/2` ranks of each other in head h's ordering"
and "within `W/2` *positions* of each other after sorting by head h's ordering" are the same set
of pairs. Identical results, different cost profile.

**Why the mask becomes head-independent** (the non-obvious bit): the ordering stops living in
the mask and starts living in the data layout. Each head still gets a totally different
ordering — the difference is now *which token sits in which slot*, not what the mask rule says.
The rule "talk to slots within W/2 of you" mentions no ordering at all, so it is the same rule
for all 24 cells. One `BlockMask`, `H=None`, built once, no per-event or per-layer rebuild.

**Cost comparison** (dim 256, 8 heads, head_dim 32, N~13k, 24 cells, one layer):

| | |
|---|---|
| Route B attention | ~124 GFLOP |
| sorted attention | ~26 GFLOP |
| gather + scatter the sort needs | ~120 MB traffic, <0.1 ms |
| extra memory for the un-permute buffer | ~40 MB |

The permutation is `O(N * dim)`; the attention is `O(N * W * dim)`. Paying a linear cost to
avoid a much larger one. Memory is a wash — Route B *already* materialises the `(B, 24, N, Dh)`
replication (`expand_heads_for_or` genuinely allocates), and the sorted version can build the
permuted tensor directly from the unreplicated q/k/v with advanced indexing (**measured and
rejected — see §0e; it is slower**)
(`q[:, head_idx, order_idx, :]`), skipping the identical-copies intermediate entirely.

**This is NOT the Route A we rejected.** Route A was "gather into fixed buckets, dense attention
within each bucket" — rejected because a bucket is not a sliding window (edge hits lose half
their neighbours). Sort-then-slide keeps the sliding window exactly. It is a third option that
was never evaluated because the block-granularity cost of Route B was not quantified until 4a.

**Why the orderings disagree so much** (worth keeping — it is the conceptual core): 2D geometry
cannot be flattened to 1D without cuts, and every ordering cuts in different places. To reach
banded density you would need a head's ordering to agree with the token layout *almost exactly*
(512 partners inside ~5 tiles = ~640 consecutive positions) — far stronger than "both are good
orderings". The LSH ordering is phi-major with coarse bins (~1300 hits per phi bin at
`num_regions=100`), and re-sorts *within* a bin by eta, which is a shuffle from a phi-sorted
layout's point of view. Plus `hept_region_counts` randomises the eta:phi split from ~3x30 to
~30x3, so some tables sort mostly by eta — near-orthogonal to a phi layout. That is the
`[0.184, 0.492]` per-cell spread, measured.

**The trade is fundamental:** the sparsity cost and the physics benefit are the same
phenomenon. Orderings must disagree to catch different neighbours; disagreement is exactly what
scatters the admitted pairs. You cannot have cheap and diverse — unless each head gets its own
layout, which is what sorting does.

**What transfers if we switch:** Step 1 (grid), Step 2 (LSE merge), and steps 4.1-4.6 (config,
asserts, grid construction, threading and permuting coords) are all identical. Only 4.8 forks:
gather along the head axis using the rank tables + one shared banded mask, instead of building a
per-layer rank BlockMask. Step 3's `per_head_window_mask_mod` becomes the fallback path.

**Recommendation:** finish 4.5-4.9 as Route B, then add the sorted variant as a third arm at
4.12 (~30 lines on top of what will already exist). Do not silently drop this.

---

## 0c — current line numbers (verified 2026-09-15, post-audit)

| Landmark | File:line |
|---|---|
| register-token / window assert | `encoder.py:190` |
| Route B guard asserts (incl. `or_impl`) | `encoder.py:191-196` |
| `attn_kwargs["window_size"]` flash-only (4.4b fix) | `encoder.py:230` |
| `num_heads` read off layer 0 | `encoder.py:236` |
| `self.ordering_grids` construction | `encoder.py:237-241` |
| `Encoder.set_backend` (window + OR guard) | `encoder.py:243-251` |
| `seq_len` captured | `encoder.py:257` |
| `x_coords` assert | `encoder.py:260` |
| mask cache, bypassed only by the masked path | `encoder.py:296-311` |
| `sorted_window_mask_mod` built here | `encoder.py:311` |
| **`seq_len` / `num_valid` written in place** (the §0e fix) | `encoder.py:322-324` |
| layer loop | `encoder.py:334` |
| `per_head_window_mask_mod` built here (masked path) | `encoder.py:348` |
| `permute_to_rank_order` | `attention.py:99` |
| `restore_token_order` | `attention.py:134` |
| flex compile decision | `attention.py:341` |
| flex rejects `kv_mask` (encoder avoids it instead) | `attention.py:517` |
| flex asserts batch size 1 | `attention.py:520` |
| the sorted / masked branch | `attention.py:522` |
| `lsh_order_values`, takes `valid` | `ordering.py:67` |
| `num_valid` kept as a tensor (no GPU sync) | `ordering.py:121` |
| `E2LSHOrderingGrid.forward`, takes `valid` | `ordering.py:204` |
| `per_head_window_mask_mod` | `flex/per_head_window.py:5` |
| `sorted_window_mask_mod` | `flex/per_head_window.py:40` |

---

## 0d — 4.12 RESULT: the sorted variant wins. The §0b fork is decided.

Measured 2026-09-14 on an A100, real ColliderML val events, N ~ 13k, 8 heads x 3 hashes, fp16,
LSH-sorted layout. `scripts/benchmark_or_amplification.py`, mean over events, **per layer**:

| arm | mask build | attention | **total** | vs flash |
|---|---|---|---|---|
| flash + window (today) | — | 0.26 ms | **0.26 ms** | 1x |
| `masked` (mask in place) | 40.3 ms | 3.70 ms | **44.0 ms** | 169x |
| `sorted` (permute per head) | 1.9 ms *(shared, cacheable)* | 1.94 ms | **3.8 ms** | 15x |

Block densities reproduced 4a exactly (`masked` 0.232, banded 0.048). The two OR arms agree
numerically on every event — the benchmark asserts it rather than assuming — so the difference is
purely cost. Memory is a wash: 0.19 GB `masked`, 0.21 GB `sorted`.

**The deciding factor is mask construction, not tile density.** On attention alone `sorted` is
1.9x faster, which is worth having but not decisive. Including the mask it is ~12x, because:

* **`masked` must rebuild its mask every layer of every event.** The rule closes over that
  event's rank tables and each layer draws its own orderings. 40 ms x 4 layers = ~160 ms/event of
  mask building against ~15 ms of actual attention.
* **`sorted`'s window is position-only.** Same mask for every head, every layer, and reusable
  across events of equal length. It is also ~19x cheaper to build in the first place (1.9 ms vs
  40 ms) simply because it has one row instead of 24.

This is structural, not an optimisation `masked` is missing. It follows from where each design
puts the ordering: in the mask (rebuilt per event) or in the data (permuted per event, then a
fixed mask).

**Why this was not seen at 4a, and the correction it forces.** §2c item 1 recorded mask building
as "under 4% of the layer" and I told Luke to strike it from the case against masking in place.
That was measured when `flex_attention` was running *uncompiled* at 271 ms/layer. Fixing that
(§2c item 3) cut attention ~27x and the same mask cost went from 4% to 92%. **The number was
right; the ratio was wrong.** Luke's instinct that mask construction mattered was correct.

**Decision: adopt the sorted variant as the default implementation of OR amplification.**
`masked` stays in the tree as the reference implementation — it is what the sorted path is checked
against, and the 4.11 tests are written in terms of it.

### 0d.1 — IMPLEMENTED (2026-09-14). `or_impl="sorted"` is the default.

Both implementations are selectable so future benchmarks can compare them:

```python
Encoder(..., or_n_hashes=3, or_impl="sorted")   # default
Encoder(..., or_n_hashes=3, or_impl="masked")   # reference / benchmarking
```

**Measured end to end on the real encoder** (4 layers, dim 256, N=13000, fp16, same weights):

| `or_impl` | ms/event |
|---|---|
| `sorted` | **23.2** |
| `masked` | 158.1 |

**6.8x**, outputs agreeing to 0.0078 (fp16 rounding). Lower than the per-layer 12x because a full
forward also runs projections, norms and the dense blocks, which are identical either way.
*(Re-measured in steady state post-audit: 13.9 ms, i.e. **5.6x vs today's flash window** — §0f.
The earlier 23.2 ms carried warm-up.)*

**What changed:**

* `attention.py` — `permute_to_rank_order` / `restore_token_order` (moved in from the benchmark
  script so there is one implementation), and `Attention.forward` gained **`or_ranks`**. Passing
  ranks selects the sorted path; omitting them keeps the masked path. Un-permuting is a `gather`
  with `ranks` itself — `ranks` *is* the inverse permutation, so no scatter is needed.
* `encoder.py` — `or_impl: str = "sorted"` guarded by `OR_IMPLS`, and **the 4.7 cache bypass was
  narrowed** to `if self.ordering_grids is None or self.or_impl == "sorted"`. This is where most
  of the win lives: the sorted path's window is position-only, which is exactly what that cache
  exists for. 4.7 had to bypass it only because the *masked* path needs a per-event mask.
* The banded mask is built with `H=None` while the permuted q/k/v have `num_heads * n_hashes`
  rows. Flex broadcasts it — verified, and it is why the mask is shareable at all.

**Tests** (`tests/models/test_encoder_or.py`): the two implementations must agree; `sorted` must
be the default *and* populate the shared cache while `masked` must not; an unknown `or_impl` is
rejected.

> [!warning] Two things this broke, both fixed
> * **Auto-compiling flex broke the CPU path.** inductor cannot lower flex with `return_lse` on
>   CPU, and the OR path needs it. The gate is now per call on `q.is_cuda` — `torch.cuda.is_available()`
>   was wrong, because a GPU machine can still be running a CPU model.
> * **`test_real_grids_build_one_distinct_mask_per_layer` was testing the wrong thing** once the
>   default moved. One mask per layer is the *masked* path's defining behaviour. Renamed to
>   `test_masked_impl_builds_one_distinct_mask_per_layer` and pinned with `or_impl="masked"`.

---

## 0e — post-implementation audit (2026-09-15)

A full read-through of every file after step 5, looking for mistakes, leftovers and
inefficiencies. One real bug, three efficiency items, and one recommendation from these notes
that turned out to be wrong.

### The bug: the cached `mask_mod` captured one event's `num_valid`

Step 5 gave the sorted path's shared mask a validity cut, `kv_idx < num_valid`. But `mask_mod` is
**cached** — built on the first forward and reused — so the closure held the *first event's*
count. Every subsequent event was masked with the wrong amount of padding: attending to junk, or
ignoring real hits, depending on which way the count moved. Nothing raised.

It had a twin: if the first event happened to have no padding at all, `num_valid` was `None` and
the closure was built **without any validity cut**, so every later padded event silently attended
its padding.

This is exactly the staleness §4.7 was written about, reintroduced one level down by step 5.

**Fix:** the same in-place tensor the encoder already uses for `seq_len` — `self.num_valid` is
allocated once and written each forward, so the cached closure reads a live view. Always set
(to `seq_len` when there is no padding), so the cut is uniform rather than conditional.

> [!warning] Why every existing test missed it
> They all ran **one event per encoder**. A staleness bug needs two. Two regression tests now
> cover it — consecutive events with different padding, and padding that only appears on a later
> event — and both were mutation-checked.

### Efficiency

* **Removed a GPU→CPU sync.** `int(valid.sum(dim=-1).max())` in `lsh_order_values` forced a device
  sync on every layer of every event. It is a tensor now; the arithmetic downstream never needed
  it to be a Python int.
* **Removed needless allocations.** `torch.where(..., torch.zeros_like(coords))` and two more
  `torch.where` calls became `masked_fill`.

### A recommendation from these notes that is wrong

§0b says to build the permuted tensor with a single advanced index
(`x[:, head_of_cell[:, None], order, :]`) instead of replicate-then-gather, "skipping the
identical-copies intermediate entirely". Measured at N~13k:

| | time | peak memory |
|---|---|---|
| replicate-then-gather (what we do) | 0.526 ms | 53.6 MB |
| advanced index | 0.575 ms | 35.1 MB |

35% less memory but 9% slower, and the OR path peaks at 0.21 GB — nowhere near memory bound. The
faster form wins. **Recorded in `permute_to_rank_order`'s docstring so nobody optimises it back.**

### Cleanliness

* "Route B" no longer appears anywhere in source (0 occurrences). The vocabulary is
  `masked` / `sorted` throughout, matching `or_impl`.
* The layer loop used `ranks = None` as an implicit signal meaning "take the masked path". Now an
  explicit `layer_ranks`.
* `expand_heads_for_or`'s docstring still proposed an alternative that the sorted path supersedes.

### Edge cases verified

N=0; `valid` all-True identical to `valid=None`; a single real hit; an all-padding event; and
padded coordinates of `inf`, `nan` or `1e30` — none disturb the real hits' ordering. That last
one is why the defensive zeroing of padded coords earns its place.

---

## 0f — GPU verification and the real speed (2026-09-15)

A100-PCIE-40GB, N=13000, 4 layers, dim 256, fp16, **steady state** (5 warm-up iterations, median
of 15):

| | median | min | max |
|---|---|---|---|
| `sorted` | **13.9 ms/event** | 13.9 | 24.3 |
| flash + sliding window (today) | **2.5 ms/event** | 2.5 | 2.6 |

**OR amplification with 3 orderings per head costs 5.6x today's sliding window, end to end.**

> [!warning] This corrects the "~15x" quoted earlier in this note
> That figure came from the per-layer attention-only benchmark (§0d), which deliberately excludes
> projections, norms and the dense blocks because they are identical across arms. Those shared
> costs are real, so measured over a whole encoder the ratio is much better. **5.6x is the number
> to quote; 15x describes only the attention.**

> [!note] Warm-up matters more than expected
> The first timed iteration is consistently 60-75% slower than the median (24.3 vs 13.9) even
> after one warm-up call. Any future benchmark needs several warm-up iterations, or it will
> overstate the cost.

Correctness on GPU, all confirmed at this size:

* `sorted` and `masked` agree, with and without padding (real tokens compared).
* Finite output everywhere, including a heavily padded event.
* A second event is unaffected by the first one's padding — the §0e bug, verified fixed at scale.
* Peak memory 0.21 GB, i.e. not a constraint.

---

## 1. Design decisions already fixed (do not relitigate mid-lesson)

* **Head-major flattening.** `h = head * n_hashes + hash`, so `H_flat = num_heads * n_hashes`.
  Both `E2LSHOrderingGrid.forward` (returns `(..., num_heads, n_hashes, N)`, `.flatten(0, 1)`)
  and `expand_heads_for_or` already assume this. Do not change it.
* **Batch size 1.** `attention.py:439` asserts it whenever a BlockMask is present, and
  `per_head_window_mask_mod` ignores `b` accordingly. Route B inherits the restriction.
* **Register tokens are already excluded.** `encoder.py:181` asserts
  `not (num_register_tokens is not None and window_size is not None)`. Since Route B requires a
  window, registers can never co-occur with it. *(Earlier in conversation I flagged
  coordinate-less register tokens as an open design problem for step 4 — that was wrong; the
  assert forecloses it. Correct this if it comes up again.)*
* **No wrapped variant.** LSH is non-periodic; `per_head_window.py` deliberately has no
  `_wrapped` sibling. `window_wrap` must be rejected with Route B.
* **DECIDED 2026-09-14: the sorted variant, not Route B.** See §0d for the measurements. Route B
  remains in the tree as the reference implementation and as the subject of the 4.11 tests.

---

## 2. Step 4 — the teaching sequence

Twelve steps. Each is small enough to write in one sitting, and each has an observable check
so Luke can tell it worked before moving on.

### 4.1 — Config surface on `Encoder.__init__`

Add three arguments after `window_wrap`:

```python
or_n_hashes: int | None = None,
or_num_regions: int = 100,
or_base_seed: int = 0,
```

Store `self.or_n_hashes = or_n_hashes`.

**Check:** `Encoder(num_layers=1, dim=16, or_n_hashes=3)` constructs without error.

### 4.2 — Guard asserts

Next to the existing asserts at `encoder.py:179-181`.

**Why:** each corresponds to a real limitation established in steps 1-3, and an assert here
turns a confusing runtime shape error into a one-line message.

**As actually written** (`encoder.py:182-186`) — the third message exceeds the 150-char line
limit, so it is parenthesised across two lines, matching the style at `attention.py:411/415/419`:

```python
assert or_n_hashes is None or attn_type == "flex", "OR amplification requires flex attention."
assert or_n_hashes is None or window_size, "OR amplification requires a window_size to be set."
assert or_n_hashes is None or not window_wrap, (
    "OR amplification has no wrapped variant: an LSH ordering treats phi as a linear "
    "coordinate, so it is not periodic."
)
```

**Teaching notes (what actually went wrong):** the implication form `not P or Q` had to be
taught from scratch — first attempts were `assert X if C` (no trailing-`if` modifier in
Python), a bare `assert` with no expression, `window_size is > 0` (`is` is identity, not a
copula), and `window_wrap is False` (works, but fails for falsy non-bools). Also worth knowing:
`window_size > 0` would raise `TypeError` on `None`, so bare truthiness is the correct test.

**Check:** `Encoder(..., or_n_hashes=3, attn_type="torch")` raises. DONE.

### 4.3 — Discover `num_heads`

The grid needs `num_heads`, which lives in `attn_kwargs` and defaults to 8 inside `Attention`.
Do **not** re-default it in the encoder — read it back off a constructed layer, after
`self.layers` is built:

```python
num_heads = self.layers[0].attn.fn.num_heads
```

(`layer.attn` is the `Residual` wrapper; `.fn` is the `Attention`. Same path
`set_backend` uses at `encoder.py:207`.)

**Why:** single source of truth. If someone sets `attn_kwargs: {num_heads: 4}`, the grid
follows automatically.

### 4.4 — Build the grid list

Still in `__init__`, after `self.layers`:

```python
self.ordering_grids = None
if or_n_hashes is not None:
    self.ordering_grids = nn.ModuleList([
        E2LSHOrderingGrid(
            num_heads=self.layers[0].attn.fn.num_heads,
            n_hashes=or_n_hashes,
            num_regions=or_num_regions,
            base_seed=or_base_seed,
            layer_idx=i,
        )
        for i in range(num_layers)
    ])
```

Import `E2LSHOrderingGrid` from `hepattn.models.ordering` at the top.

**Why one grid per layer:** `E2LSHOrderingGrid` takes `layer_idx` precisely so each layer draws
distinct orderings; different layers looking at different neighbourhoods is the point.

**Decision worth flagging to Luke, and worth benchmarking later:** a *single shared* grid
across all layers would mean one BlockMask build per forward instead of `num_layers`. Cheaper,
less diverse. Start with per-layer (it is what the design intends); measure later.

**Check:** `sum(p.numel() for p in enc.buffers())` grows; `enc.ordering_grids[0].alpha.shape ==
(num_heads, n_hashes, 2)`.

### 4.4b — stop forwarding `window_size` into `Attention` on the flex path  ← **discovered mid-lesson, pre-existing blocker**

`Encoder(attn_type="flex", window_size=8)` **cannot be constructed today**, on unmodified HEAD:

```
AssertionError: Window size can only be specified for ['flash', 'flash-varlen']
```

Chain: `encoder.py:199` (now `216`) unconditionally sets `attn_kwargs["window_size"] = window_size`, and
`attention.py:223` rejects a non-None `window_size` for any non-flash backend. Flex expresses
its window in the `mask_mod` / BlockMask, never as a kernel argument -- which is exactly what
the pre-existing TODO at `attention.py:275` ("Will need to change when supporting window with
flex") is warning about. Route B is the first caller to actually need flex + window together.

Fix in the encoder (do *not* weaken the assert in `attention.py` -- it is correct; flex really
cannot use a kernel-side window):

```python
attn_kwargs["window_size"] = window_size if attn_type in FLASH_ATTN_TYPES else None
```

`FLASH_ATTN_TYPES` is already exported from `hepattn.models.attention`; import it rather than
re-spelling the literal list.

Verified: with this line changed, the 4.4 check passes. `self.window_size` on the encoder is
untouched, so the existing sliding-window mask path (now `encoder.py:277-291`) still works.

### 4.5 — Accept coordinates in `Encoder.forward`

```python
def forward(self, x, x_sort_value=None, kv_mask=None, x_coords=None, **kwargs):
```

Assert early: `assert self.ordering_grids is None or x_coords is not None`.

`x_coords` is `(B, N, 2)` = `(eta, phi)`.

**As written** (`encoder.py:234-241`) — note `Encoder.forward` has **no docstring at all**, so
there is nothing to document `x_coords` "in the same style as the others"; a short inline
comment above the assert is the right thing:

```python
# x_coords (B, N, 2) carries the per-hit (eta, phi)
assert self.ordering_grids is None or x_coords is not None, "OR amplification requires x_coords (B, N, 2) of (eta, phi)."
```

**Order matters in that assert.** `self.ordering_grids is None` must come first so `or`
short-circuits before `x_coords` is ever truthiness-tested — a bare `assert x_coords or ...`
raises `RuntimeError: Boolean value of Tensor with more than one element is ambiguous`.

**Check:** existing call sites still work (the argument is optional and defaults to `None`);
the assert fires with the right message when coords are withheld. DONE — 15 passed, 1 skipped.

### 4.6 — Permute the coords with the tokens  ← **the subtle one**

At `encoder.py:246-252` the encoder *sorts the tokens* by `x_sort_value` and gathers `x` and
`kv_mask` into the new order. Coordinates must be gathered identically, or every rank table
will describe tokens that are no longer where it thinks they are.

**As written** (`encoder.py:254-257`) — it lives *inside* the `if x_sort_value is not None:`
block, so `x_sort_idx` is guaranteed bound and only `x_coords` needs guarding:

```python
# The coords must follow their tokens, otherwise the rank tables built from
# them describe hits that are no longer at those positions
if x_coords is not None:
    x_coords = torch.gather(x_coords, dim=-2, index=x_sort_idx.unsqueeze(-1).expand_as(x_coords))
```

Copy the shape handling from the `x` gather two lines above, **not** the `kv_mask` one:
`x_coords` has a trailing size-2 axis, so it gathers along `dim=-2` with the index unsqueezed
and broadcast so eta and phi move together.

**Luke asked a good question here worth reusing:** *"isn't the phi sort orthogonal to Route B?"*
Yes in purpose, no in effect — the sort is the only place row order changes, and Route B reads
geometry off row order. The dependency runs one way: the sort does not need coords; the coords
must survive the sort. Also worth telling him: they genuinely coexist, because §0's 4a table
shows the layout is worth ~2x to Route B (`phi` 0.30 vs `raw` 0.49), so we *want* the sort on.

**Why this is the bug most likely to be written and least likely to be noticed:** it produces
no error, no shape mismatch, no NaN. Attention just silently attends to geometric nonsense and
the model trains slightly worse. Ask Luke to predict what goes wrong *before* writing it.

**Check:** with `x_sort_value = phi`, `x_coords[0, :, 1]` comes out monotonically
non-decreasing, and sorting both sides shows the same multiset. DONE — verified by spying on
`torch.gather`; `monotonic: True`, `same multiset: True`.

*(Luke asked for this one to be written for him rather than predicting the failure mode.)*

### 4.7 — Do not let the mask cache swallow Route B

`encoder.py:280` reads `if self.mask_mod is None and ...`. That cache is correct for a plain
sliding window (a position-only rule is the same for every event) and **fatal** for Route B,
whose closure captures per-event `ranks`.

Simplest correct change: leave the cache alone and make it unreachable, by guarding the whole
`self.mask_mod` init and the `attn_mask` construction (now `encoder.py:280-291`) with
`if self.ordering_grids is None:`.

**Why not "just clear the cache each forward":** because Route B does not want *one* mask at
all — it wants a different mask per layer (4.8). Bypass, don't patch.

**Two things the lesson turned up, both worth reproducing:**

1. **`attn_mask = None` must be hoisted *above* the guard** (`encoder.py:277`). The layer loop
   passes `attn_mask=attn_mask`, so if Route B skips the whole block the name is never bound
   and you get a `NameError`. Loud, at least.
2. **Gate on `self.ordering_grids is None`, not `x_coords is None`.** Luke's first attempt used
   the latter. It is a *correlate* of the feature being on, not the feature itself: the 4.5
   assert guarantees grids ⇒ coords, but not the converse. Pass coords to a plain windowed
   encoder and it would silently lose its sliding window and fall back to global attention —
   no error, just a slower and different model. 4.10 makes that a live risk, since the natural
   way to write it passes coords unconditionally. **General rule: gate on the thing that
   decides the behaviour, never on something correlated with it.**

The `if` / `elif` pair for the torch and flex branches stays chained to *each other*, one level
deeper — it is not chained onto the new guard.

**Check:** a plain `attn_type="torch", window_size=8` encoder still populates `mask_mod`, *and*
still populates it when handed spare `x_coords`. Both print `True`; the second is the one that
fails on the wrong condition. DONE.

### 4.8 — Build the per-layer BlockMask inside the layer loop

Replace the `for layer in self.layers:` loop at `encoder.py:299-300`:

```python
for i, layer in enumerate(self.layers):
    layer_mask = attn_mask
    if self.ordering_grids is not None:
        ranks = self.ordering_grids[i](x_coords[0]).flatten(0, 1)  # (num_heads * n_hashes, N)
        layer_mask = create_block_mask(
            per_head_window_mask_mod(ranks, self.window_size),
            B=None,
            H=ranks.shape[0],
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=str(x.device),
        )
    x = layer(x, attn_mask=layer_mask, score_mod=self.score_mod,
              initial_values=initial_values, kv_mask=kv_mask,
              or_n_hashes=self.or_n_hashes, **kwargs)
```

Notes:
* `x_coords[0]` drops the batch dim — legal only because batch size is 1 (see §1).
* `H=ranks.shape[0]`, **not** `H=None`. This is the whole point of step 3.
* `seq_len` is captured at `encoder.py:238`, before register tokens; with Route B there are no
  register tokens (§1), so it is correct as-is. Worth having Luke double-check that reasoning.
* Import `per_head_window_mask_mod` from `hepattn.flex`.

**Check:** print `layer_mask.sparsity()` — it should land near the 4a numbers (~76% sparse,
i.e. ~0.24 density) and *differ between layers*.

**DONE (2026-09-11).** Only mistake in the writing was a missing import: `per_head_window_mask_mod`
has to be added to the existing `from hepattn.flex import ...` line, and ruff catches it as F821.
Everything else went in first time.

**Calibrating that check — important, I got this wrong first.** ~0.24 was measured on *real*
ColliderML geometry at N ~ 13k. On random synthetic coords the density is nothing like it, and a
small test looks broken when it is not:

| N | `x_sort_value` | blocks | density |
|---|---|---|---|
| 1024 | none | 8 | 1.000 |
| 4096 | none | 32 | 1.000 |
| 4096 | phi | 32 | 0.323 / 0.327 / 0.309 |

**The global sort does not merely improve sparsity, it creates it.** Without it the density is
1.000 at every N tried. The mask is defined over token *index*, so if index order carries no
geometric meaning every tile holds some admitted pair and nothing can be skipped. Consequence:
**Route B is useless unless the encoder is also sorting**, which is currently unenforced — worth
an assert, but add it after 4.10 when it is visible how MaskFormer supplies the sort field.
Judge 4.8 by "3 masks, `H=24`, layers differ", never by absolute density on synthetic coords.

### 4.9 — Thread `or_n_hashes` to `Attention`

Already done by 4.8's `or_n_hashes=self.or_n_hashes` in the layer call, because
`EncoderLayer.forward` and `Residual.forward` both take `**kwargs` and forward them. But
`Residual` also wraps `Dense`, which does **not** accept `or_n_hashes`.

Look at `EncoderLayer.forward`: `return self.dense(self.attn(x, **kwargs))` — kwargs only reach
the attention residual, so this is already safe. **Have Luke verify this rather than take my
word**; it is exactly the kind of thing that changes.

**Check:** a forward pass with `or_n_hashes=3` runs and the output shape is unchanged
`(B, N, dim)`.

**DONE (2026-09-11)** — it did fall out of 4.8 as predicted, and `Dense` never sees the kwarg.
Confirmed by the output coming back at `num_heads` width rather than `num_heads * n_hashes`,
i.e. the merge ran. Passing `or_n_hashes=self.or_n_hashes` unconditionally is safe: it is `None`
whenever the feature is off, which is `Attention.forward`'s default.

### 4.10 — Wire the coordinates in `MaskFormer.forward` — **DONE (2026-09-14)**

> **Tests and the full reasoning: [[test_maskformer_or_coords]].**

**`sort_inputs` traced — the worry was justified.** `sorter.py:21-57` is handed **`x`**, mutates it
in place and returns it. It never touches `inputs`. So with a sorter active, `x["key_embed"]` is
permuted and `inputs["sihit_eta"]` is not — building coords from `inputs` would have reintroduced
the 4.6 bug one level up, just as silently.

**The shape that came out of it is simpler than the plan expected.** Do *not* branch on the sorter.
Write the coord fields into `x` **before** the sorter block, and read them back at the encoder call:

```python
# before the sorter block
if self.encoder.or_n_hashes is not None:
    for field in ("eta", "phi"):
        x[f"key_{field}"] = torch.concatenate([inputs[f"{name}_{field}"] for name in self.input_names], dim=-1)

# ... existing sorter block, completely untouched ...

# at the encoder call
x_coords = torch.stack([x["key_eta"], x["key_phi"]], dim=-1) if self.encoder.or_n_hashes is not None else None
```

That one placement covers both paths: a sorter permutes the fields along with everything else, and
without one they stay put and `Encoder.forward` permutes them itself via `x_sort_value` (4.6). The
sorter stops being something the coord code reasons about. *(Luke spotted the duplicated branching
in the first version and asked for it to be compacted — he was right.)*

**Two non-obvious constraints:**
* **Two 1-D fields, never one stacked `(B, N, 2)` tensor.** `sort_inputs` gathers with a `(B, N)`
  index (`sorter.py:54`) and `torch.gather` needs matching rank; 3-D raises there.
* **`torch.stack(..., dim=-1)`, not `concatenate`** — two `(B, N)` tensors into `(B, N, 2)`. And
  **eta first**: `E2LSHOrderingGrid` reads `coords[..., 0]` as eta.

**Safe to write the fields unconditionally** (traced): `unmerge_inputs` only reads `key_embed` and
`key_is_{name}` (`model_utils.py:14-17`), and nothing downstream iterates over all of `x` — the
decoder, tasks and pooling all index it by name. Extra `key_*` entries are inert.

### 4.11 — A test  — **DONE (2026-09-11), 3 passing in `tests/models/test_encoder_or.py`**

> **Full line-by-line walkthrough of the test file: [[test_encoder_or]].** That note explains
> every piece of machinery used (stubs, monkeypatching, `state_dict` copying, forward hooks,
> `BlockMask` internals) from first principles. Read it before editing the tests.

Three tests, in increasing order of what they prove:

1. **`test_identity_ranks_match_plain_sliding_window`** — the load-bearing one. `or_n_hashes=1`
   with ranks forced to the identity collapses Route B to *exactly* the plain sliding window
   (`|rank[q]-rank[kv]|` becomes `|q-kv|`, and both mask_mods use the same `<= window//2`
   convention). Compared against a plain flex encoder carrying the same weights.
2. **`test_head_major_layout_pairs_each_head_with_its_own_ranks`** — head 0 gets the identity in
   *both* its hashes (equal `lse` -> 0.5/0.5 weights -> merges to itself), every other head a
   fixed shuffle. Head 0 must match the plain window; at least one other head must not.
3. **`test_real_grids_build_one_distinct_mask_per_layer`** — smoke test with real grids: shape,
   finiteness, one BlockMask per layer, each of height `num_heads * n_hashes`.

**Why test 2 is not redundant:** with `n_hashes == 1`, `h = head*1 + 0` and `h = 0*num_heads + head`
are the *same map*, so test 1 cannot tell head-major from hash-major. If the two flattenings ever
disagreed, every head would be silently paired with another head's ordering and test 1 would still
pass. Catching it needs `n_hashes >= 2` **and** heads made distinguishable.

**Mechanics worth reusing:**
* The grid stubs are real `nn.Module`s (`ordering_grids` is a `ModuleList`, so it will not accept a
  bare function) and must honour the contract exactly: `(N, 2)` in, `(num_heads, n_hashes, N)` out,
  because `forward` immediately does `.flatten(0, 1)`.
* Per-head outputs are invisible from the encoder's return value — `out_proj` mixes the heads. Tap
  them with a **forward pre-hook on `out_proj`**: `recombine_heads` is only `transpose(-3,-2)` +
  `flatten(-2)`, so its output still has head `h` in columns `[h*head_dim : (h+1)*head_dim]`.
  Remove the handles in a `finally`.
* `create_block_mask` is a free function, so test 3 patches it in the **encoder module namespace**;
  there is no module to hook. Restore it in a `finally`.
* Copy weights with `load_state_dict(..., strict=False)` rather than reseeding (the two configs need
  not consume the RNG identically) and **assert on the returned `missing`/`unexpected`** — bare
  `strict=False` would silently tolerate a renamed parameter.
* `.eval()` + `no_grad()` on both encoders, or drop-path makes them disagree for unrelated reasons.
* Tolerance `rtol=1e-4, atol=1e-5`, not exact: with one hash the merge is nearly an identity but it
  is still a different sequence of kernels, so the last bits can differ.

**The trap test 3 fell into, worth not repeating.** Asserting "the layers built *different masks*"
by comparing `kv_num_blocks` **fails at small N** — at `SEQ_LEN=256` there are 2 blocks of 128, every
mask is fully dense, and three genuinely different orderings are indistinguishable at block
granularity. (Comparing densities is worse: they can collide by coincidence.) Assert per-layer
independence on the **rank tables** instead, where it actually lives, and keep only the
N-independent claims at mask level (count, and `H == num_heads * n_hashes`).

**Not `@pytest.mark.gpu`.** The whole Route B path runs on CPU — flex has a CPU fallback — so the
tests do run in CI, which is where they are wanted. ~26 s on an A100, ~40 s on CPU, nearly all of it
`torch.compile` warm-up. Keep `SEQ_LEN`/`DIM`/`WINDOW` shared across tests so shapes are reused.

### 4.12 — Config and benchmark — **DONE (2026-09-14)**

**Results and the decision they force: §0d.**

**`configs/or-amp.yaml`** — a copy of `base.yaml` with only the encoder block changed. Kept as a
separate file rather than editing an existing one. The three changes that matter:

```yaml
          attn_type: flex        # a mask_mod cannot go into a fixed-function kernel
          window_wrap: false     # LSH treats phi as linear, so it is not periodic
          or_n_hashes: 3         # 8 heads x 3 = 24 orderings per layer
```

`input_sort_field: phi` is inherited from `base.yaml`, which satisfies the new "OR needs a sort"
assert. Validated by constructing the `Encoder` straight from the YAML.

**`scripts/benchmark_or_amplification.py`** — three arms on real event geometry:

* Times **attention only**. Projections, norms and the residual stream are identical across arms,
  so including them would only dilute the difference.
* Times **mask construction separately**, because it is not comparable between arms: route_b
  rebuilds per layer per event, the banded mask is built once. This turned out to be the whole
  story — do not drop it from any rerun.
* **The sorted arm is implemented inside the script**, not in the model: permute q/k/v into each
  cell's order, one shared banded window, gather back before the LSE merge. Un-permuting is a
  `gather` with `ranks` itself, since `ranks` *is* the inverse permutation — no scatter needed.
* **Asserts route_b and sorted agree numerically.** They admit identical pairs, so a difference
  would be a bug, not a result. Keep this assertion in any future version.
* Lays the sequence out in LSH order first (`sort_by_lsh`), because layout is worth ~2x (§0 4a).

Run it with `CUDA_VISIBLE_DEVICES=<idle gpu>`: GPU 0 on the shared node often has ~23 GB held by
someone else, and the flex tests OOM against it in ways that look like real failures.

---

## 2c. Measured facts from the 4.8-4.11 session (2026-09-11)

Four things were measured rather than assumed. Two of them refute claims I had made earlier in
the conversation; both corrections are load-bearing for the §0b decision.

**1. Mask construction is NOT expensive.** ~~(Refutes a worry Luke raised and I amplified.)~~
**WRONG — RETRACTED 2026-09-14. See §0d.** The conclusion below only held because the attention it
was compared against was running *uncompiled* and therefore ~27x slower than it should have been.
Once §2c item 3 was fixed, mask construction became **92% of Route B's per-layer cost**. Luke's
original worry was right and I talked him out of it on the strength of a bad denominator. The
measurement itself was fine; the ratio it was put into was not.

Per layer at N=4096, window 512, 24 cells, on an A100:

| | ms |
|---|---|
| rank tables (grid forward) | 0.75 |
| build the Route B BlockMask | 10.04 |
| flex attention + LSE merge | 270.85 |
| *[ref]* plain banded attention | 14.68 |

Mask building is under 4% of the layer. **Strike "Route B pays an extra O(N^2) mask build" from
the case against it.** (The 271 ms is close to Route B's worst case — synthetic coords gave
density 1.000 here — so do not read the 18x as a verdict either.)

**2. `torch.compile` on `create_block_mask` does NOT recompile per closure. (Refutes my own
flag.)** I claimed the `torch.compile` wrapper at `encoder.py:14` would recompile on every call
because each layer passes a freshly-built closure and dynamo guards on callable identity. Measured,
with a fresh closure *and* fresh ranks every call:

```
 compiled  per-call ms: 14572.2     9.5     9.3     9.3     9.3     9.3
    eager  per-call ms:    92.9    36.2    36.0    36.1    36.2    36.0
```

One 14.5 s compile, then ~4x faster than eager forever. Dynamo traces *through* the mask_mod and
`ranks` becomes a traced tensor input, not a guard constant. **Leave line 14 alone.**

**3. `flex_attention` itself is NOT compiled on this path. — FIXED, and it changed everything.** (Also noted in [[test_encoder_or]] §7.) The test run warns: *"flex_attention
called without torch.compile() — this will use an unfused implementation that materializes the
full scores matrix"*. Correctness is unaffected; **any timing taken from this path is
meaningless**. Must be fixed before 4.12 or Route B will look far worse than it is.

**4. `return_lse` is deprecated**, removed in torch 2.10 in favour of
`return_aux=AuxRequest(lse=True)`. We are on 2.9.1 so it still works. Affects step 2's
`attention.py` flex branch. Not urgent, but do not let it surprise anyone later.

### Environment facts settled this session

* **The V100 nodes cannot validate anything flash-related.** `tests/models/ + tests/flex/` gave
  *82 failed, 124 passed* there — and an identical, test-for-test-identical failure set on
  unmodified HEAD in a separate worktree. All FlashAttention-refusing-pre-Ampere. On an A100:
  **432 passed, 69 skipped, 0 failed.** Always confirm which node before reading a test result.
* **flex attention has a working CPU path**, and the whole Route B chain runs on it. That is why
  the 4.11 tests are not `gpu`-marked.

---

## 2b. Adjacent fixes — **ALL DONE AND COMMITTED (2026-09-14)**

All found while doing step 4, all kept out of the step-4 diff as their own commits. Each would
have made the 4.12 benchmark lie or let a model be silently misconfigured. Kept here as a record
of what was wrong and why, since the reasoning is reusable.

* **`set_backend` loses the sliding window.** `Encoder.set_backend` (`encoder.py:232`) calls
  `layer.attn.fn.set_backend(self.attn_type)` without `window_size`; `Attention.set_backend`
  then resets `self.window_size = (-1, -1)` for flash. So flipping a trained model to the
  `flash` backend at eval time **silently drops the window** and compares Route B against
  *unwindowed* flash. Pre-existing (construction is fine — `attention.py:252` does pass
  `window_size`); only runtime switching is affected. **Fixed:** `Encoder.set_backend` now
  forwards `self.window_size`. Tested in `tests/models/test_encoder_set_backend.py`, and the test
  was checked by reverting the fix.
* **`set_backend` has no Route B guard.** With `or_n_hashes` set, switching to any non-flex
  backend leaves `self.ordering_grids` populated, so `forward` still builds a BlockMask and
  hands it to a backend that rejects it (`attention.py:453`). Add an assert mirroring the
  `__init__` one. **Fixed:** `set_backend` now refuses any non-flex backend while
  `ordering_grids` is populated.
* **Route B silently does nothing without a sort.** See the table under 4.8: with no
  `x_sort_value` the density is 1.000 at every N tried. An encoder configured with `or_n_hashes`
  but no sort field is a silent no-op that still pays the full replication cost. **Fixed:**
  `MaskFormer.__init__` refuses `or_n_hashes` without either `input_sort_field` or a `sorter`.

Two more of the same kind, found later and also fixed:

* **The OR merge returned the wrong dtype.** `flex_attention` hands back `lse` in float32 whatever
  the inputs are, so the weighted sum promoted a half-precision output and `out_proj` died with
  "mat1 and mat2 must have the same dtype". Latent since step 2 and invisible to the tests,
  which all run in float32 — but training and benchmarking run in half, so **Route B would have
  crashed on its first real forward pass.** Found by running at realistic size and precision, not
  by testing. `or_merge_lse` now casts back to the input dtype.
* **flex ran uncompiled everywhere.** See §2c item 3.

## 3. Step 5 — padding — **DONE (2026-09-15)**

Events have different numbers of hits, so short ones are padded to a fixed length. Those fake
tokens have no position, and if nothing is done about them they corrupt the ordering of the real
hits — silently, as usual.

**Three separate things had to be fixed, and any one left out breaks it.** The original plan only
had one of them.

### 5.1 — Keep padding out of the *statistics* (this was missing from the plan entirely)

The ordering is not computed per hit in isolation. Two parts of it are properties of the whole
event:

* **`span`**, the max minus the min of the projection, which scales the bin offsets.
* **the quantile bins**, which are equal-occupancy — "chop the hits into 7 groups of equal
  population", so the edges depend on the distribution.

A padded coordinate (usually 0) sits inside both. It drags the span and shifts every bin edge,
which **reorders genuine hits**. No error, no NaN: the model just trains on slightly wrong
geometry forever.

Fixed in `lsh_order_values` by sending padded coordinates to `+inf` before any statistic is
computed, so they fall outside the max/min and sort past every real token, and by dividing the
bin width by the number of *real* tokens so the bins stay equal-occupancy over them.

### 5.2 — The `inf` trick, and why it is only half the answer

Padded tokens are then given an ordering value of `+inf`, so they sort to the tail of every
cell. That stops them being interleaved among real hits, where they would eat window capacity
from every real query.

**It does not stop them being attended.** They occupy the last `N_pad` slots, so the real tokens
holding the highest ranks are still within half a window of them. The plan of record said the
`inf` trick alone was sufficient; it is not.

### 5.3 — An explicit validity term in the mask

* `per_head_window_mask_mod(ranks, window_size, kv_valid=...)` for the masked path, indexed by
  **token** because nothing moves.
* `sorted_window_mask_mod(window_size, num_valid=...)` for the sorted path, indexed by
  **position**.

> [!tip] Why the sorted path can still share one mask
> Because padding sorts to the tail of *every* cell, after the permutation it occupies the **same
> trailing block of positions in all of them**. So "not padding" is `kv_idx < num_valid` — a
> statement about position that is identical for every cell, which is exactly the property the
> shared mask needs. Had padding landed in different places per cell, the sorted path would have
> needed a per-cell mask and lost its entire advantage.

**Both mask_mods always admit the diagonal.** A padded query has no valid keys, so without it the
row would be all `-inf`: an empty softmax, a NaN output, and a NaN `lse` that then spreads
through the merge into the *real* tokens. Its own output is meaningless and is discarded
downstream.

### 5.4 — The flex `kv_mask` assert

Left alone. Validity is already folded into the BlockMask, so the encoder simply does not forward
`kv_mask` to the attention layer on the OR paths — cleaner than relaxing an assert that is still
correct for plain flex, and it avoids applying the same constraint twice.

### Tests

`tests/models/test_encoder_or.py`, both parametrised over `or_impl`:

* **`test_padding_does_not_change_the_answer_for_real_tokens`** — the load-bearing one. Runs the
  same real hits with and without padding attached and demands identical outputs. This is what
  catches 5.1, which nothing else would.
* **`test_padded_queries_keep_a_finite_softmax_row`** — mostly-padding worst case; catches a
  missing diagonal.

Both were mutation-tested: reintroducing the contaminated statistics fails the first on both
implementations, and removing the diagonal escape fails the second.

> [!warning] The compile gate had to be fixed again while doing this
> `torch.cuda.is_available()` was the wrong test — it asks whether the *machine* has a GPU, not
> where the tensors are, so a CPU model on a GPU box still got a compiled flex and still hit the
> `return_lse` lowering error. `Attention` now keeps both the compiled and uncompiled callables
> and picks per call from `q.is_cuda`.

---

## 4. Standing notes on how to run this

* Luke writes the code. Give one numbered step at a time, with the *why* and an observable
  check. Do not paste the finished block unless he asks or is stuck after a try.
* Ask him to predict the failure mode before 4.6 and 4.7 — those two are the ones that fail
  silently, and the whole value of doing it by hand is meeting them.
* Working python is `.pixi/envs/default/bin/python`. `pip` and `~/.local/bin/hf` are broken.
* **Check the GPU before believing a red test run.** A V100 fails 82 attention tests (pre-Ampere,
  no flash) and a busy or 40 GB card OOMs two `tests/flex` tests when the whole suite shares one
  process. Both reproduce on unmodified HEAD. Run with `CUDA_VISIBLE_DEVICES=<idle gpu>` and, if
  something fails, reproduce it on a stashed tree before believing it is yours.
* **Run things at realistic size and precision, not just in tests.** The fp16 dtype bug in the OR
  merge (§2b) would have crashed the first real forward pass and fifteen passing tests never saw
  it, because they all run float32 at N=256. One throwaway script at N=13k in half found it
  immediately.
* **Line numbers go stale every step.** Re-grep and refresh §0c rather than quoting from
  memory; Luke has (rightly) asked twice whether the quoted numbers were still valid.
* Luke asks for conceptual re-explanations often and wants them *simple* and without code
  identifiers — he has asked several times for meeting-ready phrasing. Useful framings that
  landed: recipe-vs-ordering for "frozen"; grid cell vs quantile bin (the word "cell" was
  overloaded and caused real confusion); hash-as-recipe vs hash-as-value (the `inf`-hash
  padding trick uses the second sense); tiles are chunks of the *pair matrix*, not of the
  sequence.
* He independently re-derived the sorted variant from first principles, and separately spotted
  that mask *construction* is itself O(N^2) work per layer — **and he was right about the second
  one when I was not** (see the retraction in §2c item 1). Take his instincts seriously.
