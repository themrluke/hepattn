# 02 — OR amplification

All of this is implemented, tested and committed on `OR-amplification` (pushed, tree clean at
`3e39532`). This document exists so a new session does not have to re-derive the design from the
code.

## The idea in one paragraph

The encoder attends over hits in a sliding window after sorting them by phi. That window is
cheap but arbitrary — two hits on the same track can easily fall outside each other's window.
Instead, build several **orderings** of the hits, each of which tends to place hits that are
close in `(eta, phi)` near each other, run the same windowed attention in each ordering, and
merge the results. A pair of hits attends if **any** ordering brings them within a window, which
is the "OR" in OR amplification. The scheme is borrowed from LSH-based attention (HEPT).

## The ordering (E2LSH-style)

Source of truth: `src/hepattn/models/ordering.py`, whose module docstring is worth reading.

Each hash table is two frozen pieces:

- `alpha` — one random Gaussian projection of the 2D `(eta, phi)` coordinates.
- `region_counts` — `[eta_count, phi_count]`, equal-occupancy **quantile bin** counts along each
  axis.

The ordering value of hit `i` is

```
o_i = alpha . (eta_i, phi_i)  +  R_eta . D  +  R_phi . D . (ceil(eta_count) + 1)
```

where `R_eta`, `R_phi` are the hit's 1-based quantile-bin indices and `D` is the span of the
projection. Sorting by `o` groups hits by 2D quantile cell (phi-major, eta-minor) and, within a
cell, by the projection. The bins are coarse; the projection is a fine tie-break inside a bin.

Two points that were repeatedly misunderstood and are worth stating plainly:

- **The grid is axis-aligned and never rotated.** The bins are quantile bins on eta and on phi.
- **There is one projection line per hash table, shared by every bin in that table** — not one
  line per cell. (The presentation's slide 13 caption still says "per cell"; see the open
  questions in `01-orientation.md`.)

`argsort(o)` is the ordering; `argsort(argsort(o))` is the **rank**, which is what a sliding-window
`mask_mod` consumes. Tables are drawn once and frozen, so a given event always serialises the
same way. One table is drawn per `(layer, head, hash)`, seeded as
`base_seed + layer_idx * (num_heads * n_hashes) + head * n_hashes + hash`, so `base_seed = 0`
reproduces the benchmark tables bit-for-bit.

## The OR merge

Each ordering `c` produces, per hit, an attention output vector (`head_dim` long) and one fp32
log-sum-exp scalar. They are merged with a softmax over the LSE values:

```
out = sum_c softmax(lse)_c * out_c
```

This is exactly the merge you would get from a single attention over the union of the windows.
Implemented as `or_merge_lse` in `attention.py`; it casts back to the input dtype (an fp16
mismatch bug was fixed there).

## The two implementations

Both produce **identical** output; a test asserts this.

| | `sorted` (default) | `masked` |
|---|---|---|
| How | Permutes q/k/v into each ordering, uses one shared position-only banded mask | Leaves tokens in place, gives each ordering its own rank window inside a taller mask (height H*C) |
| Mask | Shared, cacheable, `H=None` | Rebuilt per layer, per event |
| Un-permute | Gather with ranks | Not needed |

Measured per encoder layer at N ~ 13k hits:

| | mask build | attention | total | vs window |
|---|---|---|---|---|
| flash + window (today) | — | 0.26 ms | **0.26 ms** | 1x |
| `masked` | 40.3 ms | 3.70 ms | **44.0 ms** | 169x |
| `sorted` | 1.9 ms (shared, cacheable) | 1.94 ms | **3.8 ms** | 15x |

Per full event: `sorted` **23.2 ms** vs `masked` **158.1 ms** (6.8x). Against today's window,
`sorted` is **5.6x** (13.9 ms vs 2.5 ms), and about **1.44x per training step** once the rest of
the model is included. Block densities: `masked` 0.232, banded 0.048.

`sorted` is the default because of these numbers. `masked` is kept as the reference
implementation and for benchmarking — do not delete it.

## Where the code lives

| File | Role |
|---|---|
| `src/hepattn/models/ordering.py` | `hept_region_counts`, `draw_lsh_table`, `lsh_order_values`, `E2LSHOrderingGrid` |
| `src/hepattn/models/encoder.py` | `or_*` arguments, per-layer ordering grids, rank computation, both paths |
| `src/hepattn/models/attention.py` | `permute_to_rank_order`, `restore_token_order`, `or_merge_lse`, compile handling |
| `src/hepattn/flex/per_head_window.py` | `per_head_window_mask_mod` (masked), `sorted_window_mask_mod` (sorted) |
| `src/hepattn/models/maskformer.py` | Writes `key_eta` / `key_phi` and stacks them into `x_coords` |
| `tests/models/test_encoder_or.py` | The main test file: invariance, layout, agreement, padding, caching |
| `tests/models/test_maskformer_or_coords.py`, `tests/models/test_or_merge.py` | Coordinate plumbing, merge dtype |
| `src/hepattn/experiments/colliderml/scripts/benchmark_or_amplification.py` | Three-arm benchmark; asserts `masked` == `sorted` |
| `src/hepattn/experiments/colliderml/scripts/measure_block_sparsity.py` | Block density measurements |
| `src/hepattn/experiments/colliderml/configs/or-amp.yaml` | Config with OR enabled |

## Configuration

```yaml
encoder:
  attn_type: flex        # required: the per-head window is a mask_mod, no fixed kernel can take it
  window_size: 512
  window_wrap: false     # required: LSH treats phi as linear, so it is not periodic
  or_n_hashes: 3         # None turns the feature off
  or_num_regions: 100
  or_base_seed: 0
  or_impl: sorted        # or "masked"
```

Asserts in `Encoder.__init__` enforce the first three. `MaskFormer` additionally asserts that an
`input_sort_field` or a `sorter` is configured when OR is on — which is over-strict for the
`sorted` path (see open questions).

## Things that bit us, so they do not bite again

- **flex attention must be compiled on the GPU.** Compiled: 11.2 ms and 0.11 GB; uncompiled:
  44.5 ms and 8.53 GB. But flex with `return_lse` cannot be compiled on CPU. The fix, committed
  as `ad29751`, is to choose per call from the tensor: `attn_fn = self.attn if q.is_cuda else
  self.attn_uncompiled`. Do not gate this on `cuda.is_available()` — that breaks CPU tests on a
  GPU machine.
- **A cached mask closure captured the first event's `num_valid`.** Fixed by writing into an
  in-place tensor (`self.num_valid[0] = ...`) that the closure reads through.
- **Padding.** Padded slots are excluded from span and quantile statistics, pushed to `+inf` so
  they sort to the end, and excluded by an explicit validity term in the masks. The diagonal is
  always admitted, so a padded query still attends to itself and cannot produce NaNs.
- **A presort of the hits affects tile density only for the `masked` path.** The `sorted` path
  permutes the tokens itself, so it is immune to how the input arrived. An earlier claim that a
  presort would halve tile density for the default was wrong.
- **BlockMask is not the same thing as block-sparse.** BlockMask is flex attention's data
  structure for skipping 128x128 tiles; block-sparse is a *pattern* (block-diagonal). Our model
  uses a sliding window, expressed through a BlockMask.
- **HEPTv2 uses block-sparse, not a sliding window.** Confirmed by reading their repo:
  `heptv2/model/attention.py` lines 72–87 (`sort_to_buckets` reshape), line 116
  (`_flex_attention(t_query, t_key, t_value, return_lse=True)` with no mask), lines 220 and
  236–237; `transformer.py` has no window or sliding logic. Block size 1024, m_OR = 3, m_AND = 2.
  Colleagues doubted this; the evidence above is what settles it.
- **Two tests OOM when the whole suite runs in one process.** `tests/flex/test_flex_basic.py` and
  `tests/flex/test_nested.py` fail if `tests/models/` and `tests/flex/` share a process on a
  40 GB card. `tests/flex/` alone passes 48/48, and both failures reproduce on unmodified `main`.
  Pre-existing, not caused by this work. Expect "2 failed, 455 passed, 69 skipped".

## Presentation

`src/hepattn/experiments/colliderml/notes_for_obsidian/presentation/` (carried on this branch)
holds a 33-slide deck, `OR_amplification.pptx`, built by `build_deck.py` from figures generated
by `figs_*.py`, plus a matching speaker script in `SCRIPT.md`. To rebuild, run the scripts with
the project Python; the `.venv/` there (python-pptx, pypdf) was **not** committed, so recreate it
if needed.

Two notes on the deck: runs are tagged `MSO_LANGUAGE_ID.ENGLISH_UK` only — a previous attempt to
also set `noAutofit` and `latinLnBrk` broke captions and was reverted. And the deck's contents
were frozen at Luke's request: fix layout, do not change wording or delete anything.
