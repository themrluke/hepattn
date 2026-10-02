Before training anything, I checked whether my fork had the latest upstream code for every decoder in the [[Plan]]. It did not: the code behind the June paper had never been merged into upstream `main`.

# What upstream has

| Upstream branch | Date | What it is | In my fork before the merge? |
|---|---|---|---|
| `main` | 27 Feb 2026 | Dynamic queries (#241), k-max (#251), basic LSCA | **Yes**, everything |
| **`hepattn-clean`** | 4 Jun 2026 | **The paper code.** Newer LSCA mask, much larger `task.py`, memory/timing callbacks, cleaned-up evaluation, and configs matching the paper's Table 2. Keeps k-max | No (12 commits) |
| `pippa/hepattn-dq-gcp` | 4 Jun 2026 | Sister branch of `hepattn-clean`: same configs, different `task.py`/decoder, **removes k-max** | No |
| `pippa/working-lca` | 3 Mar 2026 | LSCA with dynamic queries, open as PR #253 | No |
| `dq-cleanup` | 15 Feb 2026 | Sam's matcher fix for dynamic queries. **In no branch at all** | No |
| `pippa/dq-no-padding` | Feb 2026 | Stop padding dynamic queries; PR #249 closed unmerged | No |

- **Sparse k-max does not exist upstream.** The k-max PR had a "sparse" option that reuses mask logits instead of recomputing them; it was dropped and restored before merging, so it lives in `main` as a speed-up inside k-max, not as a separate decoder.
- I'm confident `hepattn-clean` is the paper code because its configs line up with Table 2: `tracking-eta4-pt900-loss-weights.yaml` = Pix1.0 DQ+MA (8 / 3 layers, 2000 queries, focal 25), `tracking-lca-eta4-900.yaml` = Pix1.0 DQ+LSCA (window 64).

# The merge

Merged `upstream/hepattn-clean` into `TrackML-baseline` on a new branch, `merge/hepattn-clean`. Three conflicts:

| File | Resolution | Why |
|---|---|---|
| `.gitignore` | Kept mine, dropped theirs | Theirs ignored **every** `configs/` directory, so any new config would silently never be committed |
| `utils/sorter.py` | Kept both sides | Mine: the `_sort_key` hook for `LearnedSorter`. Theirs: a real bug fix that matches keys by whole word instead of substring (`"hit"` was matching `paper_all_particle_n_hits`) |
| `tracking-eta4-pt600-epochs.yaml` | Took theirs | Keeps the paper config exactly as UCL ran it; data paths changed in a separate commit |

Also deleted `tests/flex/test_fast_local_ca.py`, which tested a module the merge removes.

# Bugs found in the paper branch

> [!bug] `hepattn-clean` crashes on every decoder forward pass
> `NameError: name 'logits' is not defined` at `decoder.py:392`. When the branch merged `main` on 7 May, the three lines that set `logits` were lost but the line using them stayed. Fixed by restoring the three lines exactly as on `main`. **Whoever produced the paper's numbers was not running this exact commit.**

- **Two outdated decoder tests:** upstream changed the decoder to record attention masks only when `debug=True`, and to record them *before* the "no allowed hits → attend everywhere" fallback, but didn't update the tests. Updated the tests to match (expected count 65 → 5).
- **Duplicated mask-building block** in `decoder.py`, left over from one of upstream's merges. It gives correct results but builds the mask twice per layer; left as is to stay identical to the paper code.

# Result

| Commit | What |
|---|---|
| `c4b1e1b` | merge the paper-era hepattn-clean branch from upstream |
| `36515a9` | restore the kmeans logits lookup lost in the hepattn-clean merge |
| `5ad4286` | pass debug to the decoder tests that check the recorded attention masks |
| `6a0174f` | point the pix1.0 trackml configs at the eos dataset (branch `trackml-pix1p0`) |

**Tests:** `tests/models` 409 / 409 passing (same count as before the merge); `tests/flex` 14 / 14 (34 removed with the deleted module). **All OR-amplification tests pass**, even though the merge changed `maskformer.py`, `task.py` and `decoder.py`.

Neither branch is pushed yet.

# Questions for UCL (Pippa)

1. Which branch produced the paper's numbers: `hepattn-clean` or `hepattn-dq-gcp`?
2. Should Sam's `dq-cleanup` matcher fix be included?
3. What is "sparse k-max", and where is its code?
4. The DQ+MA config says 10 epochs; the paper says 30. Which?
5. How did their filter handle the [[Corrupt TrackML events]]?
