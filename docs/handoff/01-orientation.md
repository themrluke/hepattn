# 01 — Orientation

## What this project is

`hepattn` is a transformer-based reconstruction framework for high-energy physics: it takes
detector hits and reconstructs particles (tracking) using a MaskFormer-style model — an encoder
over hits, then object queries that each predict a mask over the hits belonging to one particle.

This fork is Luke Johnson's (`themrluke/hepattn`, forked from `samvanstroud/hepattn`, which is
the `upstream` remote). The work here has two strands, described below.

## Strand 1 — OR amplification (implemented, not yet judged)

**The idea.** The encoder currently uses a sliding-window attention over hits sorted by phi.
A window is cheap but arbitrary: two hits on the same track can fall outside each other's
window. OR amplification, borrowed from LSH-based attention (HEPT), instead builds several
*orderings* of the hits, each of which places hits that are close in (eta, phi) near each other,
runs a windowed attention in each ordering, and merges the results. A pair of hits attends if
**any** ordering brings them together — hence "OR".

**Status: complete and committed on `OR-amplification`, which is pushed.** Implementation,
tests and benchmarks are all in place, including padding handling. Full detail in
[`02-or-amplification.md`](02-or-amplification.md).

**What is not done: knowing whether it helps.** The cost is measured (about 5.6x a plain
sliding window per encoder layer; about 1.44x per training step). The benefit is not, because
the A/B comparison was run on a ColliderML configuration whose baseline does not train well —
both arms came out equally poor, so the comparison was uninformative.

## Strand 2 — ColliderML baseline (current focus)

**The problem.** Training runs on ColliderML gave poor metrics for both the baseline and the OR
arm. The likely cause is the event selection in `configs/base.yaml`, not the model: it keeps
only pixel barrel hits (volume 17) with |eta| < 1, which leaves about **4.2 hits per particle**.
That is too few hits to reconstruct a track from, at any pile-up.

**The plan Luke chose.** Get the baseline working on a simpler no-pile-up (pu0) sample first,
with a wider selection, then step back up to pu200, and only then run the OR comparison on a
baseline that works. Detail and measurements in
[`03-colliderml-baseline.md`](03-colliderml-baseline.md).

**Progress so far** (branch `colliderml-baseline`, two commits, not yet pushed at the time of
writing — check):
- 100k pu0 training events plus 2k validation events downloaded (on the *old* machine; they must
  be downloaded again on the new one — see [`04-datasets.md`](04-datasets.md)).
- `ColliderMLDataset` gained a `pileup` argument (the `pu200` part of the directory names was
  hard-coded).
- `configs/pu0.yaml` written: same as `base.yaml` apart from the data block.
- A 200-step smoke test ran cleanly at about 4.8 steps/s.

## What comes next

In order:

1. **Download the data on the new machine** — [`04-datasets.md`](04-datasets.md) — and update the
   `train_dir` / `val_dir` / `test_dir` paths in the ColliderML and TrackML configs, which still
   point at the old cluster's `/share/rcif2/...` paths.
2. **Sanity-check the model can overfit a handful of pu0 events.** Train on 1–10 events and
   confirm efficiency approaches 100%. If it does not, something is broken in the targets, masks
   or particle–hit association, and no amount of training data will fix it. This takes minutes
   and had not been run yet.
3. **Full pu0 training run.** Watch whether validation metrics reach something respectable. Note
   the configured 10 epochs is roughly 2–3 days at the measured rate; consider fewer.
4. **Return to pu200** with the same widened selection. Expect to need a hit filter first — see
   the CSR/memory discussion in [`03-colliderml-baseline.md`](03-colliderml-baseline.md).
5. **Then, and only then, the OR A/B comparison**, on a baseline that works. Note that pu0 itself
   cannot test OR: a pu0 event has only about 3.4k hits, so a 512-wide window already covers most
   of the event and OR has nothing to add. TrackML is the other candidate host for this
   comparison, since its baseline is known to work — that discussion is in
   [`03-colliderml-baseline.md`](03-colliderml-baseline.md#where-to-run-the-or-comparison).

## Open questions awaiting Luke's decision

These were raised and never resolved. Do not act on them unilaterally.

- **Slide 13 caption** in the presentation says "one random line per cell"; it should read "one
  random line per ordering — shared by every bin in it". Offered, not approved.
- **Slide 23** shows block densities that are only meaningful for the `masked` implementation;
  it should be labelled as such. Offered, not approved.
- **Slides 13–14 attribution** to E2LSH could be softened, since our scheme uses axis-aligned
  quantile bins plus one projection rather than the paper's m_AND projections.
- **The MaskFormer assert** requiring `input_sort_field` or a `sorter` when OR is enabled is
  over-strict for the `sorted` path, which does its own permutation.
- **HEPT m_AND discrepancy** — the HEPT v1 paper (arXiv 2402.12535) was never fetched to settle
  whether the paper's m_AND E2LSH projections differ from what the repo actually does. Our
  implementation matches the *repo*.
- **`return_lse` deprecation** in torch 2.10 will need attention eventually.
