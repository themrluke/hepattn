# 03 — ColliderML baseline (current focus)

Branch: `colliderml-baseline`, branched from `main` (deliberately: none of the OR code is needed
for this, and `data.py` and `base.yaml` are identical on `main` and `OR-amplification`).

Two commits:

```
53a3454 add a pu0 ColliderML tracking config with all tracker volumes out to |eta| 2.5
d0b0e03 add a pileup argument to the ColliderML dataset
```

## The problem

Training runs on ColliderML (baseline on one GPU, OR on another, both logged to the `colliderml`
Comet project) produced metrics that were poor *and* indistinguishable between the two arms.
The A/B told us nothing, because the baseline itself was not learning.

The cause is almost certainly the event selection in `configs/base.yaml`, not the model:

```yaml
sihit_volume_ids: [17]      # pixel barrel only
sihit_max_abs_eta: 1.0
particle_max_abs_eta: 1.0
```

Measured on 200 pu0 events, that selection leaves about **4.2 hits per particle** — far too few
to reconstruct a track from, at any pile-up.

| Selection | hits/event | target particles/event | hits/particle |
|---|---|---|---|
| Old: pixel barrel (vol 17), \|eta\|<1 | 258 | 20 | **4.2** |
| New: all tracker volumes, \|eta\|<2.5 | 3,433 | 66 (max 151) | **11.8** |

(For scale, a pu200 event has about 222k raw hits, of which about 162k have |eta| < 2.5 and
about 52k have |eta| < 1.)

## The plan

Luke's plan, agreed rather than imposed: get the baseline working on a **no-pile-up (pu0)**
sample first — fewer hits and particles, so if the model cannot learn *that*, the fault is in the
pipeline rather than the difficulty — then step back up to pu200, and only then run the OR
comparison against a baseline that works.

**pu0 cannot test OR amplification.** A pu0 event has about 3.4k hits, so a 512-wide window
already covers a large fraction of the event and OR has nothing to add. pu0 is purely a
baseline-debugging stage.

## What was done

**`data.py`:** `ColliderMLDataset` had `dataset_prefix = f"{event_type}_pu200"` hard-coded, so pu0
directories could not be loaded. There is now a `pileup: int = 200` argument; existing configs
are unaffected.

**`configs/pu0.yaml`:** a copy of `base.yaml` with only the data block changed —

```yaml
train_dir/val_dir/test_dir: .../pu0/train, .../pu0/val
pileup: 0
particle_max_abs_eta: 2.5
sihit_volume_ids: null        # every tracker volume: pixel, short strip, long strip; barrel + endcaps
sihit_max_abs_eta: 2.5
max_num_particles: 256        # and hence num_queries; max seen in 200 events was 151
num_workers: 8
num_val: 1000
```

**Event count: 100k train, 2k val.** A pu0 event has only about 66 target particles against
thousands in a pu200 event, so pu0 needs far more events to see a comparable number of particles:
100k events is about 6.6M particles. It costs about 34 GB. pu0 shards hold **1000 events each**
(pu200 shards hold 100). Split used: shards 00000–00001 as validation, 00002–00101 as training,
so adding training data later just means adding more shards.

**Smoke test:** 200 training steps plus validation ran cleanly on an A100 at about 4.8 steps/s
including compile warm-up. That puts one epoch of 100k events at roughly 5–6 hours, so the
configured `max_epochs: 10` is about 2–3 days. Consider fewer epochs for a first run.

A wrinkle worth knowing: the project CLI injects a Comet-only argument into whatever logger is
configured, so `--trainer.logger false` or a CSVLogger override fails, and the LR-monitor callback
refuses to run without a logger. For a quick local smoke test, copy the config to a scratch file
with `logger: false` and the `LearningRateMonitor` callback removed.

## What to do next

1. **Overfit sanity check — do this before any long run.** Train on 1–10 events and confirm
   efficiency approaches 100%. If it does not, the bug is in the targets, masks or particle–hit
   association, and more data will not help. Minutes to run; never actually run yet.
2. **Full pu0 run** on the new machine.
3. **Then pu200** with the same widened selection.

## On CSR, dense masks, and what the real memory limit is

Luke asked whether "step 0" should be rewriting the dataloader or the model to handle CSR
matrices, since the selections are currently kept small to fit dense matrices. The answer is no,
for these reasons:

- **At pu0 it is irrelevant.** The dense particle–hit matrix is 256 x 3.4k booleans, under 1 MB.
- **At pu200 the dataloader is not the bottleneck either.** With the wider selection an event has
  about 162k hits, so the dense target matrix is about 250 MB per event on the CPU — large but
  manageable. The dataset already builds CSR components (`_build_csr_components`, using
  `scipy.sparse`) and `build_dense_masks: false` already exists to skip materialising the dense
  mask for consumers that only need the CSR.
- **The real limit is on the GPU, and CSR targets do not touch it.** The model predicts a
  queries x hits mask in *every* decoder layer, and the mask losses need gradients for it. At
  about 1,500 queries x 162k hits that is roughly 1 GB per layer in fp32, before gradients and
  before the intermediate-loss copies.
- **The standard fix is a hit filter first**, as in the TrackML pipeline: a per-hit model drops
  hits unlikely to belong to a target particle, and the tracking model then runs on what
  survives. `configs/filtering-lite.yaml` already exists for ColliderML to build on. Sparse or
  chunked mask losses would be a much bigger rewrite and are only worth it if filtering is not
  enough.

## Where to run the OR comparison

Still undecided; both options were discussed.

**TrackML** is attractive because its baseline is known to work, so a gain or loss from OR shows
up clearly, and the OR code needs no changes (MaskFormer builds the OR coordinates from
`hit_eta` / `hit_phi`, and the TrackML configs already provide `eta` and `phi`). Caveats: the
data was not present on the old machine at all; `tracking.yaml` depends on a trained hit filter's
output (`hit_eval_*` h5 files) so `tracking-lite.yaml` is the cheaper first test; and the TrackML
configs use `attn_type: flash` with `window_wrap: true`, whereas OR requires `flex` and
`window_wrap: false`. A fair comparison therefore needs the baseline arm switched to flex with no
wrap, plus ideally a third run with the original flash + wrap to confirm that the switch alone
changes nothing.

**ColliderML pu200** is the other option, and is the dataset the rest of this work is built
around. It needs the baseline fixed first, which is what this branch is for.
