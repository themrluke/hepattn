# Deeper look at the timing runs: stability within a run, outlier steps, matcher vs rest, paired step-by-step comparison.
import glob, json, sys
import numpy as np

LO, HI = 150, 600
D = sys.argv[1]
runs = {}
for f in sorted(glob.glob(D + "/*_*_[0-9].json")):
    stage, variant, rep = f.split("/")[-1][:-5].split("_")
    d = json.load(open(f))
    t = np.array(d["step_ends"]); m = np.array(d["match_times"])
    dt = np.diff(t)            # dt[i] = duration of step i+1 (step 0 has no start stamp)
    step = dt[LO - 1:HI - 1]   # steps LO..HI-1
    mt = m[LO:HI]              # matcher time of the same steps
    runs[(stage, variant, rep)] = (step, mt)

print("== 1. Stability within each run: rate in 9 blocks of 50 steps (steps 150-600)")
for k, (step, mt) in runs.items():
    blocks = [50 / step[i:i + 50].sum() for i in range(0, 450, 50)]
    cv = np.std(blocks) / np.mean(blocks)
    print(f"{'_'.join(k):20} " + " ".join(f"{b:5.2f}" for b in blocks) + f"   block spread (CV) {100 * cv:4.1f}%")

print("\n== 2. Outlier steps (> 2x the run's median) and how much time they cost")
for k, (step, mt) in runs.items():
    med = np.median(step); out = step > 2 * med
    print(f"{'_'.join(k):20} median {med:.3f}s  p90 {np.percentile(step, 90):.3f}s  max {step.max():.2f}s  "
          f"outliers {out.sum():3d}  extra time {((step - med)[out]).sum():5.1f}s of {step.sum():6.1f}s  "
          f"rate without them {len(step) / np.where(out, med, step).sum():.3f}")

print("\n== 3. Step = matcher + rest (mean seconds per step); 'rest' should be the same for every variant")
for k, (step, mt) in runs.items():
    print(f"{'_'.join(k):20} step {step.mean():.3f}  matcher {mt.mean():.3f}  rest {step.mean() - mt.mean():.3f}")

print("\n== 4. Paired comparison: same events in the same order, step by step")
for stage in ("late", "early"):
    base = [v for k, v in runs.items() if k[0] == stage and k[1] == "original"]
    for variant in ("prep", "trim"):
        for k, (step, mt) in runs.items():
            if k[0] != stage or k[1] != variant:
                continue
            for i, (bstep, bmt) in enumerate(base):
                saved = bstep - step
                print(f"{stage} original#{i + 1} vs {variant}#{k[2]}: time saved per step median {np.median(saved):.3f}s, "
                      f"saved on {100 * (saved > 0).mean():.0f}% of steps; matcher saved median {np.median(bmt - mt):.3f}s; "
                      f"corr(original step, {variant} step) {np.corrcoef(bstep, step)[0, 1]:.2f}")
print("\n== 5. Repeat-to-repeat: same variant, same events")
for stage in ("late", "early"):
    for variant in ("original", "prep", "trim"):
        reps = [v for k, v in runs.items() if k[0] == stage and k[1] == variant]
        if len(reps) > 1:
            a, b = reps[0][0], reps[1][0]
            print(f"{stage} {variant}: rep1 vs rep2 step corr {np.corrcoef(a, b)[0, 1]:.2f}, "
                  f"median |diff| {np.median(np.abs(a - b)):.3f}s, total {a.sum():.1f}s vs {b.sum():.1f}s")
