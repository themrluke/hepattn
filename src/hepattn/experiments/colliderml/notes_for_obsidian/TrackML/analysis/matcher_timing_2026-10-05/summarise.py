# Steady-state timing from the per-step timestamps: steps 150-600 only (after compile and recompiles).
import glob, json, sys
import numpy as np

LO, HI = 150, 600
rows = {}
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    stage, variant, rep = f.split("/")[-1][:-5].split("_")
    d = json.load(open(f))
    t = np.array(d["step_ends"])
    m = np.array(d["match_times"])
    dt = np.diff(t)[LO - 1 : HI - 1]          # durations of steps LO+1..HI
    rate = len(dt) / dt.sum()
    match = m[LO:HI].mean() if len(m) >= HI else float("nan")
    rows.setdefault((stage, variant), []).append((rate, np.median(dt), match, len(t)))
    print(f"{stage:5} {variant:8} rep{rep}: {rate:.3f} steps/s  median step {np.median(dt):.3f}s  matcher {match:.3f}s/step  ({len(t)} steps)")
print()
for stage in ("late", "early"):
    base = rows.get((stage, "original"))
    if not base:
        continue
    b = np.mean([r[0] for r in base])
    for v in ("original", "prep", "trim"):
        r = rows.get((stage, v))
        if r:
            rates = [x[0] for x in r]
            print(f"{stage:5} {v:8}: {np.mean(rates):.3f} steps/s (n={len(r)}, spread {min(rates):.3f}-{max(rates):.3f})  "
                  f"matcher {np.mean([x[2] for x in r]):.3f}s/step  speed-up {np.mean(rates) / b:.2f}x")
