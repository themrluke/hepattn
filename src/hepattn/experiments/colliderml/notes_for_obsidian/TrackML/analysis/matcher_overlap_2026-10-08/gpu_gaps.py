"""Where does the GPU sit idle? From a bench_run.py Chrome trace: every GPU idle gap inside the profiled steps, attributed
to what the main (step) thread was doing when the gap started (the innermost CPU op that spans that moment)."""
import bisect
import collections
import json
import sys

ev = json.load(open(sys.argv[1]))["traceEvents"]
steps = [e for e in ev if e.get("name", "").startswith("ProfilerStep#") and e.get("cat") == "user_annotation"]
if not steps:
    steps = [e for e in ev if e.get("name", "").startswith("ProfilerStep#")]
main_tid = collections.Counter(e["tid"] for e in steps).most_common(1)[0][0]
start, end = min(e["ts"] for e in steps), max(e["ts"] + e["dur"] for e in steps)
n_steps = len({e["name"] for e in steps})

gpu = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e)
gaps, cur = [], start
for s, e in gpu:
    if e <= start or s >= end:
        continue
    if s > cur:
        gaps.append((cur, s))
    cur = max(cur, e)
if cur < end:
    gaps.append((cur, end))

cpu = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in ev
             if e.get("ph") == "X" and e.get("tid") == main_tid and e.get("cat") in ("cpu_op", "user_annotation", "cuda_runtime", "python_function") and "dur" in e)
starts = [c[0] for c in cpu]
def doing(t):
    # innermost op on the main thread spanning time t (latest start among those still running)
    best = None
    i = bisect.bisect_right(starts, t)
    for s, e, name in cpu[max(0, i - 4000):i]:
        if s <= t < e and not name.startswith("ProfilerStep") and (best is None or s >= best[0]):
            best = (s, name)
    return best[1] if best else "(nothing on the main thread)"

def outer(t):
    # outermost interesting range (compiled graph, optimizer, autograd) for context
    for s, e, name in cpu:
        if s <= t < e and (name.startswith("## Call Compiled") or "backward" in name.lower() or name.startswith("Optimizer")):
            return name[:40]
    return ""

by_cause = collections.defaultdict(float)
total_gap = 0.0
for s, e in gaps:
    d = e - s
    if d < 20:  # ignore sub-20 µs launch gaps (they add up, counted separately)
        by_cause["(gaps < 20 µs: kernel launch spacing)"] += d
        total_gap += d
        continue
    by_cause[doing(s)] += d
    total_gap += d
wall = end - start
print(f"{n_steps} steps, {wall / n_steps / 1e3:.1f} ms per step, GPU idle {total_gap / n_steps / 1e3:.1f} ms per step ({total_gap / wall:.0%})")
for name, d in sorted(by_cause.items(), key=lambda x: -x[1])[:15]:
    print(f"  {d / n_steps / 1e3:6.1f} ms/step  {name[:110]}")
