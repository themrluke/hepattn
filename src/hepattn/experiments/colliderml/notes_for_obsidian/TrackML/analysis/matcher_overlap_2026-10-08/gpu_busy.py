"""From a bench_run.py Chrome trace: wall time of the profiled steps, and the share of it with a GPU kernel running."""
import json
import sys

events = json.load(open(sys.argv[1]))["traceEvents"]
kernels = sorted((e["ts"], e["ts"] + e["dur"]) for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e)
steps = [e for e in events if e.get("name", "").startswith("ProfilerStep#")]
start, end = min(e["ts"] for e in steps), max(e["ts"] + e["dur"] for e in steps)
busy, cur_s, cur_e = 0.0, None, None
for s, e in kernels:
    s, e = max(s, start), min(e, end)
    if e <= s:
        continue
    if cur_e is None or s > cur_e:
        busy += 0 if cur_e is None else cur_e - cur_s
        cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
busy += 0 if cur_e is None else cur_e - cur_s
print(f"{len(steps)} steps, {(end - start) / 1e3 / len(steps):.1f} ms per step, GPU busy {busy / (end - start):.0%}")
