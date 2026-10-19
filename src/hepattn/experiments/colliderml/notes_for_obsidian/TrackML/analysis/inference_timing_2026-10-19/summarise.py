"""Markdown tables from results_repeat*.json: per repeat mean and min (ms per event), and the spread between repeats."""
import glob
import json

import numpy as np

reps = [json.load(open(f)) for f in sorted(glob.glob("results_repeat*.json"))]
print(f"{len(reps)} repeats: " + "; ".join(f"repeat {r['repeat']} {r['start']} → {r.get('end', '(running)')}" for r in reps))
print(f"hits per event (timed events): mean {np.mean(reps[0]['hits_per_event'][10:]):.0f}, range {min(reps[0]['hits_per_event'][10:])}–{max(reps[0]['hits_per_event'][10:])}\n")


def cell(r, key, part=None):
    d = r.get(key)
    if d is None or "error" in d:
        return None
    return d[part] if part else d


def rows(keys, part=None):
    out = []
    for key in keys:
        vals = [cell(r, key, part) for r in reps]
        if any(v is None for v in vals):
            errs = [r.get(key, {}).get("error", "missing") for r in reps]
            out.append((key, None, errs[0]))
            continue
        means = [v["mean"] for v in vals]
        mins = [v["min"] for v in vals]
        spread = (max(means) - min(means)) / np.mean(means) * 100
        out.append((key, (means, mins, spread), None))
    return out


def detail_table(title, keys, label, part=None):
    print(f"### {title}\n")
    hdr = " | ".join(f"Repeat {i + 1} mean / min" for i in range(len(reps)))
    print(f"| {label} | {hdr} | Spread of means |")
    print("| :--- | " + " | ".join([":---:"] * len(reps)) + " | :---: |")
    for key, v, err in rows(keys, part):
        name = key.split(" ", 1)[1]
        if v is None:
            print(f"| {name} | " + " | ".join(["failed"] * len(reps)) + f" | {err[:60]} |")
            continue
        means, mins, spread = v
        print(f"| {name} | " + " | ".join(f"{m:.2f} / {n:.2f}" for m, n in zip(means, mins)) + f" | {spread:.1f}% |")
    print()


def grid(kind):
    print(f"### Encoder only, {kind}: mean over repeats of the per-repeat average (ms per event)\n")
    print("| Window | block 128 | block 64 | block 32 | block 16 |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for w in (512, 256, 128, 64, 32):
        cells = []
        for b in (128, 64, 32, 16):
            key = f"encoder {kind} w{w} block{b} depth8"
            v = rows([key])[0][1]
            cells.append("failed" if v is None else f"{np.mean(v[0]):.2f}")
        print(f"| {w} | " + " | ".join(cells) + " |")
    print()


for kind in ("flex", "or"):
    grid(kind)
enc_keys = [k for k in reps[0] if k.startswith("encoder ")]
detail_table("Encoder only, every configuration (ms per event)", enc_keys, "Configuration")
model_keys = [k for k in reps[0] if k.startswith("model ")]
for part in ("total", "encoder", "decoder"):
    detail_table(f"Trained model, {part} (ms per event)", model_keys, "Model", part)
print("### Mean number of dynamic queries per event\n")
for k in model_keys:
    q = [r[k].get("mean_queries") for r in reps if k in r and "error" not in r[k]]
    print(f"- {k.split(' ', 1)[1]}: {np.mean(q):.0f}" if q else f"- {k}: failed")
print("\n### GPU state when each configuration was timed\n")
procs = sorted({r[k]["gpu"]["gpu_processes"] for r in reps for k in r if isinstance(r.get(k), dict) and "gpu" in r[k]})
loads = [r[k]["gpu"]["load_avg"] for r in reps for k in r if isinstance(r.get(k), dict) and "gpu" in r[k]]
print(f"- processes on the GPU: {procs}; host load average {min(loads):.1f}–{max(loads):.1f}")
