"""Hit-filter training curves: the run that went NaN, and the retrain without the corrupt events.

    ../../../../../../.pixi/envs/default/bin/python extract_data.py comet   # refresh from Comet
    .venv/bin/python figs_training.py

Re-run both and rebuild the deck to bring the retrain up to date before the meeting.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from matplotlib import pyplot as plt

from theme import BASE, GOOD, INK, MUTED, OR, WARN, save, use_theme

DATA = Path(__file__).parent / "data"
STEPS_PER_EPOCH = {"filter_failed": 8650, "filter_retrain": 8644}
NAN_STEP = 34199  # first non-finite training loss in the failed run
TOTAL_EPOCHS = 80


def load(name):
    series = defaultdict(list)
    for r in csv.DictReader(open(DATA / f"comet_{name}.csv")):
        series[r["metric"]].append((int(r["step"]), float(r["value"])))
    for k in series:
        series[k].sort()
    return series


def _smooth(points, n=8):
    out = []
    for i in range(len(points)):
        window = [v for _, v in points[max(0, i - n + 1): i + 1] if math.isfinite(v)]
        out.append((points[i][0], sum(window) / len(window) if window else math.nan))
    return out


def fig_failure():
    """Train loss of the failed run up to the NaN, with the learning rate on a second axis."""
    use_theme()
    d = load("filter_failed")
    spe = STEPS_PER_EPOCH["filter_failed"]
    # Only up to the failure: the few finite values logged afterwards are meaningless.
    finite = [(s, v) for s, v in d["train/loss"] if s < NAN_STEP and math.isfinite(v)]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot([s / spe for s, _ in finite], [v for _, v in finite], color=BASE, lw=1, alpha=0.35)
    sm = [(s, v) for s, v in _smooth(finite) if math.isfinite(v)]
    ax.plot([s / spe for s, _ in sm], [v for _, v in sm], color=BASE, lw=2.6, label="training loss")
    ok = [(s, v) for s, v in d["val/loss"] if math.isfinite(v)]
    ax.plot([(s + 1) / spe for s, _ in ok], [v for _, v in ok], "o", ms=10, color=GOOD, label="validation loss")

    x0 = NAN_STEP / spe
    ax.axvspan(x0, 6, color=WARN, alpha=0.12)
    ax.axvline(x0, color=WARN, lw=2.4)
    ax.text(x0 + 0.08, 0.5, f"step {NAN_STEP:,}:\nloss → inf → NaN,\nNaN for the\nremaining 76 epochs",
            color=WARN, fontsize=12.5, fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_yscale("log")
    ax.set_xlim(0, 6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)

    lr = [(s / spe, v) for s, v in d.get("lr-Lion", []) if s / spe <= 6]
    if lr:
        ax2 = ax.twinx()
        ax2.plot([e for e, _ in lr], [v * 1e5 for _, v in lr], color=OR, lw=2.2, ls="--", label="learning rate")
        ax2.set_ylabel("learning rate (× 1e-5)", color=OR)
        ax2.tick_params(axis="y", colors=OR)
        ax2.set_ylim(0, 6)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(OR)
        peak = max(lr, key=lambda x: x[1])
        ax2.annotate("warm-up peak", xy=(peak[0], peak[1] * 1e5), xytext=(peak[0] - 1.6, 5.55),
                     color=OR, fontsize=12, fontweight="bold", arrowprops={"arrowstyle": "->", "color": OR})
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left")
    else:
        ax.legend(loc="upper left")
    ax.set_title("First hit-filter run: the loss climbs for ~0.8 epoch, then goes NaN", loc="left")
    return save(fig, "08_nan_failure")


def fig_retrain():
    """Validation loss per epoch: failed run vs the retrain (so far)."""
    use_theme()
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, colour, label in (("filter_failed", WARN, "first run (8,650 events)"),
                                ("filter_retrain", GOOD, "retrain (8,644 events, corrupt ones removed)")):
        d = load(name)
        spe = STEPS_PER_EPOCH[name]
        pts = [((s + 1) / spe, v) for s, v in d["val/loss"]]
        fin = [(e, v) for e, v in pts if math.isfinite(v)]
        ax.plot([e for e, _ in fin], [v for _, v in fin], "o-", color=colour, lw=2.4, ms=7, label=label)
        nan_e = [e for e, v in pts if not math.isfinite(v)]
        if nan_e:
            ax.plot(nan_e, [fin[-1][1]] * len(nan_e), "x", color=colour, ms=7, alpha=0.55)
            ax.text(nan_e[0] + 0.5, fin[-1][1] * 1.06, f"NaN from epoch {round(nan_e[0]) - 1} onwards",
                    color=colour, fontsize=12)
        if name == "filter_retrain":
            done = len(fin)
            ax.annotate(f"epoch {done} of {TOTAL_EPOCHS}", xy=(fin[-1][0], fin[-1][1]),
                        xytext=(fin[-1][0] + 3, fin[-1][1] * 0.97), color=GOOD, fontsize=13, fontweight="bold",
                        arrowprops={"arrowstyle": "->", "color": GOOD})
    ax.set_xlim(0, 20)
    ax.set_ylim(bottom=min(ax.get_ylim()[0], 0.15))
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    ax.set_title("Retrain in progress (live from Comet)", loc="left")
    fig.text(0.5, -0.04, "~12 min per epoch on one H100 → 80 epochs ≈ 16 h.  Then: evaluate on train / val / "
             "test and check 98.6% particle retention.", ha="center", color=MUTED, fontsize=11.5)
    return save(fig, "09_retrain")


def fig_grad_scale():
    """fp16 gradient-scaler scale per epoch: ratchets to zero with the corrupt events, steady without."""
    import json
    use_theme()
    d = json.load(open(DATA / "grad_scale.json"))
    fig, ax = plt.subplots(figsize=(14, 5))
    floor = 1.0  # a log axis cannot show 0; draw it on a floor line
    for name, colour, label in (("filter_failed", WARN, "first run (6 corrupt events in)"),
                                ("filter_retrain", GOOD, "retrain (corrupt events removed)")):
        rows = [r for r in d[name] if r["epoch"] <= 8]
        e = [r["epoch"] for r in rows]
        v = [max(r["scale"], floor) for r in rows]
        ax.plot(e, v, "o-", color=colour, lw=2.6, ms=8, label=label)
        for r in rows:
            if r["epoch"] <= 3 or name == "filter_retrain" and r["epoch"] in (1,):
                txt = "0" if r["scale"] == 0 else f"{r['scale']:,.0f}"
                ax.text(r["epoch"], max(r["scale"], floor) * (1.7 if r["scale"] == 0 else 0.42 if name == "filter_failed" else 1.6), txt,
                        ha="center", color=colour, fontsize=12, fontweight="bold")
    ax.axhline(floor, color=WARN, lw=1, ls=":")
    ax.text(8.1, floor * 1.25, "scale = 0  →  gradients ÷ 0  →  NaN", color=WARN, ha="right", fontsize=12)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 3e5)
    ax.set_xlim(-0.3, 8.2)
    ax.set_xlabel("epoch (end of)")
    ax.set_ylabel("fp16 gradient-scaler scale")
    ax.legend(loc="center right")
    ax.grid(alpha=0.25)
    ax.set_title("Each corrupt event halves the scale; it only regrows after 2,000 clean steps", loc="left")
    fig.text(0.5, -0.05, "6 corrupt events per epoch ≈ one every 1,400 steps, so the regrowth counter never "
             "reaches 2,000: the scale can only fall. Read from each epoch's checkpoint.",
             ha="center", color=MUTED, fontsize=11.5)
    return save(fig, "08b_grad_scale")


if __name__ == "__main__":
    fig_failure()
    fig_retrain()
    fig_grad_scale()
