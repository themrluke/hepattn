"""Data figures: the ColliderML fix, our TrackML copy against the paper, and the corrupt hits.

    .venv/bin/python figs_data.py

Reads only data/ (written by extract_data.py), so it runs in the presentation venv.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from theme import BASE, GOOD, GRID, INK, MUTED, OR, PANEL, WARN, save, use_theme

DATA = Path(__file__).parent / "data"

# ColliderML pu0, measured on 200 events (docs/handoff/03-colliderml-baseline.md)
SELECTION = {
    "old: pixel barrel, |eta| < 1": {"hits": 258, "parts": 20, "hpp": 4.2},
    "new: all tracker, |eta| < 2.5": {"hits": 3433, "parts": 66, "hpp": 11.8},
}


def fig_colliderml():
    """Left: why the old selection could not work. Right: the overfit check on the new one."""
    use_theme()
    fig = plt.figure(figsize=(16, 5.6))
    gl = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.28)

    ax = fig.add_subplot(gl[0])
    names = list(SELECTION)
    hpp = [SELECTION[n]["hpp"] for n in names]
    bars = ax.barh([0, 1], hpp, color=[WARN, GOOD], height=0.55)
    for b, n in zip(bars, names, strict=True):
        s = SELECTION[n]
        ax.text(b.get_width() + 0.25, b.get_y() + b.get_height() / 2,
                f"{s['hpp']} hits / particle\n{s['hits']:,} hits, {s['parts']} particles / event",
                va="center", color=INK, fontsize=12)
    ax.set_yticks([0, 1], names)
    ax.set_xlim(0, 21)
    ax.set_ylim(1.62, -0.45)  # inverted: old on top, room for the label underneath
    ax.axvline(3, color=MUTED, ls=":", lw=1.5)
    ax.text(3.15, 1.47, "min. hits to be a target", color=MUTED, fontsize=10.5, va="center")
    ax.set_xlabel("hits per target particle (pu0, 200 events)")
    ax.set_title("The selection was the problem", loc="left")

    ax = fig.add_subplot(gl[1])
    rows = list(csv.DictReader(open(DATA / "overfit_pu0_10ev.csv")))
    step = [int(r["step"]) for r in rows]
    for key, label, colour in (("eff_p05", "efficiency (≥50% of hits)", OR),
                               ("eff_p10", "efficiency (all hits)", GOOD),
                               ("pur_p05", "purity (≥50% of hits)", BASE)):
        ax.plot(step, [float(r[key]) for r in rows], "o-", ms=4, lw=2.4, color=colour, label=label)
    ax.axhline(1.0, color=GRID, lw=1)
    last = rows[-1]
    ax.annotate(f"{float(last['eff_p05']):.1%} / {float(last['eff_p10']):.1%}",
                xy=(step[-1], float(last["eff_p05"])), xytext=(step[-1] - 1250, 0.62),
                color=INK, fontsize=13, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": MUTED})
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("training step")
    ax.set_title("Overfit check: 10 pu0 events, train = validation", loc="left")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    return save(fig, "01_colliderml")


def fig_table3():
    """Our TrackML copy against Table 3 of arXiv 2606.17631, before any filtering."""
    use_theme()
    t = json.load(open(DATA / "table3.json"))
    paper = t["paper"]
    hits = np.array(t["hits_per_event"]) / 1e3
    reco = np.array(t["reco_per_event"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, ours, ours_err, theirs, theirs_err, title, unit in (
        (axes[0], hits.mean(), hits.std(), paper["hits"] / 1e3, None, "pixel hits per event", "k"),
        (axes[1], reco.mean(), reco.std(), paper["reco_mean"], paper["reco_std"],
         "reconstructable particles per event", ""),
    ):
        ax.bar([0], [theirs], yerr=[theirs_err] if theirs_err else None, color=BASE, width=0.55,
               capsize=8, error_kw={"ecolor": MUTED, "lw": 1.6})
        ax.bar([1], [ours], yerr=[ours_err], color=OR, width=0.55, capsize=8,
               error_kw={"ecolor": MUTED, "lw": 1.6})
        for x, v, e in ((0, theirs, theirs_err), (1, ours, ours_err)):
            txt = f"{v:,.1f}{unit}" if unit else f"{v:,.0f}"
            if e:
                txt += f" ± {e:,.0f}" if not unit else ""
            ax.text(x, v * 0.5, txt, ha="center", color="#0F1620", fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1], ["paper (Table 3)", f"ours ({len(hits)} val events)"])
        ax.set_title(title, fontsize=14)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    fig.suptitle("Before any filtering, our data matches the paper",
                 color=INK, fontsize=16, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04, "Pix1.0 selection: pixel volumes 7, 8, 9  ·  reconstructable = pT > 1 GeV, "
             "|eta| < 4, ≥ 3 pixel hits", ha="center", color=MUTED, fontsize=11.5)
    return save(fig, "03_table3")


def fig_corrupt():
    """Charge per pixel hit: everything sits below 1, six hits sit above the fp16 ceiling."""
    use_theme()
    c = json.load(open(DATA / "charge_frac.json"))
    bins, counts = np.array(c["bins"]), np.array(c["counts"])
    fp16 = c["fp16_max"]

    fig, ax = plt.subplots(figsize=(14, 5.2))
    centres = np.sqrt(bins[:-1] * bins[1:])
    ax.fill_between(centres, counts, step="mid", color=BASE, alpha=0.35)
    ax.step(centres, counts, where="mid", color=BASE, lw=2,
            label=f"every pixel hit in {c['n_events']} val events ({counts.sum() / 1e6:.1f}M hits)")
    ax.axvline(fp16, color=WARN, lw=2.2, ls="--")
    ax.text(fp16 * 0.8, 3e2, "largest value fp16\ncan store: 65,504", color=WARN, ha="right",
            fontsize=12.5, fontweight="bold")

    # Each corrupt hit is a single entry, so it sits at a count of 1; labels fan out above it.
    ys = np.geomspace(8, 6e3, len(c["corrupt"]))
    for y, h in zip(ys, sorted(c["corrupt"], key=lambda d: d["charge_frac"]), strict=True):
        ax.plot(h["charge_frac"], 1, "X", ms=13, color=OR, mec="#0F1620", mew=1, zorder=5)
        ax.annotate(f"event {h['event']}", xy=(h["charge_frac"], 1.3), xytext=(h["charge_frac"] * 1.6, y),
                    color=OR, fontsize=10.5, va="center",
                    arrowprops={"arrowstyle": "-", "color": OR, "lw": 0.8, "alpha": 0.6})
    ax.plot([], [], "X", ms=12, color=OR, label="the 6 corrupt hits (one each, training set)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 8e6)
    ax.set_ylim(0.8, counts.max() * 3)
    ax.set_xlabel("charge_frac  =  total cell charge / number of cells   (raw TrackML 'value')")
    ax.set_ylabel("hits")
    ax.legend(loc="upper left", bbox_to_anchor=(0.36, 0.98))
    ax.grid(alpha=0.2)
    ax.text(1.3, 2e3, "all genuine\nvalues ≤ 1", color=MUTED, fontsize=12)
    return save(fig, "05_corrupt_charge")


def fig_freeze():
    """What one corrupt event does to training: skipped in fp16, a silent freeze in bf16."""
    use_theme()
    d = json.load(open(DATA / "stepcheck.json"))
    events = d["events"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for ax, key, title, colour in ((axes[0], "fp16", "fp16 + gradient scaler  (the filter)", GOOD),
                                   (axes[1], "bf16", "bf16, no scaler  (the tracking configs)", WARN)):
        r = d[key]
        xs = list(range(1, len(events)))
        vals = [r["max_weight_change"][i] for i in xs]
        bars = ax.bar(xs, [max(v, 1e-9) for v in vals], color=[colour if v > 0 else GRID for v in vals], width=0.6)
        for x, v, mom in zip(xs, vals, [r["nan_momentum_frac"][i] for i in xs], strict=True):
            ax.text(x, 2e-9 if v == 0 else v * 1.4, "0" if v == 0 else f"{v:.0e}", ha="center",
                    color=INK if v else WARN, fontsize=11, fontweight="bold")
            if mom:
                ax.text(x, 3e-4, "momentum\nNaN", ha="center", color=WARN, fontsize=9.5)
            elif v == 0:
                ax.text(x, 3e-4, "step\nskipped", ha="center", color=GOOD, fontsize=9.5)
        ax.axvspan(1.6, 2.4, color=OR, alpha=0.08)
        ax.set_xticks(xs, [events[i].replace(" (corrupt)", "\ncorrupt") for i in xs], fontsize=11)
        ax.set_yscale("log")
        ax.set_ylim(1e-9, 1e-3)
        ax.set_title(title, color=colour, fontsize=14)
        ax.set_xlabel("training step on event")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("largest weight change in the step")
    fig.suptitle("One corrupt event: every gradient is NaN. fp16 skips the step; bf16 freezes the model for good",
                 color=INK, fontsize=15, fontweight="bold", y=1.04)
    fig.text(0.5, -0.06, "Lion updates by sign(momentum), and sign(NaN) = 0: the weights stay finite, the loss looks normal, "
             "and nothing ever changes again.", ha="center", color=MUTED, fontsize=11.5)
    return save(fig, "05b_freeze")


if __name__ == "__main__":
    fig_colliderml()
    fig_table3()
    fig_corrupt()
    fig_freeze()
