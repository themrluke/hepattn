"""Figures for the matcher speed-up section. Reads only data/.

    .venv/bin/python figs_speed.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from theme import BASE, GOOD, GRID, INK, MUTED, OR, PANEL, WARN, panel, save, use_theme

DATA = Path(__file__).parent / "data"
C = json.load(open(DATA / "constants.json"))


def rows(name):
    return list(csv.DictReader(open(DATA / name)))


def step_shares():
    """01: where a training step went before the fix (py-spy, 2 Oct)."""
    s = {k: v for k, v in C["perf_audit_step_shares"].items() if k != "source"}
    order = ["Waiting on the 4 solves", "Copy costs to CPU", "Copy into new shared memory", "CPU transpose",
             "CPU mask (np.where)", "Free shared memory", "GPU work, data, other"]
    colours = [MUTED, WARN, WARN, WARN, WARN, WARN, BASE]
    fig, ax = plt.subplots(figsize=(14, 5.2))
    y = np.arange(len(order))[::-1]
    ax.barh(y, [s[k] for k in order], color=colours, height=0.62)
    for yi, k in zip(y, order):
        ax.text(s[k] + 0.6, yi, f"{s[k]:g}%", va="center", color=INK, fontsize=13)
    ax.set_yticks(y, order, fontsize=13)
    ax.set_xlim(0, 38)
    ax.set_xlabel("Share of one training step (%)")
    ax.grid(axis="x", alpha=0.4)
    ax.text(37.5, 1.6, "Moving one 243 MB cost table around: ≈ 35%\nGPU busy only ≈ 16% of the time",
            ha="right", va="center", color=WARN, fontsize=14, fontweight="bold")
    ax.set_title("A step was mostly the matcher; a third of it just copying", loc="left")
    save(fig, "01_step_shares")


def copies_diagram():
    """02: the original data flow (4 full copies on the CPU) vs toggle 1 (prepare on the GPU, copy once)."""
    fig, ax = plt.subplots(figsize=(15, 5.6))
    panel(ax)
    ax.set_xlim(0, 15.15); ax.set_ylim(0, 5.6)

    def box(x, y, w, h, text, colour, fs=12.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    fc=PANEL, ec=colour, lw=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=INK, fontsize=fs)

    def arrow(x1, x2, y, colour, label=None):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=16, color=colour, lw=1.8))
        if label:
            ax.text((x1 + x2) / 2, y + 0.28, label, ha="center", va="bottom", color=colour, fontsize=11)

    ax.text(0.1, 5.25, "Before", color=WARN, fontsize=16, fontweight="bold")
    y = 3.35
    xs = [0.1, 3.25, 6.4, 9.55, 12.7]
    texts = ["GPU costs\n[4, 3900, 3900]", "CPU copy", "mask:\nnew array", "transpose:\nnew array", "new shared\nmemory, 4 solves"]
    for i, (x, t) in enumerate(zip(xs, texts)):
        box(x, y, 2.2 if i < 4 else 2.25, 1.2, t, BASE if i == 0 else WARN)
    for x in xs[:-1]:
        arrow(x + 2.3, x + 3.15, y + 0.6, WARN)
        ax.text(x + 2.72, y + 1.45, "243 MB", ha="center", va="bottom", color=WARN, fontsize=11.5)

    ax.text(0.1, 2.35, "Toggle 1: prepare_on_device", color=GOOD, fontsize=16, fontweight="bold")
    y = 0.45
    box(0.1, y, 6.4, 1.2, "On the GPU (milliseconds): mask padded queries,\ntranspose, keep only the real-particle rows", BASE)
    box(9.4, y, 5.5, 1.2, "Reused shared memory\n4 solves in parallel (unchanged)", GOOD)
    arrow(6.5, 9.4, y + 0.6, GOOD, "ONE copy, ≈ 160 MB")
    ax.set_title("The matcher copied a 243 MB table four times per step; now it copies it once", loc="left")
    save(fig, "02_copies_diagram")


def speedup():
    """03: steps/s on an idle GPU, steps 150-600, repeats as dots."""
    r = rows("timing_runs.csv")
    stages, variants = ["early", "late"], ["original", "prep", "trim"]
    names = {"original": "Original", "prep": "Toggle 1\n(identical results)", "trim": "Toggles 1 + 2"}
    colours = {"original": MUTED, "prep": GOOD, "trim": OR}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    for ax, stage in zip(axes, stages):
        base = np.mean([float(x["steps_per_s"]) for x in r if x["stage"] == stage and x["variant"] == "original"])
        for i, v in enumerate(variants):
            vals = [float(x["steps_per_s"]) for x in r if x["stage"] == stage and x["variant"] == v]
            m = np.mean(vals)
            ax.bar(i, m, color=colours[v], width=0.62)
            ax.scatter([i] * len(vals), vals, color=INK, s=14, zorder=3)
            ax.text(i, m + 0.12, f"{m:.2f}/s\n×{m / base:.2f}" if v != "original" else f"{m:.2f}/s", ha="center",
                    va="bottom", color=INK, fontsize=13, fontweight="bold" if v != "original" else None)
        n = len([x for x in r if x["stage"] == stage and x["variant"] == "original"])
        ax.set_xticks(range(3), [names[v] for v in variants], fontsize=12.5)
        ax.set_title({"early": "Early training (epoch-0 weights)", "late": "Late training (epoch-29 weights)"}[stage]
                     + f", {n} repeats", fontsize=14)
        ax.grid(axis="y", alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 5.0)
    axes[0].set_ylabel("Training steps per second")
    fig.suptitle("Toggle 1 alone: 1.9× faster early, 2.6× late, with identical matching", x=0.06, ha="left",
                 fontsize=16, fontweight="bold", color=INK)
    save(fig, "03_speedup")


def late_breakdown():
    """04: what a late-training step is made of, per variant (matcher vs rest), plus what is left in the matcher."""
    r = rows("timing_runs.csv")
    variants = ["original", "prep", "trim"]
    labels = ["Original", "Toggle 1", "Toggles 1 + 2"]
    m = [np.mean([float(x["matcher_s"]) for x in r if x["stage"] == "late" and x["variant"] == v]) * 1e3 for v in variants]
    rest = [np.mean([float(x["rest_s"]) for x in r if x["stage"] == "late" and x["variant"] == v]) * 1e3 for v in variants]
    fig, ax = plt.subplots(figsize=(14, 4.6))
    y = np.arange(3)[::-1]
    ax.barh(y, rest, color=BASE, height=0.55, label="Rest of the step (GPU backward, optimiser, …)")
    ax.barh(y, m, left=rest, color=WARN, height=0.55, label="Matcher call")
    for yi, a, b in zip(y, rest, m):
        ax.text(a / 2, yi, f"{a:.0f} ms", ha="center", va="center", color="#0F1620", fontsize=12.5, fontweight="bold")
        ax.text(a + b / 2, yi, f"{b:.0f} ms", ha="center", va="center", color="#0F1620", fontsize=12.5, fontweight="bold")
        ax.text(a + b + 8, yi, f"{a + b:.0f} ms", va="center", color=INK, fontsize=13)
    ax.set_yticks(y, labels, fontsize=13)
    ax.set_xlim(0, 860)
    ax.set_xlabel("Milliseconds per training step (late training, idle GPU)")
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title("Only the matcher shrank: the rest of the step stays ≈ 128 ms", loc="left")
    save(fig, "04_late_breakdown")


if __name__ == "__main__":
    use_theme()
    step_shares()
    copies_diagram()
    speedup()
    late_breakdown()
