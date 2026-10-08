"""Figures for the OR-amplification section. Reads only data/. Figure 15 is live: refresh with
`extract_data.py comet` (project python) and rerun this script.

    .venv/bin/python figs_or.py
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from theme import BASE, GOOD, INK, MUTED, OR, WARN, save, use_theme

DATA = Path(__file__).parent / "data"
C = json.load(open(DATA / "constants.json"))


def rows(name):
    return list(csv.DictReader(open(DATA / name)))


def or_minus_baseline():
    """13: OR (D) minus flex baseline (A′), window 512, test set pT > 1 GeV, with 1 sigma statistical errors."""
    r = rows("or_minus_baseline.csv")
    fig, ax = plt.subplots(figsize=(14, 4.8))
    y = np.arange(len(r))[::-1]
    for off, key, colour, label in ((0.14, "dm", MUTED, "DM efficiency"), (-0.14, "perfect", OR, "Perfect efficiency")):
        vals = [float(x[f"{key}_diff"]) for x in r]; errs = [float(x[f"{key}_err"]) for x in r]
        ax.errorbar(vals, y + off, xerr=errs, fmt="o", color=colour, ms=9, capsize=5, lw=2, label=label)
        for v, e, yy in zip(vals, errs, y + off):
            above = key == "dm"
            ax.text(v, yy + (0.11 if above else -0.11), f"{v:+.2f} ± {e:.2f}", ha="center", va="bottom" if above else "top",
                    color=colour, fontsize=12, bbox=dict(fc="#0F1620", ec="none", pad=1.5), zorder=4)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, [f"{x['range']}  ({float(x['share']):.0%})" for x in r], fontsize=13)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-0.75, len(r) - 0.3)
    ax.set_xlabel("OR minus baseline (percentage points); right = OR better")
    ax.set_axisbelow(True); ax.grid(axis="x", alpha=0.4)
    ax.legend(loc="lower left", fontsize=12.5)
    ax.set_title("At window 512, OR equals the baseline on DM and is better on perfect efficiency", loc="left")
    save(fig, "13_or_minus_baseline")


def seam():
    """14: DM efficiency vs distance to the phi = +-pi seam; D is the only model without wrap."""
    r = rows("seam.csv")
    models = {"paper_repro": ("Paper reproduction (wrap)", BASE), "A_prime": ("A′ baseline (wrap)", MUTED), "D_or": ("D: OR (no wrap)", OR)}
    bins = [(x["lo_rad"], x["hi_rad"]) for x in r if x["run"] == "paper_repro"]
    labels = ["< 0.05", "0.05–0.1", "0.1–0.2", "0.2–0.5", "> 0.5"]
    fig, ax = plt.subplots(figsize=(14, 4.8))
    w = 0.26
    for k, (run, (label, colour)) in enumerate(models.items()):
        sel = [x for x in r if x["run"] == run]
        e = [100 * float(x["dm_eff"]) for x in sel]; err = [100 * float(x["err"]) for x in sel]
        ax.bar(np.arange(len(sel)) + (k - 1) * w, e, w * 0.92, yerr=err, color=colour, capsize=3, label=label,
               error_kw=dict(ecolor=INK, lw=1))
    share = [int(x["n"]) for x in r if x["run"] == "paper_repro"]; share = np.array(share) / sum(share)
    ax.set_xticks(range(len(bins)), [f"{l} rad\n({s:.1%} of particles)" for l, s in zip(labels, share)], fontsize=12)
    ax.set_ylim(93, 98.6)
    ax.set_ylabel("DM efficiency (%), pT > 1 GeV")
    ax.set_xlabel("Distance of the particle's φ to the ±π seam")
    ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.4)
    ax.legend(loc="upper left", fontsize=12.5, ncol=3)
    ax.set_title("OR loses ≈ 1.5 points at the φ seam, on 1.6% of particles", loc="left")
    save(fig, "14_seam")


def sweep_results():
    """16: test-set results so far, window 512 vs 256, baseline vs OR (pT > 1 GeV, track valid 0.5, IoU 0.5)."""
    from figs_repro import _table
    r = {x["run"]: x for x in rows("test_results.csv")}
    drop = C["plateau_drop_epoch"]
    order = [("A_prime", "Baseline (flex + wrap), window 512", MUTED), ("D_or", "OR, window 512", OR),
             ("flex_w256_s42", "Baseline (flex + wrap), window 256", BASE), ("or_w256_s42", "OR, window 256", OR)]
    body, colours = [], []
    for run, label, colour in order:
        if run in r:
            x = r[run]
            body.append([label, x["dm_eff"], x["perfect_eff"], x["fake_rate"], str(drop[run])])
            colours.append(colour)
    fig, ax = plt.subplots(figsize=(14, 3.2))
    _table(ax, ["Seed 42, test set, pT > 1 GeV", "DM eff %", "Perfect %", "Fake %", "Plateau ends (epoch)"], body,
           [5.8, 1.6, 1.6, 1.4, 3.0], colours=colours, bold_rows=(2,))
    ax.set_title("Window 256: the baseline holds up (95.1% perfect); OR doesn't help in this seed", loc="left", pad=14)
    save(fig, "16_sweep_results")


def sweep_live():
    """15: validation loss vs epochs since each run left the plateau (live from Comet)."""
    r = rows("val_curves.csv")
    snap = dt.datetime.fromtimestamp((DATA / "val_curves.csv").stat().st_mtime).strftime("%d %b %H:%M")
    drop = C["plateau_drop_epoch"]
    runs = [("A_prime", "Baseline w512 (A′)", MUTED, "-", 1.6), ("D_or", "OR w512 (D)", OR, "-", 1.6),
            ("flex_w256_s42", "Baseline w256", BASE, "--", 2.6), ("or_w256_s42", "OR w256", OR, "--", 2.6)]
    fig, ax = plt.subplots(figsize=(14, 5.2))
    for run, label, colour, ls, lw in runs:
        pts = [(int(x["epoch"]) - drop[run], float(x["val_loss"])) for x in r if x["run"] == run and int(x["epoch"]) >= drop[run]]
        pts = [(k, v) for k, v in pts if v < 2]   # drop the diverged final epoch, marked separately
        ax.plot([k for k, _ in pts], [v for _, v in pts], ls, color=colour, lw=lw, marker="o" if lw > 2 else None, ms=4,
                label=f"{label} (plateau ends at epoch {drop[run]})")
    d = C["or_w256_s42_divergence"]
    k_last = 29 - drop["or_w256_s42"]
    ax.scatter([k_last], [0.668], marker="^", s=120, color=WARN, zorder=5)
    ax.annotate(f"OR w256, last epoch: val loss {d['final_val_loss']:g} (off scale), diverged at epoch {d['epoch']:g};\n"
                "the epoch-28 checkpoint was evaluated", xy=(k_last, 0.668), xytext=(k_last - 0.6, 0.615),
                ha="right", va="center", color=WARN, fontsize=12, arrowprops=dict(arrowstyle="-|>", color=WARN, lw=1.5))
    ax.set_xlim(-0.5, 27); ax.set_ylim(0.28, 0.69)
    ax.set_xlabel("Epochs since the run left the plateau"); ax.set_ylabel("Validation loss")
    ax.grid(alpha=0.4)
    ax.legend(loc="center right", fontsize=12)
    ax.set_title(f"At equal training after the plateau, OR sits slightly above its baseline (Comet, {snap})", loc="left", fontsize=15)
    save(fig, "15_sweep_live")


if __name__ == "__main__":
    use_theme()
    or_minus_baseline()
    seam()
    sweep_live()
    sweep_results()
