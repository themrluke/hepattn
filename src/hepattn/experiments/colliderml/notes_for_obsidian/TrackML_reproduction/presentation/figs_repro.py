"""Figures for the reproduction section. Reads only data/.

    .venv/bin/python figs_repro.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from theme import BASE, GOOD, GRID, INK, MUTED, OR, PANEL, WARN, panel, save, use_theme

DATA = Path(__file__).parent / "data"
C = json.load(open(DATA / "constants.json"))
MODEL = {  # run -> (label, colour)
    "paper_repro": ("Paper reproduction (flash, w512)", BASE),
    "A_prime": ("A′ baseline (flex, w512)", MUTED),
    "D_or": ("D: OR (flex, w512)", OR),
}


def rows(name):
    return list(csv.DictReader(open(DATA / name)))


def val_curves():
    """05: validation loss per epoch; every run sits on a plateau and then drops within one epoch."""
    r = rows("val_curves.csv")
    fig, ax = plt.subplots(figsize=(14, 5.4))
    drops = C["plateau_drop_epoch"]
    for run in ("paper_repro", "D_or", "A_prime"):
        ep = [int(x["epoch"]) for x in r if x["run"] == run]
        v = [float(x["val_loss"]) for x in r if x["run"] == run]
        label, colour = MODEL[run]
        ax.plot(ep, v, "-o", color=colour, ms=4, lw=2.2, label=f"{label}: drop at epoch {drops[run]}")
        d = drops[run]
        ax.annotate("", xy=(d, v[d] * 1.25), xytext=(d, v[d - 1] * 0.8),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.6))
    ax.set_yscale("log")
    ax.set_ylim(0.22, 5)
    ax.set_yticks([0.3, 0.5, 1, 2, 4], ["0.3", "0.5", "1", "2", "4"])
    from matplotlib.ticker import NullFormatter
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(-0.5, 29.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.grid(alpha=0.4)
    ax.legend(loc="upper right", fontsize=12.5)
    ax.set_title("All three sit on a plateau, then drop in one epoch: A′ only at epoch 24", loc="left")
    save(fig, "05_val_curves")


def flash_flex():
    """06: same weights through flash, flex and fp32 flex: flash/flex differ less than bf16 rounding."""
    f = C["flash_vs_flex"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    names = ["flash vs flex\n(both bf16)", "flex bf16 vs flex fp32\n(rounding alone)"]
    vals = [100 * f["flash_vs_flex_bf16_rel_diff"], 100 * f["flex_bf16_vs_fp32_rel_diff"]]
    a1.barh([1, 0], vals, color=[BASE, MUTED], height=0.55)
    for y, v in zip([1, 0], vals):
        a1.text(v + 0.25, y, f"{v:.0f}%", va="center", color=INK, fontsize=14, fontweight="bold")
    a1.set_yticks([1, 0], names, fontsize=12.5)
    a1.set_xlim(0, 14)
    a1.set_xlabel("Relative difference in encoder output (same weights, same 20 events)")
    a1.set_axisbelow(True); a1.grid(axis="x", alpha=0.4)
    panel(a2)
    a2.set_xlim(0, 1); a2.set_ylim(0, 1)
    a2.text(0.05, 0.86, "Mean loss", color=MUTED, fontsize=14)
    a2.text(0.05, 0.66, f"flash  {f['loss_flash']:.4f}", color=BASE, fontsize=17, fontweight="bold")
    a2.text(0.05, 0.50, f"flex    {f['loss_flex']:.4f}", color=BASE, fontsize=17, fontweight="bold")
    a2.text(0.05, 0.24, f"torch backend: {f['torch_backend_loss']:.1f}\n(broken; nothing we train uses it)",
            color=WARN, fontsize=13)
    fig.suptitle("Flex computes the same thing as flash, within bf16 rounding", x=0.06, ha="left",
                 fontsize=16, fontweight="bold", color=INK)
    save(fig, "06_flash_flex")


def _table(ax, header, body, col_w, colours=None, bold_rows=(), fs=14):
    """Draw a simple table on a bare axes. colours: per-row text colour."""
    panel(ax)
    n = len(body) + 1
    ax.set_xlim(0, sum(col_w)); ax.set_ylim(0, n)
    xs = np.cumsum([0] + col_w[:-1])
    for j, h in enumerate(header):
        ax.text(xs[j] + (0.1 if j == 0 else col_w[j] / 2), n - 0.5, h, ha="left" if j == 0 else "center",
                va="center", color=MUTED, fontsize=fs - 1, fontweight="bold")
    ax.plot([0, sum(col_w)], [n - 1, n - 1], color=GRID, lw=1.5)
    for i, row in enumerate(body):
        y = n - 1.5 - i
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y - 0.5), sum(col_w), 1, color=PANEL, zorder=0))
        for j, cell in enumerate(row):
            ax.text(xs[j] + (0.1 if j == 0 else col_w[j] / 2), y, cell, ha="left" if j == 0 else "center", va="center",
                    color=(colours[i] if colours else INK), fontsize=fs, fontweight="bold" if i in bold_rows else None)


def test_results():
    """07: test set at pT > 1 GeV vs the paper's corrected reference."""
    r = {x["run"]: x for x in rows("test_results.csv")}
    p = C["paper_corrected"]["MA"]
    body = [["Paper, Pix0.6 DQ+MA (corrected reference)", f"{p['dm']:.1f}", f"{p['perfect']:.1f}", f"{p['fake']:.1f}", "0.285 (UCL)"]]
    bl = C["best_val_loss"]
    for run, vl in (("paper_repro", bl["paper_repro"]), ("A_prime", bl["A_prime"]), ("D_or", bl["D_or"])):
        x = r[run]
        body.append([MODEL[run][0], x["dm_eff"], x["perfect_eff"], x["fake_rate"], f"{vl:.3f}"])
    fig, ax = plt.subplots(figsize=(14, 3.6))
    _table(ax, ["Test set, pT > 1 GeV (track valid > 0.5, IoU > 0.5)", "DM eff %", "Perfect %", "Fake %", "Best val loss"],
           body, [6.4, 1.6, 1.6, 1.4, 2.0], colours=[GOOD, BASE, MUTED, OR], bold_rows=(0, 1))
    ax.set_title("Reproduction: 0.3 below the paper on DM, 0.6 below on perfect, same fake rate", loc="left", pad=14)
    save(fig, "07_test_results")


def table4_vs_fig7():
    """08: the paper's Table 4 against its own Figure 7 integrated over each threshold."""
    r = rows("fig7_integrated.csv")
    v = {(x["decoder"], x["threshold"]): (float(x["dm_eff"]), float(x["perfect_eff"])) for x in r}
    t4 = C["paper_table4_printed"]
    body = [["Table 4, as printed", f"{t4['MA']['dm']:.1f}", f"{t4['MA']['perfect']:.1f}", f"{t4['LSCA']['dm']:.1f}", f"{t4['LSCA']['perfect']:.1f}"]]
    for th in ("pT ≥ 0.6", "pT ≥ 0.75", "pT ≥ 1"):
        ma, ls = v[("MA", th)], v[("LSCA", th)]
        cells = [f"Figure 7 integrated, {th} GeV"]
        for val, printed in ((ma[0], t4["MA"]["dm"]), (ma[1], t4["MA"]["perfect"]), (ls[0], t4["LSCA"]["dm"]), (ls[1], t4["LSCA"]["perfect"])):
            cells.append(f"{val:.1f}" + ("  ✓" if abs(round(val, 1) - printed) < 0.05 else ""))
        body.append(cells)
    fig, ax = plt.subplots(figsize=(14, 3.7))
    _table(ax, ["Paper, Pix0.6 (%)", "MA: DM", "MA: perfect", "LSCA: DM", "LSCA: perfect"], body,
           [5.2, 1.9, 1.9, 1.9, 1.9], colours=[INK, MUTED, MUTED, GOOD], bold_rows=(0, 3))
    ax.set_title("Table 4 mixes thresholds: its DM column is pT > 1, its perfect column pT ≥ 0.6", loc="left", pad=14)
    save(fig, "08_table4_vs_fig7")


def pt_counts():
    """09: particles per pT bin; most are below 1 GeV."""
    r = rows("pt_counts.csv")
    n = np.array([int(x["particles"]) for x in r]); share = n / n.sum()
    labels = [f"{float(x['lo']):g}–{float(x['hi']):g}" for x in r][:-1] + ["≥ 6"]
    below = np.array([float(x["hi"]) <= 1.0 for x in r])
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.bar(range(len(n)), n / 1e3, color=np.where(below, MUTED, BASE), width=0.75)
    for i, (c, s) in enumerate(zip(n, share)):
        ax.text(i, c / 1e3 + 1.2, f"{s:.1%}", ha="center", color=INK, fontsize=12.5)
    ax.set_xticks(range(len(n)), labels, fontsize=12.5)
    ax.set_xlabel("Particle pT bin [GeV] (test set, 100 events)")
    ax.set_ylabel("Particles [thousands]")
    ax.set_ylim(0, n.max() / 1e3 * 1.18)
    ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.4)
    ax.text(0.98, 0.9, f"{share[below].sum():.1%} of trained particles are below 1 GeV\n(not in the evaluation)",
            transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=13.5)
    ax.set_title("Most particles are low pT, where efficiency is lowest", loc="left")
    save(fig, "09_pt_counts")


def _steps(ax, r, qty_lo, qty_hi, run, metric, colour, ls):
    sel = [x for x in r if x["run"] == run and x["metric"] == metric]
    for x in sel:
        lo, hi, e, err = float(x["lo"]), float(x["hi"]), float(x["eff"]), float(x["err"])
        ax.plot([lo, hi], [e, e], color=colour, ls=ls, lw=2.2)
        ax.add_patch(plt.Rectangle((lo, e - err), hi - lo, 2 * err, color=colour, alpha=0.18, lw=0))


def eff_pt():
    """10: efficiency vs pT with the paper's Fig. 7 points; below 1 GeV shaded (trained on, not evaluated)."""
    r = rows("eff_pt_bins.csv"); fb = rows("fig7_bins.csv")
    fig, ax = plt.subplots(figsize=(16, 5.4))
    ax.axvspan(0.6, 1.0, color=PANEL, zorder=0)
    for run, (label, colour) in MODEL.items():
        _steps(ax, r, "lo", "hi", run, "dm", colour, "-")
        _steps(ax, r, "lo", "hi", run, "perfect", colour, ":")
    for metric, marker in (("DM", "x"), ("perfect", "+")):
        pts = [x for x in fb if x["decoder"] == "MA" and x["metric"] == metric]
        c = [(float(x["lo"]) + float(x["hi"])) / 2 for x in pts]
        ax.scatter(c, [float(x["eff"]) for x in pts], marker=marker, color=INK, s=70, lw=2, zorder=5)
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=c, lw=2.5, label=l) for l, c in MODEL.values()]
    h += [Line2D([0], [0], color=INK, lw=2, label="DM (solid)"), Line2D([0], [0], color=INK, lw=2, ls=":", label="Perfect (dotted)"),
          Line2D([0], [0], marker="x", color=INK, ls="none", ms=9, mew=2, label="Paper DQ+MA, DM (Fig. 7)"),
          Line2D([0], [0], marker="+", color=INK, ls="none", ms=11, mew=2, label="Paper DQ+MA, perfect (Fig. 7)")]
    from matplotlib.patches import Patch
    h.append(Patch(color=PANEL, label="0.6–1 GeV: trained, not evaluated"))
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=14)
    ax.set_xlim(0.6, 10); ax.set_ylim(0.85, 1.0)
    ax.tick_params(labelsize=15); ax.xaxis.label.set_size(16); ax.yaxis.label.set_size(16)
    ax.set_xlabel("Particle pT [GeV]"); ax.set_ylabel("Efficiency")
    ax.grid(alpha=0.4)
    ax.set_title("Per pT bin the reproduction tracks the paper's Figure 7", loc="left")
    save(fig, "10_eff_pt")


def eff_eta():
    """11: efficiency vs eta for evaluated particles (pT > 1 GeV)."""
    r = rows("eff_eta_bins.csv")
    fig, ax = plt.subplots(figsize=(16, 5.2))
    for run, (label, colour) in MODEL.items():
        _steps(ax, r, "lo", "hi", run, "dm", colour, "-")
        _steps(ax, r, "lo", "hi", run, "perfect", colour, ":")
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=c, lw=2.5, label=l) for l, c in MODEL.values()]
    h += [Line2D([0], [0], color=INK, lw=2, label="DM (solid)"), Line2D([0], [0], color=INK, lw=2, ls=":", label="Perfect (dotted)")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=14)
    ax.set_xlim(-4, 4); ax.set_ylim(0.80, 1.0)
    ax.tick_params(labelsize=15); ax.xaxis.label.set_size(16); ax.yaxis.label.set_size(16)
    ax.set_xlabel("Particle η (pT > 1 GeV)"); ax.set_ylabel("Efficiency")
    ax.grid(alpha=0.4)
    ax.set_title("The central dip (|η| < 1) is in the paper's Figure 7 too", loc="left")
    save(fig, "11_eff_eta")


def cutscan():
    """12: operating points (track-valid x IoU cut) for the paper reproduction at pT > 1 GeV."""
    r = rows("cutscan_pt1.csv")
    tv = sorted({float(x["track_valid"]) for x in r}); iou = sorted({float(x["iou"]) for x in r})
    grid = {(float(x["track_valid"]), float(x["iou"])): x for x in r}
    p = C["paper_corrected"]["MA"]
    fig, ax = plt.subplots(figsize=(14, 5.0))
    dm = np.array([[float(grid[(t, i)]["dm_eff"]) for i in iou] for t in tv])
    ax.imshow(dm, cmap="Blues", vmin=96.5, vmax=99.0, aspect="auto")
    for a, t in enumerate(tv):
        for b, i in enumerate(iou):
            x = grid[(t, i)]
            fake = float(x["fake_rate"])
            if fake <= p["fake"] + 0.01:
                ax.add_patch(plt.Rectangle((b - 0.47, a - 0.44), 0.94, 0.88, fill=False, ec=GOOD, lw=3.5))
            ax.text(b, a, f"{float(x['dm_eff']):.1f} / {float(x['perfect_eff']):.1f} / {fake:.1f}", ha="center", va="center",
                    color="#0F1620", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(iou)), [f"IoU > {i:g}" for i in iou], fontsize=13)
    ax.set_yticks(range(len(tv)), [f"valid > {t:g}" for t in tv], fontsize=13)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"DM / perfect / fake (%). Paper: {p['dm']:.1f} / {p['perfect']:.1f} / {p['fake']:.1f}. "
                 "Green box = paper's fake rate", loc="left", fontsize=14)
    fig.suptitle("Looser IoU cuts reach the paper's DM, at 2–3× its fake rate", x=0.06, ha="left",
                 fontsize=16, fontweight="bold", color=INK)
    save(fig, "12_cutscan")


if __name__ == "__main__":
    use_theme()
    val_curves()
    flash_flex()
    test_results()
    table4_vs_fig7()
    pt_counts()
    eff_pt()
    eff_eta()
    cutscan()
