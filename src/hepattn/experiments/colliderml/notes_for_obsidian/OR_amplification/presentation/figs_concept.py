"""Conceptual diagrams: why ordering exists, what the window is, how the two implementations differ.

Each function draws one figure and is independent, so you can re-run a single one while
refining it. `python figs_concept.py` regenerates all of them.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from theme import BASE, BG, GOOD, GRID, INK, MUTED, OR, OTHER, PANEL, TRACK, VIOLET, WARN, panel, save, use_theme

RNG = np.random.default_rng(7)


def fig_cost_scaling():
    """Why we cannot afford full attention: N^2 against N*W."""
    use_theme()
    fig, ax = plt.subplots(figsize=(9, 5))
    n = np.logspace(2, 5.2, 200)
    w = 512
    ax.loglog(n, n**2, color=WARN, lw=3, label="full attention  $N^2$")
    ax.loglog(n, n * w, color=BASE, lw=3, label=f"local attention  $N\\times W$  ($W$={w})")
    for x, lab in ((13_000, "our events\n~13k hits"), (100_000, "full pixel event\n~100k hits")):
        ax.axvline(x, color=MUTED, ls=":", lw=1.2)
        ax.text(x * 1.1, 3e3, lab, color=MUTED, fontsize=11, va="bottom")
    ax.annotate("", xy=(1e5, 1e10), xytext=(1e5, 5.1e7),
                arrowprops={"arrowstyle": "<->", "color": OR, "lw": 2})
    ax.text(1.15e5, 7e8, "~200x", color=OR, fontsize=15, fontweight="bold")
    ax.set_xlabel("hits per event, $N$")
    ax.set_ylabel("score-matrix entries per layer")
    ax.set_title("Attention cost is quadratic in the number of hits")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper left")
    return save(fig, "01_cost_scaling")


def fig_serialization():
    """Detector geometry flattened onto a line, and why the line's order matters."""
    use_theme()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    eta = RNG.uniform(-1, 1, 70)
    phi = RNG.uniform(-np.pi, np.pi, 70)
    track = np.array([0, 1, 2, 3, 4])
    eta[track] = np.linspace(0.15, 0.35, 5)
    phi[track] = np.linspace(0.6, 0.95, 5)

    ax = axes[0]
    ax.scatter(eta, phi, s=55, c=OTHER, edgecolors="none", zorder=2)
    ax.scatter(eta[track], phi[track], s=110, c=TRACK, edgecolors=INK, lw=0.8, zorder=3)
    ax.plot(eta[track], phi[track], color=TRACK, lw=2, alpha=0.6, zorder=2)
    ax.set_xlabel(r"$\eta$"); ax.set_ylabel(r"$\phi$")
    ax.set_title("Hits live in 2D detector space")
    ax.grid(alpha=0.2)
    ax.text(0.02, 0.96, "one track highlighted", transform=ax.transAxes,
            color=TRACK, fontsize=12, va="top")

    ax = axes[1]
    panel(ax, "Attention sees a 1D list")
    order_good = np.argsort(phi)
    order_bad = RNG.permutation(len(phi))
    for row, (order, lab, col) in enumerate([(order_bad, "unsorted: track scattered", WARN),
                                             (order_good, r"sorted by $\phi$: track contiguous", GOOD)]):
        y = 0.72 - row * 0.42
        pos = {h: i for i, h in enumerate(order)}
        for h in range(len(phi)):
            is_t = h in track
            ax.add_patch(mpatches.Rectangle((pos[h] / len(phi), y), 1 / len(phi) * 0.85, 0.16,
                                            facecolor=TRACK if is_t else OTHER,
                                            edgecolor="none", zorder=3 if is_t else 2))
        ax.text(0, y + 0.21, lab, color=col, fontsize=13, fontweight="bold")
        idx = sorted(pos[h] for h in track)
        ax.annotate("", xy=(idx[0] / len(phi), y - 0.045), xytext=(idx[-1] / len(phi) + 0.012, y - 0.045),
                    arrowprops={"arrowstyle": "|-|", "color": col, "lw": 1.6})
        ax.text((idx[0] + idx[-1]) / 2 / len(phi), y - 0.12,
                f"spans {idx[-1] - idx[0] + 1} positions", color=col, fontsize=11, ha="center")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.05, 1.0)
    fig.suptitle("Serialization: only after ordering does \"nearby in the list\" mean \"nearby in the detector\"",
                 color=INK, fontsize=16, fontweight="bold", y=1.02)
    return save(fig, "02_serialization")


def fig_window_vs_block():
    """Sliding window against block-sparse, and the seam that blocks introduce."""
    use_theme()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))
    # Wide gap: each panel carries a caption under it that must not meet its neighbour's.
    fig.subplots_adjust(wspace=0.42)
    n, w, b = 16, 5, 4

    ax = axes[0]
    m = np.array([[1.0 if abs(i - j) <= w // 2 else 0.0 for j in range(n)] for i in range(n)])
    ax.imshow(m, cmap="none" if False else plt.cm.colors.ListedColormap([PANEL, BASE]), vmin=0, vmax=1)
    ax.set_title("Sliding window", color=BASE)
    ax.set_xlabel("key position"); ax.set_ylabel("query position")
    ax.text(0.5, -0.19, "every hit gets a symmetric neighbourhood  ·  windows overlap  ·  no boundaries",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=11)

    ax = axes[1]
    m2 = np.zeros((n, n))
    for s in range(0, n, b):
        m2[s:s + b, s:s + b] = 1
    ax.imshow(m2, cmap=plt.cm.colors.ListedColormap([PANEL, VIOLET]), vmin=0, vmax=1)
    ax.set_title("Block-sparse", color=VIOLET)
    ax.set_xlabel("key position"); ax.set_ylabel("query position")
    for s in range(b, n, b):
        ax.axhline(s - 0.5, color=WARN, lw=2.2)
        ax.axvline(s - 0.5, color=WARN, lw=2.2)
    ax.annotate("seam", xy=(b - 0.5, b + 1.2), xytext=(b + 2.6, b + 3.4),
                color=WARN, fontsize=13, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": WARN, "lw": 2})
    ax.text(0.5, -0.19, "hits 3 and 4 are adjacent but in different blocks: they cannot see each other",
            transform=ax.transAxes, ha="center", color=WARN, fontsize=11)

    for a in axes:
        a.set_xticks(range(0, n, 4)); a.set_yticks(range(0, n, 4))
    fig.suptitle("Two ways to define \"the window of neighbours\"", color=INK,
                 fontsize=17, fontweight="bold", y=1.03)
    return save(fig, "03_window_vs_block")


def fig_split_track():
    """One ordering always cuts somewhere; a second ordering catches what the first split."""
    use_theme()
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.6))
    n, w = 28, 7
    track = [4, 5, 6, 7, 8]

    layouts = [
        ("Ordering 1", [0, 1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25, 26, 27, 6, 7, 8, 9, 10,
                        11, 12, 13, 14, 15, 16, 17, 18, 19], WARN, "track split: hits 3 and 4 land 8 apart"),
        ("Ordering 2", list(range(n)), GOOD, "same track, kept together"),
    ]
    for ax, (name, order, col, caption) in zip(axes, layouts, strict=True):
        panel(ax)
        pos = {h: i for i, h in enumerate(order)}
        for h in range(n):
            ax.add_patch(mpatches.Rectangle((pos[h], 0.35), 0.86, 0.3,
                                            facecolor=TRACK if h in track else OTHER, edgecolor="none"))
        tp = sorted(pos[h] for h in track)
        q = tp[0]
        ax.add_patch(mpatches.Rectangle((q - w // 2, 0.25), w, 0.5, facecolor="none",
                                        edgecolor=BASE, lw=2.4, ls="--"))
        ax.text(q - w // 2, 0.82, f"window of the first track hit (W={w})", color=BASE, fontsize=11)
        # Label sits clear of the strip, right-aligned so it never runs into the first cell.
        ax.text(-1.2, 0.5, name, color=INK, fontsize=14, fontweight="bold",
                va="center", ha="right")
        ax.text(n + 0.8, 0.5, caption, color=col, fontsize=12, va="center")
        ax.set_xlim(-8.5, n + 15); ax.set_ylim(0, 1)
    fig.suptitle("No single ordering keeps every track intact — but a different one usually does",
                 color=INK, fontsize=16, fontweight="bold", y=1.0)
    return save(fig, "04_split_track")


def fig_or_grid():
    """Where the diversity comes from: layers x heads x hashes."""
    use_theme()
    fig, ax = plt.subplots(figsize=(12, 5.6))
    panel(ax)
    layers, heads, hashes = 4, 8, 3
    cw, ch, gap = 0.42, 0.42, 0.1
    for L in range(layers):
        x0 = L * (hashes * (cw + gap) + 1.15)
        ax.text(x0 + hashes * (cw + gap) / 2, heads * (ch + gap) + 0.45, f"layer {L}",
                color=INK, fontsize=13, fontweight="bold", ha="center")
        for h in range(heads):
            for c in range(hashes):
                shade = 0.35 + 0.65 * RNG.random()
                ax.add_patch(mpatches.Rectangle((x0 + c * (cw + gap), h * (ch + gap)), cw, ch,
                                                facecolor=OR, alpha=shade, edgecolor=BG, lw=0.7))
        if L == 0:
            for h in range(heads):
                ax.text(x0 - 0.22, h * (ch + gap) + ch / 2, f"head {h}", color=MUTED,
                        fontsize=9.5, ha="right", va="center")
            ax.text(x0 + hashes * (cw + gap) / 2, -0.55, "3 hashes", color=MUTED, fontsize=10.5, ha="center")
    ax.set_xlim(-1.9, layers * (hashes * (cw + gap) + 1.15)); ax.set_ylim(-1.0, heads * (ch + gap) + 1.0)
    ax.set_title("Every square is an independent frozen ordering:  4 layers x 8 heads x 3 hashes = 96",
                 color=INK, pad=16)
    ax.text(0.5, -0.115, "the OR merge unions the 3 hashes inside one head of one layer  ·  "
                         "depth and heads add further diversity on top",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=11.5)
    return save(fig, "05_or_grid")


if __name__ == "__main__":
    for f in (fig_cost_scaling, fig_serialization, fig_window_vs_block, fig_split_track, fig_or_grid):
        f()
