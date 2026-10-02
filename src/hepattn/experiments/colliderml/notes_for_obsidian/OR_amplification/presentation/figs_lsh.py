"""How a single E2LSH ordering is actually constructed.

The deck previously asserted that each (layer, head, hash) cell owns its own ordering without
ever showing how one is built. These figures close that gap.

Formula (models/ordering.py):

    o_i = alpha . (eta_i, phi_i)  +  R_eta . D  +  R_phi . D . (ceil(eta_count) + 1)

with R_eta / R_phi the hit's 1-based equal-occupancy quantile-bin indices and D the span of the
projection. The bin terms move in steps of at least D while the projection can only vary BY D,
so the bins set the coarse order and the projection breaks ties inside a bin.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt

from theme import BASE, BG, GOOD, GRID, INK, MUTED, OR, OTHER, PANEL, VIOLET, WARN, panel, save, use_theme

RNG = np.random.default_rng(12)
N = 90


def _event():
    eta = RNG.uniform(-1, 1, N)
    phi = RNG.uniform(-1, 1, N)
    return eta, phi


def _quantile_edges(v, k):
    """Equal-occupancy edges: each bin holds the same NUMBER of hits, not the same width."""
    return np.quantile(v, np.linspace(0, 1, k + 1))


def fig_lsh_construction():
    """Three steps from 2D coordinates to a 1D ordering."""
    use_theme()
    eta, phi = _event()
    n_eta, n_phi = 3, 3
    ex, px = _quantile_edges(eta, n_eta), _quantile_edges(phi, n_phi)
    r_eta = np.clip(np.digitize(eta, ex[1:-1]), 0, n_eta - 1)
    r_phi = np.clip(np.digitize(phi, px[1:-1]), 0, n_phi - 1)

    alpha = np.array([0.80, 0.60]); alpha /= np.linalg.norm(alpha)
    proj = eta * alpha[0] + phi * alpha[1]
    span = proj.max() - proj.min()
    order_value = proj + r_eta * span + r_phi * span * (n_eta + 1)
    cell = r_phi * n_eta + r_eta

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6))
    # Each panel has a two-line caption beneath it; they collide without a generous gap.
    fig.subplots_adjust(wspace=0.38)
    cmap = plt.cm.viridis

    # --- 1. equal-occupancy quantile bins ---------------------------------------------------
    ax = axes[0]
    ax.scatter(eta, phi, s=48, c=[OTHER] * N, edgecolors="none")
    for e in ex[1:-1]:
        ax.axvline(e, color=OR, lw=2)
    for p_ in px[1:-1]:
        ax.axhline(p_, color=OR, lw=2)
    for i in range(n_eta):
        for j in range(n_phi):
            ax.text((ex[i] + ex[i + 1]) / 2, (px[j] + px[j + 1]) / 2, str(j * n_eta + i),
                    color=OR, fontsize=17, fontweight="bold", ha="center", va="center", alpha=0.55)
    ax.set_xlabel(r"$\eta$"); ax.set_ylabel(r"$\phi$")
    ax.set_title("1.  chop into quantile bins", color=OR)
    ax.text(0.5, -0.235, "equal-OCCUPANCY: every bin holds the same\nnumber of hits, so the edges "
                         "are unevenly spaced",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=11)

    # --- 2. the random projection ------------------------------------------------------------
    ax = axes[1]
    ax.scatter(eta, phi, s=48, c=proj, cmap=cmap, edgecolors="none")
    ax.annotate("", xy=(alpha[0], alpha[1]), xytext=(-alpha[0], -alpha[1]),
                arrowprops={"arrowstyle": "->", "color": WARN, "lw": 3})
    ax.text(alpha[0] * 1.06, alpha[1] * 1.06, r"$\alpha$", color=WARN, fontsize=20, fontweight="bold")
    for t in (-0.75, -0.25, 0.25, 0.75):
        ax.plot([t * alpha[0] - alpha[1], t * alpha[0] + alpha[1]],
                [t * alpha[1] + alpha[0], t * alpha[1] - alpha[0]],
                color=WARN, lw=0.9, ls=":", alpha=0.65)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel(r"$\eta$"); ax.set_ylabel(r"$\phi$")
    ax.set_title("2.  project onto a random direction", color=WARN)
    ax.text(0.5, -0.235, r"one random line per cell; each hit's position along it"
                         "\nis the tie-break INSIDE a bin",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=11)

    # --- 3. the resulting 1D sequence ---------------------------------------------------------
    ax = axes[2]
    panel(ax)
    seq = np.argsort(order_value)
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(cell[seq])) + 1, [N]])
    bin_colours = plt.cm.tab10(np.linspace(0, 1, 10))

    # Top strip: coloured by BIN - shows each bin occupies one contiguous run.
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        ax.add_patch(mpatches.Rectangle((lo, 0.62), hi - lo - 0.12, 0.26,
                                        facecolor=bin_colours[k % 10], edgecolor="none"))
    ax.text(-1.5, 0.75, "by bin", color=INK, fontsize=12, ha="right", va="center", fontweight="bold")

    # Bottom strip: coloured by PROJECTION - the ramp restarts inside every bin.
    for pos, hit in enumerate(seq):
        ax.add_patch(mpatches.Rectangle((pos, 0.24), 0.9, 0.26,
                                        facecolor=cmap((proj[hit] - proj.min()) / span),
                                        edgecolor="none"))
    for b in bounds[1:-1]:
        ax.axvline(b - 0.1, color=INK, lw=1.4, ymin=0.16, ymax=0.40)
    ax.text(-1.5, 0.37, "by projection", color=INK, fontsize=12, ha="right",
            va="center", fontweight="bold")

    ax.text(N / 2, 1.0, "bins give contiguous runs  ·  the projection ramp restarts in each one",
            color=INK, fontsize=11.5, ha="center")
    ax.set_xlim(-26, N + 2); ax.set_ylim(0.1, 1.1)
    ax.set_title("3.  sort by the combined value", color=INK)
    ax.text(0.5, -0.235, r"$o_i=\alpha\cdot(\eta_i,\phi_i)+R_\eta D+R_\phi D(\lceil n_\eta\rceil+1)$"
                         "\nthe bin terms step by $D$; the projection can only vary BY $D$",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=11.5)

    fig.suptitle("How ONE E2LSH ordering is built", color=INK, fontsize=18, fontweight="bold", y=1.04)
    return save(fig, "05a_lsh_construction")


def fig_lsh_two_cells():
    """The same hits under two different cells: different bins, different projection, different order."""
    use_theme()
    eta, phi = _event()
    track = np.argsort((eta - 0.15) ** 2 + (phi - 0.1) ** 2)[:6]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 7.4),
                             gridspec_kw={"width_ratios": [1, 1.75], "hspace": 0.55, "wspace": 0.22})

    for row, (n_eta, n_phi, a, colour, name) in enumerate(
            [(4, 2, np.array([0.9, 0.35]), BASE, "cell A"),
             (2, 5, np.array([-0.3, 0.95]), VIOLET, "cell B")]):
        a = a / np.linalg.norm(a)
        ex, px = _quantile_edges(eta, n_eta), _quantile_edges(phi, n_phi)
        r_eta = np.clip(np.digitize(eta, ex[1:-1]), 0, n_eta - 1)
        r_phi = np.clip(np.digitize(phi, px[1:-1]), 0, n_phi - 1)
        proj = eta * a[0] + phi * a[1]
        span = proj.max() - proj.min()
        seq = np.argsort(proj + r_eta * span + r_phi * span * (n_eta + 1))

        ax = axes[row][0]
        ax.scatter(eta, phi, s=34, c=OTHER, edgecolors="none")
        ax.scatter(eta[track], phi[track], s=80, c=OR, edgecolors=INK, lw=0.7, zorder=3)
        for e in ex[1:-1]:
            ax.axvline(e, color=colour, lw=1.8)
        for p_ in px[1:-1]:
            ax.axhline(p_, color=colour, lw=1.8)
        ax.annotate("", xy=(a[0], a[1]), xytext=(-a[0], -a[1]),
                    arrowprops={"arrowstyle": "->", "color": WARN, "lw": 2.2})
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}:  {n_eta}x{n_phi} bins", color=colour, fontsize=13)

        ax = axes[row][1]
        panel(ax)
        pos = {h: i for i, h in enumerate(seq)}
        for h in range(N):
            ax.add_patch(mpatches.Rectangle((pos[h], 0.4), 0.9, 0.32,
                                            facecolor=OR if h in track else OTHER, edgecolor="none"))
        idx = sorted(pos[h] for h in track)
        ax.annotate("", xy=(idx[0], 0.30), xytext=(idx[-1] + 0.9, 0.30),
                    arrowprops={"arrowstyle": "|-|", "color": OR, "lw": 1.6})
        ax.text((idx[0] + idx[-1]) / 2, 0.14, f"these 6 hits span {idx[-1] - idx[0] + 1} positions",
                color=OR, fontsize=11.5, ha="center")
        ax.set_xlim(-2, N + 2); ax.set_ylim(0.05, 0.95)
        ax.set_title(f"the sequence {name} produces", color=colour, fontsize=13)

    fig.suptitle("Same event, same six hits — two cells disagree about where they go",
                 color=INK, fontsize=17, fontweight="bold", y=1.0)
    fig.text(0.5, -0.02, "Different bin counts, different bin edges, different projection direction. "
                         "That disagreement is the point: it is what OR amplification exploits.",
             ha="center", color=MUTED, fontsize=12)
    return save(fig, "05b_lsh_two_cells")


if __name__ == "__main__":
    fig_lsh_construction()
    fig_lsh_two_cells()
