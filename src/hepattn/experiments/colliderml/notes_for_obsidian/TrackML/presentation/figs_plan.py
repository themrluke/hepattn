"""Plan figures: what we must reproduce, the four-arm design, and the upstream code audit.

    .venv/bin/python figs_plan.py
"""

from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from theme import BASE, BG, GOOD, GRID, INK, MUTED, OR, PANEL, VIOLET, WARN, panel, save, use_theme

# arXiv 2606.17631, Table 4 (Pix1.0: pT >= 1 GeV, |eta| <= 4, pixel only, filter + tracker)
TARGETS = [
    # name, double-majority eff, perfect eff, fake rate, training memory GB
    ("FQ+MA\nfixed queries,\nmasked attention", 94.1, 90.4, 0.7, 23.8),
    ("DQ+MA\ndynamic queries,\nmasked attention", 98.1, 95.8, 0.3, 3.9),
    ("DQ+LSCA\ndynamic queries,\nlocal strided CA", 97.1, 91.2, 1.0, 3.0),
]


def fig_targets():
    use_theme()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [2.2, 1]})
    x = range(len(TARGETS))
    w = 0.36
    dm = [t[1] for t in TARGETS]
    pf = [t[2] for t in TARGETS]
    b1 = ax.bar([i - w / 2 for i in x], dm, w, color=BASE, label="efficiency, double majority")
    b2 = ax.bar([i + w / 2 for i in x], pf, w, color=VIOLET, label="efficiency, perfect match")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.25, f"{b.get_height():.1f}",
                    ha="center", color=INK, fontsize=13, fontweight="bold")
    ax.set_xticks(list(x), [t[0] for t in TARGETS], fontsize=12)
    ax.set_ylim(85, 100.5)
    ax.set_ylabel("% of particles (pT > 1 GeV)")
    ax.legend(loc="upper left", ncols=2)
    ax.set_title("Tracking efficiency to reproduce", loc="left")
    ax.grid(axis="y", alpha=0.25)

    fr = [t[3] for t in TARGETS]
    mem = [t[4] for t in TARGETS]
    names = [t[0].split("\n")[0] for t in TARGETS]
    ax2.barh(names, mem, color=[WARN, GOOD, GOOD], height=0.55)
    for i, (m, f) in enumerate(zip(mem, fr, strict=True)):
        ax2.text(m + 0.4, i, f"{m:.1f} GB   ·   fakes {f:.1f}%", va="center", color=INK, fontsize=12)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 38)
    ax2.set_xlabel("training memory (GB)")
    ax2.set_title("Cost and fake rate", loc="left")
    fig.suptitle("arXiv 2606.17631, Pix1.0: the numbers our retrained baselines must hit before OR is added",
                 color=INK, fontsize=15, fontweight="bold", y=1.03)
    return save(fig, "04_targets")


def _box(ax, x, y, w, h, title, sub, colour, fill=PANEL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fill, ec=colour, lw=2.4))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", color=colour, fontsize=22,
            fontweight="bold")
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", color=INK, fontsize=12.5)


def _arrow(ax, a, b, colour, text, offset=(0, 0), rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="<|-|>", mutation_scale=18, color=colour, lw=2.2,
                                 connectionstyle=f"arc3,rad={rad}"))
    mx, my = (a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1]
    ax.text(mx, my, text, ha="center", va="center", color=colour, fontsize=12, fontweight="bold",
            bbox={"fc": BG, "ec": "none", "pad": 2})


def fig_four_arms():
    use_theme()
    fig, ax = plt.subplots(figsize=(13, 6.4))
    panel(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.6)
    ax.text(3.05, 6.25, "sliding-window encoder", ha="center", color=BASE, fontsize=15, fontweight="bold")
    ax.text(7.25, 6.25, "OR-amplification encoder", ha="center", color=OR, fontsize=15, fontweight="bold")
    ax.text(0.5, 4.55, "hit filter\n+ tracker", ha="center", va="center", color=MUTED, fontsize=13)
    ax.text(0.5, 1.55, "tracker\nonly", ha="center", va="center", color=MUTED, fontsize=13)

    _box(ax, 1.6, 3.45, 2.9, 2.2, "A", "UCL baseline\n(reproduce the paper)", BASE)
    _box(ax, 5.8, 3.45, 2.9, 2.2, "D", "OR inside\nUCL's pipeline", OR)
    _box(ax, 1.6, 0.45, 2.9, 2.2, "C", "no filter,\nsliding window", BASE)
    _box(ax, 5.8, 0.45, 2.9, 2.2, "B", "no filter,\nOR amplification", OR)

    _arrow(ax, (4.55, 4.55), (5.75, 4.55), GOOD, "OR,\nsame\ninput")
    _arrow(ax, (4.55, 1.55), (5.75, 1.55), GOOD, "OR,\nsame\ninput")
    _arrow(ax, (5.75, 2.55), (4.55, 3.55), VIOLET, "", rad=0.0)
    ax.text(5.15, 3.05, "B vs A: can OR\nreplace the filter?", ha="center", va="center", color=VIOLET,
            fontsize=12, fontweight="bold", bbox={"fc": BG, "ec": "none", "pad": 1}, zorder=4)
    ax.text(9.0, 4.55, "needs the\nhit filter", ha="left", va="center", color=MUTED, fontsize=11.5)
    ax.text(9.0, 1.55, "~57k hits /\nevent", ha="left", va="center", color=MUTED, fontsize=11.5)
    ax.text(5.15, -0.05, "Every arm: Pix1.0 selection, same data split, baselines run as flex + no wrap so the "
            "encoder is the only difference", ha="center", color=MUTED, fontsize=11.5)
    return save(fig, "07_four_arms")


def fig_code_audit():
    """Two lanes: upstream branches and our fork, joined by the merge of the paper-era branch."""
    use_theme()
    fig, ax = plt.subplots(figsize=(15, 5.8))
    panel(ax)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")

    def node(x, y, label, colour, sub=None, r=0.17):
        ax.add_patch(plt.Circle((x, y), r, color=colour, zorder=3))
        ax.text(x, y + 0.38, label, ha="center", color=colour, fontsize=12, fontweight="bold")
        if sub:
            ax.text(x, y - 0.42, sub, ha="center", va="top", color=MUTED, fontsize=10.5)

    # upstream lane
    up, fork = 4.4, 1.4
    ax.text(0.1, up, "upstream\nsamvanstroud/\nhepattn", color=BASE, fontsize=12, fontweight="bold", va="center")
    ax.plot([2.0, 6.0], [up, up], color=BASE, lw=3, zorder=1)
    node(2.6, up, "DQ", BASE, "dynamic\nqueries #241")
    node(4.3, up, "k-max", BASE, "#251")
    node(6.0, up, "main", BASE, "27 Feb")
    ax.plot([6.0, 7.6, 10.6], [up, 5.3, 5.3], color=OR, lw=3, zorder=1)
    node(10.6, 5.3, "hepattn-clean", OR)
    ax.text(11.1, 5.3, "paper code (4 Jun):\nnewer LSCA, Table 2 configs", color=OR, fontsize=11, va="center")
    ax.plot([6.0, 7.0], [up, 3.55], color=MUTED, lw=2, zorder=1)
    ax.text(7.1, 3.45, "dq-gcp · working-lca (PR #253)\n· dq-cleanup: not merged into\nmain or the paper branch",
            color=MUTED, fontsize=10.5, va="center")

    # fork lane
    ax.text(0.1, fork, "our fork\nthemrluke/\nhepattn", color=GOOD, fontsize=12, fontweight="bold", va="center")
    ax.plot([2.0, 14.0], [fork, fork], color=GOOD, lw=3, zorder=1)
    node(3.0, fork, "OR-amplification", GOOD, "OR encoder")
    node(6.0, fork, "TrackML-baseline", GOOD, "all of main")
    node(10.6, fork, "merge", GOOD, "3 conflicts\nresolved")
    node(12.0, fork, "fix", WARN, "NameError:\nlogits")
    node(13.4, fork, "tests", GOOD, "409 / 409\npassing")
    ax.add_patch(FancyArrowPatch((10.6, 5.08), (10.6, 2.12), arrowstyle="-|>", mutation_scale=20,
                                 color=OR, lw=2.6, zorder=2))
    ax.add_patch(FancyArrowPatch((6.0, up - 0.85), (6.0, 2.12), arrowstyle="-|>", mutation_scale=16,
                                 color=BASE, lw=2, ls=":", zorder=2))
    ax.text(14.1, fork, "trackml-pix1p0", color=GOOD, fontsize=12, fontweight="bold", va="center")
    return save(fig, "06_code_audit")


if __name__ == "__main__":
    fig_targets()
    fig_four_arms()
    fig_code_audit()
