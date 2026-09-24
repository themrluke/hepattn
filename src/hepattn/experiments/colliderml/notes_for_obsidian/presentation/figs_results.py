"""Measured results: block density, cost breakdown, timings, and the compilation trap.

All numbers are the measurements recorded in the notes (A100, real ColliderML val events,
N~13k, 8 heads x 3 hashes, fp16). They live in MEASUREMENTS so there is exactly one place to
update if the benchmark is re-run.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from theme import BASE, GOOD, GRID, INK, MUTED, OR, PANEL, VIOLET, WARN, save, use_theme

MEASUREMENTS = {
    # step 4a: fraction of 128x128 tiles that must be computed, by how the sequence is laid out
    "density_by_layout": {"raw (no sort)": 0.495, "sorted by phi": 0.305, "sorted by LSH": 0.237},
    "density_reference": {"banded (today)": 0.049, "element-level ideal": 0.039},
    # per layer, ms
    "per_layer": {
        "flash + window\n(today)": {"mask": 0.0, "attn": 0.26},
        "masked": {"mask": 40.3, "attn": 3.70},
        "sorted": {"mask": 1.9, "attn": 1.94},
    },
    # whole encoder, ms/event, steady state
    "end_to_end": {"flash + window\n(today)": 2.5, "sorted": 13.9, "masked": 158.1},
    "block_sweep": {32: (0.170, 0.042), 64: (0.205, 0.045), 128: (0.237, 0.049), 256: (0.288, 0.059)},
    "compile": {"uncompiled": (44.5, 8.53), "compiled": (11.2, 0.11)},
}


def fig_density():
    """Step 4a: the go/no-go. Layout matters as much as the orderings."""
    use_theme()
    fig, ax = plt.subplots(figsize=(11, 5.6))
    d = MEASUREMENTS["density_by_layout"]
    ref = MEASUREMENTS["density_reference"]
    names, vals = list(d), list(d.values())
    bars = ax.barh(names, vals, color=[WARN, VIOLET, OR], height=0.6)
    for b, v in zip(bars, vals, strict=True):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center",
                color=INK, fontsize=13, fontweight="bold")
    # The two reference lines sit 0.01 apart, so their labels must be staggered vertically and
    # pushed to opposite sides or they collide.
    placements = {"element-level ideal": (2.90, "right", -0.006), "banded (today)": (2.90, "left", 0.006)}
    for label, v in ref.items():
        colour = GOOD if "banded" in label else MUTED
        ax.axvline(v, color=colour, ls="--", lw=2)
        y, ha, dx = placements[label]
        ax.text(v + dx, y, f"{label}\n{v:.3f}", color=colour, fontsize=11.5,
                va="top", ha=ha, linespacing=1.4)
    ax.set_xlabel("fraction of tiles that must be computed  (lower is better)")
    ax.set_xlim(0, 0.58)
    ax.set_title("Step 4a: is the per-head mask actually sparse?", pad=72)
    ax.grid(axis="x", alpha=0.25)
    fig.text(0.5, -0.04, "Verdict: viable — it did NOT collapse to fully dense.  "
                         "Sorting the sequence first is worth a free 2x.",
             ha="center", color=OR, fontsize=13, fontweight="bold")
    return save(fig, "10_density")


def fig_cost_breakdown():
    """Where the per-layer time actually goes. Mask construction is the whole story."""
    use_theme()
    fig, ax = plt.subplots(figsize=(11, 6))
    d = MEASUREMENTS["per_layer"]
    names = list(d)
    mask = [d[k]["mask"] for k in names]
    attn = [d[k]["attn"] for k in names]
    x = np.arange(len(names))
    ax.bar(x, attn, 0.55, label="attention", color=BASE)
    ax.bar(x, mask, 0.55, bottom=attn, label="building the mask", color=WARN)
    for i, k in enumerate(names):
        tot = mask[i] + attn[i]
        ax.text(i, tot + 1.6, f"{tot:.1f} ms", ha="center", color=INK, fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12.5)
    ax.set_ylabel("milliseconds per layer")
    ax.set_title("Mask construction, not attention, is what separates the two implementations",
                 pad=26)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.annotate("92% of the cost", xy=(1, 22), xytext=(1.42, 30),
                color=WARN, fontsize=13, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": WARN, "lw": 2})
    fig.text(0.5, -0.03,
             "`masked` must rebuild a 24-row mask every layer of every event.  "
             "`sorted`'s window is position-only: one row, built once, reused.",
             ha="center", color=MUTED, fontsize=12)
    return save(fig, "11_cost_breakdown")


def fig_end_to_end():
    """The number to quote: whole encoder, steady state."""
    use_theme()
    fig, ax = plt.subplots(figsize=(11, 5.4))
    d = MEASUREMENTS["end_to_end"]
    names, vals = list(d), list(d.values())
    bars = ax.barh(names, vals, color=[BASE, GOOD, WARN], height=0.55)
    for b, v in zip(bars, vals, strict=True):
        ax.text(v + 2.5, b.get_y() + b.get_height() / 2, f"{v:.1f} ms",
                va="center", color=INK, fontsize=14, fontweight="bold")
    ax.set_xlabel("milliseconds per event  —  4 layers, dim 256, N=13000, fp16, A100")
    ax.set_xlim(0, 185)
    ax.set_title("Whole-encoder cost")
    ax.grid(axis="x", alpha=0.25)
    ax.text(0.30, 0.52, "11x", transform=ax.transAxes, color=GOOD, fontsize=20, fontweight="bold")
    ax.text(0.30, 0.40, "cheaper than masked", transform=ax.transAxes, color=GOOD, fontsize=12)
    fig.text(0.5, -0.05, "OR amplification costs 5.6x today's sliding window — for 3 orderings per head",
             ha="center", color=OR, fontsize=13.5, fontweight="bold")
    return save(fig, "12_end_to_end")


def fig_block_sweep():
    """Tiling cannot close the gap: the ratio is flat across every block size."""
    use_theme()
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    d = MEASUREMENTS["block_sweep"]
    sizes = list(d)
    masked = [d[s][0] for s in sizes]
    banded = [d[s][1] for s in sizes]
    x = np.arange(len(sizes))
    ax.plot(x, masked, "o-", color=WARN, lw=3, ms=10, label="masked")
    ax.plot(x, banded, "o-", color=GOOD, lw=3, ms=10, label="banded (today / sorted)")
    for i, s in enumerate(sizes):
        ax.text(i, masked[i] + 0.016, f"{masked[i] / banded[i]:.1f}x", ha="center",
                color=OR, fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{s}x{s}" for s in sizes])
    ax.set_xlabel("flex BLOCK_SIZE"); ax.set_ylabel("tile density")
    ax.set_title("Shrinking the tiles does not rescue the masked implementation")
    ax.legend(); ax.grid(alpha=0.25); ax.set_ylim(0, 0.35)
    fig.text(0.5, -0.04, "the gap sits at 4-4.9x at EVERY block size: it is a property of how scattered "
                         "the ranks are, not of the tiling", ha="center", color=MUTED, fontsize=12)
    return save(fig, "13_block_sweep")


def fig_compile_trap():
    """A methodology warning worth the slide: the denominator changed the conclusion."""
    use_theme()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    d = MEASUREMENTS["compile"]
    for ax, idx, unit, title in ((axes[0], 0, "ms", "attention time per layer"),
                                 (axes[1], 1, "GB", "peak memory")):
        names = list(d)
        vals = [d[k][idx] for k in names]
        bars = ax.bar(names, vals, color=[WARN, GOOD], width=0.5)
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.06, f"{v:g} {unit}",
                    ha="center", color=INK, fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=14)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(vals) * 1.25)
    fig.suptitle("flex attention MUST be compiled — uncompiled it materialises the whole N x N matrix",
                 color=INK, fontsize=15.5, fontweight="bold", y=1.03)
    fig.text(0.5, -0.06,
             "Before this was fixed, attention took 271 ms/layer and mask-building looked like <4% of the cost.\n"
             "Fixing it cut attention 27x and the SAME mask cost became 92%. The measurement never changed — "
             "the thing it was divided by did.",
             ha="center", color=WARN, fontsize=12)
    return save(fig, "14_compile_trap")


if __name__ == "__main__":
    for f in (fig_density, fig_cost_breakdown, fig_end_to_end, fig_block_sweep, fig_compile_trap):
        f()


def fig_waste():
    """How much of the computed work is actually wanted.

    The element-level ideal is the fraction of the score matrix the window genuinely admits -
    3.9%. It is identical for every ordering, because a rank table is a permutation. Everything
    above it is tile-granularity waste: pairs computed only because they share a tile with a
    pair we wanted.
    """
    use_theme()
    ideal = MEASUREMENTS["density_reference"]["element-level ideal"]
    rows = [("sorted / banded", 0.048, GOOD), ("masked", 0.232, WARN)]

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    for i, (name, density, colour) in enumerate(rows):
        ax.barh(i, ideal, color=colour, height=0.5)
        ax.barh(i, density - ideal, left=ideal, color=colour, alpha=0.32, height=0.5)
        ax.text(density + 0.006, i, f"{density:.3f} computed   =   {density / ideal:.1f}x more than needed",
                va="center", color=colour, fontsize=13, fontweight="bold")
    ax.axvline(ideal, color=INK, ls="--", lw=2)
    # Annotate the ideal line from inside the axes, pointing at it, rather than floating a label
    # above the top spine where it runs into the title.
    ax.annotate(f"pairs actually wanted: {ideal:.3f}", xy=(ideal, 1.33), xytext=(0.105, 1.45),
                color=INK, fontsize=12,
                arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.5})

    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=14)
    ax.set_ylim(-0.55, 1.75)
    ax.set_xlim(0, 0.40); ax.set_xlabel("fraction of the score matrix computed")
    ax.set_title("How much of the computed work is wasted", pad=22)
    ax.grid(axis="x", alpha=0.25)

    solid = plt.Rectangle((0, 0), 1, 1, fc=MUTED)
    faded = plt.Rectangle((0, 0), 1, 1, fc=MUTED, alpha=0.32)
    ax.legend([solid, faded], ["pairs we want", "computed only because they share a tile"],
              loc="lower right", fontsize=11.5)
    fig.text(0.5, -0.06,
             "Both admit the SAME pairs. The difference is entirely where those pairs land relative "
             "to the 128x128 tile grid.",
             ha="center", color=MUTED, fontsize=12)
    return save(fig, "07b_waste")
