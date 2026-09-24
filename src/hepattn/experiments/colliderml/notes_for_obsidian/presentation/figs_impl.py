"""How the two implementations work: masked vs sorted, tiles, and the LSE merge.

These are the load-bearing diagrams of the talk. `python figs_impl.py` regenerates all of them.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from theme import BASE, BG, GOOD, GRID, INK, MUTED, OR, OTHER, PANEL, TRACK, VIOLET, WARN, panel, save, use_theme

RNG = np.random.default_rng(3)
HITS = ["A", "B", "C", "D", "E"]
ORDER = ["C", "E", "A", "D", "B"]   # head 1's ordering, used by the worked example


def _chip(ax, x, y, label, colour, w=0.78, h=0.5, text_col=BG, fs=15):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                         facecolor=colour, edgecolor="none"))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=text_col, fontsize=fs, fontweight="bold")


def fig_masked_vs_sorted():
    """The five-hit worked example: same admitted pairs, two ways of asking the question."""
    use_theme()
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.2))

    # --- masked -------------------------------------------------------------------------
    ax = axes[0]
    panel(ax)
    ax.text(-0.6, 1.62, "masked", color=WARN, fontsize=19, fontweight="bold")
    ax.text(-0.6, 1.30, "hits stay put; the ordering lives in a lookup table",
            color=MUTED, fontsize=12.5)
    for i, h in enumerate(HITS):
        _chip(ax, i, 0.62, h, OTHER, text_col=INK)
    ax.text(5.35, 0.87, "memory order, untouched", color=MUTED, fontsize=12, va="center")
    rank = {h: ORDER.index(h) + 1 for h in HITS}
    for i, h in enumerate(HITS):
        ax.text(i + 0.39, 0.30, f"{rank[h]}", ha="center", color=WARN, fontsize=14, fontweight="bold")
    ax.text(5.35, 0.36, "rank table: position of each hit in MY ordering", color=WARN, fontsize=12, va="center")
    ax.text(-0.6, -0.28, r'rule:  "within 2 places of each other $\bf{in\ my\ table}$"  '
                         "— a different rule per head, so a different mask per head",
            color=WARN, fontsize=13)
    ax.set_xlim(-0.75, 12.6); ax.set_ylim(-0.55, 1.95)

    # --- sorted -------------------------------------------------------------------------
    ax = axes[1]
    panel(ax)
    ax.text(-0.6, 1.62, "sorted", color=GOOD, fontsize=19, fontweight="bold")
    ax.text(-0.6, 1.30, "hits are physically rearranged; the table disappears",
            color=MUTED, fontsize=12.5)
    for i, h in enumerate(ORDER):
        _chip(ax, i, 0.62, h, TRACK)
    ax.text(5.35, 0.87, "this head's own copy, in its own order", color=MUTED, fontsize=12, va="center")
    ax.text(-0.6, 0.18, r'rule:  "within 2 $\bf{slots}$ of each other"  '
                        "— mentions no ordering, so ONE mask serves every head",
            color=GOOD, fontsize=13)
    ax.set_xlim(-0.75, 12.6); ax.set_ylim(-0.55, 1.95)

    fig.suptitle("Both admit exactly the same pairs.  sorted pays a shuffle to make the question simpler.",
                 color=INK, fontsize=16, fontweight="bold", y=1.015)
    return save(fig, "06_masked_vs_sorted")


def _band_mask(n, w):
    return np.array([[1.0 if abs(i - j) <= w // 2 else 0.0 for j in range(n)] for i in range(n)])


def _jittered_permutation(n, strength, rng):
    """A permutation that is correlated with position, like a real LSH ordering.

    A fully random shuffle would scatter the admitted pairs over every tile and give 100% -
    but the measured figure for `masked` is 23%, because all the orderings sort the *same* 2D
    geometry with different random grids and so stay correlated with each other. `strength` is
    how far, in positions, a hit typically moves.
    """
    return np.argsort(np.arange(n) + rng.normal(0, strength, n))


def fig_tiles():
    """Why scattered pairs cost more: flex skips per tile, never per element."""
    use_theme()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0))
    n, w, tile = 96, 7, 8
    rng = np.random.default_rng(11)

    band = _band_mask(n, w)
    perm = _jittered_permutation(n, strength=9.0, rng=rng)
    masks = [(band, "sorted", "pairs hug the diagonal", GOOD),
             (band[np.ix_(perm, perm)], "masked", "same pairs, spread out", WARN)]

    for ax, (m, name, sub, col) in zip(axes, masks, strict=True):
        ax.imshow(m, cmap=ListedColormap([PANEL, col]), vmin=0, vmax=1, interpolation="nearest")
        kept = 0
        for r in range(0, n, tile):
            for c in range(0, n, tile):
                if m[r:r + tile, c:c + tile].any():
                    kept += 1
                    ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), tile, tile, facecolor=OR,
                                                    alpha=0.22, edgecolor=OR, lw=1.6, zorder=3))
        total = (n // tile) ** 2
        ax.set_title(f"{name}  —  {sub}\n{kept}/{total} tiles computed  =  {kept / total:.0%}",
                     color=col, fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("The GPU skips whole tiles, never single pairs — so WHERE the pairs land is what costs",
                 color=INK, fontsize=16, fontweight="bold", y=1.0)
    # Three separate lines at fixed offsets: a single multi-line string wraps unpredictably
    # against the figure edge and the lines end up on top of each other.
    fig.text(0.5, -0.05,
             "shaded = a tile that must be computed in full, even if it holds a single admitted pair.   "
             "Both panels admit exactly the same number of pairs.",
             ha="center", color=MUTED, fontsize=12)
    fig.text(0.5, -0.105,
             "Schematic: percentages are for this 96-token toy — the ratio transfers, the values do not. "
             "At N=13k the real figures are 4.8% and 23.2%.",
             ha="center", color=MUTED, fontsize=11, style="italic")
    return save(fig, "07_tiles")


def fig_lse_merge():
    """The OR merge: three neighbourhoods, one exact answer."""
    use_theme()
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    panel(ax)

    ax.text(0.5, 6.55, "One hit, three orderings, one exact answer", color=INK,
            fontsize=17, fontweight="bold")
    cols = [BASE, OR, VIOLET]
    for k in range(3):
        y = 5.2 - k * 1.75
        _chip(ax, 0.5, y, f"hash {k}", cols[k], w=1.5, h=0.7, fs=13)
        ax.text(2.3, y + 0.35, f"its own window  ->  $out_{k}$,  $lse_{k}$", color=cols[k],
                fontsize=13, va="center")
        # Start the arrow well right of the label; at x=6.6 it ran straight over the "lse" term.
        ax.annotate("", xy=(7.6, 3.55), xytext=(7.15, y + 0.35),
                    arrowprops={"arrowstyle": "->", "color": cols[k], "lw": 2, "alpha": 0.85})

    _chip(ax, 7.7, 3.1, "LSE merge", GOOD, w=2.5, h=0.9, fs=15)
    # Sits low and right of the merge box: at its old position the hash-2 arrow ran through it.
    ax.text(9.9, 2.15, r"$out=\sum_c \mathrm{softmax}(lse)_c \cdot out_c$", color=GOOD,
            fontsize=14, ha="center", va="top")
    ax.annotate("", xy=(12.4, 3.55), xytext=(10.3, 3.55),
                arrowprops={"arrowstyle": "->", "color": GOOD, "lw": 2.4})
    ax.text(12.6, 3.55, "exact attention over\nthe UNION of all three", color=GOOD,
            fontsize=13.5, va="center", fontweight="bold")

    ax.text(0.5, 0.75, "Why not just average?  A neighbour seen by two orderings must count for more "
                       "than one seen by a single ordering.", color=MUTED, fontsize=12.5)
    ax.text(0.5, 0.25, r"$lse$ is the log of the softmax denominator — the attention mass each "
                       "neighbourhood gathered. It is exactly the weight needed to recombine them.",
            color=MUTED, fontsize=12.5)
    ax.set_xlim(0, 17.5); ax.set_ylim(0, 7)
    return save(fig, "08_lse_merge")


def _fit_text(ax, fig, x, y, label, box_w_data, colour, *, weight="bold", start_size=12.0):
    """Draw centred text, shrinking it until it actually fits inside `box_w_data`.

    Box widths are in data coordinates but font sizes are in points, so the two do not scale
    together — eyeballing a size that fits at one figure width breaks at another. Measuring the
    rendered extent and backing off is the only reliable way to keep text inside its box.
    """
    size = start_size
    while size > 6.5:
        t = ax.text(x, y, label, ha="center", va="center", color=colour,
                    fontsize=size, fontweight=weight)
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transData.inverted())
        if bb.width <= box_w_data * 0.88:
            return t
        t.remove()
        size -= 0.5
    return ax.text(x, y, label, ha="center", va="center", color=colour,
                   fontsize=size, fontweight=weight)


def fig_pipeline():
    """End-to-end flow through the encoder, showing where each piece plugs in."""
    use_theme()
    fig, ax = plt.subplots(figsize=(15.5, 5.2))
    panel(ax)

    steps = [
        (["hits", "(eta, phi)"], OTHER, INK),
        (["global sort", "once per event"], BASE, BG),
        (["frozen LSH", "24 orderings"], OR, BG),
        (["permute", "q / k / v"], OR, BG),
        (["flex attn", "one shared mask"], GOOD, BG),
        (["un-permute", ""], OR, BG),
        (["LSE merge", "24 -> 8 heads"], GOOD, BG),
        (["encoder", "output"], OTHER, INK),
    ]

    bw, gap, y, bh = 2.55, 0.62, 1.15, 1.15
    # Limits must be set BEFORE any text is drawn: _fit_text measures through transData, which is
    # meaningless until the axes know their extent.
    ax.set_xlim(-0.4, len(steps) * (bw + gap)); ax.set_ylim(0.15, 3.1)
    for i, (lines, col, tcol) in enumerate(steps):
        x = i * (bw + gap)
        ax.add_patch(mpatches.FancyBboxPatch((x, y), bw, bh,
                                             boxstyle="round,pad=0.02,rounding_size=0.1",
                                             facecolor=col, edgecolor="none"))
        head, sub = lines[0], lines[1]
        if sub:
            _fit_text(ax, fig, x + bw / 2, y + bh / 2 + 0.20, head, bw, tcol, start_size=12.5)
            _fit_text(ax, fig, x + bw / 2, y + bh / 2 - 0.20, sub, bw, tcol,
                      weight="normal", start_size=11.5)
        else:
            _fit_text(ax, fig, x + bw / 2, y + bh / 2, head, bw, tcol, start_size=12.5)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + bw + gap - 0.08, y + bh / 2), xytext=(x + bw + 0.08, y + bh / 2),
                        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 2})

    x0 = 2 * (bw + gap) - 0.24
    x1 = 6 * (bw + gap) + bw + 0.24
    ax.add_patch(mpatches.Rectangle((x0, y - 0.3), x1 - x0, bh + 0.6, facecolor="none",
                                    edgecolor=OR, lw=2.2, ls="--"))
    ax.text((x0 + x1) / 2, y + bh + 0.46, "new in this work", color=OR,
            fontsize=13.5, fontweight="bold", ha="center")

    ax.text(0, 0.42, "everything outside the dashed box is unchanged — the rest of the model never "
                     "sees that anything happened", color=MUTED, fontsize=12)
    ax.set_title("Where OR amplification plugs into the encoder", color=INK, pad=10)
    return save(fig, "09_pipeline")


if __name__ == "__main__":
    for f in (fig_masked_vs_sorted, fig_tiles, fig_lse_merge, fig_pipeline):
        f()


def fig_tile_mechanics():
    """Background: why tiled attention computes whole tiles, and where the mask is applied.

    This is the piece of context slide 15 assumes. FlashAttention-style kernels (and flex, which
    is the same family) never build the full N x N score matrix - they walk it in tiles that fit
    in on-chip SRAM. The mask is applied to the scores *after* the tile's matmul, so a tile is
    either skipped wholesale or paid for in full.
    """
    use_theme()
    fig, ax = plt.subplots(figsize=(14.5, 6.4))
    ax.set_xlim(0, 20); ax.set_ylim(0, 9.2)
    panel(ax)

    def box(x, y, w, h, fc, label, sub=None, tc=BG, fs=12.5):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                                             boxstyle="round,pad=0.03,rounding_size=0.12",
                                             facecolor=fc, edgecolor="none"))
        ax.text(x + w / 2, y + h / 2 + (0.26 if sub else 0), label, ha="center", va="center",
                color=tc, fontsize=fs, fontweight="bold")
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.30, sub, ha="center", va="center", color=tc, fontsize=10.5)

    ax.text(0.2, 8.65, "Step 1 — the kernel walks the score matrix in tiles, never building it whole",
            color=INK, fontsize=14, fontweight="bold")
    box(0.4, 6.4, 2.5, 1.7, BASE, "Q block", "128 queries")
    box(3.6, 6.4, 2.5, 1.7, BASE, "K block", "128 keys")
    ax.annotate("", xy=(7.4, 7.25), xytext=(6.3, 7.25),
                arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 2.2})
    box(7.6, 6.4, 3.4, 1.7, VIOLET, "one tile of scores", "128 x 128 = 16,384")
    ax.text(11.3, 7.25, "loaded into on-chip SRAM and multiplied\nas a single dense matmul",
            color=MUTED, fontsize=11.5, va="center")

    ax.text(0.2, 5.55, "Step 2 — the mask is applied to those scores, AFTER the matmul",
            color=INK, fontsize=14, fontweight="bold")
    box(0.4, 3.5, 3.4, 1.6, VIOLET, "tile of scores", "already computed")
    ax.annotate("", xy=(4.5, 4.3), xytext=(3.9, 4.3),
                arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 2.2})
    box(4.7, 3.5, 3.2, 1.6, WARN, "apply mask", "rejected -> -inf")
    ax.annotate("", xy=(8.6, 4.3), xytext=(8.0, 4.3),
                arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 2.2})
    box(8.8, 3.5, 3.0, 1.6, GOOD, "softmax", "then x V")
    ax.text(12.3, 4.3, "The arithmetic has already happened.\nMasking only discards the result.",
            color=WARN, fontsize=12, va="center", fontweight="bold")

    ax.text(0.2, 2.75, "Consequence — a tile is all-or-nothing", color=INK, fontsize=14, fontweight="bold")
    for i, (frac, lab, col) in enumerate([(0.0, "no admitted pairs\nSKIPPED entirely", GOOD),
                                          (0.02, "one admitted pair\nfull 16,384 computed", WARN),
                                          (1.0, "all pairs admitted\nfull 16,384 computed", WARN)]):
        x = 0.9 + i * 5.1
        grid = np.zeros((8, 8))
        if frac == 1.0:
            grid[:] = 1
        elif frac > 0:
            grid[2, 5] = 1
        ax.imshow(grid, extent=(x, x + 1.5, 0.45, 1.95), cmap=ListedColormap([PANEL, col]),
                  vmin=0, vmax=1, aspect="auto", zorder=3)
        ax.add_patch(mpatches.Rectangle((x, 0.45), 1.5, 1.5, facecolor="none",
                                        edgecolor=col, lw=2, zorder=4))
        ax.text(x + 1.75, 1.2, lab, color=col, fontsize=11.5, va="center")

    ax.set_title("How tiled (Flash / flex) attention actually runs", color=INK, pad=14)
    fig.text(0.5, -0.015,
             "There is no per-element skip: the GPU's tensor cores do a whole tile in lockstep, and a "
             "branch per element would cost more than the multiply it avoids.",
             ha="center", color=MUTED, fontsize=11.5)
    return save(fig, "07a_tile_mechanics")


def fig_blockmask_vs_blocksparse():
    """Two different things that sound the same.

    `BlockMask` is a PyTorch data structure: a per-tile record of which 128x128 tiles of the
    score matrix hold any admitted pair. Block-SPARSE is an attention *pattern*: the sequence
    chopped into fixed blocks, dense inside each. We store a sliding window in a BlockMask, so
    we use the structure without using the pattern.
    """
    use_theme()
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2))
    n, w, b, tile = 32, 9, 8, 8

    band = _band_mask(n, w)
    blocks = np.zeros((n, n))
    for s0 in range(0, n, b):
        blocks[s0:s0 + b, s0:s0 + b] = 1

    for ax, (m, title, sub, col) in zip(axes, [
            (band, "what WE run", "sliding window, stored in a BlockMask", GOOD),
            (blocks, "what block-SPARSE means", "fixed blocks, dense inside each", VIOLET)], strict=True):
        ax.imshow(m, cmap=ListedColormap([PANEL, col]), vmin=0, vmax=1, interpolation="nearest")
        for g in range(0, n + 1, tile):
            ax.axhline(g - 0.5, color=OR, lw=1.6)
            ax.axvline(g - 0.5, color=OR, lw=1.6)
        ax.set_title(f"{title}\n{sub}", color=col, fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])

    # Inside the axes, not below them: captions under the axes make the figure taller, and the
    # slide scales figures by height, so the whole thing would render smaller.
    axes[0].text(0.5, 0.045, "windows OVERLAP — no seams", transform=axes[0].transAxes,
                 ha="center", color=GOOD, fontsize=12.5, fontweight="bold")
    axes[1].text(0.5, 0.045, "blocks are DISJOINT — seams between them", transform=axes[1].transAxes,
                 ha="center", color=VIOLET, fontsize=12.5, fontweight="bold")

    fig.suptitle("\"BlockMask\" is not \"block-sparse\"", color=INK, fontsize=18, fontweight="bold", y=1.02)
    # Dropped well clear of the coloured per-axes captions above it.
    fig.text(0.5, -0.045,
             "The orange grid is the BlockMask's 128x128 tiling — bookkeeping for which tiles to skip.",
             ha="center", color=MUTED, fontsize=12)
    fig.text(0.5, -0.095,
             "Both patterns are stored in one. Only the right-hand pattern is block-sparse attention.",
             ha="center", color=MUTED, fontsize=12)
    return save(fig, "03a_blockmask_vs_blocksparse")
