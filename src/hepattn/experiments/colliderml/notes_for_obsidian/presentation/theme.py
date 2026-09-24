"""Shared visual language for the OR-amplification deck.

One place to change colours, fonts and figure sizing so every diagram and every slide look
like they belong together. Import this from any figure script; call `use_theme()` before
plotting and `save()` to write a figure out at the deck's standard size and DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# --- palette -------------------------------------------------------------------------------
# Deep slate background with a warm accent. INK/MUTED carry text, the three hues carry meaning
# and are kept consistent across every figure: BASE = today's model, OR = the new feature,
# WARN = the thing that is expensive or wrong.
BG        = "#0F1620"   # slide + figure background
PANEL     = "#18222F"   # raised panel inside a figure
INK       = "#E8EDF4"   # primary text
MUTED     = "#8FA3BC"   # secondary text, axes
GRID      = "#243141"   # gridlines, faint rules

BASE      = "#5AC8FA"   # baseline / flash / today
OR        = "#FFB454"   # OR amplification / sorted (the accent)
WARN      = "#FF6B6B"   # masked / expensive / incorrect
GOOD      = "#6BCB77"   # confirmed / cheap / correct
VIOLET    = "#C792EA"   # a fourth category when needed

# A hit that belongs to the track under discussion, vs everything else.
TRACK     = "#FFB454"
OTHER     = "#4A5B70"

FONT = "DejaVu Sans"


def use_theme() -> None:
    """Apply the deck's matplotlib style. Call once at the top of a figure script."""
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "font.family": FONT,
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "legend.frameon": False,
        "legend.fontsize": 12,
        "figure.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig, name: str, transparent: bool = False) -> Path:
    """Write `fig` to figures/<name>.png at deck resolution and close it."""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25, transparent=transparent)
    plt.close(fig)
    print(f"  wrote {path.name}")
    return path


def panel(ax, title: str | None = None) -> None:
    """Strip an axes back to a bare drawing surface, optionally titled."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, color=INK, pad=12)
