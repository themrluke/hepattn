"""Number of target particles per pT bin on the test set (100 events, pT >= 0.6 GeV, |eta| <= 4, >= 3 pixel hits),
in the efficiency-plot bins: (a) counts per bin, (b) particles per GeV (area = count). Usage: python plot_pt_distribution.py OUTDIR"""
import pathlib, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
p = pd.read_parquet(HERE / "cache" / "paper_repro.parquet"); p = p[p.reconstructable]
edges = np.array([0.6, 0.75, 1.0, 1.5, 2, 3, 4, 6, 10])
counts, _ = np.histogram(p.particle_pt.clip(upper=9.999), bins=edges)   # >= 10 GeV (0.2%) into the last bin
share = counts / counts.sum()
labels = [f"{a:g}–{b:g}" for a, b in zip(edges[:-1], edges[1:])][:-1] + ["≥ 6"]
plt.rcParams.update({"font.family": "serif", "figure.dpi": 150})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")

below = edges[1:] <= 1.0
a1.bar(range(len(counts)), counts / 1e3, color=np.where(below, "#e8a33d", "#2c6fbb"), width=0.8)
for i, (c, s) in enumerate(zip(counts, share)):
    a1.text(i, c / 1e3 + 1, f"{s:.1%}", ha="center", va="bottom", fontsize=8)
a1.set_xticks(range(len(counts)), labels, rotation=30, fontsize=8.5)
a1.set_xlabel(r"Particle $p_\mathrm{T}^\mathrm{True}$ bin [GeV]"); a1.set_ylabel("Particles [thousands]")
a1.set_ylim(0, counts.max() / 1e3 * 1.15)
a1.set_title("(a) Particles per efficiency bin", fontsize=10)
a1.text(0.97, 0.95, f"below 1 GeV: {share[below].sum():.1%}\n(paper Table 3: ≈ 58%)", transform=a1.transAxes,
        ha="right", va="top", fontsize=9, color="#b07010")

density = counts / np.diff(edges)
a2.bar(edges[:-1], density, width=np.diff(edges), align="edge", color=np.where(below, "#e8a33d", "#2c6fbb"),
       edgecolor="white", linewidth=0.8)
a2.set_yscale("log"); a2.set_xlim(0.6, 10)
a2.set_xlabel(r"Particle $p_\mathrm{T}^\mathrm{True}$ [GeV]"); a2.set_ylabel("Particles per GeV")
a2.set_title("(b) Particles per GeV (bar area = particles in the bin;\n≥ 10 GeV, 0.2%, counted in the last bin)", fontsize=10)
fig.suptitle(f"Pix0.6 target particles, TrackML test set (100 events, {counts.sum():,} particles)", fontsize=11)
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "trackml_pix06_particles_per_pt_bin.png")
print("saved", out / "trackml_pix06_particles_per_pt_bin.png")
