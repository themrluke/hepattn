"""Efficiency vs pT and eta for our models, with UCL's evaluator (|eta| <= 4, filter + tracking, track valid > 0.5,
IoU > 0.5). Pix0.6 is evaluated at pT > 1 GeV: the eta plot uses pT > 1 GeV particles; the pT plot shows the full
trained range (>= 0.6 GeV, to compare with the paper's Fig. 7) with the region below 1 GeV shaded. Per-particle results
do not depend on the threshold, so they are evaluated once at 0.6 and cached.
Per-particle results are cached in eval_results/cache/<name>.parquet so further models overlay quickly.
Usage: python plot_eff.py OUTDIR"""
import pathlib, sys
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))
from eval_utils_simple import evaluate_file  # noqa: E402
from plot_utils import binned, profile_plot  # noqa: E402

MODELS = [  # name, label, colour, __test.h5
    ("paper_repro", "Paper reproduction (flash, w512)", "#2c6fbb", "/shared/projects/hepattn/src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-paper_20261002-T123110/ckpts/epoch=028-val_loss=0.28868__test.h5"),
    ("A_prime", "A′ (flex, w512)", "#7f7f7f", "/shared/projects/hepattn/src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-flex_20261005-T152222/ckpts/epoch=029-val_loss=0.32403__test.h5"),
    ("D_or", "D (OR, w512)", "#e8a33d", "/shared/projects/hepattn/src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-OR_20261005-T152222/ckpts/epoch=028-val_loss=0.30994__test.h5"),
]
# Paper (arXiv:2606.17631) Fig. 7a, Pix0.6 DQ+MA, extracted exactly from the PDF's vector graphics (paper_fig7a_extracted.json)
PAPER_PT = {"dm": [0.9268, 0.9723, 0.9794, 0.9815, 0.9825, 0.9810, 0.9775, 0.9749],
            "perfect": [0.8930, 0.9416, 0.9528, 0.9551, 0.9558, 0.9480, 0.9394, 0.9348]}
PT_BINS = np.array([0.6, 0.75, 1.0, 1.5, 2, 3, 4, 6, 10], dtype=np.float32)
ETA_BINS = np.arange(-4, 4.01, 0.5, dtype=np.float32)
prefilter = yaml.safe_load((HERE.parent / "configs" / "tracking-eta4-pt600-epochs.yaml").open())["data"]
cache = HERE / "cache"; cache.mkdir(exist_ok=True)
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.family": "serif", "figure.constrained_layout.use": True})


def parts_for(name, fname):
    f = cache / f"{name}.parquet"
    if not f.exists():
        _, parts = evaluate_file(fname=fname, num_events=None, eta_cut=4.0, pt_cut=0.6, track_valid_threshold=0.5,
                                 iou_threshold=0.5, truth_reference_mode="pre_filter", prefilter_data_config=prefilter,
                                 prefilter_key_mode="old", drop_same_hits_within_n=None, drop_one_hit_duplicates=False,
                                 duplicate_removal_random_seed=0)
        parts[["particle_pt", "particle_eta", "reconstructable", "metric_eff_dm", "metric_eff_perfect"]].to_parquet(f)
        print("cached", f, flush=True)
    return pd.read_parquet(f)


data = [(label, colour, parts_for(name, fname)) for name, label, colour, fname in MODELS]
for qty, bins, xlabel, lo in (("pt", PT_BINS, r"Particle $p_\mathrm{T}^\mathrm{True}$ [GeV]", 0.80), ("eta", ETA_BINS, r"Particle $\eta^\mathrm{True}$", 0.80)):
    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    for label, colour, parts in data:
        rec = parts["reconstructable"].to_numpy(bool).copy()
        if qty == "eta":
            rec &= parts["particle_pt"].to_numpy() > 1.0
        x = parts.loc[rec, f"particle_{qty}"].to_numpy()
        for metric, ls in (("metric_eff_dm", "solid"), ("metric_eff_perfect", "dotted")):
            eff, err = binned(parts.loc[rec, metric].to_numpy(bool), x, bins, underflow=False, overflow=False, binomial=False)
            profile_plot(eff, err, bins, axes=ax, colour=colour, ls=ls)
    handles = [Line2D([0], [0], color=c, label=l) for l, c, _ in data]
    if qty == "pt":
        centres = 0.5 * (bins[1:] + bins[:-1])
        ax.scatter(centres, PAPER_PT["dm"], marker="x", color="k", s=28, zorder=5)
        ax.scatter(centres, PAPER_PT["perfect"], marker="+", color="k", s=40, zorder=5)
        handles += [Line2D([0], [0], marker="x", color="k", ls="none", label="Paper DQ+MA, DM (Fig. 7)"),
                    Line2D([0], [0], marker="+", color="k", ls="none", label="Paper DQ+MA, perfect (Fig. 7)")]
    if qty == "pt":
        ax.axvspan(bins[0], 1.0, color="0.85", alpha=0.5, zorder=0, lw=0)
        ax.axvline(1.0, color="0.4", ls="--", lw=1)
        ax.text(1.06, 1.004, "evaluated: pT > 1 GeV →", fontsize=8, color="0.3", va="top")
    ax.set_ylim(lo, 1.01); ax.set_xlim(bins[0], bins[-1]); ax.set_ylabel("Efficiency"); ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25, linestyle="--")
    fig.legend(handles=handles, frameon=False, loc="outside upper center", ncol=2, fontsize=8.5)
    ax.legend(handles=[Line2D([0], [0], color="k", label="DM"), Line2D([0], [0], color="k", ls="dotted", label="Perfect")],
              frameon=False, loc="lower left", fontsize=9)
    ax.text(0.99, 0.02, "Pix0.6 DQ+MA (trained pT ≥ 0.6 GeV), test set" + (", pT > 1 GeV" if qty == "eta" else "") + "\ntrack valid > 0.5, IoU > 0.5", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="0.35")
    fig.savefig(out / f"trackml_pix06_eff_vs_{qty}.png")
    plt.close(fig)
    print("saved", out / f"trackml_pix06_eff_vs_{qty}.png")
