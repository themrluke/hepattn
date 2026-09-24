"""Live training comparison, pulled from Comet.

Run with the PROJECT python (which has comet_ml), not the presentation venv:

    ../../../../../../.pixi/envs/default/bin/python figs_training.py

Re-run any time to refresh the slide with the latest steps.
"""

from __future__ import annotations

from collections import defaultdict

from matplotlib import pyplot as plt

from theme import BASE, GOOD, INK, MUTED, OR, WARN, save, use_theme

RUNS = {
    "baseline (flash + window)": ("2ce48c0324844424bcf6b1e39d9dc0e2", BASE),
    "OR amplification (sorted)": ("c7d457a244dc4cde99d32f33796fde17", OR),
}


def fetch(key):
    import comet_ml
    series = defaultdict(list)
    for r in comet_ml.API().get_experiment_by_key(key).get_metrics():
        if r["step"] is not None:
            series[r["metricName"]].append((int(r["step"]), float(r["metricValue"])))
    for k in series:
        series[k].sort()
    return series


def fig_training():
    use_theme()
    data = {label: fetch(key) for label, (key, _) in RUNS.items()}
    panels = [("val/loss", "validation loss", None),
              ("val/p0.5_sihit_eff", "track efficiency @ 50% hits", None),
              ("val/num_flows", "objects predicted per event", 300.2)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (metric, title, truth) in zip(axes, panels, strict=True):
        for label, (_, colour) in RUNS.items():
            s = data[label].get(metric, [])
            if s:
                ax.plot([x for x, _ in s], [v for _, v in s], "o-", color=colour,
                        lw=2.4, ms=5, label=label)
        if truth is not None:
            ax.axhline(truth, color=GOOD, ls="--", lw=2)
            ax.text(0.98, truth, f" truth ≈ {truth:.0f}", color=GOOD, fontsize=11,
                    ha="right", va="bottom", transform=ax.get_yaxis_transform())
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=11)

    fig.suptitle("Both arms track each other to within noise — the comparison has no signal yet",
                 color=INK, fontsize=16, fontweight="bold", y=1.04)
    fig.text(0.5, -0.155,
             "Metrics are poor for BOTH, so this is not about OR amplification: the baseline config is "
             "pixel-barrel only (|eta|<1), giving ~4.4 hits per particle.\n"
             "Efficiency at 50% means matching 2 of 4 hits; perfect-match efficiency is 0 everywhere.",
             ha="center", color=MUTED, fontsize=11.5)
    return save(fig, "15_training")


if __name__ == "__main__":
    fig_training()
