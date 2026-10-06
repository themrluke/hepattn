"""Efficiency vs particle phi (particles with pT > 1 GeV, as Pix0.6 is evaluated) for the window-512 models: does D (OR, no phi wrap) lose tracks near the phi = +-pi seam?
Same evaluator and operating point as plot_eff.py; caches per-particle results with phi in cache/<name>_phi.parquet."""
import pathlib, sys
import numpy as np
import pandas as pd
import yaml
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval"))
from eval_utils_simple import evaluate_file  # noqa: E402
MODELS = [  # name, label, colour, __test.h5
    ("paper_repro", "Paper reproduction (flash, w512)", "#2c6fbb", "/shared/projects/hepattn/src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-paper_20261002-T123110/ckpts/epoch=028-val_loss=0.28868__test.h5"),
    ("A_prime", "A′ (flex, w512)", "#7f7f7f", "/shared/projects/hepattn/src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-flex_20261005-T152222/ckpts/epoch=029-val_loss=0.32403__test.h5"),
    ("D_or", "D (OR, w512)", "#e8a33d", "/shared/projects/hepattn/src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-OR_20261005-T152222/ckpts/epoch=028-val_loss=0.30994__test.h5"),
]

prefilter = yaml.safe_load((HERE.parent / "configs" / "tracking-eta4-pt600-epochs.yaml").open())["data"]
rows = []
for name, label, _, fname in MODELS:
    f = HERE / "cache" / f"{name}_phi.parquet"
    if not f.exists():
        _, parts = evaluate_file(fname=fname, num_events=None, eta_cut=4.0, pt_cut=0.6, track_valid_threshold=0.5, iou_threshold=0.5,
                                 truth_reference_mode="pre_filter", prefilter_data_config=prefilter, prefilter_key_mode="old",
                                 drop_same_hits_within_n=None, drop_one_hit_duplicates=False, duplicate_removal_random_seed=0)
        parts[["particle_pt", "particle_eta", "particle_phi", "reconstructable", "metric_eff_dm", "metric_eff_perfect"]].to_parquet(f)
    p = pd.read_parquet(f); p = p[p.reconstructable & (p.particle_pt > 1.0)]   # Pix0.6 is evaluated at pT > 1 GeV
    seam = np.pi - np.abs(p.particle_phi.to_numpy())          # distance to phi = +-pi
    for lo, hi in ((0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, np.pi)):
        m = (seam >= lo) & (seam < hi)
        rows.append((label, f"{lo:.2f}-{hi:.2f}", int(m.sum()), p.metric_eff_dm[m].mean(), p.metric_eff_perfect[m].mean()))
    hp = p[p.particle_pt >= 6]
    rows.append((label, "pT >= 6 GeV (all phi)", len(hp), hp.metric_eff_dm.mean(), hp.metric_eff_perfect.mean()))
df = pd.DataFrame(rows, columns=["model", "distance to phi seam [rad]", "particles", "DM", "perfect"])
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
