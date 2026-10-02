"""Snapshot every number the figures need into data/, so the deck can be rebuilt later.

Needs the PROJECT python (pandas, pyarrow, comet_ml), not the presentation venv:

    ../../../../../../.pixi/envs/default/bin/python extract_data.py              # everything
    ../../../../../../.pixi/envs/default/bin/python extract_data.py comet        # refresh Comet only

Writes:
    data/overfit_pu0_10ev.csv      ColliderML pu0 overfit check, one row per validation pass
    data/charge_frac.json          histogram of pixel-hit charge_frac, plus the six corrupt hits
    data/table3.json               our hit / particle counts per event, for the Table 3 comparison
    data/comet_<name>.csv          filter training curves (failed run and the retrain)
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

PREPPED = Path("/eos/user/l/ljohnson/datasets/TrackML/prepped")
OVERFIT_LOG = Path("/tmp/ljohnson/claude-173047/-eos-home-i00-l-ljohnson/866f6d2d-717c-47e9-8347-42cc7dbbf29a/scratchpad/overfit/overfit_10ev.log")
PIXEL_VOLUMES = [7, 8, 9]
FP16_MAX = 65504.0

COMET_RUNS = {
    "filter_failed": "d6fe946db8274229a3c14362b6bcf00f",   # diverged to NaN at step 34,199
    "filter_retrain": "c466da9add12497e92a01a2bb76153fd",  # 8,644 events, corrupt ones removed
}


def overfit() -> None:
    """Parse the `[overfit] step=... ` lines printed by the overfit callback."""
    if not OVERFIT_LOG.exists():
        print(f"  skip overfit: {OVERFIT_LOG} is gone (data/overfit_pu0_10ev.csv kept as is)")
        return
    rows = []
    for line in OVERFIT_LOG.read_text().replace("\r", "\n").splitlines():
        if not line.startswith("[overfit]"):
            continue
        vals = dict(re.findall(r"(\S+)=([-\d.]+)", line))
        # The callback printed train/loss then val/loss, both under the key "loss"; recover the order.
        losses = re.findall(r"\bloss=([-\d.]+)", line)
        rows.append({
            "step": int(vals["step"]),
            "train_loss": float(losses[0]),
            "val_loss": float(losses[1]),
            "eff_p05": float(vals["p0.5_sihit_eff"]),
            "eff_p10": float(vals["p1.0_sihit_eff"]),
            "pur_p05": float(vals["p0.5_sihit_pur"]),
            "num_flows": float(vals["num_flows"]),
            "num_parts": float(vals["num_parts"]),
        })
    with open(DATA / "overfit_pu0_10ev.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote overfit_pu0_10ev.csv ({len(rows)} validation passes)")


def charge_and_counts() -> None:
    """Charge histogram over the val split, the six corrupt hits, and Table 3 style counts."""
    import pandas as pd

    bins = np.logspace(-4, 6.5, 85)
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    hits_per_event, reco_per_event = [], []
    for p in sorted((PREPPED / "val").glob("*-hits.parquet")):
        h = pd.read_parquet(p, columns=["volume_id", "particle_id", "charge_frac"])
        h = h[h.volume_id.isin(PIXEL_VOLUMES)]
        counts += np.histogram(h.charge_frac.clip(lower=bins[0]), bins=bins)[0]
        parts = pd.read_parquet(str(p).replace("-hits", "-parts"))
        pt = np.hypot(parts.px, parts.py)
        eta = np.arctanh(parts.pz / np.sqrt(pt**2 + parts.pz**2))
        n = parts.particle_id.map(h.particle_id.value_counts()).fillna(0)
        hits_per_event.append(len(h))
        reco_per_event.append(int(((pt > 1.0) & (eta.abs() < 4.0) & (n >= 3)).sum()))

    corrupt = []
    for p in sorted((PREPPED / "excluded_train").glob("*-hits.parquet")):
        h = pd.read_parquet(p, columns=["volume_id", "layer_id", "charge_frac"])
        bad = h[h.charge_frac > FP16_MAX].iloc[0]
        corrupt.append({"event": int(p.name[5:14]), "charge_frac": float(bad.charge_frac),
                        "volume": int(bad.volume_id), "layer": int(bad.layer_id)})

    json.dump({"bins": bins.tolist(), "counts": counts.tolist(), "n_events": len(hits_per_event),
               "fp16_max": FP16_MAX, "corrupt": corrupt}, open(DATA / "charge_frac.json", "w"), indent=1)
    json.dump({"hits_per_event": hits_per_event, "reco_per_event": reco_per_event,
               "paper": {"hits": 56700, "reco_mean": 1100, "reco_std": 160, "post_filter_hits": 9700,
                         "retention": 98.6}},
              open(DATA / "table3.json", "w"), indent=1)
    print(f"  wrote charge_frac.json ({counts.sum():,} pixel hits, {len(corrupt)} corrupt) and table3.json")


def grad_scale() -> None:
    """fp16 GradScaler scale saved in each epoch checkpoint of both filter runs."""
    import glob
    import torch

    logs = Path("/shared/projects/hepattn/src/hepattn/experiments/trackml/logs")
    runs = {"filter_failed": "FAILED-nan-HF-900MeV-eta4_20260930-T163553",
            "filter_retrain": "HF-900MeV-eta4_20261001-T100156"}
    out = {}
    for name, d in runs.items():
        rows = []
        for f in sorted(glob.glob(str(logs / d / "ckpts" / "epoch=*.ckpt"))):
            st = torch.load(f, map_location="cpu", weights_only=False).get("MixedPrecision", {})
            rows.append({"epoch": int(Path(f).name[6:9]), "scale": float(st.get("scale", float("nan")))})
        out[name] = rows
    json.dump(out, open(DATA / "grad_scale.json", "w"), indent=1)
    print(f"  wrote grad_scale.json ({', '.join(f'{k}: {len(v)} epochs' for k, v in out.items())})")


def comet() -> None:
    import comet_ml

    api = comet_ml.API()
    for name, key in COMET_RUNS.items():
        exp = api.get_experiment("themrluke", "trackml-filtering", key)
        rows = []
        for metric in ("train/loss", "val/loss", "val/valid_recall", "val/valid_precision", "lr-Lion"):
            for m in exp.get_metrics(metric):
                rows.append({"metric": metric, "step": int(m["step"]), "value": float(m["metricValue"])})
        with open(DATA / f"comet_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["metric", "step", "value"])
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote comet_{name}.csv ({len(rows)} points)")


if __name__ == "__main__":
    which = sys.argv[1:] or ["overfit", "charge", "comet", "scale"]
    if "overfit" in which:
        overfit()
    if "charge" in which:
        charge_and_counts()
    if "comet" in which:
        comet()
    if "scale" in which:
        grad_scale()
