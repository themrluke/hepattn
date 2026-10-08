"""Snapshot every number the 8 Oct deck needs into data/, with its source.

Run with the PROJECT python (pandas, pyarrow, comet_ml):
    /shared/projects/hepattn/.pixi/envs/default/bin/python extract_data.py            # everything
    /shared/projects/hepattn/.pixi/envs/default/bin/python extract_data.py comet      # refresh the live training curves
Sections: constants, eval, fig7, timing, comet. Figures read only data/.
"""

from __future__ import annotations

import csv
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
TRACKML = Path("/shared/projects/hepattn/src/hepattn/experiments/trackml")
EVAL = TRACKML / "eval_results"
ANALYSIS = HERE.parent.parent / "TrackML" / "analysis"

RUNS = {  # name: Comet key (project trackml-tracking)
    "paper_repro": "231882fd8e504f768ac312fc81a0e4e6",
    "A_prime": "16086b485e904dd597a69295467f860c",
    "D_or": "ba80ab750e7a49c69c1e4c842e3dfe81",
    "flex_w256_s42": "b301733d7b2445a7baf7e46a7784d301",
    "or_w256_s42": "3dd7ab3989844748812f83a92a3d6e61",
}


def write_csv(name, header, rows, source):
    with open(DATA / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    (DATA / f"{name}.source").write_text(source + "\n")
    print(f"  wrote data/{name} ({len(rows)} rows)")


def comet():
    """Validation loss and efficiency per epoch for every run (live for the sweep runs)."""
    import comet_ml

    api = comet_ml.API()
    keys = ["val/loss", "val/p0.75_eff", "val/final_track_hit_valid_mask_dice"]
    rows = []
    for name, key in RUNS.items():
        exp = api.get_experiment_by_key(key)
        vals = {}
        for k in keys:
            for p in exp.get_metrics(k):
                if p.get("epoch") is not None:
                    vals.setdefault(int(p["epoch"]), {})[k] = float(p["metricValue"])
        for ep in sorted(vals):
            rows.append([name, ep] + [vals[ep].get(k, "") for k in keys])
    write_csv("val_curves.csv", ["run", "epoch", "val_loss", "val_eff075", "val_dice"], rows,
              "Comet project trackml-tracking, keys in extract_data.RUNS; per-epoch validation metrics")


def _parts(name):
    import pandas as pd

    p = pd.read_parquet(EVAL / "cache" / f"{name}_phi.parquet")
    return p[p.reconstructable]


def eval_():
    """Test-set results (pT > 1 GeV), cut scan, per-bin efficiencies, seam, OR vs baseline, pT counts."""
    rows = []
    for name, f in (("paper_repro", "paper_repro_ep28.txt"), ("paper_repro_iou0", "paper_repro_ep28_iou0.txt"),
                    ("A_prime", "flex-w512_iou0.5.txt"), ("A_prime_iou0", "flex-w512_iou0.0.txt"),
                    ("D_or", "OR-w512_iou0.5.txt"), ("D_or_iou0", "OR-w512_iou0.0.txt"),
                    *[(n.replace("-", "_").replace("OR", "or").replace("flex_w", "flex_w") + sfx, f"{n}_iou{iou}.txt")
                      for n in ("flex-w256-s42", "OR-w256-s42", "flex-w128-s42", "OR-w128-s42",
                                "flex-w256-s43", "OR-w256-s43", "flex-w128-s43", "OR-w128-s43")
                      for iou, sfx in (("0.5", ""), ("0.0", "_iou0"))]):
        if not (EVAL / f).exists():
            continue                                    # sweep run not evaluated yet
        for line in open(EVAL / f):
            m = re.search(r"\s1\.0\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%", line)
            if m:
                rows.append([name, int(m[2]), float(m[3]), float(m[4]), float(m[5])])
    write_csv("test_results.csv", ["run", "particles", "dm_eff", "perfect_eff", "fake_rate"], rows,
              f"UCL evaluator at pT > 1 GeV, track valid > 0.5, IoU > 0.5 (or 0); {EVAL}/*.txt")

    rows = []
    for line in open(EVAL / "paper_repro_ep28_cutscan_pt1.txt"):
        m = re.match(r"\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%", line)
        if m:
            rows.append([float(x) for x in m.groups()])
    write_csv("cutscan_pt1.csv", ["track_valid", "iou", "dm_eff", "perfect_eff", "fake_rate"], rows,
              f"{EVAL}/paper_repro_ep28_cutscan_pt1.txt (paper reproduction, pT > 1 GeV)")

    pt_edges = [0.6, 0.75, 1.0, 1.5, 2, 3, 4, 6, 10]
    eta_edges = list(np.round(np.arange(-4, 4.01, 0.5), 2))
    rows_pt, rows_eta, rows_seam = [], [], []
    for name in ("paper_repro", "A_prime", "D_or"):
        p = _parts(name)
        pt = p.particle_pt.to_numpy()   # particles above 10 GeV fall outside the bins, as in the paper's Fig. 7 (0.6-10 GeV)
        for lo, hi in zip(pt_edges[:-1], pt_edges[1:]):
            m = (pt >= lo) & (pt < hi)
            for metric in ("dm", "perfect"):
                e = p[f"metric_eff_{metric}"].to_numpy()[m].mean(); n = int(m.sum())
                rows_pt.append([name, lo, hi, metric, e, np.sqrt(e * (1 - e) / n), n])
        q = p[p.particle_pt > 1.0]
        eta = q.particle_eta.to_numpy()
        for lo, hi in zip(eta_edges[:-1], eta_edges[1:]):
            m = (eta >= lo) & (eta < hi)
            for metric in ("dm", "perfect"):
                e = q[f"metric_eff_{metric}"].to_numpy()[m].mean(); n = int(m.sum())
                rows_eta.append([name, lo, hi, metric, e, np.sqrt(e * (1 - e) / n), n])
        seam = np.pi - np.abs(q.particle_phi.to_numpy())
        for lo, hi in ((0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, np.pi)):
            m = (seam >= lo) & (seam < hi)
            e = q.metric_eff_dm.to_numpy()[m].mean()
            rows_seam.append([name, lo, hi, e, np.sqrt(e * (1 - e) / m.sum()), int(m.sum())])
    write_csv("eff_pt_bins.csv", ["run", "lo", "hi", "metric", "eff", "err", "n"], rows_pt,
              "per-particle cache eval_results/cache/*_phi.parquet (track valid 0.5, IoU 0.5); all trained particles 0.6 <= pT < 10 GeV (as the paper's Fig. 7)")
    write_csv("eff_eta_bins.csv", ["run", "lo", "hi", "metric", "eff", "err", "n"], rows_eta,
              "same cache, particles with pT > 1 GeV (the evaluation)")
    write_csv("seam.csv", ["run", "lo_rad", "hi_rad", "dm_eff", "err", "n"], rows_seam,
              "same cache, pT > 1 GeV; distance of particle phi to the +-pi seam")

    a, d = _parts("A_prime"), _parts("D_or")
    a, d = a[a.particle_pt > 1], d[d.particle_pt > 1]
    rows = []
    for lo, hi, label in ((1, 3, "1–3 GeV"), (3, np.inf, "≥ 3 GeV"), (1, np.inf, "all (pT > 1)")):
        ma = (a.particle_pt >= lo) & (a.particle_pt < hi); md = (d.particle_pt >= lo) & (d.particle_pt < hi)
        n = int(ma.sum())
        row = [label, n / len(a)]
        for col in ("metric_eff_dm", "metric_eff_perfect"):
            x, y = a[col][ma].mean(), d[col][md].mean()
            row += [100 * (y - x), 100 * np.sqrt((x * (1 - x) + y * (1 - y)) / n)]
        rows.append(row)
    write_csv("or_minus_baseline.csv", ["range", "share", "dm_diff", "dm_err", "perfect_diff", "perfect_err"], rows,
              "D (OR) minus A′ (flex baseline), window 512, test set, pT > 1 GeV, percentage points")

    p = _parts("paper_repro")
    counts, _ = np.histogram(p.particle_pt.clip(upper=9.999), bins=pt_edges)
    write_csv("pt_counts.csv", ["lo", "hi", "particles"], [[a_, b_, int(c)] for a_, b_, c in zip(pt_edges[:-1], pt_edges[1:], counts)],
              "test set (100 events), particles with pT >= 0.6, |eta| <= 4, >= 3 pixel hits; >= 10 GeV in the last bin")


def fig7():
    """Paper Figure 7(a) values (exact, from the PDF's vector graphics) integrated over three thresholds."""
    fig = json.load(open(EVAL / "paper_fig7a_extracted.json"))
    counts = np.array([int(r[2]) for r in list(csv.reader(open(DATA / "pt_counts.csv")))[1:]], dtype=float)
    rows = []
    for dec in ("MA", "LSCA"):
        dm = np.array([e for _, _, e in fig[dec]["DM"]]); pf = np.array([e for _, _, e in fig[dec]["perfect"]])
        for label, first in (("pT ≥ 0.6", 0), ("pT ≥ 0.75", 1), ("pT ≥ 1", 2)):
            w = counts[first:] / counts[first:].sum()
            rows.append([dec, label, 100 * np.dot(w, dm[first:]), 100 * np.dot(w, pf[first:])])
    write_csv("fig7_integrated.csv", ["decoder", "threshold", "dm_eff", "perfect_eff"], rows,
              "paper Fig. 7(a) bin values (eval_results/paper_fig7a_extracted.json) weighted by test-set particle counts")
    rows = [[dec, k, lo, hi, e] for dec in fig for k in fig[dec] for lo, hi, e in fig[dec][k]]
    write_csv("fig7_bins.csv", ["decoder", "metric", "lo", "hi", "eff"], rows, "paper Fig. 7(a), extracted from the PDF")


def timing():
    """Idle-GPU matcher timing: steps/s over steps 150-600 and matcher time per step, per run."""
    rows = []
    for f in sorted(glob.glob(str(ANALYSIS / "matcher_timing_2026-10-05" / "*_[0-9].json"))):
        stage, variant, rep = Path(f).stem.split("_")
        d = json.load(open(f))
        t = np.array(d["step_ends"]); m = np.array(d["match_times"])
        dt = np.diff(t)[149:599]
        rows.append([stage, variant, int(rep), len(dt) / dt.sum(), m[150:600].mean(), dt.mean() - m[150:600].mean()])
    write_csv("timing_runs.csv", ["stage", "variant", "rep", "steps_per_s", "matcher_s", "rest_s"], rows,
              "notes_for_obsidian/TrackML/analysis/matcher_timing_2026-10-05/*.json, steps 150-600, idle H100")


def constants():
    """Numbers typed in from documents, with where they come from."""
    c = {
        "paper_table4_printed": {"source": "UCL paper (vault _attachments/Efficient_sparse_reco_paper.pdf), Table 4, Pix0.6",
                                 "MA": {"dm": 98.0, "perfect": 93.2, "fake": 0.3}, "LSCA": {"dm": 97.6, "perfect": 91.5, "fake": 0.4}},
        "paper_corrected": {"source": "vault note 'Paper Table 4 inconsistency': DM and fake from Table 4, perfect from Fig. 7 over pT > 1",
                            "MA": {"dm": 98.0, "perfect": 95.3, "fake": 0.3}, "LSCA": {"dm": 97.6, "perfect": 93.9, "fake": 0.4}},
        "perf_audit_step_shares": {"source": "py-spy profile of the paper-reproduction run, 2 Oct (note 'Arm A DQ+MA training')",
                                   "Waiting on the 4 solves": 30, "Copy costs to CPU": 20, "Copy into new shared memory": 16,
                                   "CPU transpose": 8.5, "CPU mask (np.where)": 5.8, "Free shared memory": 4.2, "GPU work, data, other": 15.5},
        "gpu_busy_fraction_per_run": {"source": "nvidia-smi pmon, 2 Oct", "value": 0.16},
        "late_step_breakdown_ms": {"source": "breakdown.py on captured epoch-29 costs + training timings (note 'Hungarian matching overhead')",
                                   "GPU forward + costs (matcher waits)": 69, "GPU prep + one copy": 6, "4 solves on CPU (slowest)": 38,
                                   "Backward, optimiser, rest": 127},
        "flash_vs_flex": {"source": "TrackML/analysis/flex_vs_flash_2026-10-05/compare_*.log, paper-repro epoch-29 weights",
                          "flash_vs_flex_bf16_rel_diff": 0.06, "flex_bf16_vs_fp32_rel_diff": 0.11,
                          "loss_flash": 0.3214, "loss_flex": 0.3225, "events": 20, "torch_backend_loss": 6.8},
        "plateau_drop_epoch": {"source": "Comet val/loss, vault note 'Flex vs flash and the training plateau'",
                               "paper_repro": 3, "A_prime": 24, "D_or": 8, "flex_w256_s42": 3, "or_w256_s42": 9},
        "or_w256_s42_divergence": {"source": "Comet train/loss, run 3dd7ab3989844748812f83a92a3d6e61",
                                   "step": 253199, "epoch": 29.6, "loss_before": 0.46, "loss_after": 660829,
                                   "final_val_loss": 7.16, "evaluated_checkpoint": "epoch 28 (val 0.3146)"},
        "best_val_loss": {"source": "checkpoint names", "paper_repro": 0.28868, "ucl_pix06_run": 0.28455, "A_prime": 0.32403, "D_or": 0.30994,
                          "flex_w256_s42": 0.29608, "or_w256_s42": 0.31462},
        "session_expiry": "notebook session lifetime 7 days; current session expires 12 Oct 10:08 CEST",
    }
    (DATA / "constants.json").write_text(json.dumps(c, indent=1, ensure_ascii=False))
    print("  wrote data/constants.json")


SECTIONS = {"constants": constants, "eval": eval_, "fig7": fig7, "timing": timing, "comet": comet}

if __name__ == "__main__":
    for s in (sys.argv[1:] or ["constants", "eval", "fig7", "timing", "comet"]):
        print(f"[{s}]")
        SECTIONS[s]()
