"""Evaluate our runs' paper-format __test.h5 files with UCL's simple evaluator as UCL evaluates Pix0.6:
models trained at pT >= 0.6 GeV, evaluated on particles with pT > 1 GeV (their evaluator's default), |eta| <= 4,
pre-filter truth (filter + tracking combined), track valid > 0.5, IoU threshold from $IOU (default 0.5).

Usage: python eval_runs.py NAME=path/to/__test.h5 [NAME=...]
"""
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "eval"))
from eval_utils_simple import evaluate_file, summarize_results  # noqa: E402

DATA_CONFIG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "tracking-eta4-pt600-epochs.yaml"
IOU = float(__import__("os").environ.get("IOU", "0.5"))
prefilter = yaml.safe_load(DATA_CONFIG.open())["data"]

print(f"{'run':28} {'pT cut':>6} {'events':>6} {'particles':>9} {'DM eff':>7} {'perfect':>7} {'fake':>6} {'dup (DM)':>8}")
for arg in sys.argv[1:]:
    name, fname = arg.split("=", 1)
    for pt_cut in (1.0,):
        tracks, parts = evaluate_file(
            fname=fname, num_events=None, eta_cut=4.0, pt_cut=pt_cut, track_valid_threshold=0.5, iou_threshold=IOU,
            truth_reference_mode="pre_filter", prefilter_data_config=prefilter, prefilter_key_mode="old",
            drop_same_hits_within_n=None, drop_one_hit_duplicates=False, duplicate_removal_random_seed=0,
        )
        s = summarize_results(tracks, parts)
        print(f"{name + f" iou{IOU}":28} {pt_cut:6.1f} {s.n_events:6d} {s.n_reconstructable_particles:9d} {s.dm_efficiency:7.1%} "
              f"{s.perfect_efficiency:7.1%} {s.fake_rate:6.1%} {s.same_majority_duplicate_rate_dm:8.1%}", flush=True)
