"""Scan the two cuts the paper does not state (track-valid threshold, IoU threshold) for Pix0.6
(trained at pT >= 0.6, evaluated at pT > 1 GeV, |eta| <= 4), with UCL's evaluator. Usage: python scan_cuts.py file.h5"""
import pathlib, sys
import yaml
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "eval"))
from eval_utils_simple import evaluate_file, summarize_results  # noqa: E402

prefilter = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1] / "configs" / "tracking-eta4-pt600-epochs.yaml").open())["data"]
print(f"{'valid thr':>9} {'IoU thr':>7} {'DM eff':>7} {'perfect':>7} {'fake':>6}")
for tv in (0.1, 0.3, 0.5, 0.7):
    for iou in (0.0, 0.25, 0.5):
        tracks, parts = evaluate_file(fname=sys.argv[1], num_events=None, eta_cut=4.0, pt_cut=1.0, track_valid_threshold=tv,
                                      iou_threshold=iou, truth_reference_mode="pre_filter", prefilter_data_config=prefilter,
                                      prefilter_key_mode="old", drop_same_hits_within_n=None, drop_one_hit_duplicates=False,
                                      duplicate_removal_random_seed=0)
        s = summarize_results(tracks, parts)
        print(f"{tv:9.1f} {iou:7.2f} {s.dm_efficiency:7.1%} {s.perfect_efficiency:7.1%} {s.fake_rate:6.1%}", flush=True)
