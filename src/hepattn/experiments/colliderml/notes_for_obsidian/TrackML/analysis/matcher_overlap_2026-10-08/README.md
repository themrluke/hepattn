# Matcher overlap, cost trimming, speed-ups and step profile (8–14 Oct 2026)

Code: branches `matcher-layer-overlap` (`overlap_layers`, `trim_target_costs`, fixed `lap1015_late`) and `training-speedups` (`fast_permute`, `slice_valid_targets`, single-product focal cost, `scipy`). Write-up: vault note
*Hungarian matching overhead* (sections from "8 Oct: toggle 3").

| What | Files |
|---|---|
| Real-model identity checks (matchings, losses, gradients; toggle 1/2 vs +3, `trim_target_costs`) | `identity.py`, `compare.py`, `run_identity*.sh`, `*_{off,on,trimoff,trimon,rowstrimoff,rowstrimon}_*.log` (the `.pt` files with saved gradients, 2.6 GB, are not committed) |
| `trim_target_costs` against the full costs | `cost_rows_check.py`, `cost_rows_check*.log` |
| `scipy` vs `lap1015_late` timing (and what the adaptive check times) | `solver_timing.py`, `solver_timing*.log` |
| Speed benchmark (idle GPU, 11 Oct) | `run_bench.sh`, `bench_run.py`, `make_cfg.py`, `configs/`, `timing/`, `summarise.py`, `bench_status` |
| Step profile | `profile/*.txt` (torch.profiler tables), `gpu_busy.py` and `gpu_gaps.py` (from the Chrome traces, 150 MB each, not committed), `op_sites.py` and `profile/op_sites_late.txt` (GPU time by calling line) |
| 14 Oct: deterministic whole-model bit checks (`identity.py` with `+`-joined options, e.g. `perm+slice+det+eager`) | `run_identity5.sh`, `*+det*.log` |
| 14 Oct: single-product focal cost on real events, scored against float64 | `focal_cost_check.py`, `focal_cost_check.log` |
| 14 Oct speed benchmark of the speed-ups | `run_bench2.sh`, `run_bench2b.sh` (extra repeat after a stalled run), `configs/otr*.yaml`, `timing2/`, `summarise2.py`, `bench2_status`, `profile/late_otrpss.txt` |
| Import shim so scripts run the worktree's code while jobs use the main checkout | `shim/sitecustomize.py` (with `HEPATTN_SRC`) |

Summaries: `python summarise.py timing`; `python gpu_gaps.py profile/late_overlaptrimrows.trace.json`.
