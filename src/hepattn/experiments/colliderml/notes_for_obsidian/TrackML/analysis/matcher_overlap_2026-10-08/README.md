# Matcher overlap, cost trimming and step profile (8–14 Oct 2026)

Code: branch `matcher-layer-overlap` (`overlap_layers`, `trim_target_costs`, fixed `lap1015_late`). Write-up: vault note
*Hungarian matching overhead* (sections from "8 Oct: toggle 3").

| What | Files |
|---|---|
| Real-model identity checks (matchings, losses, gradients; toggle 1/2 vs +3, `trim_target_costs`) | `identity.py`, `compare.py`, `run_identity*.sh`, `*_{off,on,trimoff,trimon,rowstrimoff,rowstrimon}_*.log` (the `.pt` files with saved gradients, 2.6 GB, are not committed) |
| `trim_target_costs` against the full costs | `cost_rows_check.py`, `cost_rows_check*.log` |
| `scipy` vs `lap1015_late` timing (and what the adaptive check times) | `solver_timing.py`, `solver_timing*.log` |
| Speed benchmark (idle GPU, 11 Oct) | `run_bench.sh`, `bench_run.py`, `make_cfg.py`, `configs/`, `timing/`, `summarise.py`, `bench_status` |
| Step profile | `profile/*.txt` (torch.profiler tables), `gpu_busy.py` and `gpu_gaps.py` (from the Chrome traces, 150 MB each, not committed), `op_sites.py` and `profile/op_sites_late.txt` (GPU time by calling line) |
| Import shim so scripts run the worktree's code while jobs use the main checkout | `shim/sitecustomize.py` (with `HEPATTN_SRC`) |

Summaries: `python summarise.py timing`; `python gpu_gaps.py profile/late_overlaptrimrows.trace.json`.
