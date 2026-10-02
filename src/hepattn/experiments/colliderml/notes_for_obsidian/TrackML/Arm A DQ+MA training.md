Training arm A of the [[Plan]]: the Pix0.6 **DQ+MA** baseline (dynamic queries, masked attention, with the hit filter), to reproduce the paper's **98.0%** efficiency before any OR arm is run. Data, filter and cuts: [[TrackML UCL data and cuts]].

# Config

`tracking-eta4-pt600-epochs.yaml` on branch `trackml-pix1p0` (pushed), run name `TRK-Pix0.6-DQMA-paper`.

Changed from UCL's file, to match the paper (decided 2026-10-02):

| Setting | UCL config | Now (paper) |
|---|---|---|
| Max queries / particles (`event_max_num_particles`, `num_queries`) | 4,000 | **3,900** |
| Focal loss weight (loss and matching cost) | 100 | **25** |
| Matcher processes `n_jobs` | 3 | **4** (speed only, see below) |

The DQ+LSCA config got the same changes plus **LSCA window 128 → 64**.

# Why training is slow at the start: the matching

A 400-step timing run went at **0.37 steps/s with the GPU at 0%**. Sampling the process stack showed it **always waiting on the particle matcher**.

- Every step matches **4 cost matrices** (3 decoder layers + final), each **3,900 queries × ~2,400–2,600 particles**, with an exact assignment solver on the CPU
- At initialisation **all 3,900 dynamic queries are valid** (the untrained first-hit classifier keeps every slot) and the costs are nearly uniform, which is close to the worst case for these solvers
- Timed on **real captured matrices**:

| Matrix | scipy | lap1015 |
|---|---|---|
| ~2,400 × 3,900 | 15.5 s | **0.94 s** |
| ~2,600 × 3,900 | 3.7–18.9 s | **0.8–1.3 s** |

- The matcher's adaptive mode benchmarks at step 0 and picks `lap1015`, so runs use it; but with `n_jobs: 3` the 4 matrices took **two rounds**. `n_jobs: 4` solves all four at once: **0.28 → 0.50 steps/s** at step 50, with identical results (it only changes parallelism)
- Random test matrices are ~100× easier than real ones (0.1 s), so don't benchmark solvers on random costs

> [!note] Expected to speed up
> As the model learns, fewer queries stay valid (towards the ~2,600 real particles) and the costs separate, both of which make matching faster. Watch the steps/s in the first epochs before estimating the total time: at the starting rate, 30 epochs would take ~6 days.

# Run

| | |
|---|---|
| Started | 2026-10-02 12:31, detached (`setsid nohup`) |
| Comet | `trackml-tracking/231882fd8e504f768ac312fc81a0e4e6` |
| Folder | `src/hepattn/experiments/trackml/logs/TRK-Pix0.6-DQMA-paper_*`, stdout `logs/TRK-Pix0.6-DQMA-paper_full.out` |
| Speed | 0.50 steps/s at step 50 (≈ 4.7 h per epoch at that rate) |
| GPU memory | ≈ 11.6 GB (paper: 11.1 GB for Pix0.6 DQ+MA) |

# Done criteria

- Test-set double-majority efficiency ≈ **98.0%**, perfect ≈ 93.2%, fake rate ≈ 0.3% (paper Table 4, Pix0.6 DQ+MA)
- Evaluate with UCL's own scripts from the merged branch (`eval/run_simple_eval.py`) so the definitions match
