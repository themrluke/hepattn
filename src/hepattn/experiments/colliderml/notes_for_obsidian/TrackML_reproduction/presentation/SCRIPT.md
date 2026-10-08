---
sticker: lucide//scroll-text
---
# Speaker script: *Reproducing UCL's Pix0.6 tracker*

Read alongside [[08.10.2026 Powerpoint.pptx]]. Slide numbers match the deck (25 slides; the number is printed bottom-right).

**Runtime:** ≈ 15 min. About 35 s per slide; longer on 6, 12–13, 19 and 21–22; section slides 5 s each.

**Shape of the argument:** last week the filter broke → switched to UCL's own data and filter → training was slow, so made it 2–3× faster without changing results → reproduced the paper, and found that its Table 4 mixes two pT thresholds → first fair OR-vs-baseline comparison → window 256: the baseline holds up, OR doesn't help yet in one seed.

**The one thing to land:** *we now reproduce the paper (97.7 / 94.7 / 0.3 against 98.0 / 95.3 / 0.3), and against a fair baseline OR matches on DM and improves perfect efficiency; at window 256 the baseline loses nothing and OR doesn't help in the one seed so far, so seed 43 and window 128 decide it.*

---

## 1 — Title

> "This week: getting a reproduction of UCL's Pix0.6 tracker, found a way to make the training a lot faster, an inconsistency in Table 4 of the paper, and the first OR result."

---

## 2 — Where we left off last week

> "Last week the hit-filter run diverged due to these 6 corrupt pixels in the data.
> 
> This week: used the data on Hypatia instead to circumvent that issue, trained DQ+MA to the paper's number, then OR model as per the original plan. Plus along the way I made a change that lead to 2–3× training speed-up"

---

## 3 — We now use UCL's data and their 600 MeV hit filter

> "Rather than messing around forever with the filter, I switched to UCL's copy of TrackML dataset and Pippa's 600 MeV filter outputs. I didn't know this but there were different releases of TrackML and the event IDs on Hypatia were different to the ones I was using.
> 
> The copy of TrackML and 600 MeV filter outputs on NGT Both match the paper's Table 3 exactly: 56.7k hits, 2,600 particles per event, 20.3k hits after the filter.

**If asked about the configs:** UCL's files had 4,000 queries, focal weight 100, LSCA window 128; I set them to the paper's 3,900, 25 and 64.

---

## 4 — Part 1: Making training 2–3× faster

> "First, a change I made to improve the training speed."

---

## 5 — Training was waiting on the Hungarian matcher

Point at the red bars.

> "Every step, the Hungarian matching algorithm runs multiple times on the CPU and is the largest portion of the compute.
> 
> I profiled a run: the GPU was busy only about 16% of the time. And a third of every step was just moving one 243 MB cost table around."

---

## 6 — Fix: do the reshuffling on the GPU, copy once

Point top row, then bottom row.

> "In the hepattn-clean branch the original code copies the cost table to the CPU, then makes a new copy to mask padded queries, a new copy to transpose it. Then because the next step (the solve itself) runs in parallel across 4 processes and so far this array is living in the main process' private memory, another copy has to be made into shared-memory for them all to access it.
> 
> In total: Four full copies per step.
> 
> I added 2 toggles in the config:
>  
> Toggle 1 does the masking and transposing on the GPU, keeps only the rows the solver reads, and copies once into a buffer that is reused. The solver gets exactly the same numbers, so the matching is identical: I tested that against the original code, index by index, on real events.
> 
> Toggle 2 also drops the padded query columns, which makes the solve itself smaller. That one isn't bit-identical: it changes which leftover queries get the 'no object' loss, in 0.16% of cases early in training, none late."


Each decoder layer gets its own cost matrix:
- The decoder is trained with *deep supervision*: not only the final layer's predictions get a loss, each earlier layer predicts a full set of tracks and gets its own loss which pushes every layer towards useful predictions and speeds up training.
- Query 17 at layer 1 may describe a different particle from query 17 at the final layer so each layer neds its own matching.
- Each layer needs its own matching so one cost matrix per layer (4 matrices) and the **"4 solves"** corresponds to these 4 matrices
- Then each process slices one dimension to solve of the `[4, 3900, 3900]` array


Reducing memory footprint:
- Toggle 1 trims the rows on the GPU before the copy so the padding never leaves the GPU. the solver sees the exact same numbers
- Toggle 2 trims the columns which makes the copy and each solve smaller. The solver no longer sees the padded queries

---

## 7 — Measured: 1.9–2.6× faster, 3.1× with both toggles

> "On an idle GPU, same config, same seed and events, timed after compile: toggle 1 alone is 1.9× faster early in training and 2.6× late. Both toggles: 2.2 to 3.1×.
> For a 30-epoch run that's roughly 53–70 hours down to 20–36, with identical results.
> 
> All of these gains are with no change to the solver (uses lap1015 solver before and after), only how the costs *get* to that solver"

**If asked why it's bigger late:** the solves get faster as the model learns, so the copying was a bigger share of the step.

---

## 8 — Only the matcher got faster, and it's robust

> "Two checks. The rest of the step stays at about 128 ms in every run, so only the matcher changed; and comparing step by step on the same events, the new code was faster on every single step. Repeats agree within 1%.
> 
> What's left in the matcher is mostly waiting for the GPU forward pass, plus about 40 ms of actual solving. The next gain would be overlapping those."

**If asked about other users on the machine:** we're in a container and can't see them, but I logged host load: it went from 12 to 60 during the runs and the numbers didn't move.

---

## 9 — Part 2: Reproducing the paper

---

## 10 — Every run plateaus, then its masks snap into place

Point at each drop.

> "Validation loss for three runs:
> 1. the paper reproduction with flash attention
> 2. A′, the same thing with flex attention which OR needs
> 3. and D, which adds OR.

All 3 sit on a plateau and then drop in a single epoch. What snaps into place is the track masks; the first-hit finder is already fine. The paper run dropped at epoch 3, D at 8, A′ only at 24. So A′ got six epochs of real training instead of twenty-seven."

---

## 11 — A′ isn't worse because of flex attention

> "I was a bit confused why flex attetntion looks worse but I think there is some difference in bf16 rounding and when they drop out the plateau. So the one that leaves the plateau soonest gets more epochs to improve afterwards

**If asked about the red line:** the plain PyTorch attention backend gives garbage with the same weights. Nothing we train uses it, but it's broken upstream.

---

## 12 — Test set: the reproduction is close to the paper

> "On the test set, with the evaluator in the repo at pT > 1 GeV: our reproduction gets 97.7% double-majority efficiency against the paper's 98.0, 94.7% perfect against 95.3, and the same 0.3% fake rate. Validation loss 0.289 against UCL's own run at 0.285, both best at epoch 28.

---

## 13 — Table 4's perfect efficiency is a pT ≥ 0.6 number

**LOOK AT SLIDES HERE**

> "Pix0.6 is trained down to 0.6 GeV but evaluated above 1 GeV. I took the paper's own Figure 7, pulled the exact bin values out of the PDF, and integrated them over each threshold.
> Table 4's DM column matches the figure above 1 GeV. Its perfect column only matches if you include everything down to 0.6. And the same is true for LSCA. So the perfect efficiencies in the table are at the wrong threshold: they should be 95.3 and 93.9, not 93.2 and 91.5.
> The updated version of the paper from September still has this, and even repeats 93.2 in the conclusion."

**If asked whether it could be a different operating point:** the pattern reproduces all four numbers exactly with two thresholds; a different model or cut wouldn't hit every one to the decimal.

---

## 14 — Why the threshold matters so much

> "Here is a distribution of the data and 57% of the particles we train on are below 1 GeV, where efficiency is lowest. That's how 95.3 becomes 93.2."

---

## 15 — Per pT bin we track the paper's Figure 7

> "Bin by bin against the paper's figure: double-majority efficiency is within about half a point up to 6 GeV. Perfect efficiency is about half a point below at 1 to 3 GeV."

---

## 16 — Efficiency vs η matches the paper's shape

> "Against η, same story: the dip in the barrel, below |η| of 1, is in the paper's figure too. The flash run has the best perfect efficiency almost everywhere."

---

## 17 — Operating points: the IoU cut is the lever

> "The paper doesn't say which cuts it uses on track validity and predicted IoU, so I did a scan. The validity cut barely matters; the IoU cut trades efficiency for fakes.
> Loosen it and we reach 98.2–98.3% DM, above the paper, but at two to three times its fake rate. At the paper's fake rate we're at 97.7."

---

## 18 — Part 3: OR amplification

---

## 19 — OR vs its fair baseline at window 512

> "Now OR. The fair comparison is D against A′: same flex attention, same window, same seed, only OR differs.
> Double-majority efficiency is equal: minus 0.07, plus or minus 0.07. Perfect efficiency is better with OR: plus 0.44, plus or minus 0.11, so about four sigma. The high-pT deficit is small and only about 7% of particles.
> At window 512 the plain window already sees enough neighbours, so I wouldn't expect much more. The interesting question is what happens when the window shrinks."

---

## 20 — OR's one weakness: the φ seam

> "One cost: OR's hash orderings treat φ as a straight line, so each table has a seam at ±π, where the baselines wrap around. Within 0.05 radians of the seam OR loses about 1.5 points. But that's 1.6% of particles, so overall it's 0.02 points.
> A fix would be to rotate φ randomly per hash table, so the seams land in different places and the OR merge repairs them."

---

## 21 — Window 256: the baseline holds up, OR doesn't help (yet)

> "The sweep: windows 256 and 128, baseline and OR, two seeds each, eight runs. This takes a long time to train so it is still ongoing
> 
> window 256 is done and evaluated. It seems halving the window costs the baseline nothing. 97.9 DM, 95.1 perfect, 0.3 fake: as good as the flash reproduction, and better than the same baseline at 512 within noise.
> 
> OR at 256 is 0.4 below it on DM and 1.3 below on perfect efficiency. So in this seed OR doesn't help.
> 
> But if you look at the plateau again, the baseline left it at epoch 3, OR at 9. At 512 it was the other way round, and OR won. So it seems that whoever gets off the plateau first just gets more epochs to refine itself and wins. I am retrying with a different seed and will post the results."

**If asked about the IoU cut:** same evaluator and cuts as before (track valid 0.5, IoU 0.5, pT > 1). At IoU 0: baseline 98.2 / 95.2 / 0.8, OR 97.9 / 93.9 / 0.9; same ordering.

---
## 22 — Next

> "Next: finish the sweep. Seed 43 and window 128 tell us whether the window-256 result is OR or just the plateau. Then find out why OR at 256 diverged, DQ+LSCA, a periodic-φ OR, and the next matcher speed-up."

---

## Anticipated questions

- **"How do you know toggle 1 doesn't change results?"** Unit tests against the original `matcher.py` from git, plus 16 real events from the paper run (early and late weights): the full permutation is identical in every one.
- **"Why not do the whole matching on the GPU?"** Exact GPU solvers exist, but with only 4 matrices of ~2,500 × 3,000 per step `lap1015` is already 0.1–0.4 s per solve, and results wouldn't be bit-identical. `lap1015`'s own multi-threaded mode is broken (crashes, or wrong answers).
- **"Is 97.7 vs 98.0 significant?"** One seed each, and the paper gives no spread. Bin by bin we're within half a point; seed-to-seed variation could easily be that size.
- **"Why is A′ so far behind on the plateau?"** Most likely the different rounding sends it down a different path from the same seed; the plateau length is very sensitive to that. One run per configuration can't say more, hence two seeds in the sweep.
- **"How did you get the paper's Figure 7 values?"** The figure is vector graphics in the PDF; I extracted the exact line coordinates and calibrated against the gridlines. By-eye readings agreed within 0.0025.
- **"Does OR help yet?"** At window 512 it matches on DM and improves perfect efficiency by 0.44 ± 0.11. At 256 (one seed) it is 0.4 below on DM and 1.3 below on perfect, but it also left the plateau six epochs later; seed 43 and window 128 will separate the two.
- **"Why is the baseline better at 256 than at 512?"** Most likely the plateau again: the 256 run left it at epoch 3, A′ at 24. Its test numbers match the flash reproduction, which also left at epoch 3.
- **"What's the batch size?"** One event per step, by design: the data loader has no batching, and the paper uses one event too.
