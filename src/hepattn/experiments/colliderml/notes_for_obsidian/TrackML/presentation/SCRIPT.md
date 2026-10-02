# Speaker script: *OR Amplification vs Baseline*

Read alongside `TrackML_baseline.pptx`. Slide numbers match the deck (12 slides; the number is printed bottom-right).

**Runtime:** ≈ 6 min. Roughly 25–30 s per slide, a little longer on slides 8–11.
**Shape of the argument:** the OR result was uninformative because the baseline didn't learn → fixed ColliderML's setup → chose a published TrackML baseline to reproduce → got its data and code right → started training → hit a failure, and found something real in the data on the way.

---

## 1 — Title

> "Last time I showed OR amplification working as a mechanism, with no physics result. This is what happened next: building a baseline good enough to judge it against, and what went wrong on the way."

---

## 2 — Where we left off

> "OR amplification is built and costs about 1.44x per training step. No meaningful comparison, because both arms trained equally badly.
> So 2 ways forward:
> - fix ColliderML at zero pile-up
> - and in parallel reproduce a published TrackML baseline.

---

## 3 — ColliderML: the selection was the problem

Point left, then right.

> "Before moving onto TrackML avenue, I really quickly wanted to try something to see why ColliderML was training so uselessly.
> 
> The old config kept pixel barrel hits with eta below one. That leaves 4.2 hits per particle, barely more than the three you need to count as a target. Widening to the whole tracker out to eta 2.5 at zero pile-up gives 11.8. And the overfit check, ten events trained and validated on the same events, reaches 98% efficiency.
> 
> So targets, masks and matching all work. The poor results was the bad selection I made to fit events into memory, not a bug."

**If asked about the 2% still missed:** about 1.2 of 73.5 particles per event; not investigated, and the learning rate had already decayed to its floor.

---

## 4 — TrackML: to reproduce

> "The June paper compares 3 decoders on 1 filter and encoder
> - fixed queries with masked attention
> - dynamic queries
> - and dynamic queries with local Strided cross-attention
>
> Dynamic queries at 98.1% is the number OR has to beat."

**If asked why not the 2024 TrackML paper:** it doesn't give the optimiser, learning rate or precision, and the repo's configs have since moved on to this newer paper's setup.

---

## 5 — Verified: TrackML data on NGT Cluster matches the paper

> "All 8,850 Kaggle training events, prepped, with 100 validation and 100 test as in the paper. Before any training, pixel hits per event and reconstructable particles per event agree with the table in the paper."

---

## 6 — Getting the code the paper actually used

> "The paper's code wasn't on upstream main. So correct me if I am wrong but I think, hepattn-clean is the branch to use. I merged it into my branch and had to deal with a few conflicts.
> 
> All 409 model tests pass, including all the OR tests."

---

## 7 — The experiment: 4 arms

> "2 choices: filter or no filter, sliding window or OR.
> 
> A versus D is the clean one: OR inside our existing Maskformer pipeline, same input, only the encoder changes.
> 
> B versus C asks the same with no filter, about 57 thousand hits per event, which is where a 512-hit window misses the most and OR should help most.
> 
> Every baseline is rerun as flex with no wrap, so the encoder is the only difference.
> 
> The purple comparison is probably the most interesting as this answers the question *can OR replace the hit-filter?*"

---

## 8 — First hit-filter run: diverged after 3 epochs

The filter comes first, because arms A and D need it.

> "I thought I could quickly retrain the hit filter to reproduce the UCL paper, got 3 healthy epochs, then the loss climbs for most of an epoch and goes NaN at step 34k, and stays NaN for the remaining 76 epochs.
> 
> The dashed line is the learning rate: it's warming up to its peak, and the blow-up lands right on the peak."

> "My first guess was the learning rate. I am not sure what LR was used before, I don't think it's in the paper so I just used what was already in the config as well as fp16. 
> 
> But I retrained with exactly the same schedule and it went straight through that peak, so it isn't the learning rate. The real cause turned out to be some corrupt data, which I'll show next."

---

## 9 — Six corrupt pixels in the TrackML data

> "Max and Mungo didn't encounter this but when I trained there were 6 events in the published TrackML training data each have one pixel with a charge over 100k which is problematic. Every other value in the dataset is at most 1.
> 
> I re-downloaded the originals and checked the checksums, so it's in Kaggle's files, not something I did. They're not in Kaggle's blacklist and they all sit on the edges of sensor chips."

> "No dataloader in hepattn, ours or any upstream branch, cleans this. Inputs are stored in fp16, whose maximum is 65k, so these values become infinity, and on those events every gradient comes out NaN."

---

## 10 — Why the filter broke: the fp16 gradient scale hit zero

Slow down here; this is the answer to slide 8.

> "fp16 can't store very small numbers, and gradients are often tiny. So to use more of the fp16 range it multiplies the loss by a big number, the scale. This is calculated automatically: too big and the gradients go past the max range and become infinity, too small and they become 0. If there are no problems for 2k steps in a row it doubles the scale to use more of the range.
> 
> For these problematic events the step is correctly skipped, but the scale gets halved to try and fit it into range next time. There are 6 of them, so it happens roughly every 1,400 steps, and it never gets to the 2k clean steps it needs to double. So the scale keeps getting halved and halved."

Point at the first-run column.

> "These are read straight from each epoch's checkpoint: 4,096, 256, 16, 0. As it falls, more and more small gradients round to zero, but Lion keeps taking full-size steps anyway, so the loss climbs. That's the slow climb on slide 8. When the scale hits zero, the gradients get divided by zero, and that's the NaN.
> 
> The learning rate only looked guilty because the scale ran out just as the warm-up peaked. With the six events removed, the scale sits at 32 thousand the whole way."

> "Looks much better **SHOW COMET**. At the start I thought it would be easy to retrain but yeah I think I will probably just grab an already trained from Mungo off of Hypatia."

---

## 11 — In bf16 they would silently freeze training instead

> "So in fp16 each bad step gets skipped, and it's the build-up of skips that kills the run.
> 
> If you were to use bf16, which every tracking config uses, there's no scaler at all. Lion updates each weight by the *sign* of its momentum, and the sign of NaN is zero. So one bad event turns the momentum NaN, every weight update after that is exactly zero, and the model stops learning. Nothing goes NaN, the loss looks fine, it just never improves."

> "I've removed the six events: six out of 8,650."

---

## 12 — Next, and what we need from UCL

> "Next: confirm the filter matches the paper, train dynamic queries to 98.1%, then the OR arms. And some questions for UCL…"

The questions aren't on the slide any more; ask them aloud:
- Which branch produced the paper's numbers: `hepattn-clean` or `hepattn-dq-gcp`?
- Should Sam's `dq-cleanup` matcher fix be in? Where is sparse k-max?
- The DQ+MA config says 10 epochs, the paper says 30. Which?
- Did you ever hit the six corrupt pixels, and could I take an already-trained filter?

---

## Anticipated questions

- **"Why not just clip the charge in the dataloader?"** Could, and probably should upstream. I kept the input pipeline identical to UCL's for the replication, so I removed the events instead.
- **"Is it the learning rate?"** No: the retrain used the identical schedule and passed the same peak cleanly. The checkpoints show the first run's gradient scale falling to zero instead.
- **"Doesn't the gradient scaler protect fp16?"** For one bad step, yes. For a bad step every ~1,400 steps it can't: each skip halves the scale and resets the 2,000-step count it needs to double back.
- **"How do you know it's the scale?"** Lightning saves the scaler's state in every checkpoint; the numbers on slide 10 are read straight from them, for both runs.
- **"Does OR help yet?"** Still no physics result. That waits for arm A to reproduce 98.1%.
