Index for the work after [[1. Per-Head Ordering|OR amplification]] was finished: getting baselines that train well, so the OR encoder can finally be judged. Background on the two datasets is in [[ColliderML vs TrackML]].

# Notes, in reading order

1. [[Plan]]: the goal (a short paper), the four-arm comparison, and status
2. [[Getting Started]]: the new machine, environment, and the ColliderML and TrackML downloads
3. [[Retraining TrackML model]]: which UCL paper we reproduce, and its target numbers
4. [[Upstream code audit]]: finding the paper's code, merging it, and the bugs in it
5. [[Hit filter training]]: the first filter run, why it failed, and the retrain
6. [[Corrupt TrackML events]]: six corrupt pixels in the published TrackML data
7. [[TrackML UCL data and cuts]]: **the data, filter and cuts used from now on**; check before every training run
8. [[Arm A DQ+MA training]]: the baseline run, and why matching makes the first steps slow

# Where things stand

| | State |
|---|---|
| ColliderML pu0 | Data on EOS; overfit check passed (98.2% efficiency); full run deferred |
| TrackML data | UCL's copy (CodaLab release): 8,543 train / 100 val / 100 test in `TrackML_UCL/`; matches the paper's Table 3 exactly. Our Kaggle copy deleted 2 Oct. See [[TrackML UCL data and cuts]] |
| Code | Paper branch merged into `trackml-pix1p0` (on GitHub); 409 / 409 tests pass |
| Hit filter | **Using UCL's 600 MeV filter outputs** (99.13% retention, 20.3k hits/event: matches the paper). Our own 900 MeV filter (Kaggle data) finished but is no longer needed. See [[Hit filter training]] |
| Trackers (arms A–D) | Configs set to the paper's Pix0.6 values. **Arm A (DQ+MA) training since 2 Oct 12:31**, see [[Arm A DQ+MA training]] |

# Branches

| Branch | Where | What |
|---|---|---|
| `OR-amplification` | GitHub | The OR encoder |
| `colliderml-baseline` | GitHub | ColliderML pu0 config, EOS paths |
| `TrackML-baseline` | GitHub | From `OR-amplification` |
| `merge/hepattn-clean` | GitHub | + upstream paper code, logits fix, test updates |
| `trackml-pix1p0` | GitHub | + Pix0.6 configs (paper values) pointing at `TrackML_UCL`. **Training runs from here** |
| `notes-and-presentation` | GitHub | These notes and the presentations |

# Presentation

`presentation/` holds the deck for the meeting after OR amplification (`TrackML_baseline.pptx`), its speaker script (`SCRIPT.md`), and the scripts that make every figure from real data.
