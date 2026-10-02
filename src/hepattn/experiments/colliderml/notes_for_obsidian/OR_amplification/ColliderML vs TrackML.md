---
sticker: lucide//git-compare
---

# ColliderML vs TrackML

The two simulated HL-LHC tracking datasets this project sits between. Both simulate proton–proton collisions at **μ ≈ 200 pile-up**, but they belong to different eras and serve very different scopes: TrackML is a 2018 *tracker-only, fast-simulation* challenge; ColliderML is a 2025 *full-detector, full-simulation* benchmark.

> [!SUMMARY]
> | | **TrackML** | **ColliderML** |
> |---|---|---|
> | Released | 2018 (Kaggle/Codalab challenge) | Dec 2025 (arXiv:2512.15230, NeurIPS 2025) |
> | Detector | generic, simplified silicon tracker | **OpenDataDetector** (ODD) — realistic full detector |
> | Simulation | **fast** sim (ACTS) | **full** sim + realistic digitisation |
> | Scope | silicon **tracker only** | tracker **+ EM + hadronic calorimeters** |
> | Physics | ttbar only | **10 SM + BSM** processes + single particles |
> | Pile-up | μ ≈ 200 | μ = 200 (also 20 / 40 / 60 / none) |
> | Events | ~O(10⁴) events | **~1 M** full-PU + 10 M hard-scatter + 10 M single-particle |
> | Hosting | Kaggle / CERN OpenData | Hugging Face (CERN org), EDM4hep / Parquet |

---
## TrackML

The **TrackML Particle Tracking Challenge** — a 2018 open competition (Kaggle "accuracy" phase, then a Codalab "throughput" phase) run by CERN physicists together with the ML community, to crowd-source machine-learning solutions to track reconstruction at the **High-Luminosity LHC**.

- **Detector:** a *generic, idealised* silicon tracker (pixel + strip layers), not a real experiment — designed to be broadly representative of an HL-LHC tracker. Produced with a **fast simulation** (ACTS), so detector response is simplified.
- **Physics:** top quark–antiquark (ttbar) hard-scatter events with an average of **~200 pile-up** minimum-bias collisions layered on top.
- **Scale:** roughly **10,000 particles per event** (~10⁴ reconstructable tracks) and of order **10⁵ hits per event**; the challenge shipped thousands of events, with up to ~100k events available for training.
- **What's included:** 3D tracker **hits** (with cell/charge information), **truth** particle→hit associations, and true particle kinematics. **Tracker only — no calorimetry.**
- **Purpose:** a pure *track-finding* benchmark. It became the de-facto standard dataset for ML tracking research (GNNs, transformers, etc.).

---
## ColliderML

A **2025 full-simulation detector benchmark** (arXiv:2512.15230, "first release", presented at NeurIPS 2025), built by CERN, Mainz, the Niels Bohr Institute, LBNL and Copenhagen (corr. Daniel Murnane) to give ML researchers *realistic, detector-level* HL-LHC data rather than a simplified tracking toy.

- **Detector:** the **OpenDataDetector (ODD)** — an experiment-agnostic but *realistic* geometry combining ATLAS Phase-2 ITk, CMS HGCal and FCCee features. Full stack: **inner tracker** (50×50 μm pixels + short/long strips), **EM calorimeter** (~27 X₀ silicon), **hadronic calorimeter** (scintillator), inside a **3 T** solenoid. Uses **full simulation** with realistic digitisation, so occupancy matches ATLAS ITk statistics.
- **Physics:** **10 production channels** — 7 Standard Model (ttbar, Z→ee, Z→μμ, di-Higgs, diphoton, multijet/QCD, ggF Higgs) and 3 Beyond-Standard-Model (GMSB SUSY with displaced vertices, Z′ resonance, Hidden-Valley semi-visible jets) — plus **single-particle** samples (μ, e, π, K, γ from 300 MeV–1 TeV).
- **Scale:** **~1 million** full-pile-up (μ=200) events, **10 million** hard-scatter-only events, **10 million** single-particle events, and ~300k challenge-condition events. Alternative pile-up points (μ = 20/40/60) are also provided. Total ~**400 TB** (EDM4hep) / **40 TB** (Parquet), on **Hugging Face**.
- **What's included (Release 1):** low-level **tracker hits**, **calorimeter hits**, **truth particles**, and **reconstructed tracks**. Later releases add particle-flow candidates, jets, and miscalibration/challenge sets.
- **Purpose:** a *broad* detector-level benchmark spanning track finding, calorimeter clustering, jet reconstruction and end-to-end / foundation models — not just tracking.

---
## The differences that actually matter

1. **Detector scope.** TrackML is silicon-tracker-only; ColliderML is a *whole* detector (tracker + calorimeters), enabling tasks beyond tracking.
2. **Simulation fidelity.** TrackML uses a *fast* simulation with simplified response; ColliderML uses *full* simulation with realistic digitisation — its hit occupancy matches real ATLAS ITk, whereas TrackML's is systematically lower and less realistic.
3. **Pile-up is not the difference.** Both run at μ ≈ 200. What differs is how *realistically* that pile-up is simulated, not whether it exists.
4. **Physics breadth.** TrackML is ttbar-only; ColliderML spans 10 SM+BSM processes plus clean single-particle samples.
5. **Era & scale.** TrackML (2018) is ~10⁴ events aimed at one challenge; ColliderML (2025) is a ~10⁶–10⁷-event, multi-hundred-TB resource meant as a long-lived, general ML benchmark.

> [!NOTE]
> A common shorthand — "TrackML has no pile-up" — is **not correct**: the public TrackML dataset is ttbar **+ μ≈200 pile-up**. If a note or config here treats TrackML as a small, low-occupancy sample, that reflects a *reduced/filtered subset* used in the codebase, not the dataset itself. See where these numbers get used in [[3. Padding Particles]] and [[Hit Filtering]].

---
## Data stats
![[Pasted image 20260710120639.png]]

- Mungo and Pippa use 8,500 TrackML events for training
- At the moment we only have 4,100 ColliderML events saved on disk so need to download more if want to compare

## Downloading more events

HuggingFace download:
```bash
# Fresh, self-owned copy
DEST=/share/rcif2/ljohnson/data/colliderml/v1/pu200_fresh
mkdir -p "$DEST"
# Run it detached so it survives disconnects (144 GB)
nohup hf download CERN/ColliderML-Release-1 --repo-type dataset --local-dir "$DEST" \
  --include "data/ttbar_pu200_particles/train-000[0-9][0-9]-of-01000.parquet" \
            "data/ttbar_pu200_tracker_hits/train-000[0-9][0-9]-of-01000.parquet" \
            "data/ttbar_pu200_tracks/train-000[0-9][0-9]-of-01000.parquet" \
  > "$DEST/download.log" 2>&1 &
```
- `train-000[0-9][0-9]-of-01000.parquet` matches shards 00000–00099 exactly — 100 shards × 100 events = **10,000 events**. (The quotes matter: they stop bash from expanding the `[0-9]` itself.)
- Files land at `$DEST/data/ttbar_pu200_<collection>/train-000XX-of-01000.parquet` — the repo's `data/` layout is preserved.
- Size ≈ 144 GB (particles 747 MB + tracker_hits 676 MB + tracks 14 MB per shard × 100). You have 11 TB free, so fine — but it's a long transfer.

**Can watch the download with:**
```bash
# clean progress, one line every 30s
while true; do
  echo "$(date +%T)  $(ls /share/rcif2/ljohnson/data/colliderml/v1/pu200_fresh/data/*/*.parquet 2>/dev/null | wc -l)/300"
  sleep 30
done
```
- This will output `<progress>/300` shards completed


**Verify when done (should print 100 for each):**
```bash
for c in particles tracker_hits tracks; do
  echo "$c: $(ls /share/rcif2/ljohnson/data/colliderml/v1/pu200_fresh/data/ttbar_pu200_$c/*.parquet 2>/dev/null | wc -l) shards"
done
```

Then, to split manually (8,500 train / 1,500 val = shards 00000–00084 and 00085–00099), the shards are inside `$DEST/data/ttbar_pu200_<collection>/`; move the ranges into your `train/` and `val/` folders per collection. Shout if you want the exact `mv` loop for the split.


---
## Sources
- ColliderML — arXiv:2512.15230 (`arxiv.org/abs/2512.15230`); dataset at `huggingface.co/datasets/CERN/ColliderML-Release-1`
- TrackML — *TrackML: A High Energy Physics Particle Tracking Challenge* (CERN, 2018)
