Start with [[ColliderML vs TrackML#TrackML|TrackML]] to see whether [[Per-Head Ordering Steps 1-3]] improves things. It is the easiest short-term comparison, because a good MaskFormer baseline already exists on TrackML. To reach the end goal on [[ColliderML vs TrackML#ColliderML|ColliderML]] we first need to fix the base model there, starting with 0 pile-up.

# Goal: a short paper

A short (~3 page) paper pairing the OR-amplification encoder with UCL-developed MaskFormer decoders:

| Decoder | In the code? |
|---|---|
| Basic masked attention (FQ+MA) | Yes, `main` |
| Dynamic queries (DQ+MA) | Yes, `main` (PR #241) |
| LSCA (local strided cross-attention) | Yes; newer version on `hepattn-clean`, merged into our branch |
| k-max | Yes, `main` (PR #251) |
| Sparse k-max | **Not found anywhere upstream.** Ask UCL |

Three of these (FQ+MA, DQ+MA, DQ+LSCA) are compared in the latest UCL paper, which is what we are reproducing. See [[Retraining TrackML model]].

# Comparisons

The short-term studies are [[ColliderML vs TrackML#TrackML|TrackML]] pixel hits only, testing whether [[Per-Head Ordering Steps 1-3|OR-amplification]] helps before going back to [[ColliderML vs TrackML#ColliderML|ColliderML]].

Four arms:

![[Pasted image 20260928115539.png]]

| | Sliding-window encoder | OR-amplification encoder |
|---|---|---|
| **Hit filter + tracker** | **A**: UCL baseline (reproduce the paper) | **D**: OR inside UCL's pipeline |
| **Tracker only** | **C**: no filter | **B**: no filter, OR |

- **A vs D** isolates OR inside UCL's pipeline. Same input, so the cleanest result
- **B vs C** isolates OR without a filter
- **B vs A** asks whether OR can replace the filter

> [!note] Making the comparison fair
> OR needs `attn_type: flex` and `window_wrap: false`. The UCL configs use flash with wrap, so the baseline arms are rerun as flex + no wrap, so the encoder is the only difference. Ideally one extra run of A in its original flash + wrap form confirms the switch alone changes nothing.

> [!info] What to expect
> After the 600 MeV filter there are ≈ 20.3k hits per event (Pix0.6), so the gain from OR in **D** may be modest. Without the filter there are ≈ 57k hits, where a 512-hit window misses most of each event, so **B** is where OR should matter most. Both results are worth reporting.

# Status

- [x] Data downloaded, checked against the paper ([[Getting Started]])
- [x] Paper code merged and tested ([[Upstream code audit]])
- [x] Hit filter: UCL's 600 MeV filter outputs, checked against the paper ([[TrackML UCL data and cuts]])
- [x] Switched to **Pix0.6** (pT > 0.6 GeV) on UCL's data; configs pointed at it, 5-step test passes
- [ ] Confirm the config/paper mismatches with UCL (focal weight, max queries, LSCA window)
- [ ] Arm A: DQ+MA baseline reproduces **98.0%** (Pix0.6)
- [ ] Arms B, C, D
- [ ] Repeat for LSCA and FQ+MA
