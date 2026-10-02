We need to train baseline models on [[ColliderML vs TrackML#TrackML|TrackML]] ourselves: UCL's trained models and filter outputs live on their cluster and we have no access. To check our models train correctly, compare them against the latest UCL paper.

# Which paper

**Duckett, Van Stroud, Hart, Facini, Scanlon (UCL, June 2026), *Better Queries, Cheaper Attention: Adapting Transformers for Efficient Sparse Reconstruction*, [arXiv 2606.17631](https://arxiv.org/abs/2606.17631)**

Chosen over the earlier TrackML paper ([arXiv 2411.07149](https://arxiv.org/abs/2411.07149)) because:
- It compares **3 decoders on the same filter and encoder**, three of the five in our [[Plan]]:
	1. **FQ+MA**: fixed queries with masked attention (the original MaskFormer)
	2. **DQ+MA**: dynamic queries with masked attention
	3. **DQ+LSCA**: dynamic queries with local strided cross-attention
- It states the training details the older paper leaves out
- Its configs exist in the code (`hepattn-clean`, see [[Upstream code audit]]), and match its Table 2

## Training details (Section 5.1)

- **Optimiser:** Lion, one-cycle schedule, 1e-5 → 5e-5 → 1e-5, weight decay 1e-5
- **Precision:** bf16
- **Length:** 30 epochs of 8,500 events
- **Loss:** 0.1 · valid + 2 · dice + 25 · focal + quality + DQ

![[Pasted image 20260929134425.png]]

## Pix1.0 targets (pT ≥ 1 GeV, |η| ≤ 4, pixel only)

| Decoder | Efficiency (double majority) | Perfect efficiency | Fake rate | Training memory |
|---|---|---|---|---|
| FQ+MA | 94.1% | 90.4% | 0.7% | 23.8 GB |
| DQ+MA | **98.1%** | 95.8% | 0.3% | 3.9 GB |
| DQ+LSCA | 97.1% | 91.2% | 1.0% | 3.0 GB |

Table 3, before any tracking: **56.7k** pixel hits per event → **9.7k** after the filter, with **98.6%** of reconstructable particles retained, and **1100 ± 160** reconstructable particles per event.

> [!note] Two caveats
> - The FQ+MA row is carried over from the older paper (identical numbers), not retrained. Use the older paper's architecture for it (8 decoder layers, fixed queries).
> - The paper does not state the filter threshold. **0.314** comes from UCL's tracking configs, not the paper.

> [!note] Switched to Pix0.6 on 2026-10-02
> UCL's data comes with **600 MeV** filter outputs only, so the comparison now uses **Pix0.6** (pT ≥ 0.6 GeV; DQ+MA 98.0%, DQ+LSCA 97.6%) instead of Pix1.0. Cuts and the config-vs-paper comparison are in [[TrackML UCL data and cuts]].

# Configs

On branch `trackml-pix1p0` (local). **In use now (Pix0.6, pointing at `TrackML_UCL/`):**

| Config | What |
|---|---|
| `tracking-eta4-pt600-epochs.yaml` | Pix0.6 **DQ+MA**: 8 encoder / 3 decoder layers, 4000 queries, focal 100, threshold 0.1 |
| `tracking-lca-eta4-600.yaml` | Pix0.6 **DQ+LSCA**: LSCA window 128, 3700 queries, focal 100 |

**No longer used (Pix1.0, still point at the deleted Kaggle copy):**

| Config | What |
|---|---|
| `filtering.yaml` | Hit filter `HF-900MeV-eta4`: 8 layers, window 1024, pT > 0.9 GeV, \|η\| < 4, 80 epochs, fp16 |
| `tracking-eta4-pt900-loss-weights.yaml` | Pix1.0 **DQ+MA**: 8 encoder / 3 decoder layers, 2000 queries, focal 25 |
| `tracking-lca-eta4-900.yaml` | Pix1.0 **DQ+LSCA**: LSCA window 64, 1800 queries |

> [!warning] The DQ+MA config says 10 epochs, not 30
> Check with UCL before training it.

# Plan

1. [ ] Train the hit filter for Pix1.0 and check it keeps **98.6%** of particles. See [[Hit filter training]]
2. [ ] Train DQ+MA and aim for **98.1%**
3. [ ] Add OR runs: DQ+MA with flex and no wrap, against DQ+MA with OR
4. [ ] Repeat for LSCA and FQ+MA to cover the other decoders in the paper
