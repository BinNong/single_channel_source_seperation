# Paper 3 — Open-Set SC-BSS: Outline

> **Target venue:** ICASSP 2027 / IEEE TSP
> **Length:** ~6 pages (ICASSP) or ~12 pages (TSP)
> **Revised 2026-08-21** after completing the 5-seed baseline, center-loss
> and embedding-dim ablations, and the SNR-routed ensemble analysis (see
> `EXPERIMENT_LOG.md`).  The narrative below supersedes the original
> "≥ 0.9 AUROC" framing, which the data did not support.

---

## Title (tentative)

**"Open-Set Single-Channel Blind Source Separation: Per-Source Modulation Detection is SNR-Dependent"**

(alternative: keep the original "Don't Know What You'll Get" title, but the
SNR-dependence finding is the paper's real hook and could go in the title)

---

## Abstract (~150 words) — REVISED

Existing single-channel blind source separation (SC-BSS) systems for
co-frequency communication signals assume the modulation format of every
source is known a-priori; a practical receiver has no such guarantee.  We
reframe SC-BSS as an **open-set recognition problem**: joint separation and
per-source modulation identification with rejection of unseen modulations.
On a new open-set SC-BSS benchmark (4 known / 4 held-out modulations,
three evaluation protocols), we show that standard OOD scorers
(Energy, Prototype, VOS) over per-source embeddings from a lightweight
complex-valued backbone (~69K params) are **at chance when pooled across
SNR — but strongly complementary when conditioned on SNR**: Prototype/VOS
reach 0.85 AUROC at 10 dB yet collapse below 0 dB, where Energy instead
peaks at 0.74.  Exploiting this, a training-free **SNR-routed ensemble**
lifts weighted-average AUROC from 0.50 (best single scorer) to **0.625**,
within 0.02 of the per-SNR oracle, and degrades gracefully (≥ 0.576) under
up to 6 dB of SNR-estimation error.  Ablation studies show the effect is a
property of the backbone features, not of head capacity or training loss.

---

## 1. Introduction

- Motivation: real receivers don't know modulation priors
- Gap: existing SC-BSS literature assumes closed-set; existing OOD
  literature assumes a single global operating point — neither expects
  detectability to flip sign with SNR
- Contributions:
  1. **Problem + benchmark**: first open-set formulation of SC-BSS;
     per-source OOD detection under kk/ku/uu protocols with
     64QAM / π/4-DQPSK / MSK / OFDM-QPSK held out
  2. **Finding**: per-source OOD detectability is *SNR-dependent*;
     pooled AUROC ≈ 0.50 masks a strong complementary structure
     (Prototype/VOS good ≥ 5 dB, Energy good ≤ 0 dB)
  3. **Method**: SNR-routed ensemble — training-free, +0.12 weighted
     AUROC over the best single scorer, robust to SNR-estimation error
- Outline of paper

## 2. Related Work

- SC-BSS for communication signals (Hou & Gao 2022, Guo 2024, etc.)
- Complex-valued neural networks (paper1's C-SE, Trabelsi 2018, etc.)
- Open-set recognition / OOD detection (ODIN, Energy, VOS, OpenMax)
- SNR-dependent behaviour in modulation classification literature
- Highlight: no prior work on open-set SC-BSS; no prior OOD work routes
  scorers by an operating-condition variable

## 3. Method

### 3.1 Problem Setup
- Mixture model: $y(t) = \alpha_1 s_1(t) + \alpha_2 s_2(t) + n(t)$
- Closed-set assumption relaxed: $s_i$ from $\mathcal{K} \cup \mathcal{U}$
- Goal: recover $(s_1, s_2)$ AND $(\hat{m}_1, \hat{m}_2) \in (\mathcal{K} \cup \{\text{unknown}\})^2$
- Evaluation protocols: kk (both known), ku (one unknown), uu (both unknown)

### 3.2 OpenSetCSE Architecture
- Backbone: C-SE (paper 1) — complex Conv + SE blocks (~69K params total)
- Per-source head: mask-weighted bottleneck features → 64-d embedding → logits
- Joint training: $\mathcal{L} = \mathcal{L}_{\text{SI-SDR}} + \alpha \mathcal{L}_{\text{CE}}$ with PIT

### 3.3 Per-Source OOD Scoring
- Energy score: $E(x) = -T \log \sum_k \exp(f_k(x)/T)$
- Prototype distance: $\min_k \| z - \mu_k \|_2$
- VOS: synthesise virtual outliers, distance to nearest

### 3.4 SNR-Routed Ensemble (main method)
- Rule (fixed a-priori from the training/validation SNR profile, NOT tuned
  on test): Energy for SNR ≤ 0 dB, Prototype for SNR ≥ 5 dB
- Metric: per-SNR-bin AUROC (both pools at the same SNR), weighted average
  — an SNR-conditioned detector, reported with per-bin operating points
- Practical deployment: route on estimated SNR; sensitivity analysed in §4.6

## 4. Experiments

### 4.1 Setup
- Backbone: OpenSetCSE (~69K params)
- Known: BPSK, QPSK, 8PSK, 16QAM; Unknown: 64QAM, π/4-DQPSK, MSK, OFDM-QPSK
- Training SNR: [-5, 20] dB; Test SNR: [-10, ..., 20] dB
- 5 seeds (42–46); every reported number is multi-seed (single-seed
  low-SNR results proved unreliable twice — worth one honest sentence)

### 4.2 Closed-Set Sanity
- SI-SDR −0.88 dB pooled, positive at SNR ≥ 10 dB (+0.78 @ 20 dB);
  cls_acc 0.41 (chance 0.25)
- Message: backbone works; the difficulty is genuinely in OOD detection

### 4.3 The Pooled-Metric Trap (negative result #0)
- Table: AUROC / AUPR / FPR@95 / OSCR for 3 scorers, all ≈ 0.50
- Per-SNR breakdown reveals the complementary structure (Fig 1)
- Message: pooled open-set metrics are misleading for SC-BSS

### 4.4 SNR-Routed Ensemble (main result)
- Weighted-avg AUROC 0.625 ± 0.031 vs 0.502 best single; oracle 0.641
- Fig 1 (per-SNR profile), Fig 2 (overall bars), Table (main results)

### 4.5 Ablations (both negative, both informative)
- **Center loss** (λ=0.1, 5 seeds): no aggregate gain; weakens the
  high-SNR prototype signal the router needs (routed 0.568 vs 0.625)
  → OOD margin is not shaped by classification-side losses
- **Embedding dim** 16/32/64/128 (3 seeds): closed-set flat, routed
  AUROC within noise (0.588–0.631, non-monotonic)
  → OOD performance is backbone-feature-limited, not capacity-limited

### 4.6 Robustness to SNR-Estimation Error
- Routed AUROC: 0.625 (ideal) → 0.597 (σ=1) → 0.593 (σ=3) → 0.576 (σ=6 dB)
- Always ≥ 0.07 above best single scorer; Fig 3

### 4.7 Qualitative
- PCA of per-source embeddings, low vs high SNR (Fig 4): structure
  emerges only at high SNR — visual confirmation of the core finding

## 5. Discussion

- Why does Energy invert at high SNR? (over-confident logits → spread
  collapse; Prototype's geometry only exists once embeddings de-noise)
- Connection to SNR-dependent phenomena in modulation classification
- Generality: routing needs only a scalar operating-condition estimate —
  same trick applies to other condition-dependent OOD settings
- Computational overhead: per-source head ~5K params; routing is free

## 6. Limitations & Future Work

- Routing boundary uses (estimated) SNR; sensitivity analysed, but a
  learned/soft router could do better
- Symbols assumed aligned; synthetic data only; 2-source scenarios
- Training-side fixes that directly optimise the known/unknown margin
  (e.g. VOS with synthetic outliers during training) remain unexplored
- Mixture-level (instead of per-source) OOD as an alternative formulation

## References (~30)

- SC-BSS literature (paper1's references + new ones)
- Open-set recognition (ODIN, Energy, VOS, OpenMax, etc.)
- Complex-valued neural networks
- Communication signal processing

---

## Figures / Tables

Figures (all generated by `make_figs.py` into `figures/`):
1. **Fig 1.** `fig_per_snr_auroc` — per-SNR AUROC: Energy vs Prototype vs
   SNR-routed (the money figure; shows complementarity + routing boundary)
2. **Fig 2.** `fig_overall_auroc` — weighted-avg AUROC bars: 3 scorers vs
   routed vs oracle bound
3. **Fig 3.** `fig_snr_noise_robustness` — routed AUROC vs SNR-est noise σ
4. **Fig 4.** `fig_embedding_pca` — embedding PCA, low vs high SNR
5. (to draw) Block diagram of OpenSetCSE + routing inference path

Tables:
1. Closed-set separation + classification (per-SNR SI-SDR, per-class acc)
2. Open-set OOD detection — pooled (the "trap" table, all ≈ 0.5)
3. Main result: per-SNR AUROC + weighted avg, single vs routed vs oracle
4. Ablations: center loss; embedding dim
5. SNR-noise robustness

---

## Submission timeline (tentative)

- Experiments: COMPLETE for ICASSP scope (2026-08-21)
- Internal draft: next step — LaTeX skeleton does not exist yet
- ICASSP 2027 submission: ~September 2026 (typical deadline)
- TSP submission: rolling (fallback / extended version)
