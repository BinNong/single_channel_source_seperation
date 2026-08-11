# Reviewer Response Letter — R1 → R2

**Manuscript**: A Lightweight Complex-Valued CNN with Complex Squeeze-and-Excitation Attention for Single-Channel Blind Source Separation of Co-Frequency Communication Signals

**Authors**: Bin Nong, Weihong Fu, Zhuoyun Jiang

**Date**: 2026-07-26

---

We thank the reviewer for the thorough and constructive critique. The detailed review identified genuine weaknesses in the manuscript that we have addressed in this revision. We respond to each comment below in order. The original review text is in *italics*; our response is in roman.

---

## Section 1 — 必改项 ("硬伤")

### R1.1 — Reference citation unresolved ([?], Fig.??)

*Original*: ~14 occurrences of `[?]`, ~9 of `Fig. ??`, `Table ??`, `Eq. (??)`, including `[?,?,?]` in the abstract.

**Action taken**: The manuscript now builds cleanly with two `xelatex` passes; we verified that the PDF contains **0 occurrences** of `[?]` or `??`. All citation keys match a `\bibitem` entry in `thebibliography`.

### R1.2 — Author information placeholder

*Original*: "Bin Nong · Independent Researcher · E-mail: nongbin@example.com"; "the author thanks the anonymous reviewers and the editor".

**Action taken**: Author block now lists all three authors with their institutions:
- Bin Nong — Tianfu College of Southwestern University of Finance and Economics, Mianyang, China
- Weihong Fu — School of Telecommunications Engineering, Xidian University, Xi'an, Shaanxi, China
- Zhuoyun Jiang — School of Health Science and Engineering, University of Shanghai for Science and Technology, Shanghai, China
Corresponding email replaced with `nongbin@tfswufe.edu.cn` (real address provided by the authors). The Acknowledgements section has been removed; the paper no longer thanks reviewers before submission. Statements and Declarations section has been added with Funding, Competing Interests, Author Contributions, Data Availability, and Code Availability.

### R1.3 — Reference entries that could not be verified

*Original*: Entries #22, #23, #24, #25, #27 marked as suspicious (Hou & Gao "CNSE" paper, "Domain-specific modulation classification via BSS", "SepFormer-style attention", "Complex-valued U-Net for co-channel", "Complex-domain NNs for coherent optical").

**Action taken**: All five entries removed and replaced with two verifiable cross-domain baselines that we re-implemented from primary sources:
- **Trabelsi et al. 2018** (Deep Complex Networks, ICLR) — replaces refs #22 and #24
- **Zhao et al. 2021** (Deep-Waveform, IEEE JSAC, DOI: 10.1109/JSAC.2021.3087241) — replaces ref #25

These are now cited in Section 2 (complex-domain building blocks) and Section 3 (related work). Refs #23 and #27 (unverifiable) have been deleted.

### R1.4 — "First" / "SOTA" overclaims

*Original*: "this is the first SE-style attention explicitly designed and validated for the complex domain in wireless BSS"; "Conv-TasNet (SOTA)".

**Action taken**: "First" is now softened throughout to "to the best of our knowledge, the first phase-preserving, real-gated complex-valued SE design specifically evaluated for lightweight single-channel blind separation of near-co-frequency communication signals" (Abstract and Introduction). "SOTA" replaced with "Baseline" in Table 6. The Discussion section explicitly acknowledges that prior work exists (CSENet 2022, complex-valued SE 2024/2025) and that the C-SE's specific contribution is the **phase-preserving real-gated** design. "Optimal" and "is in fact optimal" replaced with "is sufficient" in the scale-mode ablation discussion.

### R1.5 — Equation (1) does not match the data generator

*Original*: Eq. (1) has only `α_1 s_1(t) + α_2 s_2(t) + n(t)`, ignoring multipath, timing, carrier offset.

**Action taken**: Eq. (1) now reads `y(t) = α_1 (h_1 ∗ s_1)(t) + α_2 (h_2 ∗ s_2)(t) + n(t)`, with explicit definitions for `h_k`, `α_k`, the symbol structure, and the unit-power normalisation. A new paragraph after the equation acknowledges that the generator uses **symbol-aligned** mixtures with per-source carrier offset within ±5 Hz — explicitly stating the scope of the current model (relaxing symbol alignment is left to future work).

### R1.6 — "Co-frequency" too narrow

*Original*: only tested at 5 Hz offset; reviewer asked about 0 Hz performance.

**Action taken**: We trained two new models (proposed C-SE and matched no-SE) with the carrier-frequency gap drawn uniformly from `U(0, 5)` Hz per sample. Evaluating at gaps from 0 Hz to 500 Hz (Section 4.7, Fig. micro_freq) shows the proposed C-SE holds `SDR ≈ 3.3 dB` across the entire range, including the **previously unseen co-frequency limit Δf = 0 Hz** (SDR = 3.33 dB, SIR = 6.47 dB). The no-SE baseline trained under the same schedule still collapses to SIR ≈ 21 dB at every gap, confirming the gap is structural.

### R1.7 — Baselines too soft

*Original*: only 3 baselines (no-SE ablation, real-valued baseline, Conv-TasNet).

**Action taken**: We added two parameter-matched baselines and two cross-domain communication-signal baselines:
- **Param-matched no-SE** (H=70, 242K params): same architecture as proposed minus the SE blocks. Tests whether the +1.19 dB gain is from C-SE or from extra parameters.
- **Param-matched real-valued** (H=80, L=12, 237K params): tests whether the choice between real and complex domain dominates the SE addition.
- **CNSE (Hou & Gao 2022, scaled-down to 6.69 M params)**: re-implemented from the source PDF (provided by the authors of this response).
- **S4-UNET (Gao et al. 2026, scaled-down to 1.57 M params)**: re-implemented from the source PDF.

Both CNSE and S4-UNET were scaled down to fit the 8 GB GPU (the originals used RTX 5090D 32 GB); full hyperparameters in Section 3 and `EXPERIMENT_LOG.md`.

### R1.8 — Statistical significance insufficient

*Original*: only 3 seeds, no std, no p-values.

**Action taken**: We extended every evaluable configuration to **5 seeds** (42-46). Table 1 is now reported with sample std over seeds. A new section (§ Statistical Significance and Model Variance) reports paired **Wilcoxon signed-rank tests** between C-SE and every baseline on per-seed SDR. The Wilcoxon p-values are also embedded in Table 1 (rightmost column). Honest discussion: **no comparison against C-SE reaches statistical significance at α = 0.05** (smallest p = 0.062 against the matched real-valued baseline). We argue that this is itself an important finding and reposition the paper's narrative around per-parameter efficiency and per-seed reproducibility rather than raw SDR dominance.

### R1.9 — Table 1 vs Table 4 SDR inconsistency

*Original*: Table 1 says SDR = 2.75 dB, Table 4 says 3.33 dB (both T=4096, SNR=10 dB).

**Action taken**: We confirmed that the two numbers come from different evaluation splits (3 seeds × 500 samples vs 1 seed × 50 samples) and added a long footnote to Table 4 explaining the protocol difference. Table 1 is now the authoritative 5-seed result (SDR = 2.31 ± 0.62 dB).

### R1.10 — Table 6 memory contradiction

*Original*: text says "much smaller than Conv-TasNet"; Table 6 shows the opposite.

**Action taken**: Text rewritten to acknowledge the opposite direction: Conv-TasNet occupies **less** peak memory (60 MB) than the proposed (170 MB) in our setting. The efficiency paragraph now correctly states that the proposed model improves **single-sample latency** and **parameter count** while activation memory and batched throughput are not universally superior to Conv-TasNet.

### R1.11 — Table 3 missing s42/s43 entries

*Original*: s42/s43 = `--` for the proposed real-mode row.

**Action taken**: Filled in from the actual checkpoints: s42 = 2.79, s43 = 2.74. Caption updated to clarify that the proposed row uses three seeds while the ablation variants use two (compute budget).

### R1.12 — Acknowledgements thanks reviewers before submission

**Action taken**: Removed. The Acknowledgements section is now empty; Statements and Declarations section added per Springer WPC requirements.

---

## Section 2 — 决定能否过外审

### R2.1 — Eq. (1) must include multipath

**Action taken**: Multipath channel `h_k` included in Eq. (1) with explicit 3-tap complex Gaussian definition, normalisation, and unit-energy constraint.

### R2.2 — Symbol timing offset

**Action taken**: Acknowledged as a limitation in the scope paragraph after Eq. (1): "the generator uses symbol-aligned mixtures... relaxing symbol alignment is left to future work". No experimental change in this revision (would require retraining every model from scratch).

### R2.3 — Micro-frequency 0-5 Hz experiment

**Action taken**: See R1.6 above. Dedicated Section 4.7 with Fig. micro_freq.

### R2.4 — Output collapse discussion

**Action taken**: Section 4.4 (Ablation: C-SE) now includes a numerical-stability note explaining why a high SIR combined with low SDR is the signature of the collapse regime. Five-seed results show this collapse regime affects different fractions of seeds for different models (4/5 for no-SE, 2/5 for C-SE and Real, 1/3 for Conv-TasNet, 2/3 for CNSE, 0/3 for S4-UNET).

### R2.5 — SER pipeline

**Action taken**: A new paragraph in Section 4.4 documents the SER pipeline: down-conversion, RRC matched filter with fixed symbol-aligned offset, normalisation, **single global complex rotation** via inner product with reference (absorbs scale and constant-phase ambiguities), minimum-distance decision. The paragraph also states the limitation that this global rotation cannot compensate per-sample residual carrier offset, explaining the high 16QAM SER.

### R2.6 — Real-valued at matched params

**Action taken**: Real-Valued CNN widened to H=80, L=12 (237K params, +0.6% over proposed). Result: SDR = 2.20 ± 0.59 dB (5 seeds), comparable to C-SE (2.31 ± 0.62 dB). The Discussion now explicitly states that at the ~235K parameter scale the choice between real and complex domain is roughly comparable.

### R2.7 — Pooling ablation

**Action taken**: New Table 4 (Pooling ablation, 2 seeds per variant). All three single-statistic variants (mean/power/magnitude) reach SDR ≈ 2.75 dB with stable SIR; the mean+power variant consistently collapses with the same SIR signature as the no-SE baseline. Section 4.5 concludes that the choice between these is not performance-critical.

### R2.8 — Table 1 vs Table 4 protocol

**Action taken**: See R1.9 above.

### R2.9 — Complex BN / Complex ReLU

**Action taken**: Renamed in text as "split complex-BN" and "split complex-ReLU"; explicitly cited Trabelsi et al. 2018 as the proper complex whitening; noted that modReLU / zReLU / CReLU are left for future work. Section 3 (Complex-Valued Building Blocks) now contains the full caveat.

### R2.10 — "Phase is unimportant" too strong

**Action taken**: All occurrences softened to "within the tested scaling family the real-valued weight design is sufficient; this does not establish that phase is irrelevant". Discussion section 4.6 adds the polar-parameterisation caveat.

---

## Section 3 — 提升档次

### R3.1 — Real-radio / public I/Q datasets

**Action taken**: We provide a `data_radioml.py` loader for RadioML 2016.10A in the released code. The reviewer did not request a headline external-dataset result; we left this as future work to avoid reporting results we have not run.

### R3.2 — Open-source release

**Action taken**: `README.md`, `requirements.txt`, and `LICENSE` (MIT) added at the repository root. `docs/EXPERIMENT_LOG.md` provides a complete reproducibility record with exact commands and per-seed numbers.

### R3.3 — Figure 6 too sparse

**Action taken**: Fig. 6 (params vs. SDR) regenerated with all 8 evaluated models and 5-seed error bars; an annotation arrow highlights the proposed model. The figure now carries the cross-domain baseline information (CNSE / S4-UNET) the reviewer asked for.

---

## Summary of changes to the manuscript

| Section | Change |
|---------|--------|
| Abstract | Updated with 5-seed numbers and honest significance discussion |
| Section 1 (Introduction) | "First" claim softened; cross-domain baselines added to related work |
| Section 2 (System Model) | Eq. (1) now includes multipath channel; scope limitations stated |
| Section 3 (Method) | Split BN/ReLU explicitly named; CNSE / S4-UNET architectures described |
| Section 4.4 (Ablation: C-SE) | Numerical-stability note added; SER pipeline documented |
| Section 4.5 (Ablation: Scale Mode) | Softened to "is sufficient" |
| Section 4.6 (Ablation: Pooling) | **New** Table 4 and discussion |
| Section 4.7 (Micro-frequency) | **New** with Fig. micro_freq |
| Section 4.8 (Statistical Significance) | **New** Wilcoxon test subsection |
| Section 5 (Efficiency) | Memory-contradiction paragraph rewritten |
| Section 6 (Discussion) | 5 paragraphs covering C-SE value, param matching, cross-domain, pooling, micro-freq |
| Section 7 (Conclusion) | Updated with 5-seed numbers |
| Section 8 (Statements) | **New** per Springer WPC requirements |
| Acknowledgements | Removed |
| Bibliography | 5 unverifiable entries replaced with 2 verified entries |
| Table 1 | 5-seed mean ± std, with Wilcoxon p-value column |
| Table 3 | s42/s43 filled in |
| Table 4 (new) | Pooling ablation |
| Table 5 (length gen) | Footnote explaining Table 1 vs Table 4 protocol difference |
| Fig. 6 | Regenerated with 8 models and 5-seed error bars |
| Fig. micro_freq (new) | C-SE trained on U(0,5) Hz generalises to 0 Hz |

---

## Per-seed numbers

All per-seed numbers are available in `results/phase5_results/_aggregated/table1_5seed.json` and `docs/EXPERIMENT_LOG.md`.

The manuscript is now 12 pages, builds without errors, and contains 0 unresolved cross-references. We believe these changes address all the reviewer's concerns and respectfully request a re-evaluation.