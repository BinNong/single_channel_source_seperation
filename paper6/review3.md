# Review 3 — main(7).pdf (13 pp), IEEE TSP criteria (received 2026-10-04)

Recommendation: **Major Revision** (with near-Reject risk at TSP standards).

The review was read against the full manuscript: body, theorems/propositions, E1–E6, tables, figures, and both appendices, with emphasis on theory–experiment closure.

## Core observation praised

Waveform separation quality, independent carrier-sync quality, and final bit-detection performance are NOT the same optimization objective. E3 (high SI-SDR separated waveform ≠ good bit detection) + E4 (separate→sync→detect pushed to joint sync+detection) form a complete story. Negative results (K=3, blind modulation recognition, count head) are honestly reported.

**Main problem: some headline claims run further than the evidence; Proposition 2 is not at TSP-level rigor.**

## 1. "Structure, Not Loss" core claim is under-evidenced (highest priority)

- Abstract: "The remedy is structural, not a loss weight." But Methods states "Fine-tuning the separator was scoped but not run" — no demodulation-aware loss / joint loss / receiver-aware separator control experiment was actually run.
- What is proven: for the current "separator output = target + residual interference, then separate detection" architecture, a detection floor exists. What is NOT proven: that no loss-redesign of the separator can break the floor (a receiver-aware separator need not satisfy Thm 2's residual-interference model). CNSE (2022) already couples a separator with an LC-PSP demodulator and reports SER.
- Fix option A (recommended): narrow the claim to "For linear-residual separate-then-detect receivers, waveform-level fidelity alone cannot remove the interference-induced detection floor."
- Fix option B: add a controlled experiment (baseline SI-SDR separator; demodulation-aware loss; separator + differentiable detector; joint detector; V4) under identical budget/data.

## 2. Proposition 2 — biggest theoretical risk

- Claim: off constellation collision, effective complex coefficients are recoverable up to finite rotations 2π/M_k and permutation symmetry.
- Appendix A proof shows (1) which constellation rotations are symmetries and (2) collision breaks identifiability — but "I know the ambiguities" ≠ "no other ambiguities exist". The sketch ("any other rotation moves codewords off the alphabet...") does not exclude other coefficient transformations.
- Asked: formal equivalence relation (a1,a2)~(ã1,ã2) with G = Z_M1 × Z_M2 (plus S2 for equal alphabets); theorem: two effective-coefficient pairs inducing the same noiseless joint constellation belong to the same equivalence class, under explicit generic conditions. Cover BPSK/QPSK/8PSK, square QAM, equal/unequal alphabets, equal/unequal amplitudes, collision rotations, finite-N vs asymptotic identifiability.
- Fallback: if the proof cannot be strengthened, downgrade Proposition → Observation/Lemma.

## 3. Proposition 3 CRB analysis — theory–experiment loop not closed

- 3.1 Fig. 1 right validates a K=1 synchronizer, not Prop 3 (two-tone mixture). Asked: new figure with K=2 two-tone frequency RMSE/bias/variance vs |Δf| vs theoretical CRB, across SNR and amplitude ratio. Fig. 3(b) shows V4 joint-loop RMSE, which is not a BlindCarrierSync two-tone CRB validation.
- 3.2 BlindCarrierSync at K≥2 has 2–4 Hz SNR-flat bias; comparing a biased estimator against the unbiased CRB is questionable. State MSE = Var + Bias² explicitly; separate "CRB benchmark" from "algorithm efficiency".

## 4. "without any oracle" overstates

- E4 scoring is "PIT- and M-fold-rotation-resolved"; limitations admit genie resolution of M-fold ambiguity in scoring. Oracle-free estimation is fine; "without any oracle at all" (abstract/discussion/conclusion) is not. System also assumes known modulation, nominal carrier, known search range, fixed timing grid.
- Suggested wording: "oracle-free parameter estimation under a semi-blind operating assumption"; report full genie-free differential-decoding K=2 V4 results.

## 5. Baselines too weak for TSP

- SlotSepNet's 3.2 dB SI-SDRi at K=2 is hard to accept as a "strong learned separator". Prior art: Hou & Gao CNSE 2022 (separator + LC-PSP, SER), CTDCRN 2024 (BER), S4-UNET 2026 (sim + measured data, vs ConvTasNet/CTDCRN/TDE-ICA), hybrid knowledge-data 2026.
- Asked: (1) strong SC-BSS baselines (CNSE, CTDCRN, S4-UNET, SlotSepNet); (2) receiver-aware separation (hybrid knowledge-data); (3) direct joint/neural detectors. Otherwise: "the proposed receiver wins because the learned separator baseline is weak."

## 6. E4 statistics need unified reporting

- pooled SER 0.5310 CI [0.5261,0.5360] (five test grids) vs paired per-burst +0.003 CI [−0.011,+0.018] Wilcoxon p=4.8e-4 (700 shared cells): state unit of inference (= mixture burst), estimand, bootstrap/aggregation method; unify effect = SER_A − SER_B with mean/median difference, 95% CI, paired test.

## 7. V4 "ECM + monotonicity" wording

- With ISI, per-symbol hypothesis + local ISI cancellation is generally not an exact coordinate minimization over the discrete sequence. "exact coordinate minimiser" needs proof or rewording. Recommended: "generalized EM / monotone coordinate-refinement procedure" (accept-on-decrease ⇒ non-increasing energy).

## 8. E3→E4 is the paper's strength — keep as the main axis

- V1 raw-mixture ECM → V4 ISI-aware ECM → V3 oracle-frequency ladder (0.5392 → 0.5310 → 0.4106) cleanly separates memoryless-model error, ISI mismatch, carrier acquisition. Do not weaken; make it the spine.

## 9. K=3 / blind modulation negative results — keep, reword

- "the failure is once again structural, not parametric" overstates: only "the tested model-selection strategies fail" / "the tested K=3 acquisition strategy fails" is proven. Reword accordingly.

## 10. Scope still narrow — position precisely

- synthetic benchmark; 256 symbols; fixed timing grid; known modulation; comparable power; K=1/2 main; 3-tap fading; residual CFO [−5,10] Hz. Fig. 3 shows timing collapse at 0.25 Ts, weak source unsolvable at ±10 dB imbalance, K≥3 unsolved. Position as "a structural receiver analysis for short-burst, comparable-power, semi-blind co-frequency mixtures".

## 11. BER should be more prominent

- "the receiver's deliverable is bits", but headline metric is SER (V4 pooled 0.5310). Add BER, Gray-coded BER, differential-decoding BER, modulation-pair-specific BER to complete waveform→symbol→bit.

## 12. Complexity — position as offline proof-of-concept

- ≈1.6 s/burst vs 0.256 s burst duration; hypothesis set exponential in K. Do not imply deployability. Appendix B should contain FULL V4 pseudocode (currently prose for V1 + one incremental paragraph for V4).

## 13. Related Work — reposition novelty

- Novelty is NOT "we propose joint detection" (classical multiuser detection; [8] single-channel co-frequency joint ML; CNSE 2022 separator+PSP; CTDCRN 2024 BER). Distill: (1) exact separate-detection floor under a discrete interferer; (2) generic effective-coefficient identifiability; (3) two-tone carrier-only Fisher coupling + full-waveform contrast; (4) empirical architecture diagnosis (separate→per-slot sync→detect fails structurally); (5) training-free joint sync/detection operationalizing the analysis.

## 14. Reference [7] EDSNet — verify title/journal/DOI (not a technical rejection point)

## Scores

| Dimension | Score |
|---|---|
| Technical significance | 7.5/10 |
| Novelty | 7/10 |
| Theoretical rigor | 6/10 |
| Algorithmic novelty | 6/10 |
| Experimental quality | 7.5/10 |
| Baseline completeness | 5.5/10 |
| Reproducibility | 8.5/10 |
| Writing / organization | 8/10 |
| TSP suitability | 7/10 |
| Overall | 6.8/10 |

**Final: Major Revision.** With a theory-focused reviewer, Prop 2 + lack of strong receiver-aware baselines + "not a loss" overclaim could combine into Reject.

## Suggested revision priorities

1. Tighten "Structure, Not Loss" or add demodulation-aware loss baseline.
2. Make Prop 2 a real identifiability theorem.
3. Add K=2 two-tone frequency-estimator CRB/MSE/bias experiment.
4. Add strong SC-BSS / receiver-aware baselines.
5. Replace "no oracle" with accurate wording; add genie-free differential-decoding V4 results.
6. Unify E4 statistical tests and CI definitions.

Closing remark: the paper's "story" is mature; what is missing is lifting several nice conclusions from "convincing engineering observations" to "proofs and experiments that survive line-by-line scrutiny by a TSP theory reviewer."
