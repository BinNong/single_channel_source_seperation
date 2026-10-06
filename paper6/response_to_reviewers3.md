# Response to Review 3 — paper6 (main(7).pdf), 2026-10-04

We thank the reviewer for the careful, line-by-line reading. The review identifies
three issues that could combine into a rejection — the "Structure, Not Loss"
headline claim, the rigor of Proposition 2, and the baseline/story closure — plus a
series of wording, statistical-reporting, and positioning points. We address each
below. All section/line references are to the revised manuscript.

## 1. "Structure, Not Loss" overclaim — FIXED (claim narrowed AND control experiment added)

We did both of the reviewer's options:

- **(A) Claim narrowed.** The abstract now reads: "For such linear-residual
  separate-then-detect receivers, waveform-level fidelity alone cannot remove this
  interference-induced floor; the remedy we pursue is structural." The same
  scoping was applied to the three sibling phrasings in the Introduction,
  Discussion, and Conclusion. The title is retained as a (clearly labelled)
  architectural slogan whose precise, scoped version is now stated in the abstract.
- **(B) Control experiment run (new E-C).** We fine-tuned the baseline separator
  with an additional differentiable demodulation-aware loss (soft constellation
  cross-entropy on the PIT-assigned pairs; vendored `soft_demod.py`), same
  benchmark, same data stream, same architecture, five seeds,
  $\lambda_{\mathrm{ser}} \in \{0.1, 1.0\}$. On the strongest waveform route
  (separate → oracle sync → PSP/Viterbi, K=2, test seed 99999), the compensated
  SER moves by at most 0.12 points: pooled 0.5337 (baseline) → 0.5334 (λ=0.1) →
  0.5325 (λ=1.0), with Gray BER 0.2660 → 0.2656 / 0.2666 and SI-SDRi
  3.17 / 3.17 / 2.99 dB — all within seed noise. Loss-level re-weighting of the
  separator does not reach the floor; the "Fine-tuning was scoped but not run"
  paragraph is replaced by this measured outcome.

## 2. Proposition 2 rigor — FIXED (proof rewritten as moment inversion + numerical certificate)

The proposition is restated with the equivalence relation made explicit
($(a_1,a_2)\sim(\tilde a_1,\tilde a_2)$ iff $\tilde a_k = a_k e^{j2\pi q_k/M_k}$,
plus the $S_2$ swap for equal constellations; group
$G=\mathbb{Z}_{M_1}\times\mathbb{Z}_{M_2}$) and with explicit genericity
conditions (collision-free sum constellation; avoidance of a finite exceptional
algebraic set). The Appendix A proof is no longer a sketch:

- Step 1: the group acts, and no other rotation does (set-symmetry order argument).
- Step 2: amplitudes from $\mathbb{E}|s|^2$, $\mathbb{E}|s|^4$ (quadratic system,
  unordered-pair recovery; BPSK via $\mathbb{E}[s^2]$).
- Step 3: phases from circular moments $\mathbb{E}[s^m]$ — the divisibility
  structure $\mu_m^{(k)}=0$ unless $M_k|m$ isolates $a_\ell^{M_\ell}$ exactly for
  nested unequal orders, and reduces equal orders to one quadratic with at most
  two ordered solutions (exactly the swap for equal constellations; the spurious
  root is generically rejected by the Step-2 moduli).
- Step 4: the exceptional set (collisions = zero contacts of Theorem 1, quadratic
  degeneracies, modulus ties) is finite/algebraic, hence measure-zero; finite-$N$
  follows by consistency of empirical moments.

New numerical certificate `paper6_sync_jd/theory_identifiability.py` solves the
moment system (analytic moments, no Monte Carlo) on 200 random generic draws for
each of the ten benchmark constellation pairs and checks that every candidate
solution is group-equivalent to the truth: **10/10 pairs, 200/200 trials each**,
with the Step-2 modulus constraint rejecting the spurious quadratic root in
200/200 QPSK+16QAM trials (never do both roots pass). The two negative controls
behave as the proof predicts: at a collision rotation (QPSK+QPSK, equal powers,
$\varphi=90^\circ$) the sum constellation collapses (16→9 distinct points,
$d_{\min}=0$) and the coefficient recovery becomes $\sqrt{\varepsilon}$-ill-conditioned
(degenerate double root), while continuous Gaussian alphabets retain the full
gauge freedom of Proposition 1 (all rotated candidates match the moments).

## 3. Proposition 3 — theory–experiment loop closed (new Fig.)

New figure (Fig.~\ref{fig:twotone}) and text: a spectral-line two-tone estimator
on the $K{=}2$ carrier-only mixture is compared directly against the exact
two-tone CRB across $|\Delta f|$, SNR $\in \{0,10,20\}$ dB, and amplitude ratios
$\{0,6\}$ dB (200 MC per cell; `eval_twotone_crb.py`). Measured RMSE is within
$1.0$–$1.3\times$ of $\sqrt{\mathrm{CRB}}$ for
$|\Delta f|\gtrsim 1.6/T_{\mathrm{burst}}$; below $\approx 1/T_{\mathrm{burst}}$
the estimator leaves the CRB regime entirely (SNR-flat RMSE saturation,
$\pm 0.6$–$0.7$ Hz per-tone bias, 95–98% acquisition failure at $|\Delta f|=1$ Hz)
— the capture effect the inflation predicts.

On the bias/CRB point: the text now states explicitly that the CRB is used as a
benchmark for *unbiased* estimation, that
$\mathrm{MSE}=\mathrm{Var}+\mathrm{Bias}^2$, and that the observed high-SNR gaps
are squared bias (fading ISI / interferer capture), which no unbiasedness
assumption prices.

## 4. "without any oracle" — FIXED (wording) + genie-free results (new)

All occurrences ("without any oracle", "no oracle of any kind", "no oracle at
all") are replaced by "no oracle carriers, phases or symbols during inference";
the abstract now states that the residual rotational ambiguity in *scoring* is
genie-resolved and that genie-free differential decoding is reported separately.

New measurement (`eval_e4_ber_diff.py`, full run: 700 reference-grid bursts,
differential-equivalent scoring on the n=383 PSK-pair bursts, paired with the
genie-resolved scores): V4 is flat — 0.381 → 0.372 SER, 0.239 → 0.244 BER —
while the strongest oracle-assisted waveform route degrades from 0.440 to 0.537
SER (+9.8 pts) and 0.250 to 0.357 BER (+10.7 pts). Under deployment-faithful,
fully genie-free scoring the headline tie becomes a clear V4 win
(0.372 vs 0.537 on PSK pairs). The numbers are now in E4 and Limitations (i).

## 5. Baseline strength — PARTIALLY ADDRESSED

- "strong learned separator" wording removed everywhere; the separator is now
  described by its measured quality (SI-SDRi 3.2 dB at K=2) and the claims are
  scoped to "the separator quality we can currently reach" (already the paper's
  own boundary-not-a-law framing, with the true-stream control SER 0.041).
- The demodulation-aware fine-tuning control (item 1B) is exactly the
  receiver-aware-separator baseline the reviewer asked for, and it changes the
  outcome by ≤0.12 SER points.
- Full CNSE/CTDCRN/S4-UNET re-training through our receiver pipeline is deferred:
  CTDCRN has no public implementation, and the paper's claim is explicitly
  scoped to the reachable separator quality with the true-stream control pricing
  what a near-perfect separator would buy (SER 0.041). We state this scoping in
  the text rather than claiming a universal impossibility.

## 6. E4 statistics — FIXED

A "Statistical reporting" paragraph now defines the unit of inference (the mixture
burst), the two estimands (pooled level vs paired per-burst gap), and the tie
criterion (paired interval contains zero). No numbers changed.

## 7. V4 monotonicity — FIXED

The manuscript no longer claims exact ECM for V4: the ISI-cancellation decision
step is described as an approximate conditional minimiser, V4 as a monotone
coordinate-*refinement* (generalised-EM sense) whose guarantee is implemented
(accept/fallback against the V1 energy; never triggered). Full V4 pseudocode is
now Algorithm 1 in Appendix B (the `algorithm` float the reviewer asked for).

## 8. E3–E4 spine — kept as the main axis (unchanged, per the reviewer's advice).

## 9. E5 negative-result wording — FIXED

"The failure is once again structural, not parametric" → "The tested
evidence-based selectors are insufficient"; the K=3 paragraph already read "the
*tested* acquisition strategy ... is unreliable; ... no impossibility is claimed";
the Conclusion's "must again come from structure" is reworded accordingly.

## 10. Scope — the Conclusion already restricts all quantitative claims to the
short-burst, comparable-power, known-modulation, fixed-timing regime; the
complexity limitation now explicitly calls V4 an "offline proof-of-concept
receiver" (1.6 s/burst vs 0.256 s burst, exponential in K).

## 11. BER — ADDED

Table tab:e4 now carries the pooled Gray-mapped BER for every route (same
decisions as the SER), and the text reports the honest nuance the reviewer
anticipated: pooled BER favours the PSP route (0.266 vs 0.306 — the
16QAM-involving pairs carry 4 bits/symbol and no route recovers them), while
PSK-only BER favours V4 (0.236 vs 0.250), mirroring the SER structure.
Differential-decoding BER is reported alongside (item 4).

## 12. Complexity / reproducibility — V4 full pseudocode added (Algorithm 1,
Appendix B); offline-proof-of-concept positioning stated; all new scripts ship in
the public repo (`eval_twotone_crb.py`, `eval_e4_ber_diff.py`,
`theory_identifiability.py`, `train_finetune_demod.py`).

## 13. Related Work — a "Positioning" paragraph now closes the section,
distilling the five contributions exactly along the reviewer's suggested axes.

## 14. Reference [7] (EDSNet) — VERIFIED

The entry is correct: Zhang J., Wu J., Zhu H., Jin B., "EDSNet: Joint Detection
and Separation of Time-Frequency Aliasing Signals with Unknown Number", *Modern
Radar*, 2026 (online first), DOI 10.16592/j.cnki.1004-7859.2026111 — resolvable
via the journal site (xdld.xml-journal.net) with matching authors/title
(张靖/吴佳杰/朱红桥/晋本周,《现代雷达》网络首发). We will add volume/issue once the
formal publication data appears.

## Page budget

All additions were absorbed within the 13-page TSP target by compression
(captions, float sizes, redundant restatements), with the manuscript rebuilding
cleanly (xelatex, no unresolved references).
