# Response to Reviewer 2

We thank the reviewer for an unusually constructive review. The
recommendation — "scientific repositioning" rather than more
experiments — is exactly what this revision does. The manuscript has
been restructured around the review's own framing: **from "we propose
a joint receiver algorithm" to "we reveal and quantify the structural
mismatch between waveform-fidelity optimisation, independent
synchronisation, and per-source detection in SC-BSS, and derive the
receiver architecture that closes it."** Concretely:

1. **Related work corrected and novelty repositioned** (Major
   Concern 1). The claim "all of these works stop at the waveform"
   was wrong and has been removed. Hou & Gao 2022, Chen et al. 2020,
   Yu et al. 2024, Luo et al. 2026 (C2ESDNet) and Gao et al. 2026
   (IQUMamba-1D) are now cited and characterised accurately, and the
   gap statement is rebuilt on what is actually missing: a quantified
   account of *when and why* separate-then-detect breaks, with exact
   floors/bounds, on a controlled variable-count benchmark.
2. **Proposition 2 restated** (Major Concern 2): what is identifiable
   is the *effective complex coefficient* (channel ∘ carrier phase ∘
   gain), up to the constellations' finite rotational symmetry and
   source permutation — never the carrier phase in isolation; a
   channel-phase reference convention is now stated explicitly.
3. **The decoupling claim regime-qualified** (Major Concern 3): the
   inseparability of synchronisation and joint detection is now stated
   for the co-frequency regime ($|\Delta f|\,T_{\mathrm{burst}}
   \lesssim O(1)$, comparable powers, short bursts), with the
   large-separation decoupling acknowledged as Proposition 3 itself
   predicts.
4. **Proposition 3 bridged to the communication waveform** (Major
   Concern 4) by the review's own route B: a Fisher analysis of the
   full benchmark waveform (known symbols, known pulse shaping, unknown
   CFOs and unknown 3-tap channels). The result is stronger than a
   defence: the waveform FIM shows *no* two-source frequency coupling
   (median inflation 1.01× at every $|\Delta f|\in[0,16]$ Hz), so the
   disambiguating information lives in the modulation structure —
   unreachable for non-data-aided spectral-line synchronisation, which
   is exactly what the pure-tone surrogate bounds, and partially
   reachable by our ECM through decision feedback. The surrogate is now
   labelled a "carrier-only diagnostic" wherever it appears (including
   the Figure 3 caption).
5. **Theorem 1 grid statement corrected** (§VI of the review): the
   pointwise bound is now said to upper-bound the phase-*integrated*
   ensemble; finite grids are called approximations with
   construction-dependent (downward, near zero contacts) bias.
6. **"Blind" renamed throughout** (Major Concern 5): the title now
   reads "*Structure, Not Loss: Sync-Aware Joint Detection for
   Semi-Blind Single-Channel Co-Frequency Reception*", and the genie
   M-fold resolution is *measured*, not assumed: a new
   differential-decoding experiment (E1-D, 200 bursts per
   modulation–SNR cell, two grids) prices the deployment answer —
   differential decoding removes the genie entirely, is *more* robust
   than genie-resolved absolute decoding at low SNR (per-symbol
   rotation ≈1.8°/Hz; 8PSK at −5 dB: 0.458 vs. 0.747), and costs
   +0.03–+0.08 SER points at high SNR — a penalty the oracle receiver
   pays equally (8PSK at 20 dB: 0.018→0.120), i.e. the intrinsic
   differential penalty under the fading channel, not a synchronisation
   deficiency. 16QAM is excluded (no standard differential scheme).
7. **Paired statistics** (Major Concern 7): all headline contrasts are
   now tested paired, per burst, on shared cells.
8. **New experiments** (Major Concerns 6, 8, 9, 14): a PSP/Viterbi
   sequence-detection upgrade of the waveform route, a cross-family
   separator reference, separator-quality metrics, enlarged modulation
   probe, and three robustness sweeps (frequency separation, timing
   offset, amplitude ratio).
9. **V3 renamed** from "oracle-carrier bound" to "oracle-frequency
   reference" everywhere (Major Concern 11), the counting/localisation
   contradiction is resolved by splitting the two claims (Major
   Concern 9/§11), the complexity discussion now reports the
   real-time factor, memory and CPU/GPU split honestly (Major Concern
   12), and all conclusions are qualified to the benchmark regime
   (Major Concern 10).

Point-by-point responses follow; **all** changes are in the revised
manuscript, and every new number is reproducible from the public code
(new scripts: `theory_waveform_fim.py`, `analyze_paired_stats.py`,
`eval_robustness_sweeps.py`, `eval_psp_baseline.py`,
`eval_diff_coding.py`, `eval_separator_quality.py`).

---

## Major Concern 1 — Related work / novelty positioning

**We accept this fully; the reviewer is right.** The sentence "All of
these works stop at the waveform" was factually wrong, and the framing
"past work focuses on waveform, this paper first introduces receiver
structure" does not survive the works the reviewer lists — several of
which were already in our reference list and were nonetheless
mischaracterised.

Changes in Section II-A:

- Removed the "stop at the waveform" claim. The related-work paragraph
  now states explicitly that Hou & Gao couple their convolutional
  separator to an LC-PSP demodulator and report SER; that Chen et al.
  recover information bits directly with a BRNN; that hybrid
  knowledge–data designs train separators with demodulation-aware
  losses (Luo et al. 2026); and that long-sequence state-space
  separators (IQUMamba-1D) target the same front-end.
- Added the missing references: Yu et al., *J. Commun.* 2024 (optimal
  receiver and blind-demodulation performance bounds for single-channel
  co-frequency mixtures); Luo et al., *IEEE TCCN* 2026; Gao et al.,
  *JKSUCIS* 2026. Yu et al. is additionally discussed in Section II-C,
  where we position our V1/V3 arms as the same joint-ML principle with
  the offsets and coupling matrix estimated blind inside the loop — the
  end-to-end, no-oracle evaluation they do not pursue.
- The novelty claim is rebuilt on the review's own formulation: the
  contribution is *the systematic, exactly quantified mismatch between
  waveform separation, carrier synchronisation and joint symbol
  detection* — when separate-then-detect floors (Theorem 2, exact,
  SNR-independent), why independent sync fails (Propositions 2–3), and
  how a joint synchroniser–detector on the raw mixture closes the gap —
  not the first receiver-aware SC-BSS system. The Discussion now
  opens with this three-layer statement (Section VII, first paragraph).

On the requested baseline families (PSP/Viterbi, particle filtering,
CNSE+LC-PSP, bit-oriented DL, the 2024 optimal receiver, complex-domain
DL, C2ESDNet, IQUMamba): we added what can be made a *fair* comparison
on our benchmark and say plainly why the rest is not claimed (see
Concern 6 below). In particular, Yu et al.'s optimal receiver and our
V1/V3 are the same detection principle; re-implementing their receiver
would duplicate V3 with a different synchronisation assumption, not
test our claims.

## Major Concern 2 — Proposition 2 identifiability

**Accepted.** The proposition now states that the identifiable objects
are the *effective per-source complex coefficients* — the product of
channel response, carrier phase and mixing gain — up to per-source
rotations by $2\pi/M_k$ and source permutation, and that carrier-phase
recovery is meaningful only relative to a fixed channel-phase reference
convention (Section III-C, Proposition 2 and the following paragraph).
The appendix proof sketch was updated to match. The abstract,
introduction and conclusion no longer say "carrier phases are
recoverable"; they say the effective coefficients are generically
identifiable through the joint discrete structure.

## Major Concern 3 — "cannot be decoupled at K ≥ 2" is too strong

**Accepted.** The statement is now regime-qualified everywhere it
appears (abstract, introduction, Section III-C, conclusion):
independent synchronisation and joint detection are inseparable *in the
co-frequency regime* — small offset separation
($|\Delta f|\,T_{\mathrm{burst}} \lesssim O(1)$), comparable powers,
short bursts. Proposition 3's large-$|\Delta f|$ decay to unity is
cited explicitly as the reason $K \geq 2$ alone is not sufficient.
The new frequency-separation sweep (Concern 14) measures the
transition directly: the end-to-end SER is nearly flat in
|Δf| (the joint likelihood search substitutes for spectral-line
acquisition), while the frequency RMSE stays elevated at co-frequency
and the V4−V3 acquisition gap is largest at Δf=0 (0.047 at 20 dB),
vanishing by 8 Hz — the coupling is a property of the regime, not of
the source count.

## Major Concern 4 — pure-tone CRB vs. the communication waveform

**Accepted, and addressed by the review's route B.** We computed the
Fisher matrix of the actual benchmark waveform under the minimal
data-aided model the review specifies — known symbols, known
modulation, known pulse shaping; unknown carrier offsets and unknown
3-tap complex channels (14 real parameters; the generator's passband
fading convention is respected). New Remark 3 in Section III-C and the
new red curve in Figure 3 (left): the two-source frequency CRB shows
**no coupling** — median inflation 1.01× at every
$|\Delta f| \in [0,16]$ Hz, 90th percentile ≤ 1.02×,
modulation-agnostic (BPSK/QPSK/8PSK/16QAM pairs all within
1.007–1.009) — against the pure-tone divergence (≈6.4×10³ as
$\Delta f \to 0$, 1.59× at $1/T_{\mathrm{burst}}$).

This turns the review's "exact for the wrong statistical model" into
the paper's sharpest result: the pure-tone FIM is exactly the
information available to *non-data-aided* (spectral-line)
synchronisation, and its divergence is *why* per-slot blind sync fails
at $K{=}2$; the waveform FIM shows the disambiguating information lives
in the modulation structure, which a non-data-aided receiver cannot
reach — and which our ECM partially reaches through its hard decisions
(semi-pilots). The V4→V3 residual is thereby attributed to decision
errors, not to frequency coupling. The surrogate is now labelled
"carrier-only" in Proposition 3's title, in the text, and in Figure 3's
caption, and is called a diagnostic of the non-data-aided route, not a
bound on the communication receiver.

## §VI — Theorem 1 "any phase grid" statement

**Accepted; corrected.** Theorem 1 now says: the bound holds pointwise,
hence upper-bounds the phase-*integrated* ensemble; a finite grid only
approximates that integral, with construction-dependent bias —
downward near the zero contacts, consistent with Theorem 2's own
finite-grid caveat (which the paper already carried).

## Major Concern 5 — "blind" is semi-blind

**Accepted.** The title is now "*Structure, Not Loss: Sync-Aware Joint
Detection for Semi-Blind Single-Channel Co-Frequency Reception*"; the
terminology note in Section V already defined the estimate-path sense
of "blind", and the abstract now says "semi-blind" in its first
paragraph. On the genie M-fold resolution: we no longer assert
"differential coding in deployment" — we *measure* it (new E1-D
experiment, `eval_diff_coding.py`; 200 bursts per modulation–SNR
cell, two grids): differential decoding removes the genie entirely,
is *more* robust than genie-resolved absolute decoding at low SNR
(the per-symbol rotation 2πΔfT_s ≈ 1.8°/Hz makes it insensitive to
acquisition error — 8PSK at −5 dB: 0.458 vs. 0.747), and costs
+0.03–+0.08 SER points at high SNR (20 dB: BPSK 0.007→0.032, QPSK
0.014→0.070, 8PSK 0.045→0.120) — a penalty the oracle receiver pays
equally (8PSK: 0.018→0.120), i.e. the intrinsic differential penalty
under the fading channel, not a synchronisation deficiency
(Limitation (i)). 16QAM is excluded from differential decoding (no
standard differential scheme) and this is stated.

## Major Concern 6 — no SOTA comparison

**Partially accepted, with a scope statement.** The reviewer is right
that E4's contrast is architectural: *on this benchmark*, joint ECM on
the raw mixture vs. our own separator's waveform route. The revised
text says exactly this and no longer implies "state of the art"
(Section VI-D, Results paragraph: "We stress what this comparison is
not..."). To strengthen the baseline side we added:

- **PSP/Viterbi waveform route** (new): the separated slots under the
  oracle receiver are now also decoded by a per-slot Viterbi sequence
  detector with decision-directed channel estimation (L=3 taps; the
  PSP family, à la Liu et al. 2015), not only by symbol-wise
  demodulation. **This baseline changed our headline, and we report it
  as measured**: the PSP upgrade improves the oracle-received waveform
  route from 0.5527 to 0.5337 pooled SER — within 0.3 points of V4
  (paired per-burst difference +0.003, 95% CI [−0.011, +0.018]). The
  revised claim is correspondingly sharper: the blind, training-free
  joint loop *ties the strongest oracle-assisted waveform route we can
  build* (winning at SNR ≥ 5 dB by +0.03–+0.10, losing at SNR ≤ 0 dB,
  where the separate routes' advantage is their oracle synchroniser,
  not the architecture), while the architecture contrast isolated by
  the oracle-frequency arm is 12 points (V3 0.4106 vs. PSP route
  0.5337). In other words: sequence detection recovers part of what
  joint detection buys, but only with an oracle; the joint loop is the
  only arm that is both blind and near-oracle-level.
- **Cross-family separator reference** (new): paper-1's C-SE CNN — a
  convolutional separator of a different family than our slot
  separator — evaluated zero-shot on the same K=2 cells: SI-SDRi
  3.49 dB (SlotSepNet: 3.24 dB on the same cells), so the E3/E4
  premise does not hinge on one architecture.
- **Yu et al. 2024** is cited and positioned (Section II-C); our V1/V3
  arms *are* the joint-ML optimal-receiver principle with blind
  estimation of the quantities they assume.

What we did not do: re-implement C2ESDNet or IQUMamba-1D. Both are
separator architectures whose training recipes and hyperparameters are
tuned to their own benchmarks; a faithful re-implementation and
re-training is a separate study, and a straw-man re-implementation
would be worse than no comparison. Our claim is deliberately narrower
and, we believe, now unambiguous: *on a controlled variable-count
benchmark, receiver architecture — not separator fidelity — decides
the bits at K ≥ 2*.

## Major Concern 7 — the 0.8-point V4 gain lacks statistical strength

**Accepted; resolved — the paired test shows the gain is highly
significant.** New `analyze_paired_stats.py`: per-burst paired
bootstrap (10⁴ resamples) and Wilcoxon signed-rank on shared cells.

- V1 − V4: pooled +0.0082 over five independent test grids,
  paired 95% CI [+0.0074, +0.0089], Wilcoxon p ≈ 1×10⁻⁹⁰
  (n = 3500); every grid individually p ≤ 7×10⁻¹⁷.
- The paired per-SNR decomposition is the ISI signature: ≈0 at
  −10 dB, +0.003 at 0 dB, +0.013 at 10 dB, +0.016 at 15–20 dB —
  the gain exists exactly where the memoryless model's ISI floor
  lives, which is why independent CIs (which include grid variance)
  understated it.
- V4 − waveform route (headline contrast, 700 shared cells):
  vs. the symbol-wise route +0.0222 pooled, but SNR-structured
  (−0.108 at −10 dB to +0.099 at 20 dB — the pair-rule reversal at
  low SNR, oracle sync included); vs. the PSP-upgraded route
  +0.0032, 95% CI [−0.0112, +0.0178], Wilcoxon p = 4.8×10⁻⁴ — a
  statistical tie, which is exactly what the revised text claims
  ("ties the strongest oracle-assisted waveform route", winning at
  SNR ≥ 5 dB).

These numbers are now in Section VI-D and the abstract.

## Major Concern 8 — E5 sample size

**Accepted; enlarged 4×.** The modulation-pair probe now runs 100
cells per SNR (300 cells total, up from 75): pair accuracy
0.11/0.53/0.40 at 0/10/20 dB (binomial 95% CIs ±0.06/0.10/0.10;
chance 0.10), end-to-end SER 0.987/0.610/0.762 (pooled 0.786 vs. 0.483
for known-modulation V4 on the same cells), conditional on a correct
pair 0.880/0.264/0.404. The per-pair breakdown is the diagnostic: at
10 dB the PSK-only pairs are identified reliably (0.86–0.88) while
16QAM-involving pairs sit at 0.11–0.78; at 0 dB only the densest
pairs are ever recovered; and at 20 dB the selector drifts toward
16QAM-involving hypotheses (the evidence measure's density bias). We
also retitled the claim: E5 is a *feasibility probe*, and
the text says the negative is about the tested selectors, not about
modulation classification in general.

## §XI — counting vs. localisation contradiction

**Accepted; the two claims are now split everywhere** (E3's
end-to-end accounting, Discussion, conclusion, and the RS-SlotSepNet
contribution item): the separator is measured to be useful for *slot
generation and localisation* (matched streams, rare misses), while its
*learned count head is unreliable at K=2* (4.2% accuracy; 93%
false-slot rate) — and any counting use must threshold the occupancy
head, not trust the count readout. The sentence "separation serves
source counting and localisation" no longer appears.

## Major Concern 9 — separator quality baseline

**Accepted; measured.** New `eval_separator_quality.py`: on the test
grid, SlotSepNet reaches SI-SDR improvement 9.6/3.2/2.8 dB at
K=1/2/3 with slot occupancy recall 0.98/0.98/1.00 (precision
0.89/0.68/1.00 — the K=2 precision is the surplus-slot behaviour), and
the cross-family zero-shot reference (paper-1 C-SE CNN, three
checkpoints, no retraining) reaches SI-SDRi 3.49 dB at K=2 on the same
cells. These numbers now anchor the "strong learned slot separator"
claim in Sections IV and V: at this measured, cross-validated quality,
the E3 oracle arm still sits at the separate-detection floor — which
is precisely the reviewer's point that the waveform metric does not
correlate with SER, now documented with the metric itself.

## Major Concern 10 — benchmark qualification

**Accepted.** The conclusion and abstract now carry the explicit
qualifier "under the considered short-burst, comparable-power,
known-modulation, fixed-timing co-frequency regime", and the new
robustness sweeps (Concern 14) probe exactly the boundaries the
reviewer names.

## Major Concern 11 — V3 is not a "bound"

**Accepted; renamed.** V3 is now the "**oracle-frequency reference**"
throughout (Table III, Section VI-D, Discussion, conclusion, abstract),
with the explicit note that it is not a fundamental bound: gains and
symbols are still estimated under the finite-tap model. The ladder is
accordingly described as decomposing the distance to the
oracle-frequency *reference*, not to a bound.

## Major Concern 12 — 1.6 s/burst practicality

**Accepted; now reported completely.** Limitation (vi) now states: the
per-burst cost is ≈1.6 s (V1+V4, one CPU core) against a 0.256 s
burst — ≈6× real time, offline-analysis cost; the measured peak
resident set is ≈0.2 GB (including the Python/torch runtime); the
per-candidate cost is linear in N and M₁M₂, the frequency grid
quadratic in the resolution, and the joint hypothesis set exponential
in K (∏ Mₖ — the concrete scaling reason K=3 must change acquisition
structure); and the loop is numpy CPU code, so a GPU accelerates only
the separator forward pass, not the joint loop. The ≈750 converged
classification-EM invocations and ≈6.3×10³ decision/gain rounds per
burst were already reported.

## §XIV — generalization sweeps

**Accepted; new E6 with one four-panel figure.** All three requested
sweeps are now measured end-to-end (K=2, QPSK+QPSK, 100 bursts per
cell, deterministic, `eval_robustness_sweeps.py`):

- **Frequency separation** |Δf| ∈ {0, 0.5, 1, 2, 4, 8} Hz — the sweep
  that connects Proposition 3 to receiver performance: SER is nearly
  flat in |Δf| (the joint likelihood search substitutes for
  spectral-line acquisition — the route Remark 3 says the modulation
  structure enables); the frequency RMSE stays elevated at
  co-frequency; and the V4−V3 acquisition gap is largest at Δf=0
  (0.047 at 20 dB), vanishing by 8 Hz, where the ISI-aware model beats
  the memoryless oracle by 3.5 points.
- **Timing offset** τ ∈ {0, 0.1, 0.25, 0.5} T_s: graceful to 0.1 T_s
  (+0.005–+0.04 SER), collapsing at 0.25 T_s (0.58 at high SNR),
  near-random at 0.5 T_s — timing recovery is a required deployment
  stage, and its sensitivity is now measured rather than assumed
  (Limitation (iv)).
- **Amplitude ratio** ρ ∈ {−10,…,10} dB (also the end-to-end SIR
  sweep): graceful within ±5 dB, weak-source-limited at ±10 dB
  (0.43–0.48 at high SNR). The symbol-level SIR sweep the reviewer
  saw in the previous version is superseded by this end-to-end
  version; the SIR ≳ 15 dB undecodability of the weak stream is
  unchanged.

## §XVII — reorganise the theory around the three-layer mismatch

**Accepted; done in the Discussion** (new opening paragraph): waveform
fidelity, independent synchronisation, and per-source detection
optimise three different statistical objects, so a high-SI-SDR
separator need not be an effective detection front-end — Layer 1
(waveform≠bits, E3), Layer 2 (sync ambiguity, Propositions 2–3 +
waveform FIM), Layer 3 (detection geometry, Theorems 1–2). The
introduction's contributions were reworded to the same structure. We
kept the four theoretical results as separate formal statements (they
have separate proofs and validations) rather than merging them into a
single theorem, which would obscure rather than sharpen the logic.

## Minor comments

1. **Abstract density**: rewritten as problem → insight → method →
   main result; secondary numbers (16QAM floor, marginal-MAP, E5/K=3
   detail) removed.
2. **"First pipeline"**: removed. The claim is now "the first study on
   this benchmark to quantify and close the waveform-to-bit gap at
   K=2".
3. **"Exact CRB"**: now "exact two-tone *carrier-only* CRB under the
   specified nuisance model" (Proposition 3 title, text, Figure 3
   caption).
4. **Figure 3 assumptions**: the caption now says "carrier-only
   surrogate (not the full communication waveform)" and points to the
   waveform-FIM curve.
5. **Figure 4 collision example**: the caption now gives the exact
   QPSK collision at φ=90°: the mixture point c₁ + j·c₂ is invariant
   under (c₁, c₂) ↦ (c₁ + jδ, c₂ − δ) for every difference-set
   element δ.
6. **Layout density**: the revision adds content under a hard 13-page
   limit; we compressed figures/tables and prose throughout to stay
   within it.

---

We believe the revised manuscript now meets the review's own bar: the
novelty is positioned where the evidence is (the quantified mismatch
and the receiver structure that closes it), the theory statements say
exactly what is proven (effective coefficients, carrier-only
diagnostic, phase-integrated bounds), the statistics are paired and
per-burst, the baselines cover the architecture families the review
names, and every conclusion is qualified to the regime where it was
measured.
