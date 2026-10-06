# Response to Review 5 — paper6 (main.pdf, 13 pp), 2026-10-06

We thank the reviewer for the exceptionally careful verification against the
shipped code and artifacts. Every quantitative defect was confirmed before
repair; the few places where our measurement and the reviewer's differ are
noted explicitly. Organised by the review's own numbering.

## M1. Abstract length and abbreviations — FIXED

The abstract is rewritten to **249 words** (IEEE 150–250 window) and now
contains no unexpanded abbreviations: SC-BSS/SER/SNR/SIR/MAP/CRB/BPSK/QPSK/
ISI/ECM/PSP are either spelled out ("symbol error rate", "maximum a
posteriori", "phase-shift-keyed") or removed. The rewrite simultaneously
absorbs the M3/M4/M5 wording repairs (per-constellation floor; tie at 5 dB /
win from 10 dB; the differential-score headline removed, see M4).

## M2. Headline receiver promoted to the Method section — FIXED

Section V is retitled **"Method: the joint synchroniser–detector"** and
reorganised:

- **V-A (new): "The headline receiver: ECM joint synchronisation–detection on
  the raw mixture"** — the mixture observation model, the classification-EM
  loop on the ≤16×16 grid, the coarse-to-fine frequency search with restart
  ranking, the exact-coordinate-minimiser monotonicity (memoryless), and the
  V4 ISI-aware extension (centred L=5 taps, parallel-iterated cancellation,
  2L-tap refit, generalised-EM fallback). Appendix B keeps the full procedure
  and Algorithm 1; its prose was compressed to remove the now-duplicated
  monotonicity discussion.
- **V-B: BlindCarrierSync, the K=1 branch** (content unchanged).
- **V-C/V-D: the diagnosed baseline** — SlotSepNet and JointPairDetector,
  explicitly framed as the separate → per-slot sync → joint re-detection
  chain that E3/E4 reject.

The E4 variant bullets for V1/V4 shrink to pointers into Section V-A. The
**K-decider gap** is now stated in Limitations (viii): nothing in the paper
decides K (count head 4.2 % at K=2; E5's K=3 acquisition fails), so the
K∈{1,2} envelope assumes K known or externally supplied.

## M3. "Wins at SNR ≥ 5 dB" — FIXED (was a mixed-population reading)

The E4 Results text now reads: at 5 dB the joint loop **ties** the PSP arm
(+0.035, CI [−0.000, +0.070], Wilcoxon p = 0.19) and beats only the
symbol-wise arm (+0.068, CI [+0.029, +0.106], p = 0.006); from 10 dB upward
it wins the PSP arm by +0.07–+0.08 per cell. A parenthetical states that
Table III's V1/V3/V4 cells average five grids while the route rows use the
reference grid, and that all win/tie statements are same-grid paired
contrasts — never row-wise differences. Abstract and Introduction say
"winning beyond 5 dB". We thank the reviewer for the correction of their own
earlier draft; the low-SNR route advantage (11.5/9.0/4.0 pts at −10/−5/0 dB)
is retained and cited as such.

## M4. Differential score reframed as a phase-slip rate — FIXED

The reviewer is right on every point. The E4 paragraph now states explicitly
that the rotation-invariant differential-equivalent label
(d_n = ĉ_n·conj(ĉ_{n−1}) quantised to the M-th roots) **measures the change
rate of the per-symbol phase error — a phase-slip rate, not a
differential-receiver SER** — that it rewards slowly-varying phase errors
regardless of magnitude, and that it is therefore not on the same axis as the
genie-resolved SERs. It is reported only as a secondary rotation-free view
(0.372/0.244 vs 0.537/0.357), with both caveats stated: the data are not
differentially encoded (absolute bits remain rotation-ambiguous), and genuine
continuous-stream differential detection on clean streams *costs*
+0.03–+0.10 SER at 20 dB, paid identically by oracle and blind arms
(Limitations (i)). The causality-reversed sentence ("tolerates … but
punishes …") is removed; "turns the tie into a clear win" is removed; the
abstract and conclusion no longer headline the 0.372 vs 0.438 comparison;
the CNSE contrast (iii) now carries the estimand caveat. The sample-size
ambiguity is fixed (n = 383 reference-grid bursts for the deterministic
V1/V4; the route arms average five checkpoints, 1915 evaluations), and the
"same decisions pipeline" claim is dropped (the EM path sensitivity the
reviewer measured, max|Δ| ≈ 0.014–0.018, made it overstated). In Limitations,
the K=1 differential measurement is now locatable
(`eval_diff_coding.py` cited by name; the dangling "E1-D" label is gone) and
the units bug "SER points" → "SER" (+0.03–+0.10) is fixed (minor x).

## M5. "Sits near the floor" restricted to per-constellation statements — FIXED

The pooled 0.5534 is no longer described as "near the 1/2 floor". E3 now
decomposes the K=2 oracle arm per source modulation from the shipped
per-pair records: **QPSK sources 0.454 vs the exact SIR = 0 dB expression
averaged over the same SNR grid, 0.467; 16QAM sources 0.821 vs the 0.838
floor; BPSK sources (floor 0) 0.251 on noise alone** — the pool is the
modulation-weighted mixture of per-constellation levels. The abstract,
introduction, related work, discussion and conclusion all now say "track the
exact *per-constellation* separate-detection floor", and the abstract's
"cannot remove" is now "cannot eliminate this floor **at reachable separator
quality**", removing the tension with the CNSE 0.553 → 0.444 gain.

## M6. Reproducibility — artifacts shipped / code fixed / text aligned

All items resolved; new artifacts live in `paper6_sync_jd/results/` and the
commands are logged in `EXPERIMENT_LOG.md` (2026-10-06):

- **Table I.** The n400 runs were found on the server and are now shipped:
  `e1_blind_sync_k1_n400_a.json` / `_b.json` (seed bases 99999/31337, ~200
  bursts per cell per grid). They reproduce Table I exactly, including the
  8PSK 2–4-pt gap (0.2125/0.1244/0.0878/0.0602/0.0538 vs oracle
  0.1793/0.1011/0.0656/0.0322/0.0180). One wording repair this surfaced: the
  QPSK blind-minus-oracle gap at −5 dB is 0.60 pts (one grid 0.78), so
  "≤ 0.5 pts" is now "≤ 0.6 pts" in E1, the introduction and the conclusion.
- **E3 unconditional accounting.** `eval_e2e_e3.py` now dumps
  `e3_unconditional_accounting.json` (re-run over the five checkpoints):
  oracle 0.16491/0.56031/0.58579, blind 0.21819/0.64768/0.67879 — the
  printed 0.165/0.560/0.586 and 0.218/0.648/0.679 to the digit.
- **True-stream control.** New `eval_true_stream.py` +
  `true_stream_control.json` (700 K=2 cells × 7 arms, plus a dedicated
  100-burst QPSK×QPSK @ 20 dB mode, since the grid holds only ~6 such cells
  per 100). The headline control confirms: V1 ECM on the two *true* streams
  = **0.0418** (paper: 0.041). Two numbers did *not* survive re-measurement
  and are corrected in the text: the mixture-arm reference is 0.130 (was
  0.121, a different small burst set), and the L=5-vs-memoryless-oracle
  check is **0.062 vs 0.107** (was "0.106 vs 0.164", a 4-burst smoke value
  with a swapped semantic) — the correction strengthens the claim: the ISI
  taps capture *more* of the capturable penalty than stated. As a by-product,
  V4-oracle-df now pools 0.391, *below* the V3 memoryless oracle-frequency
  reference 0.4105; zero fallback triggers throughout.
- **0.483 → 0.488** (same 300 cells, `e4_joint_k2_v4.json`), fixed.
- **`diagnose_v2.py`** — the `import json, os` inside `main()` moved to the
  top; re-run (CPU-pinned) regenerates `e4_v2_diagnosis.json` with the exact
  published medians (0.0274/0.0122/0.0297/0.0148; per-burst-ratio medians
  1.84×/1.60×). The JSON now carries the summary medians, not just stdout.
- **Remark 1 cross-modulation.** `theory_waveform_fim.py --crossmod` ships
  `waveform_fim_crossmod.json`: known-symbol median inflation over all ten
  unordered benchmark pairs is 1.0072–1.0098 (QPSK×QPSK 1.0087) —
  "modulation-agnostic over the benchmark pairs" now has an artifact.
- **Line-strength correlation.** No original script existed (interactive
  measurement); new `eval_line_strength.py` + `line_strength_corr.json`
  (400 bursts, 20 dB, plus a noiseless control): the pooled correlation of
  |E[c⁴]Σh_l⁴| with the measured symbol-level line strength is **0.75**
  (Pearson; Spearman 0.73), carried largely by the cross-modulation
  structure — within one modulation the tap factor alone correlates weakly
  (0.03–0.19). The sentence is rewritten to exactly this statement and the
  unsupported "0.65" is gone.

## M7. Theory-validation figure added — FIXED

Fig. 2 is new: joint vs separate SER vs SNR at SIR = 0 dB (QPSK+QPSK and
16QAM+16QAM panels), showing the exact separate expression riding the
separate Monte-Carlo curve, the union bound above the joint curve, the
marginal-MAP low-SNR fix, and the absence of 16QAM headroom. The sentence
deferring these curves to "the public code's figure outputs" is removed.

## M8. Identifiability certification at unit ratio — FIXED

`theory_identifiability.py` no longer excludes |ratio − 1| < 0.02: for
equal-κ pairs the unit-ratio degeneracy is the S₂ swap already in G, and the
certification now treats it as in-class; an exact-ratio-1 block is certified
separately. Re-run: generic sweep 10/10 pairs at 200/200 draws (max identity
residual 2.1e-13) **and** exact unit ratio 10/10 pairs at 200/200 (≤ 5.1e-12);
the QPSK+16QAM spurious roots are still rejected 200/200 by the higher-moment
test. (The unit-ratio regime exposed a genuinely ill-conditioned triangular
construction for BPSK+BPSK at E[s²] → 0, replaced by the algebraically
equivalent direct quadratic; verified on the previously failing draws.) The
text below Proposition 2 now states the certification explicitly includes the
benchmark's SIR ≈ 0 dB unit-amplitude-ratio point.

## M9. Headline-contrast statistics — FIXED

The Statistical-reporting paragraph now states why the route arm is
reference-grid-only (a PSP/Viterbi pass per checkpoint per grid was not
budgeted), and reports the difference *shape*: V1−V4 is positive on 59 % of
its 3500 bursts (per-grid 0.57–0.60), the headline route contrast on 36 % of
its 700 — a diffuse small mean shift, not a subset effect.

## Minor numerical corrections (a–x) — all applied

a: caption now "2.4× smaller in the plotted d_su-normalised units (5.4×
raw)". b: "within 0.0036 (47/48 cells within 0.003)". c: the 0.488 figure is
now identified in place as the 32-point phase-grid evaluation at 20 dB
(0.479 with 12 points); the dangling self-reference is gone. d: ≤0.013 Hz.
e: inlier-RMSE qualifier + 8PSK all-burst RMSE 0.135 Hz (1 % outliers).
f: "1.0–1.3× in 15 of 18 cells (worst 2.4×)". g: 89–98 %. h: BPSK/QPSK-only
0.86–0.88; 8PSK-involving PSK 0.36–0.50; six-class PSK pool 0.65.
i: "13 and 8 bursts (6 and 5 distinct triplets)". j: "no co-frequency
catastrophe (8.3-point U-shaped spread at 20 dB, minimum near 4 Hz)" —
"remarkably flat" removed. k: +0.014–+0.047. l: +7.6–+10.9 per-SNR span,
"7–11" in abstract/intro/conclusion. m: +9.4. n: 0.292. o: 0.64–0.70.
p: "+2–4 at SNR ≥ 5 dB (+1.7 at 0 dB)". q: "median of the per-burst ratio"
stated. r: 0.4102. s: phase drift 1.3–1.6×, magnitude drift 1.0–1.3×.
t: 0.6078 vs 0.5527 (in-table). u: +5.7 pts vs the tabulated 0.5392.
v: "monotonically" qualified as a reference-grid signature (five-grid average
non-monotone). w: the 36–45 % figure is now described as the measured
>1 Hz acquisition-error fraction ("the signature of interferer-line
capture"), not a measured label. x: "+0.03–+0.10 SER" (not "SER points").

## Dangling references and drafting artefacts — all fixed

All six "(complete)" subsection titles cleaned; the "E1-D" label replaced by
an explicit description + code citation; `eq:mix` and `eq:dmin` are now
\eqref'd where discussed; Fig. 6(b)'s "acquisition probability" is defined in
the caption (fraction of bursts with final frequency error < 1 Hz).

## Citations and keywords — fixed

Added: Le Roux et al. 2019 (SI-SDR), Raheli–Polydoros–Tzou 1995 (PSP),
Forney 1972 (MLSE), Meng & Rubin 1993 (ECM), Kay 1993 (two-tone
frequency-estimation context), Proakis & Salehi 2008 (differential
detection), and the companion C-SE separator manuscript. Keywords now read:
blind source separation, co-frequency communication, carrier synchronisation,
joint detection, symbol detection, identifiability, Cramér–Rao bound.

## Page budget

All repairs were absorbed within the 13-page initial-submission limit
(appendices included) by the abstract compression, the Appendix B prose
compression (content moved to Section V-A, not duplicated), figure-width
reductions, and \linespread 0.885 → 0.83. The manuscript rebuilds clean:
13 pp, no unresolved references, no overfull boxes.
