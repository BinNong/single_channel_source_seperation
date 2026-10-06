# Paper 6 — Experiment Log

Code dir: `paper6_sync_jd/`. Plan: `docs/PAPER6_SYNC_JD_PLAN.md`.
All local runs use `.venv_verify/bin/python` (torch 2.2.2 CPU, numpy 1.26.4).

## 2026-09-13 — E1 skeleton + BlindCarrierSync development

### Files
- `config.py` — paper-5-style config (identical SignalConfig) + `SyncConfig`
  (sync knobs incl. per-mod `power_mode`, candidate windows, `dd_iterations`)
  + `JointConfig` (sigma2, em_rounds, max_const).
- `sync.py` — BlindCarrierSync (numpy). `blind_sync` (hypothesis selection),
  `blind_sync_known_mod` (E1 path). Smoke test in `__main__`.
- `joint_detect.py` — JointPairDetector: `joint_detect` (vectorised joint ML
  over the ≤16×16 grid), `estimate_A_em` (EM from A=I + restart grid over
  off-diagonal magnitude/phase, max-likelihood selection), `build_oracle_A`.
- `probe_blind_sync_k1.py` — E1: nominal proxy / oracle (ser_comp) / blind,
  per mod × SNR, JSON to `results/e1_blind_sync_k1*.json`.

### Sync design iterations (all measured, K=1 bursts, before E1)

Plan §3.1 as literally specified fails in three places; each was diagnosed
and fixed or documented:

1. **M-th power on the RAW burst fails for 8PSK/16QAM even at 20–0 dB.**
   Root cause: transition-slew spurs after amplitude normalisation beat the
   carrier line (observed spurious lines >2 kHz from the carrier stronger
   than the true line at 77 Hz). Fix: RRC-matched-filter FIRST (10.7 dB
   noise-bandwidth reduction), then M-th power. Residual failure rate at
   0 dB (20 trials × 2 seeds): BPSK 0, QPSK 0, 8PSK 3/20 (ampnorm),
   16QAM 0/20 (raw). Per-mod amplitude mode (`power_mode`): 'ampnorm' for
   constant-modulus PSK, 'raw' for 16QAM.
2. **The DD phase line fit (spec step 4) is fold-biased and HURTS.**
   θ_n = angle(z_n conj(ĉ_n)) folds wrong decisions' phases toward the
   nearest grid point; with fading-ISI spread the fitted intercept is
   biased ~6° (measured: SER-optimal rotation −0.9° vs DD estimate −7.3°).
   At 20 dB, 10 trials: DD (any variant: LS / circular-mean / IRLS /
   confidence-weighted) increased SER for every modulation
   (e.g. 8PSK 0.028→0.087, BPSK 0.002→0.007). NDA alternatives are no
   better under the 3-tap channel: symbol-level Viterbi-Viterbi phase
   fluctuates several degrees per burst (cross terms of h*c products),
   and the min-distance phase scan minimises at the same biased angle
   (its min-distance value was IDENTICAL to the V&V angle's). A
   decision-directed TONE refinement (FFT of z·conj(ĉ)) is worse still
   (decision-flip phase steps create false lines; 8PSK 0.014→0.319 at
   20 dB). => `dd_iterations=0` by default; spec'd DD kept for ablation.
   Scoring resolves the leftover constant phase via the M-fold ambiguity
   (genie rotation resolution, standard blind-receiver convention).
3. **Lock score cannot gate or search frequency.** (a) With a DD line fit
   inside, unwrap absorbs any ramp → every candidate locks (useless).
   (b) Without it, the arbitrary channel phase dominates for BPSK/QPSK.
   (c) With constant-phase-only removal, 16QAM's score is systematically
   LOWER at ±0.5–1 Hz offsets than at the truth (tilted clouds land closer
   to the dense grid — min-distance absorption). (d) Slow spins are
   invisible to decision-referred statistics entirely: hard decisions
   absorb them into a zero-mean sawtooth (at a 2.6 Hz wrong pick, lock and
   residual slope match the true sync).
   Working alternative (implemented): two-stage estimation —
   sample-level coarse Δf̂₁ (MF→M-th power→FFT×8→parabolic), then THREE
   candidates {Δf̂₁, narrow ±2 Hz symbol-level refine, wide ±13 Hz
   symbol-level estimate}, arbitrated by the PRODUCT of sample-level line
   strength × symbol-level |mean(z^M)| coherence (decision-free; the two
   scores' failure modes are disjoint — deep-fade bursts keep a strong
   sample-level line while symbol-level coherence gets fooled by ISI, and
   vice versa for transition-slew spur bursts).
   Also found en route: my first smoke test drew the carrier from
   U(−5,10) — wrong; the generator takes carrier_base + U(0,5) and adds
   ±5 Hz jitter internally (residual ∈ [−5,+10]).

### Known limitations (reported, not fixed)
- **M-fold phase ambiguity**: blind decisions = true labels up to k·2π/M;
  E1 scores min over the M rotations (genie resolution).
- **16QAM weak-line bursts** (plan R2): occasionally the per-draw data
  average Σc⁴ nearly cancels (even on a CLEAN burst — observed in the
  smoke test); then no 4th-power method has a line and the coherence
  arbitration also fails (~1/16 bursts at 20 dB in the 16-trial probes).
- **8PSK at SNR ≤ −5 dB**: M=8 line below the noise floor; wholesale
  failure (physical limit of the M-th-power method at this burst length).
- **Hypothesis selection (`blind_sync`) is biased toward denser
  constellations**: lock score mean|z−ĉ|² favours 16QAM/8PSK over
  QPSK/BPSK on clean bursts (smoke test (d): QPSK→8PSK/16QAM,
  BPSK→16QAM confusions). E1 uses the true-mod variant; fixing the
  selection bias is E5-relevant TODO (e.g. normalise by constellation
  density).

### E1 results

Command: `python probe_blind_sync_k1.py --n_per_cell 20` (smoke;
results/e1_blind_sync_k1.json) and
`python probe_blind_sync_k1.py --n_per_cell 100 --out results/e1_blind_sync_k1_n100.json`.

n_per_cell=100 (≈25 bursts per modulation cell; oracle / blind SER,
blind scored with genie M-fold rotation resolution):

```
   SNR |    BPSK o/b     |    QPSK o/b     |    8PSK o/b     |   16QAM o/b
 ----------------------------------------------------------------------
 -10.0 |  0.168 / 0.158  |  0.270 / 0.520  |  0.523 / 0.763  |  0.697 / 0.866
  -5.0 |  0.036 / 0.037  |  0.169 / 0.181  |  0.334 / 0.761  |  0.522 / 0.644
   0.0 |  0.066 / 0.069  |  0.051 / 0.050  |  0.152 / 0.159  |  0.303 / 0.316
   5.0 |  0.042 / 0.043  |  0.039 / 0.041  |  0.090 / 0.118  |  0.163 / 0.185
  10.0 |  0.012 / 0.009  |  0.027 / 0.036  |  0.061 / 0.062  |  0.104 / 0.127
  15.0 |  0.005 / 0.005  |  0.010 / 0.011  |  0.012 / 0.019  |  0.054 / 0.083
  20.0 |  0.005 / 0.005  |  0.010 / 0.014  |  0.019 / 0.050  |  0.022 / 0.063
```

Nominal proxy (no sync) overall: 0.62–0.69 at every SNR (broken as
expected — the reference point).

Verdict vs the plan's success criterion (blind within ~1–2 SER points of
oracle at SNR ≥ 0 dB): MET for BPSK, QPSK and 8PSK at 0 dB (gaps
≤ +1.1 pts; BPSK blind ≤ oracle everywhere, even at −10 dB).  The
residual high-SNR gaps for 8PSK (+3.1 pts at 20 dB) and 16QAM
(+2…+4 pts at ≥ 5 dB) are dominated by ~1 weak-line burst per ~25
(per-draw Σc^M cancellation → no spectral line; plan R2) plus 16QAM's
sensitivity to the small residual constant phase (no phase refinement —
the DD/NDA alternatives measured worse, see above).  Where it breaks:
8PSK wholesale at ≤ −5 dB and QPSK at −10 dB (M-th power line under the
noise floor — physical limit at 256-symbol bursts).

### JointPairDetector smoke (synthetic K=2, SIR 0 dB)
- A=I clean → exact decisions. Oracle-A joint detection at SIR 0 dB nearly
  error-free (QPSK×QPSK SER 0.000–0.036 at SNR 10–30 dB) while separate
  detection floors at 0.42–0.49 — the Thm 1/Thm 2 gap is real and large.
- EM-A beats separate everywhere but converges only partially for
  SAME-modulation pairs (QPSK×QPSK: 0.33–0.42 vs oracle 0.00): the
  symmetric-identical-alphabet case is ambiguous (both slots QPSK,
  |off-diag|=|diag| — swap/degenerate minima; plan R3). Mixed pairs
  (QPSK×16QAM) identify A essentially perfectly (EM-A 0.005).
  => E4 will need better EM initialisation or the net-A variant for
  same-mod pairs; oracle-A bound is the headline.

## 2026-09-13 (later) — E2 theory validation (`theory_validation.py`)

Command: `python theory_validation.py` (full: 1e6/5e5 MC symbols per point
for QPSK/16QAM, 12 fixed-phi MC blocks offset from zero contacts, 64-phi
union bound, 30 bursts/mod/SNR for the frequency sweep; ~80 s CPU).
`--fast` (20× smaller MC) and `--skip_freq_meas` for quick passes.
Numbers: `results/e2_theory_validation.json`; figures:
`figures/fig_e2_ser_vs_snr_sir0.{pdf,png}`, `fig_e2_ser_vs_sir_snr20.*`,
`fig_e2_crb_inflation.*`, `fig_e2_dmin_phi.*`.

Model: scalar symbol-rate mixture r = c1 + a2 e^{jφ} c2 + w (a1 = 1 WLOG),
SNR := 1/σ² per unit-power symbol.

### Part 1 — Thm 1/2 vs Monte Carlo (QPSK×QPSK and 16QAM×16QAM)
- **Protocol fix during development**: an early version drew ONE random φ
  per MC point; the point landed near a zero-contact rotation and looked
  like a bound violation.  Thm 1 is a per-φ (a.e.) statement — MC now runs
  on a fixed 12-φ grid (offset from contacts) and reports median-φ
  (typical rotation) and mean-φ (burst ensemble) separately; the union
  bound is asserted pointwise (bound(φ) ≥ MC(φ)) at every grid phase with
  ≥10 error events.
- **QPSK×QPSK, SIR 0**: joint MC median ≈ single-user AWGN bound at high
  SNR (SNR 15: 0.0143 vs bound-of-median 0.0156; SNR 20: ~1e-4) — Thm 1
  confirmed for typical φ.  The φ-MEAN lags (0.0316 at 20 dB): small
  neighbourhoods of the 4 zero-contact rotations dominate the ensemble at
  high SNR — quantifies "a.e." at finite SNR.  Separate detection is flat
  in SNR at 0.44–0.49 (Thm 2 saturation ✓).
- **Thm 2 numeric correction**: the plan's Gaussian-interference floor
  (Q(1) ≈ 0.159/axis → SER ≈ 0.29) UNDERESTIMATES the measured floor
  (0.44–0.49) — a discrete constant-modulus interferer is more damaging
  than Gaussian at equal power.  Added `separate_exact_ser`: the EXACT
  separate-detection SER under the discrete interferer (axis-aligned
  boundaries ⇒ per-(c2, φ) Q-sums); matches the MC to ≤0.003 everywhere
  (asserted) — this is the curve to quote in the paper.
- **16QAM×16QAM**: the sum constellation is too dense — joint MC median
  0.32 at 20 dB SIR 0 (UB 0.40); joint still beats separate (0.82) but
  both are unusable.  Joint detection of two 16QAM needs ≫ 20 dB.
- **Low-SNR inversion**: at SNR ≤ 0 dB joint-ML per-symbol SER is slightly
  WORSE than separate (QPSK 0 dB: 0.478 vs 0.446) — joint ML optimises the
  pair, not the marginals; crossover ≈ 5 dB.
- **High-SIR behaviour** (fig 2): at SIR ≥ 15 dB the weak stream becomes
  undecodable from the scalar mixture and the (stream-averaged) joint SER
  rises again (QPSK 20 dB SIR: 0.146) — the SIR≈0 operating point is
  genuinely the sweet spot for joint detection.
- Union bound is loose (vacuous > 1) below ~7 dB SNR as expected; tracks
  the MC mean within ~2× at ≥ 15 dB for QPSK.

### Part 2 — frequency CRB (§2.3)
- Numeric Fisher (unknown phases) == Rife–Boorstyn closed form to 1e-6
  (asserted).  Single-tone CRB std: 0.042 Hz @ −5 dB → 0.0024 Hz @ 20 dB
  (N=4096, fs=16 kHz).
- Two-tone inflation over the generator's Δf = |U(0,5)+U(−5,5)| law:
  median 1.82×, p10 1.04×, p90 10.5×, mean 22.8× (heavy tail below
  Δf ≲ 1 Hz).  Exact Fisher has NO null at Δf = 1/T (inflation 1.59
  there) — the plan's sinc² heuristic dips to exactly 1.000 at 1/T; the
  null is an artifact of the unweighted kernel (with unknown phases the
  n²-weighted kernel stays correlated).  Both blow up as Δf → 0 — the
  architectural argument (separate → sync → joint re-detect) stands.
- Measured BlindCarrierSync |Δf̂| (K=1, 30 bursts/mod/SNR): BPSK/QPSK are
  within 1.1–1.8× of the CRB std at −5…0 dB (near-efficient when
  noise-limited); ALL mods flatten to an SNR-independent floor of
  ~0.01–0.07 Hz at high SNR (fading-ISI data-dependence, not noise) —
  e.g. 16QAM 0.053 Hz = 22× CRB at 20 dB.  8PSK at −5 dB: 26/30
  wrong-peak outliers (the known M=8 low-SNR failure).  Framing: CRB is
  the lower bound; our estimator is not efficient and is ISI-floored —
  both support the plan's Prop 5 narrative.

### Part 3 — d_min(φ) landscape (SIR 0)
- QPSK: median d_min(φ) = 0.552 = 0.39·d_su; exact zero contacts at
  {0°, 90°, 180°, 270°} (4 points — only EQUAL-magnitude difference pairs
  can cancel; an early guess of 8 contacts (k·45°) was wrong: the
  diagonal-diagonal cancellations land on the same 4 angles).
- 16QAM: median 0.102 = 0.16·d_su; p5 = 0.009; 30% of rotations have
  d_min < 0.1·d_su.  Grid-aligned exact contacts {0°,90°,180°,270°}, plus
  dense near-zero dips at atan-ratio angles (e.g. 143.13°) — all
  measure-zero, but they explain why 16QAM×16QAM joint detection needs
  ≫ 20 dB SNR: the typical minimum distance is 6× smaller than QPSK's.

## 2026-09-15 — Manuscript skeleton

Created `paper6/` (repo root): `main.tex` (elsarticle review format,
paper-5 conventions; corrected E2 statements: 4 QPSK zero contacts, exact
discrete-interferer floor 0.488, exact-Fisher CRB inflation), `build.sh`
(XeLaTeX two-pass), `elsarticle.cls` (copy), `figures/` (4 E2 figure PDFs
copied from `paper6_sync_jd/figures/`). Compiles clean, 27 pages, no
undefined refs. E3–E5 are TODO-marked placeholders; no E3–E5 numbers
fabricated.

## 2026-09-18 — E3: end-to-end blind pipeline (server, complete)

Code: `eval_blind_pipeline.py` (paper5 evaluate.py conventions copied
verbatim: test seed 99999, occupancy gating 0.5 with slots ordered by
occupancy, Hungarian matching with cost = -SI-SDR, matched pairs only).
Synced via rsync to `/data/experiment/paper6_sync_jd/`; run with
`/data/experiment/venv_bss/bin/python`.

Commands:
  pilot: python eval_blind_pipeline.py --n_per_cell 25 --configs mse --sic
  full : python eval_blind_pipeline.py --n_per_cell 100 --configs mse ser_mse --sic
  (full log: e3_full_n100.log on the server; runtime ~7 min total)

Results: results/e3_blind_pipeline_{mse,ser_mse}.json (+ _pairs.json),
results/e3_mixture_blind_k2.json.

**Oracle reproduction check (config b, mse, 5 seeds, n=100):** pooled
compensated SER 0.5027 (per-seed 0.5019–0.5037) and BER 0.2651 —
BIT-IDENTICAL to paper 5's published 0.5027±.0007 / 0.2651±.0005; per-K
oracle 0.1454/0.5534/0.5858 vs the published family 0.146/0.561/0.587
(mixture baseline) — the evaluation port is exact. Config (d) ser_mse
oracle pooled 0.5035 vs published 0.5034±.0013 ✓.

**Blind vs oracle (config b, 5 seeds; config d in parentheses):**
pooled delta +0.0853±.0009 (+0.0802±.0010) SER, 5/5 seeds positive;
BER 0.2651→0.3519. Per-K SER delta: K=1 +0.0545±.0031, K=2
+0.0887±.0010, K=3 +0.0930±.0007. Per-K×SNR deltas: K=1 is +0.7..+2.3
pts at SNR ≥ 0 dB (E1-level) and ~+15 pts at −10/−5 dB (the known
low-SNR sync failure); K=2/K=3 deltas are SNR-FLAT (+8..+11 pts at every
SNR from −10 to 20 dB).

**K≥2 leakage diagnosis:** the blind slot penalty at K≥2 is NOT spectral
line burial by the interferer: (i) the crude SIC variant (sync the slot
after subtracting the other matched slot estimates from the mixture)
does NOT help (K=2: 0.6484 vs plain blind 0.6422 — slightly worse);
(ii) the median blind frequency error on K≥2 slots is 2–4 Hz at ALL
SNRs (vs 0.007–0.04 Hz at K=1, SNR ≥ 5 dB) — the residual leakage's
M-th-power line PULLS the estimate off the slot's carrier (structured,
SNR-independent bias), exactly the phase/frequency-entanglement regime
of Props 4–5, i.e. the regime E4's joint re-detection targets.
K=2 blind slots (0.6422) ≈ K=2 blind raw-mixture baseline (0.6427):
separation+independent-sync gains nothing over blindly demodulating the
mixture — sync must be joint or leakage-aware.

## 2026-09-18 — E4: joint sync+detection at K=2 (V1/V3 complete; V0/V2 pending server)

New code: `joint_sync_detect.py` (ECM core), `eval_joint_k2.py`
(variant driver).  Design is measurement-driven; two hard-learned rules
documented in the module docstring: (i) EM must run to CONVERGENCE at
every frequency candidate used for ranking (truth E=0.63 vs phantom
E=1.74 converged, INVERTED at 2-4 rounds); (ii) frequency moves need EM
re-estimated at the candidate (decision-frozen NLS can't leave a
bad-decision basin).  Bugs found and fixed en route: gain-phase init
grid must cover one symmetry sector [0, 2π/M) finely (not {k·2π/M},
which collapses to the same point modulo symmetry); a swapped-conjugate
bug in the batched 2x2 complex-LS inverse (caught by a polish-stage
monotonicity check + numpy.linalg.lstsq parity check).
Rejected alternatives (measured): M-th-power extreme-lines two-tone
estimator (cross terms dominate at SIR 0; pair errors 2-16 Hz);
M-power phase init for the gains (interferer's power dominates at SIR 0);
decision-directed frequency tone refinement (false lines from decision
flips).

### ECM pipeline (joint_sync_detect.ecm_joint)
Coarse 1.5 Hz 2-D grid (d1<=d2) with converged phase-grid EM on the
stride-4 subsampled symbol stream -> fine 0.2 Hz grid (+-1.2 Hz) around
the top-2 with warm-started converged EM -> full-rate converged
phase-grid EM -> NLS polish (+-0.15 Hz, 0.05 Hz).  ~2.3 s/burst (local
CPU).  Smoke test: QPSKxQPSK at 20 dB ECM 0.106 = oracle-df 0.103
(separate 0.467); monotone energy; noiseless QPSK pair 0.012.

### Results so far (test seed 99999, K=2 cells, n=100/SNR)
Commands:
  pilot (local): python eval_joint_k2.py --n_per_cell 25 --snr_points 0 10 20 --variants V1 V3
  full  (local): python eval_joint_k2.py --n_per_cell 100 --variants V1 V3 --out results/e4_joint_k2_mixture.json

V1 = ECM on the raw mixture (headline, training-free);
V3 = oracle-frequency bound.  Per-source SER (PIT + rotation-resolved):

| variant | -10 | -5 | 0 | 5 | 10 | 15 | 20 | pooled |
| V1 | 0.7016 | 0.6286 | 0.5875 | 0.4657 | 0.4531 | 0.4718 | 0.4574 | **0.5380** |
| V3 | 0.6200 | 0.5273 | 0.4488 | 0.3275 | 0.3075 | 0.3305 | 0.3114 | **0.4104** |

References: mixture baseline 0.5611, waveform route 0.5475.
V1 beats BOTH references (pooled -2.3 pp vs waveform route, -0.9 pp vs
baseline... vs mixture; per-pair-class split: PSK-only V1 0.3920 / V3
0.2570; 16QAM-involving V1 0.7143 / V3 0.5958 (n=383/317) — the gain
concentrates on PSK pairs exactly as E2's d_min analysis predicted.
The blind-vs-oracle-df gap (0.538 vs 0.410) is the residual frequency
acquisition cost; the SNR curve is non-monotone (best at 10 dB) because
fading ISI smears the sum constellation at high SNR (model-mismatch
floor), noise at low.

V0 (E3-chaining baseline) and V2 (slot-aided ECM) need the paper-5
checkpoints -> server; the server dropped mid-session (SSH timeouts from
~15:05 local); retry loop running.  Will append when done.

### E4 V0/V2 completed (2026-09-18, server back ~17:10 local)

Server came back after ~2 h.  Code rsynced up; the 5 config-(b) *_best.pt
checkpoints were pulled DOWN to local paper5_task_oriented/checkpoints/
(so V2-style runs now go local).  Commands (all local CPU):
  pilot: python eval_joint_k2.py --n_per_cell 25 --snr_points 0 10 20 --variants V0 V2 --out results/e4_joint_k2_slots_pilot.json
  full : python eval_joint_k2.py --n_per_cell 100 --variants V0 V2 --out results/e4_joint_k2_slots.json
Runtime: pilot 13.4 min; full 1124 s/seed x 5 seeds ≈ 1.6 h.

Full V0–V3 table (K=2 cells, n=100/SNR, 5 seeds for V0/V2):

| variant | -10 | -5 | 0 | 5 | 10 | 15 | 20 | pooled |
| V0 slots->blind sync->EM-A | 0.674 | 0.628 | 0.634 | 0.559 | 0.590 | 0.590 | 0.580 | 0.6078 |
| V1 ECM on raw mixture      | 0.702 | 0.629 | 0.588 | 0.466 | 0.453 | 0.472 | 0.457 | 0.5380 |
| V2 ECM on slot streams     | 0.702 | 0.660 | 0.634 | 0.547 | 0.531 | 0.550 | 0.548 | 0.5960 |
| V3 oracle-df ECM           | 0.620 | 0.527 | 0.449 | 0.328 | 0.308 | 0.331 | 0.311 | 0.4104 |

Per modulation-pair class (pooled): V0 PSK-only 0.512 / 16QAM 0.724;
V2 PSK-only 0.483 / 16QAM 0.733;  (V1/V3: see previous entry).

Findings:
- Ordering V3 < V1 < V2 < V0 — sanity holds; naive chaining (V0) is
  worse than the mixture baseline (0.6078 > 0.5611), confirming that
  independent sync kills the slots before detection.
- **V2 (slot-aided) does NOT beat V1 (mixture)**: +0.058 SER on all
  5 seeds; vs the 0.5475 waveform-route reference V2 is +0.0485±.0013
  (5/5 seeds positive).  Diagnosis (direct measurement): the
  occupancy-top-2 slot streams carry target/leakage SIR of only
  0.3-2.4 dB and their blind-sync carrier estimates are biased (both
  slots often lock the same source's line); the two-source ECM model is
  exactly matched by the raw mixture but mismatched by the
  nonlinearly-distorted slots.  Positive control: the SAME 2-channel ECM
  on the two TRUE source streams gives SER 0.041 (vs mixture ECM 0.121
  on those bursts) — the joint loop gains a lot IF separation is good;
  at the current SI-SDRi ~+4 dB it is not good enough to help.
- Conclusion for the paper: the mixture route (V1) is the operating
  point; slot-aided ECM is a measured dead end at the current separator
  quality; V3-V1 gap (0.41 vs 0.54) is the frequency-acquisition
  residual; both gaps quantify what better separation / better sync
  would buy.

### E4 V2-vs-V1 diagnosis + canonical server run (2026-09-18, later)

Canonical server run (config-(b), seeds 42-46, n=100/SNR, K=2 cells):
  cd /data/experiment/paper6_sync_jd && setsid nohup /data/experiment/venv_bss/bin/python \
    eval_joint_k2.py --n_per_cell 100 --variants V0 V2 \
    --out results/e4_joint_k2_slots_server.json > e4_full_v0v2.log 2>&1 < /dev/null &
  (launcher PID 355679, python child 355681; ~0.9 s/cell -> ~55 min ETA)
NOTE: an identical full V0/V2 run had already completed LOCALLY
(results/e4_joint_k2_slots.json — the table in the previous entry);
the server run is the canonical record.

Diagnosis of why V2 (slot-aided ECM) loses to V1 (mixture ECM)
(diagnose_v2.py, local, config-b s42 checkpoint, 40 K=2 cells at SNR 20 dB
— 22 PSK-pair + 18 16QAM-involving):
  (a) Converged-ECM residual energy per channel-symbol: slots carry
      1.6-1.84x the mixture's residual (PSK pairs: 0.0274 vs 0.0122
      median; 16QAM-involving: 0.0297 vs 0.0148) — the mask-separated
      slot streams fit the linear two-source model WORSE than the raw
      mixture (nonlinear mask distortion), and the ECM likelihood
      surface degrades accordingly.
  (b) Split-burst gain drift (4 sub-blocks, phase/magnitude spread of
      the LS-fitted A columns): slots drift ~1.5x more than the mixture
      (PSK col0: 0.97deg/1.6% vs 0.60deg/1.3%) — a secondary effect, not
      the dominant mechanism.
  (c) Both classes show the same ordering, so the mechanism is
      class-independent (not a 16QAM-specific artefact).
  Positive control (earlier): the same 2-channel ECM on the TRUE source
  streams gives SER 0.041 vs 0.121 mixture ECM on the same bursts —
  the joint loop would win big if separation were good; at SI-SDRi
  ~+4 dB the slot distortion costs more than the leakage suppression
  buys.  This is the paper's "when to separate at all" answer: at the
  current separator quality, joint detection should run on the mixture.

### E4 cross-check + V4 (ISI-aware ECM) — 2026-09-18 (later)

Cross-check: the canonical server run (e4_joint_k2_slots_server.json)
matches the local full run to 1e-4 (V0 0.6078=0.6078, V2 0.5961 vs
0.5960 — float-level; server 603 s/seed vs local 1124 s). PASSED; local
numbers canonical.

**V4: ISI-aware joint ECM** (joint_isi.py, warm-started from V1):
model r_n = sum_k e^{j2πΔf_k n Ts} sum_l g_k[l] c_{k,n-m}, m = l − L//2
— CENTRED lags (the generator's fading conv is mode='same'/zero-delay,
so the composite symbol-rate response is symmetric; a causal-only tap
layout captured half the ISI and gave ~half the gain — measured during
development).  L = 5 (L=7 saturates, measured).  Loop: parallel-iterated
ISI cancellation + joint grid decisions with centre-tap gains, closed-form
LS refit of all 2L taps, small 2-D NLS frequency polish with taps refit
per candidate; energy-monotone, fallback to V1 if worse (0 fallbacks in
all runs).  __main__ smoke: 4 faded QPSK pairs @20 dB, V4 ≤ V1 on each.

Commands:
  smoke : python joint_isi.py
  pilot : python eval_joint_k2.py --n_per_cell 25 --snr_points 0 10 20 --variants V1 V3 V4 --out results/e4_joint_k2_v4_pilot2.json
  full  : python eval_joint_k2.py --n_per_cell 100 --variants V4 --out results/e4_joint_k2_v4.json   (24 min, local CPU)

Full-grid V4 (n=100/SNR; V1/V3 from e4_joint_k2_mixture.json, same cells):

| SNR | V1 | V4 | V3 |
| -10 | 0.7016 | 0.7016 | 0.6200 |
|  -5 | 0.6286 | 0.6269 | 0.5273 |
|   0 | 0.5875 | 0.5850 | 0.4488 |
|   5 | 0.4657 | 0.4581 | 0.3275 |
|  10 | 0.4531 | 0.4386 | 0.3075 |
|  15 | 0.4718 | 0.4575 | 0.3305 |
|  20 | 0.4574 | 0.4412 | 0.3114 |
| pooled | 0.5380 | **0.5299** | 0.4104 |

PSK-only pairs, gap closure (V1→V3): 1%/1%/3%/7%/15%/14%/20% from −10 to
20 dB — the gain grows with SNR exactly as the ISI-floor story predicts;
the 20 dB PSK floor moved 0.2408→0.2169.  PSK-only pooled: V1 0.3920 →
V4 0.3796; 16QAM-involving unchanged (0.711 vs 0.714 — dense sum
constellation, no gain available, as E2 predicted).  V4-oracle-df with
L=5 on QPSK pairs @20 dB: 0.106 vs 0.164 memoryless-oracle (true-stream
control 0.041) — the taps capture most of the ISI penalty that IS
capturable at the fixed zero-delay grid; the rest is long-tail/non-model
residual.

## 2026-09-19 — E5: blind modulation, K=3 probe, ablations

All local CPU.  Files: `e5_modsel_probe.py`, `eval_joint_k3.py`,
`joint_sync_detect_k3.py`, variants V4B/V4NOPOLISH/V4SINGLE/V4L3 in
`eval_joint_k2.py`.  Results: results/e5_modsel_probe.json,
e5_subset_v3.json, e5_k3_probe.json.

### 1. Blind modulation-pair selection (V4B)

Selector tournament on shared bursts (each arm = full ECM per
hypothesis, 10 unordered pairs):
  - raw converged energy: always picks (16QAM,16QAM) — acc 0.11
    (densest grid absorbs everything; margins −0.5…−0.9 vs truth);
  - held-out (even/odd symbol) energy: identical failure (acc 0.11) —
    the overfit is in the per-symbol min over a dense grid, present at
    test time too;
  - two-stage per-source lock score (E1): acc ~0.2 (density-biased as
    known from E1);
  - unnormalised soft evidence (logsumexp over the grid): still always
    (3,3) — the log M1·M2 term count bonus;
  - per-hypothesis sigma2 evidence: always (BPSK,BPSK) (acc 0.22) —
    overcorrects;
  - NORMALISED soft evidence (uniform prior over the hypothesis grid,
    shared sigma2 from the best fit): acc ~0.5 on the probe sample —
    the only working selector; adopted for V4B.

V4B (subset: SNR {0,10,20}, n=25, test seed 99999):
mod-pair accuracy 0.16 / 0.36 / 0.16 at SNR 0/10/20 (chance = 0.10).
End-to-end SER with the joint classify+detect convention (wrong pair ⇒
SER 1): 0.982/0.674/0.946, pooled 0.867 vs known-mod V4's
0.558/0.423/0.542 (pooled 0.508) on the same cells.  Conditional on a
correct pair decision (17/75 cells), SER = 0.415 ≈ V4-level — the
pipeline is fine WHEN the pair is right; the classifier is the
bottleneck.  Honest verdict: joint blind modulation classification at
SIR≈0 with 256-symbol bursts is unreliable with every selector we
tried; known-mod stays the paper's operating assumption, and this is a
stated limitation.  (Note: an early V4B run scored garbage because
run_v4b double-applied the matched-filter front-end — fixed; the
selector-tournament numbers above were unaffected.)

### 2. K=3 feasibility probe (PSK-only triplets, SNR {5,15}, n=25)

joint_sync_detect_k3.py: sequential-extraction init (K=2 ECM -> subtract
-> residual M-th-power line for source 3) + 3-source converged
phase-grid EM + centred-tap (L=3) ISI + per-source NLS polish; PIT over
6 assignments.  Result — honest NEGATIVE: ECM-K3 0.622/0.572 vs the
same-cells oracle mixture baseline 0.516/0.459 at SNR 5/15 dB
(+10-11 pts WORSE).  Failure mode: the frequency init — the K=2 ECM on
a 3-source mixture locks a phantom pair, and the residual line is then
meaningless (measured dfs far off truth).  K=3 joint detection stays
future work; the paper covers K=1/K=2.

### 3. Ablations (subset SNR {0,10,20}, n=25; V4 = 0.5579/0.4227/0.5424,
pooled 0.5077)

  - V4NOPOLISH (no fine NLS): 0.5596/0.4229/0.5464, pooled 0.5096 —
    the NLS polish contributes ~0 (the fine grid already suffices).
  - V4SINGLE (top-1 coarse candidate, no restart set):
    0.5590/0.4409/0.5464, pooled 0.5154 — init robustness costs ~0.8 pp
    pooled; restarts matter little at these SNRs.
  - V4L3 (3 centred taps): 0.5584/0.4323/0.5484, pooled 0.5130 —
    dose-response L=1 (V1) 0.5202 < L=3 0.5130 < L=5 0.5077.

## 2026-09-19 — paper6_sync_jd made self-contained (vendoring)

Vendored (copy-not-share; provenance comments in each file):
  - signal_utils.py   <- paper1_cnn_se/data_generator.py (rrc_filter,
    generate_symbols, generate_single_signal) + paper1_cnn_se/utils.py
    (CONSTELLATIONS, _get_constellation) + soft_demod.py's CONST_LISTS as
    numpy CONST_MAT/CONST_MASK.
  - data_generator.py <- paper5_task_oriented/data_generator_vark.py
    (imports the vendored signal_utils; no sys.path tricks remain).
  - ser_comp.py       <- paper5_task_oriented/ser_comp.py (evaluation truth).
  - models.py         <- paper5_task_oriented/models.py slot-relevant
    classes (ComplexConv1d/BatchNorm/ReLU/SE/ResidualBlock, CSEBackbone,
    OccupancyHead, CountHead, SlotSepNet; JointLLRHead/StopHead/
    OneAndRestNet/SpecialistBankNet dropped; use_joint_head raises).
  - losses_baseline.py <- paper5_task_oriented/losses.py (si_sdr pair
    matrix, assignment_table, variable_k_pit_loss; the paper-5 lambda_ser/
    lambda_llr branches raise NotImplementedError — not vendored).
  - train_baseline.py <- trimmed paper5 train.py (slot arch, baseline
    recipe lambda_sep=1/lambda_mse=1/lambda_occ=1/lambda_cnt=0.1, 100
    epochs, --seed, --resume, --smoke).
  - config.py: added VarKConfig (values identical to paper4/5) for the
    vendored ser_comp.main.
  - checkpoints/: the 5 config-(b) *_best.pt copied in (canonical
    evaluation artifacts; do NOT retrain for headline numbers).

All existing scripts rewired to the vendored imports (no runtime imports
from paper1_cnn_se / paper4_open_world / paper5_task_oriented remain;
grep-verified).

Equivalence verification (pre- vs post-vendoring, same subsets):
  - E1: probe_blind_sync_k1.py --n_per_cell 10 --snr_points 0 20 —
    per-mod tables BIT-IDENTICAL (JSON diff clean).
  - E4-V4: eval_joint_k2.py --n_per_cell 10 --snr_points 10 --variants V4
    — per-record (ser, df errors, res_energy) BIT-IDENTICAL.
  - ser_comp vendored main (n=2/cell): overall 0.4723, per-K
    0.147/0.570/0.516 — consistent with the published family
    0.146/0.561/0.587 within small-cell noise.
  - All __main__ smokes pass: signal_utils, data_generator, models
    (254,093 params), losses_baseline, sync, joint_detect, joint_isi,
    joint_sync_detect, joint_sync_detect_k3.
  - train_baseline.py --smoke: 1 epoch / 64 samples on CPU runs and
    saves checkpoints (loss 3.92 -> val SI-SDR -4.8 dB as expected for
    one epoch on noise-init).

Server sync: server unreachable at the time (SSH timeout); a background
retry loop (rsync-when-back) is armed.  Local copies are complete and
canonical.

## 2026-09-21/22 — Major-revision revision round (review1.md): theory fixes + statistics

Reviewer report (`../paper6/review1.md`, Major Revision).  All hard
issues addressed; every new number below is reproduced by the code in
this directory (no hand edits).

### R1/4. Prop 3 offset law corrected (theory_validation.py)
- BUG FOUND AND FIXED: the inflation statistics were averaged over
  |u + eps| of a SINGLE source (support [0,10] Hz), mislabeled as the
  two-source separation.  The correct benchmark quantity is
  Df = |delta_1 - delta_2|, delta_k = u_k + eps_k i.i.d.
  (support [0,15] Hz).
- New: df_diff_pdf/df_diff_cdf — EXACT piecewise-quadratic density and
  CDF (symbolically derived; KS vs sampler < 2e-3; CDF(15)=1;
  E|Df| = 89/24 Hz; median 3.225; p90 7.556; p95 8.772;
  P(Df < 1/T = 3.906 Hz) = 0.5871).
- New inflation stats: median 1.698x, p90 7.624x (phases-only FIM);
  median 4.611x, p90 286.9x (gains unknown — the benchmark's actual
  condition; two_tone_inflation(..., unknown_amplitudes=True) 6x6 FIM).
  Qualitative conclusion unchanged; numbers now rigorous.

### R2. Theorem 2 floor: continuous-phase ensemble = 1/2 (closed form)
- separate_floor_limit(): high-SNR limit under the continuous uniform
  phase ensemble: QPSK 0.4999 (CLOSED FORM 1/2, proof added to
  Appendix A), 16QAM 0.8380 (numerical).  The published 0.488 was the
  20 dB / finite-phase-grid evaluation (contact phases carry floor
  7/16 < 1/2; 32-grid saturation at 0.4922 verified symbolically).
- Main text now separates P_sep(phi), E_phi[P_sep] and the finite-grid
  average explicitly; abstract/intro say "1/2".

### R5. EM renamed and specified (Algorithm 1 in the paper)
- classification-EM (hard decisions + exact LS refit) = coordinate
  descent on one least-squares objective -> energy-monotone; nested in
  the frequency search = ECM.  Convergence criteria stated (1e-5
  energy tolerance, 30 rounds; polish stall 1e-4).
- marginal_decisions(): per-stream marginal-MAP from a converged ECM
  fit (sigma2 = residual energy / N).

### E2 rerun (local CPU, both rounds; asserts all pass)
- MC now scores joint-ML, separate AND marginal-MAP (marginal fixes the
  low-SNR reversal: 0 dB SIR 0 QPSK: joint 0.479 > separate 0.447;
  marginal 0.446 <= separate; mmap == joint at >=5 dB).
- Frequency sweep re-measured at n=100 bursts/(SNR,mod): failures
  P(|err|>1 Hz): 8PSK 84% @ -5 dB; <=10% elsewhere in range; 0 at 20 dB
  for all.  Full bias / all-burst RMSE / inlier RMSE recorded.

### E4 multi-seed (server, 5 test grids x (V1V4+V3), n=100/cell)
eval_joint_k2.py: --test_seed; V1V4 combined variant (V1 search shared);
marginal-MAP + EM-complexity counters recorded per burst.
  pooled SER over 5 grids: V1 0.5392 [0.5342, 0.5441],
  V4 0.5310 [0.5261, 0.5360], V3 0.4106 [0.4044, 0.4169] (95% CI,
  t-dist, 5 seeds).  V4 vs waveform route 0.5534: -2.2 pts.  Seed 99999
  reproduces the paper numbers (V1 0.5384 / V4 0.5305 / V3 0.4105 vs
  published 0.5380 / 0.5299 / 0.4104; <=0.0006 from BLAS threading).
  Marginal-MAP: V1 0.5388, V3 0.4101 (joint rule not limiting >=0 dB).
  EM stats/burst: 747 converged-EM calls, ~6.3e3 rounds; V1+V4 =
  1.58 s/cell on an i5-12500 core (grid 1108 s).

### E1 reinforced (server, probe_blind_sync_k1.py --n_per_cell 400,
  seeds 99999 & 31337 -> 200 bursts per modulation-SNR cell)
- IMPORTANT CORRECTION at 8x more data: 8PSK blind-oracle gap at
  SNR>=0 is 2-4 pts (0.212/0.124/0.088/0.060/0.054 vs oracle
  0.179/0.101/0.066/0.032/0.018), NOT the <=1.1 pts the n=25 pilot
  suggested.  BPSK/QPSK gaps <=0.5 pts at SNR>=-5 (two grids agree).
  Table e1 + the "1.1 pts" claims rewritten accordingly.
- probe_blind_sync_k1.py: added --seed_base.

### E3 end-to-end scoring (new: eval_e2e_e3.py, server, 5 ckpts)
Per-sample accounting: count acc 0.909/0.042/1.000 at K=1/2/3 (pooled
0.650); per-source miss 2.3/1.5/0%; false-slot rates 6.8/92.7/0%.
End-to-end (missed source = SER 1): K=2 oracle 0.5603 vs conditional
0.5534; blind 0.6477 vs 0.6422 -> matched-pairs conditioning moves the
conclusions by <=0.7 pts only.  Paper text updated.

### Sync to server
4 modified files + config + probe_blind_sync_k1.py rsync'd; server runs
launch with OMP_NUM_THREADS=2 per job.  All artifacts under
results/ (e4_joint_k2_mse_ts*.json, e3_e2e_mse*.json,
e1_blind_sync_k1_n400_*.json, e2_theory_validation.json).

### Final packaging (2026-09-22)
- paper6/main.tex builds to 13 pages (TSP target), all new content
  included: Algorithm 1 (classification-EM/ECM, monotonicity +
  convergence), marginal-MAP rule + curves, E3 end-to-end accounting,
  multi-seed E4 CIs, corrected E1 table, appendix proofs (Thm 2 closed
  form + |Df| piecewise-quadratic density + dual-nuisance FIM).
- Page budget measures (kept in the repo for reproducibility):
  fig_e2_ser_vs_sir_snr20 and fig_e2_dmin_phi regenerated single-column;
  ser_snr/CRB figures compressed; appendix \footnotesize; bibliography
  \scriptsize with DOIs dropped (vol/no/pp identify every entry; the
  online-first edsnet2026 keeps its DOI); ~90 lines of prose tightened.
  theory_validation.py now produces exactly the shipped figure formats.
- Server code re-synced (theory_validation.py, joint_sync_detect.py,
  eval_joint_k2.py, eval_e2e_e3.py, probe_blind_sync_k1.py, config.py).
- FINAL: main.pdf = 13 pages (TSP limit), builds clean via build.sh,
  zero undefined references. Point-by-point response letter at
  ../paper6/response_to_reviewers.md.

## 2026-09-22 (late) — Review-2 revision, round 2

### Infrastructure (RNG-stream preserving)
- signal_utils.py: generate_single_signal(..., freq_jitter=None) — a
  given float REPLACES the U(-5,5) jitter draw (draw skipped; all
  pre-existing callers bit-identical, asserted).  Needed for controlled
  true carrier offsets in the frequency-separation sweep.
- joint_sync_detect.py: mixture_symbols(mix, tau_samples=0.0) —
  fractional symbol-grid offset via linear interpolation (tau=0
  bit-identical, asserted).  Needed for the timing-offset sweep.

### Communication-waveform FIM (review 2, Major Concern 4; NEW
  theory_waveform_fim.py, local CPU, deterministic seed 20260922)
- Model: full benchmark waveform, passband fading convention,
  s_n = sum_k conv(u_k x_k, h_k)[n]; known symbols/pulse, unknown
  (Df1, Df2, Re/Im h_1(0..2), Re/Im h_2(0..2)) = 14 real params.
- RESULT: median two-source frequency-CRB inflation 1.01x at EVERY
  |Df| in [0,16] Hz (p90 <= 1.02x; modulation-agnostic: BPSK/QPSK/
  8PSK/16QAM pairs all 1.007-1.009 at Df=0/0.5 Hz, 12-trial check)
  vs the pure-tone surrogate divergence (6446x at Df->0, 1.59x at
  1/T_burst).  Interpretation (Remark 3 in the paper): the
  frequency-disambiguating information at co-frequency lives in the
  modulation structure (data-aided), unreachable for non-data-aided
  spectral-line sync; the ECM's decisions act as semi-pilots, which is
  why the joint loop beats per-slot sync and why the V4->V3 residual is
  the price of decision errors, not of frequency coupling.
- results/waveform_fim.json; theory_validation.py overlays the curve
  on fig_e2_crb_inflation (left panel) and re-validates all E2 numbers
  (full rerun, logs_theory_rerun.log).

### Paired per-burst statistics (review 2, Major Concern 7; NEW
  analyze_paired_stats.py; inputs: the five e4_joint_k2_mse_ts*.json
  pulled from the server + results/psp_baseline.json)
- C1 (V1 - V4, same cells): pooled over 5 grids +0.0082, paired
  bootstrap 95% CI [+0.0074, +0.0089], Wilcoxon p = 1.3e-90 (n=3500).
  Per-SNR gain grows monotonically: ~0 at -10 dB to +0.016 at 15-20 dB
  (the ISI-budget signature).  All five grids individually
  p <= 7e-17.
- C2 (waveform route - V4): PENDING eval_psp_baseline.py (symbol-wise
  arm is the pairing source; E3's pairs JSON cannot be
  position-paired — Hungarian matching drops unmatched pairs).

### Review-2 experiments (2026-09-22/23; all deterministic, commands below)
- theory_waveform_fim.py (local): communication-waveform known-symbol
  FIM (14 real params: Df1, Df2, Re/Im 3-tap channels, passband fading
  convention).  RESULT: median two-source frequency-CRB inflation
  1.01x at every |Df| in [0,16] Hz (p90 <= 1.02x; modulation-agnostic
  over BPSK/QPSK/8PSK/16QAM pairs, 12-trial check 1.007-1.009) vs the
  pure-tone divergence (6446x at Df->0, 1.59x at 1/T_burst).  The
  disambiguating information lives in the modulation structure
  (data-aided); the pure-tone surrogate bounds only the non-data-aided
  spectral-line route.  Paper Remark 3 + red curve in fig_e2_crb_inflation.
- analyze_paired_stats.py (local; inputs: 5x e4_joint_k2_mse_ts*.json
  from the server + psp_baseline.json):
  C1 V1-V4: +0.0082, paired bootstrap 95% CI [+0.0074,+0.0089],
  Wilcoxon p=1.3e-90 (n=3500); per-SNR gain ~0 at -10 dB -> +0.016 at
  15-20 dB (ISI signature).
  C2 route-V4 (700 shared cells): symbol-wise +0.0222 (SNR-structured,
  -0.108 at -10 dB to +0.099 at 20 dB); PSP route +0.0032, CI
  [-0.0112,+0.0178], p=4.8e-4 — a statistical tie.
- eval_psp_baseline.py (local, 5 ckpts, reference grid): waveform route
  + per-slot Viterbi (L=3, DD channel estimation): symbol-wise 0.5527,
  PSP 0.5337 pooled.  The PSP upgrade ties V4 (0.5310) — the headline
  was rewritten to "matches the strongest oracle-assisted waveform
  route; V3 (oracle freq) beats it by 12 pts".
- eval_diff_coding.py (local, 200 bursts/cell x 2 grids, BPSK/QPSK/8PSK):
  genie M-fold vs differential decoding.  Blind DIFF ~= blind ABS at
  high SNR modulo the intrinsic differential penalty (+0.03..+0.08;
  oracle pays it too: 8PSK 20 dB 0.018->0.120); at low SNR DIFF is
  BETTER (per-symbol rotation ~1.8 deg/Hz; 8PSK -5 dB 0.458 vs 0.747).
- eval_separator_quality.py (local): SlotSepNet SI-SDRi 9.6/3.2/2.8 dB
  at K=1/2/3; occupancy recall 0.98/0.98/1.00, precision
  0.89/0.68/1.00; paper-1 C-SE CNN zero-shot SI-SDRi 3.49 dB at K=2.
- E5 enlarged (SERVER: eval_joint_k2.py --variants V4B --n_per_cell 100
  --snr_points 0 10 20 --test_seed 99999 --out results/e5_modsel_large.json):
  pair acc 0.11/0.53/0.40 (binomial CI +-0.06/0.10/0.10), e2e SER
  0.987/0.610/0.762 (pooled 0.786 vs 0.483 V4 same cells), conditional
  0.880/0.264/0.404; per-pair: PSK-only 0.86-0.88 at 10 dB,
  16QAM-involving 0.11-0.78; density drift toward 16QAM at 20 dB.
- Robustness sweeps (SERVER: eval_robustness_sweeps.py, sequential;
  infra: signal_utils.py freq_jitter param (RNG-preserving),
  joint_sync_detect.py mixture_symbols tau_samples param (tau=0
  bit-identical, asserted); local and server amp runs bit-identical):
  freq |Df| in {0,0.5,1,2,4,8} Hz: SER flat; f-RMSE elevated at
  co-frequency; V4-V3 gap 0.047 at Df=0 (20 dB) -> -0.03 at 8 Hz.
  tau in {0,1.6,4,8} samples: 0.442/0.131/0.097 -> 0.686/0.679/0.678
  (0/10/20 dB V4) — graceful to 0.1 Ts, collapse at 0.25 Ts.
  rho in {-10..10} dB: graceful within +-5 dB, weak-source-limited
  0.43-0.48 at +-10 dB.  Figure: make_robustness_fig.py ->
  figures/fig_e6_robustness.pdf (+ copy in paper6/figures/).

### Page-budget measures for the 13-page target (this revision)
fig ser_snr REMOVED from the paper (numbers all in text; figure file
kept in repo); fig ser_sir removed (superseded by the E6 amplitude
sweep); crb figure 2.6->2.35 in; dmin 3.4->3.0 in; robust 2.35->2.05
in; tab:e5 ablations folded into text; tab:e1/tab:e3 single-column;
Algorithm 1 moved to Appendix B as a compact paragraph; appendix
\scriptsize; \linespread 0.96; float seps 7pt; ~120 lines of prose
tightened; 4 marginal bibliography entries dropped (lo2019metricgan,
ochiai2017multichannel, su2017underdetermined, andrews2005interference,
hyvarinen2000ica).  NOTE: paper6/figures/ must be refreshed from
paper6_sync_jd/figures/ after any figure regeneration (the build reads
paper6/figures/) — a stale copy once broke the build silently.

## 2026-10-04: E-A two-tone CRB validation (review 3)

Direct experimental closure of Prop 3 (two-tone CRB inflation,
carrier-only model): measured estimator error vs the theoretical
two-tone CRB on the SAME pure-tone model, K=2.

- NEW eval_twotone_crb.py (local CPU, ~3 min full grid).  Signal:
  r_n = a1 e^{j(2 pi f1 n Ts + phi1)} + a2 e^{j(2 pi f2 n Ts + phi2)} + w_n,
  N = 4096, fs = 16000 Hz (T_burst = 0.256 s, 1/T = 3.906 Hz), f1 = 0,
  f2 = Df WLOG, phi ~ U[0,2pi), SNR per strong tone
  (sigma2 = a1^2/10^(snr/10), the crb_single_tone_hz2 convention),
  amplitude ratio rho = 20log10(a1/a2).
- Estimator: minimal two-tone spectral-line estimator implemented in the
  script (BlindCarrierSync's public API is strictly single-carrier — no
  K=2 mode).  Reuses sync._fft_line (same x8 zero-padded FFT peak +
  parabolic interpolation as BlindCarrierSync step 2, M=1 since the
  tones are already spectral lines) + projection subtraction with a
  masked second-peak pick (exclusion +-0.6/T) + 2-sweep Gauss-Seidel
  golden-section periodogram (single-tone ML) refinement.  No learning,
  no oracle.  Deterministic per-burst seeding (SEED_BASE 94000).
- Theory overlay: exact numeric Fisher via theory_validation._fisher_tones
  with congruence amplitude scaling J -> S J S / sigma2; reduces to
  two_tone_inflation x crb_single_tone_closed at rho=0 dB (asserted,
  rel. err < 1e-9).  sync.py / theory_validation.py unchanged.
- Commands:
    python eval_twotone_crb.py --smoke --out results/eval_twotone_crb_smoke.json \
        --fig figures/fig_twotone_crb_smoke.pdf   # 3 Df x 3 SNR x 2 rho x 20 MC
    python eval_twotone_crb.py                    # full: 10 Df x 3 SNR x 2 rho x 200 MC
  Grid: |Df| = geomspace(0.25, 16, 10) Hz (paper Fig. 3(b) range,
  bracketing 1/T), SNR in {0,10,20} dB, rho in {0,6} dB.
  Outputs: results/eval_twotone_crb.json (per-tone bias/var/RMSE, pooled
  RMSE, P(|err|>1 Hz), per-tone CRB stds),
  figures/fig_twotone_crb.pdf/.png (2x2: (a) measured pooled RMSE vs
  sqrt(CRB), (b) per-tone bias symlog; columns = amplitude ratio).
- Headline numbers (200 MC/cell):
  * Large separation (|Df| >= 6.35 Hz ~ 1.6/T): estimator is essentially
    EFFICIENT — pooled RMSE/sqrt(CRB) = 1.0-1.3x at all SNRs and both
    amplitude ratios (e.g. 16 Hz, 20 dB, rho 0: 0.0025 vs 0.0024 Hz).
  * |Df| = 1 Hz, rho = 0 dB: measured RMSE vs sqrt(CRB) =
    1.387/0.0423 Hz (32.8x) at 0 dB, 1.179/0.0134 (88.1x) at 10 dB,
    1.202/0.0042 (283.9x) at 20 dB; P(fail > 1 Hz) = 0.95-0.98 —
    the RMSE plateaus at ~1 Hz SNR-INDEPENDENTLY (acquisition/capture
    failure: merged main lobes, estimates repel: bias1 ~ -0.7..-0.8 Hz,
    bias2 ~ +0.6 Hz at rho 0).
  * |Df| = 4 Hz (~1/T): still 20x/60x/170x at 0/10/20 dB (rho 0);
    the crossover to CRB-efficiency sits between 1/T and 1.6/T.
  * Amplitude ratio 6 dB barely changes the strong tone (bias1 shrinks:
    less leakage pull) but the weak tone fails harder (bias2 +1.2-1.6 Hz
    at 1 Hz separation); the CRB gap between tones (6 dB) is only
    realised at |Df| >= 6.35 Hz.
- Interpretation for the paper: the measured spectral-line estimator
  CONFIRMS Prop 3's qualitative collapse — below ~1/T the measured error
  saturates at O(Df)-scale regardless of SNR, so the SNR-flat receiver
  penalty at co-frequency (E6 freq sweep) is a frequency-acquisition
  effect, not a detection effect.  The measured-vs-bound gap at small
  Df is the classic threshold effect: the CRB remains a local bound
  (ML would need joint 2-D fit), the practical estimator hits it only
  beyond ~1.6/T.

## 2026-10-04: E-B E4 BER + genie-free differential (review 3)

NEW eval_e4_ber_diff.py (local CPU; `__main__` = self-test + tiny smoke).
Two parts, both requested by review 3.

### (a) Gray-mapped BER from the CACHED E4 decisions (no re-run)
Every E4 scoring path already records ser_comp.GRAY_BITS BER from the SAME
hard decisions as the SER; this part re-aggregates the caches
(e4_joint_k2_mse{,_ts*}.json = V1/V3/V4 over 5 test grids;
e4_joint_k2_slots.json = V0/V2 x 5 ckpt seeds; psp_baseline.json =
sym/psp x 5 ckpt seeds).  Aggregation reproduces every published pooled
SER exactly (V4 0.5310, psp 0.5337, sym 0.5527, V3 0.4106, V1 0.5392,
V0 0.6078, V2 0.5960 — sanity anchor).

Pooled Gray BER (PSK-only / 16QAM-involving split in results/e4_ber_diff.json,
plus a full 10-pair table):

| route | pooled SER | pooled BER | PSK-only BER | 16QAM-inv BER |
| V4    | 0.5310±.0036 | 0.3060±.0019 | 0.2358 | 0.3955 |
| V1    | 0.5392±.0036 | 0.3116±.0020 | 0.2442 | 0.3976 |
| V3    | 0.4106±.0045 | 0.2149±.0022 | 0.1500 | 0.2976 |
| psp   | 0.5337±.0009 | 0.2660±.0007 | 0.2501 | 0.2851 |
| sym   | 0.5527±.0008 | 0.2849±.0006 | 0.2730 | 0.2993 |
| V2    | 0.5960±.0013 | 0.3538±.0008 | 0.3128 | 0.4034 |
| V0    | 0.6078±.0007 | 0.3530±.0006 | 0.3247 | 0.3873 |

NOTE the BER ranking differs from SER: psp BEATS V4 on pooled BER
(0.2660 vs 0.3060) — 16QAM pairs carry 4 bits/symbol and V4's extra
symbol errors there are multi-bit; on PSK-only BER V4 still leads
(0.2358 vs 0.2501).  Both directions must be reported honestly.

### (b) Genie-free differential decoding (differential-equivalent scoring)
E1-D convention (eval_diff_coding.py) adapted to the E4 hard decision
streams: d_n = c[dec[n]] conj(c[dec[n-1]]) vs the reference differential
stream, canonically labeled by the nearest M-th root of unity; a per-burst
M-fold rotation cancels EXACTLY in the product (self-test asserts ser=0
under all k rotations; parity with E1-D's continuous-stream diff_ser
asserted; chance level asserted).  DIFF BER uses the same GRAY_BITS table
on the differential index.  16QAM-involving pairs excluded (no standard
square-QAM differential coding — E1-D convention); PIT over assignments
kept.  Detectors RE-RUN (decisions are not cached); --n_subset takes the
FIRST n bursts per SNR block of the SAME deterministic n_per_cell=100,
seed-99999 grid, so subsets are strict subsets of the published runs.
Cross-check of the re-run ABS scores vs the cached records: waveform
route EXACT (0.00000, all 5 seeds x 70 bursts); V1/V4 mean |delta|
0.0007-0.0009 with 3/70 bursts >0.01 (EM path sensitivity under BLAS
threading — the known <=0.0006 pooled effect).

Subset run (n_subset=10/SNR = 70 K=2 bursts, 41 PSK-only; waveform route
5 ckpt seeds -> 205 PSK bursts), ~4 min local CPU.  PAIRED on the same
PSK-only bursts (ABS* = genie-rotated, DIFF = genie-free):

| route | ABS* SER | DIFF SER | ABS* BER | DIFF BER |
| V4    | 0.3742 | 0.3763 | 0.2365 | 0.2515 |
| V1    | 0.3882 | 0.3975 | 0.2472 | 0.2668 |
| psp   | 0.4235 | 0.5190 | 0.2434 | 0.3489 |
| sym   | 0.4458 | 0.5347 | 0.2656 | 0.3651 |

Headline for the response letter: going genie-free costs V4 almost
nothing (+0.2 pp SER / +1.5 pp BER) but costs the strongest oracle-
assisted waveform route +9.6 pp SER / +10.6 pp BER (the PSP's decisions
rely on absolute phase over the ISI trellis; differentially, burst-level
phase drift and decision-run structure hurt it).  Under the deployment
metric V4 BEATS the PSP route by 14 pts on PSK pairs (0.376 vs 0.519) —
the review-2 "statistical tie" becomes a clear V4 win.  Per-SNR: V4's
DIFF<=ABS* from 0 dB up (drift cancellation), pays +2..+5 pp at -10/-5 dB.
SUBSET numbers (n=41 PSK bursts for V1/V4) — full-grid confirmation
pending on the server:
  python eval_e4_ber_diff.py --mode all --n_subset 100 \
      --out results/e4_ber_diff.json        # ~35 min ECM + ~7 min waveform
Outputs: results/e4_ber_diff.json (ber_cached + diff sections),
results/e4_ber_diff_smoke.json (smoke), logs_e4_ber_diff.log.
No existing script modified; main.tex / paper6/ untouched.
  Addendum: compact 1x3 textwidth variant for the page budget —
  eval_twotone_crb.py --from_json results/eval_twotone_crb.json --wide
  (new make_figure_wide + --wide/--from_json CLI; MC not re-run) ->
  figures/fig_twotone_crb_wide.pdf/.png: (a) rho=0 RMSE vs sqrt(CRB),
  (b) rho=6 dB, (c) rho=0 per-tone bias.  2x2 fig_twotone_crb.pdf
  unchanged (same code path, same data).

## 2026-10-04: E-C demodulation-aware fine-tuning control (review 3)

Reviewer-3 question: paper 6 claims "the remedy is structural, not a loss
weight" (the E4 waveform-route detection floor survives +8 dB SI-SDRi), but
a demodulation-AWARE loss was never run as a control.  E-C adds exactly
that control on the unchanged benchmark.

### Design
- **Vendored** `soft_demod.py` from `paper5_task_oriented/soft_demod.py`
  (2026-10-04; paper6 self-contained).  Minimal adaptation only: the
  paper1 sys.path trick -> vendored `signal_utils.rrc_filter`; the
  `__main__` smoke imports rewired to `signal_utils` / `data_generator` /
  `ser_comp`.  Numerical core UNCHANGED (the smoke test asserts
  torch==numpy decisions against the vendored `ser_comp._demod_pair_labels`
  symbol-for-symbol on clean sources — passed locally).
- **`train_finetune_demod.py`** (new): loads the E3/E4 baseline checkpoints
  `checkpoints/slot_h64_l4_k13_bs16_lr0.001_mse_s{seed}_best.pt` and
  fine-tunes with the baseline recipe PLUS the paper-5 soft-SER term:
  `losses_baseline.variable_k_pit_loss` (lambda_sep=1.0, lambda_mse=1.0,
  lambda_occ=1.0, lambda_cnt=0.1 — unchanged) `+ lambda_ser *
  soft_demod.soft_ser_pairs` on the SAME PIT-assigned pairs (reuses
  `aux['assigned']`, no second assignment enumeration — the paper-5
  losses.py construction).  Same architecture/parameter budget (254K),
  same generator and RNG convention (`CommBSSVarKDataset` with
  `return_carriers=True` — the mixture stream is identical to baseline
  training; the flag only exposes the true carriers).  Hyperparameters
  follow the task brief / paper-5 usage: `--epochs 15 --lr 1e-4
  --ser_sigma2 0.1`, lambda_ser swept over {0.1, 1.0}, seeds 42-46.
  Checkpoint selection stays by val SI-SDR (lambda_sep=1.0 — this is the
  paper-5 config-(d) regime, NOT the reward-hacked pure-task config (e)).
  Output ckpts: `slot_h64_l4_k13_bs16_lr0.0001_mse_ftser{lser}_s{seed}_*.pt`.
  `losses_baseline.py` untouched (its lambda_ser guard stays); no existing
  script's default behavior changed.
- **`eval_finetune_demod.py`** (new): both arms (baseline ckpts vs
  fine-tuned ckpts) through the SAME K=2 separate -> oracle-sync -> detect
  pipeline as the E4 waveform route (`eval_psp_baseline.py`): occupancy
  top-2 slots -> Hungarian match on SI-SDR -> oracle front-end
  (true-carrier down-conversion -> RRC MF -> 0::sps grid -> unit power,
  i.e. the `ser_comp` compensated receiver) -> `sym` and `psp` detection ->
  `eval_joint_k2.score_decisions` (PIT + M-fold rotation genie) ->
  compensated SER + Gray BER from the same decisions
  (`ser_comp.GRAY_BITS`).  Deterministic test grid seed 99999, K=2 cells,
  n_per_cell=100 (same as E3/E4).  Per-pair SI-SDR/SI-SDRi of the matched
  slots recorded alongside SER/BER.  Output: `results/finetune_demod.json`
  (per-seed records + seed-averaged summary per arm x variant).

### Local smoke (Mac, CPU)
- `python soft_demod.py` — PASSED (constellation parity vs
  signal_utils.CONSTELLATIONS, matched-filter rel err 3.5e-07 vs
  np.convolve, Gray 1-bit property, identity CE < 0.5 with >=4x margin and
  torch==numpy decisions on 6 clean sources, backward finite).
- `python train_finetune_demod.py --smoke --lambda_ser 1.0 --batch_size 8
  --name smoke` — PASSED: loaded baseline s42 (epoch 93, best_val
  5.330 dB), 1 epoch / 64 samples: TRAIN loss=-1.616 si_sdr=5.601 dB
  ser_soft=3.406 (17.9 s), VAL si_sdr=2.449 dB ser_soft=3.834
  count_acc=0.562; checkpoint saved.  (Smoke ckpt deleted after use.)
- `python eval_finetune_demod.py --n_per_cell 1 --snr_points 10 --seeds 42
  --lser_values 1.0 --variants sym|psp` (temp ckpt dir mapping the smoke
  ckpt onto the canonical name) — PASSED for both variants; both arms
  evaluated, `results/finetune_demod_smoke.json` written (1 burst, sanity
  only: baseline sym SER 0.570 vs 1-epoch-ft 0.565 — meaningless at n=1,
  chain verification only).

### Server run
Connection `ssh -p 3009 vr301@172.16.65.85` FAILED at the SSH banner
(`kex_exchange_identification: Connection closed by remote host`) on all
attempts from 2026-10-04 (host pings, port 3009 accepts TCP; sshd drops
the session — same failure mode as the 2026-09-18 outage, which recovered
later that day).  **RESOLVED 2026-10-05 09:23 (server back; jobs launched
via cron-triggered retry — see the 2026-10-05 completion entry at the
end of this log for final numbers).**

Prepared commands (run when the server is back):
```
rsync -avz -e "ssh -p 3009" \
  paper6_sync_jd/{soft_demod.py,train_finetune_demod.py,eval_finetune_demod.py,run_finetune_demod.sh} \
  vr301@172.16.65.85:/data/experiment/paper6_sync_jd/
ssh -p 3009 vr301@172.16.65.85 \
  'cd /data/experiment/paper6_sync_jd && nohup bash run_finetune_demod.sh \
   > finetune_demod_20261004.log 2>&1 < /dev/null & echo $!'
```
Driver `run_finetune_demod.sh`: 10 fine-tune runs (5 seeds x lambda_ser
{0.1, 1.0}; 15 epochs, lr 1e-4, 2000 train samples) then
`eval_finetune_demod.py --n_per_cell 100 --seeds 42 43 44 45 46
--lser_values 0.1 1.0 --variants sym psp`.  Log:
`/data/experiment/paper6_sync_jd/finetune_demod_20261004.log`.
No existing files modified; main.tex / paper6/ / paper5_task_oriented/
untouched.

## 2026-10-04 — Prop 2 identifiability certificate (review 3)

### Files / command
- `theory_identifiability.py` (NEW; numpy-only, no torch/scipy dependency;
  constellation tables duplicated from signal_utils.CONSTELLATIONS with a
  parity assert) — constructive numerical certificate for Proposition
  prop:discrete (generic identifiability of the effective coefficients up
  to Z_{M_1} x Z_{M_2}, +S_2 for equal alphabets).
- Command: `cd paper6_sync_jd && ../.venv_verify/bin/python
  theory_identifiability.py` (n_trials=200/pair, seed 20261004; local CPU
  ~4 s) -> `results/theory_identifiability.json`.

### Method (mirrors the proof step by step; all moments EXACT by
### product-alphabet enumeration, no Monte Carlo)
- Step 1 magnitudes: p = E|s|^2, q = E|s|^4 quadratic (mu_2 = 0 pairs);
  E[s^2] = a_B^2 (BPSK + mu_2=0 partner); q - |E s^2|^2 = 4xy
  (BPSK+BPSK).  Step 2 phases: nested-order separation via
  E[s^{M_min}]/E[s^{M_max}] (pairs (2,4),(2,8),(4,8)); equal orders: one
  quadratic in u2 from E[s^M] = u1+u2 and E[s^{2M}] = alpha u1^2 +
  binom(2M,M) u1 u2 + beta u2^2; BPSK+BPSK: triangle + E[s^4] mirror
  disambiguation.  Per-trial residuals of every cross-term-vanishing
  identity are recorded (max over the sweep: 2.1e-13).
- Filters applied to every solver candidate, rejections counted:
  F1 modulus constraint vs Step-1; F2 all raw moments m<=24 + |s|^2/4/6
  (each vs its own term scale); F3 exact mixture-multiset equality
  (support + multiplicities).  Success = every F3 survivor
  group-equivalent AND the truth's class recovered.  Generic sampling:
  |a1|~U(0.7,1.3), ratio~U(0.5,1.5) with |ratio-1|<0.02 ties excluded
  (counted), phases uniform.

### Results (all in results/theory_identifiability.json)
- ALL 10 unordered pairs: 200/200 each, success rate 1.000
  (BPSK+BPSK, BPSK+QPSK, BPSK+8PSK, BPSK+16QAM, QPSK+QPSK, QPSK+8PSK,
  QPSK+16QAM, 8PSK+8PSK, 8PSK+16QAM, 16QAM+16QAM); tie draws excluded
  5-10/pair; zero solver-degenerate trials.
- Modulus-rejection check (QPSK+16QAM, the equal-order unequal-alphabet
  case): 200/200 spurious quadratic roots rejected by F1; in ZERO trials
  did both roots satisfy the modulus constraint — the proof's rejection
  step works exactly as stated (no paper-wording caveat needed).
- Control (a) collisions (QPSK+QPSK): (1, e^{j90deg}) equal power: 16
  pairs -> 9 distinct mixture points, d_min = 0; excluded by the tie
  rule; the certificate still returns the right class but through a
  DEGENERATE double root, with measured sqrt(eps) ill-conditioning
  (median class error 2.2e-6/2.2e-3 at moment-perturbation eps =
  1e-12/1e-6 vs 2.5e-12/2.5e-6 = linear at a generic point).
  (1, sqrt(2) e^{j45deg}): d_min = 0 (12/16 distinct points) but NOT
  tie-excluded; the certificate succeeds and stays well-conditioned —
  identifiability SURVIVES this collision; the failure is symbol-level.
  2-D pmf search (r' x phi', 106x720 grid, MMD^2 over the multiset):
  exact matches (MMD^2 < 1e-8) only on the group orbit (4/76320 grid
  points at the equal-power collision), i.e. no non-equivalent
  parameter pair with an identical mixture pmf found.
- Control (b) continuous alphabet (c_k ~ CN(0,1)): E[s^m] = 0 for all
  m >= 1 and q = 2p^2 — 40 random phase-rotated / power-reallocated
  candidates match all moments to 2.7e-15 while none is group-related
  (continuous gauge freedom, Prop 1); MC sanity (2e5 samples) confirms
  rotated candidate indistinguishable within 1 SE.

### Note for the paper wording
At the tested collision configurations the coefficient equivalence class
remains determined by the mixture pmf (no non-equivalent exact solutions
found); the collision caveat of prop:discrete manifests (i) at the symbol
level as the non-injective hypothesis map (d_min = 0, support collapse)
and (ii) at the coefficient level as sqrt(eps) ILL-CONDITIONING at the
equal-power zero contacts.  This is consistent with the proposition's
"generic, finite-symmetry" wording as written — no text change required.

## 2026-10-05: E-B full run + E-C full run COMPLETE (review 3)

Server SSH recovered 2026-10-05 09:23 (user intervention); both jobs
launched from the cron-armed retry plan.

### E-B full genie-free differential scoring — COMPLETE
```
cd /data/experiment/paper6_sync_jd && nice -n 10 \
  CUDA_VISIBLE_DEVICES="" /data/experiment/venv_bss/bin/python \
  eval_e4_ber_diff.py --mode all --n_subset 100 --out results/e4_ber_diff.json
```
(First GPU attempt died to CUDA OOM — another tenant held 4.2 GiB and
E-C training held 2.3 GiB; E-B is ECM/CPU-bound so it was rerun with the
GPU hidden.  Runtime 1180 s + waveform-route scoring.)
FULL grid (reference grid, seed 99999, 700 bursts; differential on the
n=383 PSK-pair bursts, PAIRED with the genie-resolved scores;
results/e4_ber_diff.json, `diff.config.subset=false`):

| route | ABS SER (PSK) | DIFF SER | ABS BER (PSK) | DIFF BER |
| V4  | 0.3807 | 0.3722 | 0.2385 | 0.2437 |
| V1  | 0.3927 | 0.3966 | 0.2467 | 0.2609 |
| psp | 0.4396 | 0.5374 | 0.2501 | 0.3568 |
| sym | 0.4620 | 0.5566 | 0.2730 | 0.3765 |

Full-grid confirms the subset finding, stronger: V4 is FLAT under
genie-free differential scoring (−0.9 pp SER / +0.5 pp BER), the PSP
route pays +9.8 pp SER / +10.7 pp BER.  The review-2 "tie" becomes a
clear V4 win in deployment terms (0.372 vs 0.537).  Integrated into
main.tex (E4 Results after the BER sentence; Limitations (i)).

### E-C demodulation-aware fine-tuning — COMPLETE
```
cd /data/experiment/paper6_sync_jd && bash run_finetune_demod.sh \
  > finetune_demod_20261005.log 2>&1
```
(First launch hit a CPU/CUDA indexing bug at train_finetune_demod.py
soft_ser_term — carriers/mods are CPU tensors by design and were indexed
by a CUDA index tensor; fixed with idx.cpu() before indexing; the Mac
CPU smoke could not see it.  10 fine-tunes, 5 seeds x lambda_ser
{0.1,1.0}, 15 epochs lr 1e-4, ~8 min each; then
eval_finetune_demod.py on the E4 waveform-route pipeline, K=2 test seed
99999, n=100/SNR, both sym and psp variants.)

Pooled results (results/finetune_demod.json, summary section;
5-seed mean ± std):

| arm | route | SER | BER | SI-SDRi (dB) |
| baseline   | sym | 0.5527±.0008 | 0.2849±.0006 | 3.170±.052 |
| baseline   | psp | 0.5337±.0009 | 0.2660±.0007 | 3.170±.052 |
| ft_lser0.1 | sym | 0.5526±.0009 | 0.2847±.0006 | 3.173±.039 |
| ft_lser0.1 | psp | 0.5334±.0010 | 0.2656±.0007 | 3.173±.039 |
| ft_lser1   | sym | 0.5507±.0012 | 0.2830±.0008 | 2.989±.060 |
| ft_lser1   | psp | 0.5325±.0010 | 0.2666±.0007 | 2.989±.060 |

Baseline psp row reproduces the published E4 number (0.5337) exactly —
pipeline cross-check.  Demodulation-aware fine-tuning moves the
compensated SER by at most 0.12 pp (0.5337 → 0.5334 / 0.5325) with BER
and SI-SDRi unchanged within seed noise: on this benchmark the
separate-then-detect floor of Thm 2 is NOT reachable by re-weighting
the separator's loss — the review-3 "Structure, Not Loss" objection is
now answered by a controlled experiment, not just scoping.  The
"Fine-tuning scoped but not run" paragraph in main.tex is replaced by
this result.

## 2026-10-05: Review 4 (main(8).pdf) — text revisions + CNSE strong baseline

Review file: `paper6/review4.md` (Major Revision, 7.7/10, "close to
conditionally acceptable"). Response: `paper6/response_to_reviewers4.md`.

### Text revisions to main.tex (DONE; 13 pp, 0 `??`, no overfull)
1. **P0-A Prop 2 Step 2** (Appendix A): "positive solutions are the unordered
   pair" was wrong for κ1≠κ2 (two distinct positive roots possible). Step 2
   rewritten as "amplitudes: a finite candidate set" (one quadratic in x ⇒ at
   most two positive unordered solutions; spurious candidate rejected in Step 3
   by substitution into a further known moment, e.g. E|s|^6 — an extra algebraic
   equation folded into Step 4's exceptional set). The invalid "spurious root
   generically rejected by Step 2's moduli" sentence removed.
2. **P0-B finite-N**: "Finite-N identifiability follows from the consistency of
   empirical moments" REMOVED (consistency ≠ finite-sample identifiability).
   Replaced by: "The proposition concerns the population mixture law; for finite
   bursts the empirical-moment inversion is consistent as N→∞, not exact."
   Prop 2 statement now reads "identifiable from the population mixture law up
   to G".
3. **Loss wording**: "re-weighting the separator's loss does not reach the
   floor...; the remedy remains structural" → "the tested loss-level
   modification does not remove the floor ...: within the tested
   waveform-separator objective family, the remedy remains structural."
4. **"disambiguating information" wording** (Intro, contributions bullet,
   Conclusion): now "carried by the modulation-bearing waveform, but exploiting
   it requires symbol-aware or decision-directed processing" — matches what the
   known-symbols Fisher analysis actually proves.
5. **E4 statistical reporting** (P1-D): paragraph now states exact units —
   7 SNR × 100 bursts = 700 per grid; V1–V4 contrast pools 3500 paired bursts
   (n=3500 inline); headline waveform contrast = 700 paired bursts of the
   reference grid, route arm averaged over 5 checkpoints/burst. CI vs Wilcoxon
   coexistence explained as different estimands (bootstrap CI on the MEAN paired
   difference vs Wilcoxon on symmetry about zero). Verified against
   results/e4_paired_stats.json: headline n=700 (mean +0.0032, CI
   [-0.0112,+0.0178], p=4.77e-4); V1−V4 n=3500 (+0.0082 [0.0074,0.0089],
   p=1.3e-90). "700 shared cells" phrasing replaced by "n=700 paired bursts".
6. Page budget: additions absorbed by \linespread 0.925 → 0.912; build stable at
   13 pages.

### CNSE strong baseline (review-4 P1-C) — oracle route COMPLETE, blind route running
```
# server: cd /data/experiment/paper6_sync_jd
for s in 42 43 44 45 46; do /data/experiment/venv_bss/bin/python \
  train_cnse_baseline.py --batch_size 8 --seed $s > cnse_train_s$s.log 2>&1; done
/data/experiment/venv_bss/bin/python eval_cnse_baseline.py --n_per_cell 100 \
  > cnse_eval.log 2>&1
```
New files: `cnse.py` (verbatim vendored CNSE from paper1_cnn_se/models.py:600-735;
6,693,509 params = 26x SlotSepNet), `train_cnse_baseline.py` (K=2-only training,
2-source PIT SI-SDR + MSE anchor; recipe of train_baseline.py except
**batch_size 8** — only 2539 MiB GPU free at launch, shared tenant; no OOM),
`eval_cnse_baseline.py` (mirrors eval_psp_baseline.py exactly: same 700
reference-grid K=2 cells, oracle front-end, sym/psp arms, score_decisions;
adds per-cell sisdr1/2 + sisdr_mix1/2).

Training (best val SI-SDR): s42 3.462 / s43 3.173 / s44 3.428 (96 ep early-stop)
/ s45 2.201 (55 ep) / s46 2.805 dB; ~18 min/seed, total ~80 min.

Eval (results/cnse_baseline.json, 7000 records = 700 cells x 5 seeds):
- **SI-SDRi @ K=2 = 6.013 dB** (SI-SDR 1.723 vs mixture -4.290 dB; per-SNR
  9.66/8.22/5.91/5.15/3.91/4.83/4.42 at -10..20 dB) vs SlotSepNet 3.2 dB.
- CNSE + oracle sync + PSP: per-SNR SER 0.5801/0.5077/0.4882/0.3718/0.4098/
  0.3775/0.3729, **pooled SER 0.4440, pooled BER 0.2130** (sym arm 0.4568/0.2230).
  PSK-only psp: SER 0.3374, BER 0.1852 (n=1915). Per-seed psp SER 0.4384..0.4561.
- **Narrative consequence**: the oracle-assisted CNSE route (0.4440) beats blind
  V4 (0.5310) pooled AND on PSK-only (0.337 vs 0.378 SER) — the "V4 ties the
  strongest waveform route" headline must be rescoped to the paper's separator
  quality. Surviving claims: (i) at matched oracle frequency V3 (0.4106) still
  beats CNSE-psp (0.4440) — architecture comparison separator-independent;
  (ii) the waveform route is separator-quality-limited (3.2 dB -> 0.534,
  6.0 dB -> 0.444), confirming Thm 2's floor is a property of the
  linear-residual regime it models; (iii) blind-vs-blind pending.

### CNSE blind route — COMPLETE (eval_cnse_blind.py)
```
/data/experiment/venv_bss/bin/python eval_cnse_blind.py --n_per_cell 100 \
  > cnse_blind_eval.log 2>&1   # ~2.5 min total on server (27 s/seed)
```
New file `eval_cnse_blind.py`: same 700 K=2 cells / checkpoints / matching as
eval_cnse_baseline.py; 4 variants (oracle_sym, oracle_psp, blind_sym,
blind_psp) x 2 scorings (ABS via score_decisions; DIFF genie-free via
eval_e4_ber_diff.score_diff_pair, PSK pairs n=1915). blind arms use
sync.blind_sync_known_mod (returns z symbol stream -> PSP-able, no deviation).
Oracle arms reproduce cnse_baseline.json EXACTLY (0.4568/0.4440 SER,
0.2230/0.2130 BER) — pipeline identity confirmed.
Results (results/cnse_blind.json, 14000 records):
- blind_sym 0.5484/0.3103, **blind_psp ABS 0.5356/0.2999** (per-seed
  0.5300..0.5480) — **+9.2 pts over oracle_psp 0.4440**, inside E3's +8-11
  capture band. Blind-arm freq error: median 1.23 Hz pooled; fraction >1 Hz
  0.355/0.453/0.402/0.380 at SNR 5/10/15/20 dB — capture persists despite
  6 dB cleaner slots. E3 failure confirmed CROSS-ARCHITECTURE.
- **Blind-vs-blind: V4 0.5310 ties blind CNSE-psp 0.5356** — the oracle-CNSE
  advantage was entirely the oracle carrier.
- DIFF (genie-free, PSK pairs): V4 0.372/0.244 leads every waveform route —
  CNSE oracle_psp 0.4380/0.2826, CNSE blind_psp 0.4415/0.2856,
  SlotSepNet-psp 0.537/0.357.

### Integration into main.tex (DONE; 13 pp, 0 undefined refs, 0 overfull)
- Headline rescope everywhere ("strongest waveform route we can build" ->
  "strongest waveform route at the paper's separator quality"): abstract, intro
  contributions bullet, E4 Results headline, Discussion, Conclusion.
- Abstract/intro/conclusion carry the CNSE summary: oracle route 0.444, blind
  0.536, V3 0.411 vs 0.444 at matched oracle, genie-free 0.372 vs 0.438.
- E4 gains "Cross-architecture check (CNSE)" paragraph (9-pt route improvement
  0.5337->0.4440 = separator-quality-limited; three qualifications (i) V3 wins
  matched-oracle, (ii) blind CNSE +9.2 pts inside E3 band, capture 36-45% at
  >=5 dB, blind tie 0.5310 vs 0.5356, (iii) genie-free V4 leads all routes).
- Fine-tune control paragraph: "strongest waveform route" -> "the E4 waveform
  route" + pointer to the CNSE cross-check.
- Page budget: \linespread 0.905 -> 0.885; double-column figures shrunk
  (twotone 0.78->0.68, e2_crb 0.66->0.57, e6 0.66->0.57 \textwidth);
  Implementation-remarks and E5 selector sentences trimmed; bib left at
  scriptsize (a baselineskip hack was tried and reverted).
