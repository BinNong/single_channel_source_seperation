# Review 5 — paper6 / main.pdf (13 pp.), 2026-10-06

Independent fifth-round review. **Every quantitative claim below was checked against
the shipped code, `results/*.json`, `*.log` and `EXPERIMENT_LOG.md` in
`paper6_sync_jd/`**; each verdict states whether it is supported, contradicted, or
unverifiable. Line references are to `paper6/main.tex`.

---

## 0. Summary and recommendation

**Overall recommendation: Major Revision** (≈7.5/10)

This is an honest, well-organised manuscript. Its core message — waveform fidelity,
synchronisation quality and detection quality are three *different* statistical
objects — is memorable, and the E3→E4 ladder V3 < V4 < V1 < V2 < V0 is the paper's
best asset: it is an *architecture diagnosis*, not a "our method works" report. The
E5 negative results are worded with the right maturity, and the theory checks out —
I re-derived Theorem 1's union bound, Theorem 2's per-axis exact expression and its
QPSK floor of 1/2, the 7/16 zero-contact floor, the 16QAM 0.8380, every CRB statistic
(1.70×/7.6×/4.6×/287×, E|Δf| = 89/24, P(|Δf|<1/T) = 0.5871), and the d_min landscape
(0.39/0.16 d_su, 30 %) — all correct and all matching `e2_theory_validation.json`.

What blocks acceptance is a cluster of defects, none of which overturns the thesis but
all of which need one revision round:

| # | Issue | Severity |
|---|---|---|
| M1 | Abstract is ~394 words against a hard 150–250-word IEEE limit, and uses 11 unexpanded abbreviations | **Editorial / blocking** |
| M2 | The headline algorithm (V1/V4) has no Method section; the Method section describes the architecture the paper's own experiments reject | Major (structure) |
| M3 | Table E4 silently mixes two burst populations; "wins at SNR ≥ 5 dB" reads as false from the printed cells | Major (claim) |
| M4 | The genie-free differential score measures a *phase-slip rate*, not a differential-receiver SER; it is then compared against absolute SERs | Major (interpretation) |
| M5 | "Sits near the exact separate-detection floor" is not established for the pooled number, and Theorem 2 does not cover half the test grid | Major (claim) |
| M6 | Reproducibility gaps: Table I unverifiable; three load-bearing numbers (0.041, 0.106/0.164, and the whole V2-diagnosis block) have no artifact or have non-running code | Major (reproducibility) |
| M7 | The central theory validation has no figure; two generated figures are unused | Minor |
| M8 | The identifiability certification excludes the benchmark's own SIR ≈ 0 dB operating point | Minor (rigour) |
| M9 | Statistical reporting asymmetry for the headline contrast | Minor (reporting) |

---

## 1. Major comments

### M1. The abstract violates two explicit IEEE requirements

From the IEEE SPS *Information for Authors* page:

> "The abstract must be self-contained, **without abbreviations**, footnotes, displayed
> equations, or references.
> The abstract **must be between 150-250 words**."

The submitted abstract measures **~394 words** (57 % over the cap) and contains 11
undefined abbreviations: SC-BSS, SER, SNR, SIR, MAP, CRB, BPSK, QPSK, ISI, ECM, PSP.
This is the kind of defect an editor returns without review.

Related, and it constrains the fix: **the manuscript is exactly at the page limit.**
The initial-submission cap is 13 double-column pages *including* "appendices and
proofs"; the PDF is 13 pages with Appendix A starting on p. 12. There is no slack. The
*revised* limit is 16 pages with appendices counted as supplemental material, so the
correct way to make room for M7 is to move the appendices into supplemental, not to
delete content.

### M2. The headline receiver is not in the Method section

Section V is titled "Method: RS-SlotSepNet" and contains (a) a baseline separator the
text explicitly disowns ("it is not the contribution"), (b) BlindCarrierSync, and
(c) JointPairDetector. The receiver that produces the paper's headline 0.531 — the
ISI-aware ECM joint synchroniser-detector on the raw mixture — is defined only in a
bullet list inside E4 and in Appendix B.

Worse: the architecture Section V *does* describe, "separate → per-slot sync → joint
re-detection", is exactly V0 — the worst row in Table III (0.6078). The Method section
therefore documents the architecture the paper's own experiments refute.

Recommended fix: promote the mixture-route joint synchroniser-detector to the body of
Section V (ISI tap model, coarse-to-fine frequency search, ECM monotonicity and the
fallback test), and demote SlotSepNet to "the diagnosed baseline". BlindCarrierSync
becomes the K = 1 branch.

This reorganisation also surfaces a real logical gap: **how does the receiver choose
between the K = 1 and K = 2 branches?** The separator's count head is 4.2 % accurate at
K = 2 (occupancy counts in `separator_quality.json`), and E5's K = 3 acquisition fails,
so nothing in the paper provides a K-decider — yet K ∈ {1,2} is presented as the
operating envelope. State this explicitly in the Limitations.

### M3. Table E4 silently mixes two burst populations, which makes "wins at SNR ≥ 5 dB" read as false

The caption (l. 1196) discloses that V1/V3/V4 are averages over **five test grids**,
while the reference rows are **reference-grid only** — but the sentence at
ll. 1283-1284 makes a row-wise claim. At 5 dB the printed cells give V4 = 0.510 vs
PSP = 0.493 (V4 *worse* by 1.7 pts), whereas on the shared reference grid V4 = 0.4582
vs 0.4930 (V4 better by 3.5 pts). Likewise "V3 beats the oracle-received PSP route by
12 pooled points (0.4106 vs. 0.5337)" pairs a 5-grid V3 with a ref-grid PSP.

By the paper's **own** tie rule (l. 1268: "A tie is claimed only when the paired
interval contains zero"), 5 dB is a **tie** for the PSP arm: the paired mean is
+0.0348 with CI [−0.0004, +0.0703] and Wilcoxon p = 0.19
(`e4_paired_stats.json` → `psp_minus_V4.per_snr`). Only the *symbol-wise* arm is a
genuine V4 win at 5 dB (+0.0677, CI [+0.0286, +0.1064], p = 0.0057).

Suggested rewording: "at SNR ≥ 10 dB the joint loop wins the PSP arm by +0.07 to
+0.08 pts, ties it at 5 dB, and beats the symbol-wise arm at 5 dB (+0.068)". Present
one same-grid comparison table, or run the route arm on all five grids.

*Correction to an earlier draft of this review:* I initially read the table row-wise
and concluded that "at SNR ≤ 0 dB the oracle-assisted routes win by 4–11 points"
fails at 0 dB. That was wrong — the paired same-cells analysis gives the routes
winning by 11.5 / 9.0 / 4.0 pts at −10 / −5 / 0 dB, which supports the claim. The
row-wise reading is simply not valid for these rows.

### M4. The genie-free differential score is genuine — but it measures the wrong quantity

To be clear about what holds: `diff_labels()` (`eval_e4_ber_diff.py:246-256`) is
textbook conjugate-product decoding, `d_n = ĉ_n·conj(ĉ_{n−1})`, quantised to the M-th
roots of unity. There is **no M-rotation search** anywhere in the scoring path (the
only `range(M)` loop is in the self-test), no true-symbol information fixes the phase,
and the rotation invariance is mathematically exact for BPSK/QPSK/8PSK. The word
"genie-free" is defensible.

The problem is the estimand. Because the product acts on **already-quantised hard
decisions** (unlike E1-D, where it acts on the continuous symbol stream), the
differential label at position n is wrong **iff φ_n ≠ φ_{n−1}**. Hence:

> `diff_ser` ≡ P(φ_n ≠ φ_{n−1}) — a **phase-slip rate**, not a differential-receiver SER.
> ABS SER ≡ P(φ_n ≠ 0).

Reproduced on a real cell (seed 99999, SNR 10 dB, QPSK/8PSK, source 1): 252 of 256 V4
decisions carry a non-zero phase error, the genie-resolved ABS SER is 0.508, yet DIFF is
0.094 because the phase error only *changes* 24 times. A purely drifting decision stream
can score DIFF ≈ 0 with every symbol wrong.

Three consequences:

1. **"0.381 → 0.372" (V4 holds) and "0.440 → 0.537" (PSP degrades) are not on the same
   axis** — a slip rate is being compared against symbol error rates. Differential
   scoring *rewards* routes whose phase error is slowly varying and *punishes* routes
   whose errors are noise-driven; the latter is what good synchronisation looks like.
   The sentence "differential detection tolerates the joint loop's residual acquisition
   error but punishes the waveform route's error structure" (ll. 1294-1295) has the
   causality backwards.
2. **The paper's own deployment-grade control gives the opposite sign.** E1-D
   (`eval_diff_coding.py`, continuous stream, genuine differential detection) at 20 dB
   is ABS 0.0176 → DIFF 0.1196 (8PSK), 0.0096 → 0.0698 (QPSK), 0.0058 → 0.0324 (BPSK) —
   paid identically by the oracle arm. The deployment cost of removing the genie is
   real, and cannot be presented next to "V4 holds".
3. **A differentially scored stream still cannot deliver bits.** The data are not
   differentially encoded, so absolute symbols — and the user's bits — remain
   rotation-ambiguous regardless of how good the differential labels are.

Recommended: report DIFF as a *rotation-invariant / differential-equivalent* score,
state explicitly that it measures the change rate of the per-symbol phase error and is
therefore not comparable to the genie-resolved SER, cite E1-D for the actual
differential-detection penalty, and drop "turns the tie into a clear win". The abstract
and Conclusion both repeat the strongest version of the claim ("under genie-free
differential scoring it leads every waveform route (0.372 vs. 0.438)") — this is the
weakest-supported sentence in the paper and should not be its headline.

Secondary: this experiment's "same decisions pipeline" holds only to EM path
sensitivity — `logs_e4_ber_diff.log` reports cross-check V1 max|ABS − cached| = 0.01367
and V4 = 0.01758, against an expectation of ≤ 0.0006. Also, 383 is the per-seed count;
0.537/0.357 are 5-seed means over n = 1915, which the text ("n=383") does not say.

### M5. "Sits near the exact separate-detection floor" is not established for the pooled number

Theorem 2 is explicitly restricted to "constellations with axis-aligned decision
boundaries (QPSK, square QAM)". The E3/E4 grid also contains BPSK pairs (whose floor is
actually 0 — the interferer offset never exceeds the signal half-distance — and which
Theorem 2 does not cover) and 16QAM pairs (floor 0.838, which is precisely what pulls
the pooled figure up to 0.5534). The PSK-only subset of the *same* measurement is
0.440, i.e. **below** the 1/2 QPSK floor; the CNSE route's 0.444 is also below 1/2.

So 0.5534 is neither "near 1/2" nor the average of any derived floor. Please report the
E3/E4 oracle-arm SER restricted to QPSK+QPSK cells alongside that subset's exact floor,
or restrict the "near the floor" statement to QPSK pairs. The abstract's "waveform-level
fidelity alone cannot remove this interference-induced floor" should read "cannot
*eliminate*" and keep the "within the reachable quality range" qualifier the body
already uses — otherwise it sits in direct tension with the same abstract's report that
a 26× larger separator lifts the route from 0.553 to 0.444 (an 11-point gain from
separator quality alone).

### M6. Reproducibility gaps

The data-availability statement promises "every evaluation script behind Tables
I–III". In fact:

- **Table I cannot be verified.** The caption claims 200 bursts per cell on two grids
  (`probe_blind_sync_k1.py --n_per_cell 400`, seeds 99999 & 31337), but `results/` ships
  only the n=20 and n=100 single-grid pilots, which disagree with the published table at
  roughly 20 of 56 cells (low-count noise; largest gap BPSK oracle @ −5 dB, 0.091 vs
  0.036). The 8PSK SNR ≥ 0 column is independently confirmed via
  `EXPERIMENT_LOG.md:606-608`; the rest is not. Ship the n400 artifacts or align the
  caption with what ships.
- **E3's unconditional accounting** (0.165/0.560/0.586 and 0.218/0.648/0.679) has no
  JSON in `results/`. The numbers reconstruct exactly from
  `eval_e2e_e3.py:92-93` (0.56031 / 0.64768), so they are almost certainly right — but
  the artifact is missing.
- **The true-stream control SER 0.041 (vs 0.121) has no artifact at all.** It appears
  only in `EXPERIMENT_LOG.md:350`; no script implements an ECM-on-true-sources arm and
  no results file contains it. This number is load-bearing — it appears in E4, the
  Discussion *and* the Conclusion as the evidence that "the joint loop would win
  decisively once a separator preserves the linear mixture model faithfully".
- **"0.106 vs 0.164 for the memoryless oracle"** (l. 1373) also exists only in the log
  and supports Limitations (v).
- **`diagnose_v2.py` cannot run as committed.** `main()` calls `os.path.join`, but
  `import json, os` appears *inside* `main()` (~l. 131), so `os` is function-local →
  `UnboundLocalError` on entry. The JSON is dated 9月18 and the script 9月19, so the
  whole V2-diagnosis block (four medians, the 1.6–1.84× ratio, the drift figures) cannot
  be regenerated from the committed source.
- **Remark 1's "modulation-agnostic over the benchmark pairs"** is unsupported:
  `theory_waveform_fim.py:55-57` hard-codes `MODS = ['QPSK','QPSK']`, the results file
  contains only QPSK×QPSK, and the `__main__` block the comment points to never checks
  QAM. The 1.007–1.009 cross-modulation figure is log-only.
- **The 0.65 line-strength correlation** (`E[c^4] Σ h_l^4` vs measured strength) is in
  no script or log. (The four constants in the same paragraph, −0.68 / 3.35 dB / 2.66 /
  7.4 %, I derived analytically and MC-verified — all correct.)
- **E5's selector tournament is narrative-only.** The held-out even/odd energy,
  per-hypothesis-σ² and unnormalised-soft-evidence selectors exist only in
  `EXPERIMENT_LOG.md:446-461`; only `energy` and `lock` are implemented. The sentence
  "Three selectors collapse onto the densest hypothesis" overstates what was measured.
- **"0.483 for known-modulation V4 on the same cells"** is not reproducible: the same
  300 cells give 0.4883 (`e4_joint_k2_v4.json`) / 0.4902. The qualitative contrast
  survives; please use a reproducible value.

### M7. The central theory validation has no figure

`fig_e2_ser_vs_snr_sir0.pdf` and `fig_e2_ser_vs_sir_snr20.pdf` are generated but
unused. Theorems 1 and 2 are the paper's theoretical spine — the union bound, the exact
floor riding the Monte-Carlo curve, the marginal-MAP fix for the low-SNR reversal, and
the 16QAM joint 0.36 vs separate 0.82 — yet their Monte-Carlo curves live only in "the
public code's figure outputs". Add the joint-vs-separate SER-vs-SNR figure; the space is
available once the appendices move to supplemental (M1).

### M8. The identifiability certification excludes the benchmark's operating point

`theory_identifiability.json` samples `|a1|~U(0.7,1.3), ratio~U(0.5,1.5)` with
`|ratio−1| < 0.02` excluded — i.e. it systematically excludes |a₂| ≈ |a₁|, which is
exactly the benchmark's "SIR ≈ 0 dB by construction" point. For same-κ constellations
that degeneracy *is* the S₂ swap symmetry already in G, so the exclusion is unnecessary.
Re-run the certification around unit ratio so that "certified numerically for all
benchmark constellation pairs" is not stronger than the evidence.

### M9. Statistical reporting of the headline contrast

The headline tie pairs 700 bursts from a single grid against a five-checkpoint-averaged
route arm, while V4's CI is over five grids. Provide the 3500-burst paired contrast, or
explain why the route arm did not run on all five grids. Also report `frac_positive`
(V1 − V4 is positive on only 0.59–0.73 of bursts), which describes the shape of a
0.008-point margin far better than the pooled mean does.

---

## 2. Minor comments and numerical corrections

Every row was verified against code. Left: current text. Right: actual value.

| # | Current text | Verified value |
|---|---|---|
| a | Fig. 2 caption: "16QAM's typical minimum distance is 6× smaller than QPSK's" | Median ratio is **5.41×** (raw) / **2.42×** (normalised by d_su, which is what the figure plots). No statistic equals 6×; 5.92× is the p5 ratio |
| b | "matches Monte-Carlo simulation to within 0.003 at every grid point" | 47/48 cells ≤ 0.003; QPSK 20 dB / SIR 20 dB = **0.00359**. Tightest true bound 0.0036 |
| c | "the 0.488 figure our simulations quote is the 20 dB, **12**-phase-grid evaluation" | 0.48817 comes from the **32**-point grid (the `separate_exact_ser` default; 12 points gives 0.4791). Also **0.488 is never quoted anywhere in the paper** — the self-reference dangles |
| d | E1: "≤0.01 Hz bias" for BPSK/QPSK at −5…0 dB | QPSK @ −5 dB: **−0.0126 Hz** |
| e | E1: 20 dB "SNR-independent 0.01–0.07 Hz floor" | Holds on **inlier** RMSE only; 8PSK all-burst RMSE @ 20 dB = **0.135 Hz** (1 % outliers) |
| f | Two-tone estimator "tracks the exact CRB within 1.0–1.3× once \|Δf\| ≳ 1.6/T" | 3 of 18 cells exceed it: **2.39×, 1.61×, 1.50×** |
| g | "95–98 % failure at 1 Hz" | Measured **89.5–97.5 %**; two cells below 95 % |
| h | E5: "the PSK-only pairs are identified reliably (0.86–0.88)" at 10 dB | True only for BPSK/QPSK classes; 8PSK-involving PSK-only are **0.500 / 0.500 / 0.364**; pooled over all six PSK classes **0.65** |
| i | E5 K = 3: "on PSK-only triplets (**13 and 8 cells** at SNR 5 and 15 dB)" | These are **surviving burst counts**, not cells. Only 10 unordered PSK-only triplets exist (C(5,3)), and only **6 (5 dB) / 5 (15 dB)** distinct triplets actually appear. Suggest "13 and 8 PSK-only bursts (6 and 5 distinct triplets)" |
| j | E6: "the end-to-end SER is remarkably flat in \|Δf\|" | 20 dB across Δf ∈ {0…8} Hz shows an **8.3-point** U-shaped spread (0.1631 at Δf = 0 vs 0.0802 at 4 Hz). "No catastrophe at co-frequency" is the true and defensible claim |
| k | E6: τ = 0.1 T_s "+0.005–+0.04 SER" | Measured **+0.047 / +0.041 / +0.014** at 0/10/20 dB |
| l | E3: "+8–+11 pts at every SNR from −10 to 20 dB" | Minimum is **+7.38** (K = 3 @ −10 dB); per-SNR span is +7.6 to +10.9 |
| m | Table II, K = 2, SNR = 0: +9.5 | +9.44 (pool all pairs) / +9.45 (per-seed mean) — rounding |
| n | "Gaussian interferer … predicts a QPSK floor of only 1−(1−Q(1))² ≈ 0.293" | Exact value **0.2921** |
| o | "The nominal no-sync proxy sits at 0.62–0.69 at every SNR" | Measured 0.642–**0.698** (exceeded at two SNRs) |
| p | "16QAM runs +2–4 points above the oracle at SNR ≥ 0 dB" | 1.7 points at 0 dB |
| q | E4 V2: "1.6–1.84× the mixture's residual energy per channel-symbol (PSK: 0.0274 vs 0.0122; 16QAM: 0.0297 vs 0.0148)" | **All four medians are exactly right.** 1.6–1.84× is the *per-burst median ratio* (a legitimate statistic), but the ratio of the quoted group medians is **2.25× / 2.01×** — juxtaposing them invites the reader to compute a different number. State "median of the per-burst ratio" |
| r | Caption: "marginal-MAP … V3 0.4101 vs 0.4106" | Recomputes to **0.4102** |
| s | "within-burst gain drift is also ≈1.5× larger on the slots" | Phase-spread ratios are 1.33–1.62, but **magnitude-spread ratios are only 1.04–1.27**. "≈1.5×" is the phase component only |
| t | "V0 is worse than the waveform route it augments (0.6078 vs. 0.5534)" | 0.5534 is E3's variable-grid K = 2 oracle pooled SER — a different pipeline and grid. The in-table comparison is 0.6078 vs **0.5527** |
| u | "V2 loses to V1 by +5.8 pts" | Holds only against V1's ref-grid value 0.5384; against the tabulated 0.5392 it is **+5.7** |
| v | Gap-closure "monotonically … 3/7/15/14 % in between" | Reference-grid only. Averaged over five grids it is **non-monotone** (−0.6/0.6/3.2/7.4/14.6/19.6/**15.2** %), so "monotonically" is a single-grid statement |
| w | "the interferer's line still capturing 36–45 % of bursts at SNR ≥ 5 dB" | The diagnostic counts |Δf̂| > 1 Hz (5/10/15/20 dB = 0.355/0.453/0.402/0.380 ✓); "captured by the interferer's line" is an inference, not a measured label |
| x | Limitations (i): "at 20 dB it costs +0.03–+0.08 **SER points**" | Throughout the paper "pts" means *percentage points*; read that way this is off by two orders of magnitude. The measured cost in SER units is +0.027…+0.102 — write "SER", not "SER points" |

### Dangling references and drafting artefacts

- **"E1-D"** is cited in Limitations (i) as the evidence for "8PSK at −5 dB, 0.458 vs
  0.747" and for the 20 dB differential cost, but **there is no E1-D subsection anywhere
  in the paper**. I re-derived those numbers from `diff_coding.json` and they are
  correct — they are simply unlocatable in the manuscript. Add the subsection, or move
  the numbers into Limitations with an explicit code/appendix citation.
- **All six experiment subsection titles carry "(complete)"** — E1 … E6. A drafting
  marker that should not appear in a submission.
- Unused labels `eq:mix` and `eq:dmin`: the mixture model (1) and the minimum-distance
  definition (2) are never referenced, despite being discussed at length.
- "acquisition probability" in Fig. 6(b) is never defined in the text.

### Missing citations

- **SI-SDR itself is uncited**, while SI-SDRi is the paper's headline metric. Cite
  Le Roux et al., "SDR — half the truth is not enough", ICASSP 2019.
- **PSP / per-survivor processing is uncited**, although the PSP/Viterbi arm is the
  strongest baseline in E4. Cite Raheli, Polydoros & Tzou (1995) or equivalent; same for
  Viterbi equalisation.
- **ECM** (the paper names its own scheme expectation-conditional-maximisation) —
  Meng & Rubin (1993).
- **The companion C-SE separator** ("our companion work", used as a zero-shot
  cross-family check) has no bibliography entry.
- Non-data-aided frequency estimation is supported only by the single-tone
  Rife–Boorstyn CRB; add one or two classical references for the two-tone capture effect.
- Differential detection, the proposed deployment answer in Limitations (i), is uncited.

### Keywords

`complex-valued neural networks` appears in IEEEkeywords although the text explicitly
declares the learned separator "not the contribution". Replace with terms that carry the
actual contribution (joint detection, symbol detection, identifiability, carrier
synchronisation).

---

## 3. Claims I verified as correct

Listed explicitly so that none of them gets "fixed" by mistake:

- **Theorem 1** (union bound): the inner sum runs over all (c₁′, c₂′) with c₁′ ≠ c₁,
  which is the correct standard per-symbol union bound; the proof is sound.
- **Theorem 2** in full: the per-axis exact expression, the QPSK floor 1/2 (including
  the e(u) = ½·1{|u| > 1/√2} limit and the bounded-convergence step), the 7/16
  zero-contact floor, the 0.8380 16QAM value, and the 0.292 Gaussian-interferer
  comparison.
- **Proposition 2**: the four-step moment inversion, the finite-candidate-set
  formulation, and the 200-draw-per-pair numerical certification (identity residuals
  ~1e-15).
- **Every §III-D statistic**: median 1.70× / p90 7.6× (phases only), median 4.6× /
  p90 287× (unknown gains), E|Δf| = 89/24, median 3.23 Hz, p90/p95 7.56/8.77 Hz,
  P(|Δf| < 1/T) = 0.5871, single-tone CRB 0.042 Hz / 0.0024 Hz. Independently
  re-derived by Monte Carlo (4×10⁶ draws).
- **d_min landscape**: 0.39 / 0.16 d_su medians, 30 % of 16QAM rotations below
  0.1 d_su, QPSK zero contacts exactly at {0°, 90°, 180°, 270°}, and the φ = 90°
  invariance example in the caption.
- **Remark 1 numerics**: 1.007–1.009 median inflation and ≤1.02 p90 (QPSK only).
- **M-th-power line analysis**: E[c⁴] = −0.68, 3.35 dB, Var(c⁴) = 2.66,
  P(|Σ h_l⁴| < 0.15) ≈ 7.4 % — all arithmetically correct.
- **10.7 dB matched-filter noise-bandwidth reduction** (16000/1350 Hz).
- **Table II**: the K = 1 and K = 3 rows exact; pooled +8.5 ± 0.1 with 5/5 seeds
  positive; BER 0.2651 → 0.3519.
- **E3 aggregation**: 0.5027 is per-source weighting over matched pairs, not the
  unweighted mean over K (which would be 0.4282); per-K values, per-seed range and all
  unconditional-accounting numbers reconstruct exactly.
- **E5**: pair accuracy 0.1100/0.5300/0.4000, e2e SER 0.9868/0.6097/0.7617, pooled
  0.7860, conditional 0.8801/0.2635/0.4042, 16QAM-involving 0.111–0.778, the 20 dB
  drift toward 16QAM hypotheses.
- **E5 ablations**: NLS polish +0.19 pp, top-1 coarse candidate +0.77 pp, and the
  monotone L = 1/3/5 dose-response 0.5202 / 0.5130 / 0.5077.
- **E6 amplitude sweep**: graceful within ±5 dB, 0.431–0.483 at ±10 dB, minimum at
  ρ = 0.
- **Table III in full**: every V0–V4 row and pooled value, all three reference rows,
  blind mixture demodulation 0.6427, the marginal-MAP contrasts, the PSK/16QAM split,
  the −11.2 pts, V3-beats-PSP by 12 pooled / +22 at 20 dB, and all caption CIs.
- **CNSE cross-check**: SI-SDRi 6.013 dB vs 3.2 dB, 26.3× parameters, 0.4440 oracle-PSP,
  0.5356 blind-PSP (+9.16 pts), 36–45 % acquisition failure, 0.438/0.441 under
  differential scoring.
- **E1's central claim**: no BPSK/QPSK blind-minus-oracle gap exceeds 0.5 points at any
  SNR ≥ −5 dB on either the paper's table or the available pilot (QPSK @ −5 dB is 0.6
  points — the single marginal case).

---

## 4. Disposition

Priority order: **M1 (abstract) → M4 (reframe the differential score) → M2 (restructure
Section V) + M3 (reword the SNR claim) → M5/M6 (floor argument, artifacts) → M7 (add the
figure) → the minor table and the citations**.

With these addressed I would expect the paper to reach an acceptable standard. The
reason is not that it patches a handful of numbers; it is that the manuscript already
has what a TSP paper needs — a falsifiable thesis, a set of exact results that pin the
thesis down, and an honest architectural diagnosis. What remains is to state the work
that was already done correctly in language that does not outrun its evidence. The
common pattern across roughly fifteen defects above is precisely this: **the claims are
almost all true, but each is stated a half-step stronger than the evidence supports**
("remarkably flat", "reliably", "6×", "every grid point", "wins at SNR ≥ 5 dB", "clear
win"). Pulling each back to its evidence boundary will make the paper more persuasive,
not less.
