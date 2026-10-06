# Response to Reviewer 1 — Major Revision

Dear Reviewer,

We thank you for an unusually thorough review. Every substantive point
below has been addressed in the revised manuscript, most by fixing the
theory itself rather than re-wording it; where we respectfully disagree,
we say so explicitly. All new numbers are reproduced by the public code
(`paper6_sync_jd/`, linked in the Data Availability statement); no
manuscript number was changed by hand.

## The three most dangerous points (your §二–四)

**1. Proposition 3's offset law (your §二). You were right — and it was
worse than you suspected.** The inflation statistics had been averaged
over $|u+\epsilon|$ of a *single* source (support $[0,10]$ Hz),
mislabeled as the two-source separation. The correct quantity is
$|\Delta f| = |\delta_1 - \delta_2|$, $\delta_k = u_k + \epsilon_k$
i.i.d., whose support is $[0,15]$ Hz. The manuscript now derives the
exact density: $\delta$ is a trapezoid on $[-5,10]$, and $|\Delta f|$
is its autocorrelation — a piecewise-quadratic spline (explicit
polynomials and CDF in Appendix A, verified against the sampler to KS
distance $<2\times10^{-3}$). Under this law: $\mathbb{E}|\Delta f| =
89/24 \approx 3.71$ Hz, median $3.23$ Hz, $P(|\Delta f| < 1/T_{\rm
burst}) = 0.587$, and the CRB inflation has median $1.70\times$ /
90th percentile $7.6\times$ — replacing the incorrect "triangular on
$[0,10]$, median $1.82\times$, p90 $10.5\times$".

**2. Proposition 2 (your §三). Weakened, exactly as you suggested.**
"Discrete alphabets restore joint identifiability" now reads "finite
symmetry, generic identifiability", with explicit caveats for the
constellation-collision rotations (the zero contacts, where the floor
drops to $7/16$), source permutation, and equal-alphabet degeneracy.
The abstract, introduction, and the proposition's proof sketch are
consistent with the new statement.

**3. Theorem 2 and the 0.488 "exact floor" (your §四). Separated by
ensemble, and the continuous-ensemble limit is now closed-form.**
Appendix A adds the full proof of Theorem 2 and states the three
quantities separately: $P_{\rm sep}(\varphi)$ (pointwise),
$\mathbb{E}_\varphi[P_{\rm sep}]$, and the finite-grid average. For
QPSK pairs at SIR 0 dB the continuous-ensemble floor is **exactly
$1/2$** (closed form in the appendix: of $u_I=\cos\theta$,
$u_Q=\sin\theta$, exactly one exceeds the boundary distance a.e.); the
16QAM floor is $0.838$ by numerical integration. The $0.488$ figure is
now explicitly labeled the $20$ dB / 12-phase-grid evaluation — the
grid mean is pulled below $1/2$ by the contact phases and by finite-SNR
smearing (the 32-point grid saturates at $0.4922 =
(28\cdot\tfrac12+4\cdot\tfrac{7}{16})/32$, which we verified
symbolically). The abstract now says "$1/2$".

## Nuisance parameters in the CRB (your §十)

We now report **both** Fisher models: $(f_1,f_2,\phi_1,\phi_2)$ and the
$6\times 6$ Fisher with unknown per-tone gains — the benchmark's actual
condition, since $w_k$ and the fading taps are unknown to the receiver.
With gains unknown the inflation median rises to $4.6\times$ and the
90th percentile to $287\times$ (Fig. 3, left, dash-dotted). The text
also labels the analysis explicitly a *pure-tone surrogate* bounding
the carrier-structure information, not the FIM of the full
communication waveform.

## Decision rule vs. metric (your §五): marginal MAP added

You asked for a marginal-MAP baseline; it is now everywhere. The theory
section derives the per-source Bayesian rule $\hat c_1 = \arg\max_{c_1}
\sum_{c_2} p(r|c_1,c_2)$; E2's Monte-Carlo adds it as a curve (it
eliminates the low-SNR reversal you flagged — at $0$ dB, SIR 0 dB:
joint-ML $0.479$ > separate $0.447$, marginal-MAP $0.446 \le$ separate —
and coincides with joint ML at $\mathrm{SNR}\ge 5$ dB); and E4 re-decides
every converged ECM fit with it (V1: $0.5388$ vs. $0.5392$ joint pooled;
V3: $0.4101$ vs. $0.4106$). Your observation was correct: the joint rule
optimises the pair; at the benchmark's SNRs the distinction costs
nothing, and the paper now says so.

## "EM" renamed and specified (your §六)

The algorithm is now named precisely: **classification-EM** (hard joint
decisions + exact LS refit — both steps are exact coordinate
minimisations of one least-squares objective, hence energy-monotone),
nested in the frequency search as an **ECM** scheme. **Algorithm 1**
gives the full procedure: the $66+4\times169+1$ converged-EM hypothesis
stages, the NLS polish with accepted-only-at-lower-energy moves, the
restart ranking, and the convergence criteria ($10^{-5}$ relative
energy decrease, 30 rounds; polish stall at $10^{-4}$).

## CRB comparison without cherry-picking (your §七)

Fig. 3 (right) now reports the estimator in full: bias, all-burst RMSE,
inlier RMSE, and the acquisition-failure probability $P(|\hat f - f| >
1$ Hz) at $100$ bursts per modulation–SNR cell. The failure rates are
now in the text: 8PSK $84\%$ at $-5$ dB (the $M{=}8$ line under the
noise floor), $\le 10\%$ for 8PSK/16QAM at mid SNRs (weak-line bursts),
zero elsewhere.

## Matched-pairs-only scoring (your §八): end-to-end accounting added

A new paragraph in E3 prices counting errors. On the same pipeline:
count accuracy $0.909/0.042/1.000$ at $K{=}1/2/3$ (the occupancy head
lights a surplus third slot on $93\%$ of $K{=}2$ mixtures); per-source
miss probability $2.3\%/1.5\%/0\%$. With a missed source priced as a
total detection failure (SER 1, BER 0.5), the end-to-end SER moves the
oracle arm to $0.165/0.560/0.586$ per $K$ (vs. conditional
$0.145/0.553/0.586$) and the blind arm to $0.218/0.648/0.679$ (vs.
$0.200/0.642/0.679$). The matched-pairs conditioning moves nothing by
more than $0.7$ points — and we say so.

## "Blind" tightened (your §九)

The Experiments preamble now defines the term: blind qualifies the
*estimate path* (no oracle carriers, phases, symbols or modulations in
any estimator; training-free receiver stages); the operating
assumptions are a known nominal carrier and offset range, the
benchmark's fixed timing grid, and known modulations — "a semi-blind,
modulation-conditioned receiver".

## Statistics (your §十四): multi-grid E4, reinforced E1

- E4's deterministic variants now run on **five independent test grids**
  (seeds $99999/88888/77777/55555/31337$): pooled SER V1 $0.5392$
  [$0.5342, 0.5441$], V4 $0.5310$ [$0.5261, 0.5360$], V3 $0.4106$
  [$0.4044, 0.4169$] (95% seed-level CIs). Seed $99999$ reproduces the
  original numbers to $\le 0.0006$ (BLAS threading). The headline
  contrast V4 vs. waveform route is $-2.2$ points.
- E1 is re-measured at **200 bursts per modulation–SNR cell over two
  independent grids** (your $\approx 25$/cell concern was justified):
  BPSK/QPSK gaps are $\le 0.5$ points at SNR $\ge -5$ dB, but 8PSK runs
  $2$–$4$ points above the oracle at SNR $\ge 0$ — the original
  "$\le 1.1$ points" claim was a small-sample artifact, and the
  manuscript, abstract, and conclusion now state the corrected numbers.

## Scope of conclusions (your §十一–十三, 十五)

- The headline is now framed as "structurally meaningful but
  quantitatively modest", benchmark-conditioned throughout, and the
  true-stream control (SER $0.041$) is presented as the boundary:
  separation returns to the detection path once it preserves the linear
  mixture model — "a boundary, not a law".
- $K{=}3$: the text now states precisely that the *tested* acquisition
  strategy (sequential extraction + $M$-th-power line search) fails; no
  impossibility is claimed, and a joint three-tone grid is named as the
  open alternative.
- The 16QAM weak-line bursts now have a mechanism, not just an
  observation: the line amplitude factorises as
  $|\mathbb{E}[c^4]|\cdot|\sum_l h_l^4|$; for 16QAM
  $\mathbb{E}[c^4] = -0.68$ with per-symbol self-noise
  $\mathrm{Var}(c^4)=2.66$, and under the 3-tap Rayleigh channel
  $P(|\sum_l h_l^4| < 0.15) \approx 7\%$ — those bursts have no usable
  line for *any* fourth-power method. The measured symbol-level line
  strength tracks the prediction (correlation $0.65$).

## Complexity (your §十八.3 and §六)

The limitation item now reports the measured platform (12th-gen Intel
i5), $\approx 750$ converged classification-EM invocations and
$\approx 6.3\times 10^3$ decision/gain rounds per burst, the
hypothesis counts ($66$ coarse, $\le 676$ fine, $1$ full-rate), the
$\mathcal{O}(N M_1 M_2)$ per-candidate cost, and the
$\prod_k M_k$ (exponential-in-$K$) hypothesis growth — the concrete
scaling reason $K{=}3$ needs a different acquisition structure. Measured
cost: $\approx 1.6$ s per burst per V1+V4 evaluation; the full
700-cell grid takes $19$ min on one core.

## Points we respectfully push back on

- **Related work (your §十六, last part):** the manuscript already
  cites and differentiates Guo 2024 (complex-domain WCL), S4-UNET 2026
  (long-sequence state-space) and EDSNet 2026 (unknown source count),
  and §II-B positions the joint-transceiver line; we have added an
  explicit sentence grouping them.
- **Full classical-baseline suite (SIC / MMSE / BCJR):** two of your
  requested baselines were already the paper's V3 (joint ML with true
  carriers) and V1 (joint ML with estimated carriers); the marginal-MAP
  rule is now added on top. A complete SIC/MMSE/BCJR suite is beyond a
  13-page TSP paper's budget and, given E3's measured result that
  per-stream sync is captured by the interferer's line (SIC's estimates
  inherit that bias), we expect it to land between the blind-mixture
  and waveform-route arms; we state this as future work.
- **GMI / mutual-information metrics and fractional-timing sweeps:** the
  first is nice-to-have rather than load-bearing (the compensated-SER
  protocol with BER is already decision-identical); the second conflicts
  with the paper's declared scope (timing recovery out of scope, stated
  in the model and limitations). We prefer to keep the benchmark
  honest and bounded.
- **Real data:** every published SC-BSS baseline operates on synthetic
  benchmarks; we scope the claims accordingly rather than claim
  real-world validity.

We believe the revision now meets the rigor bar you set, and the
theory is better for it — the corrected offset law strengthens the CRB
argument (median inflation $1.70\times$, $4.6\times$ with unknown
gains), and the closed-form $1/2$ floor replaces the grid-dependent
$0.488$.

Yours sincerely,
The authors
