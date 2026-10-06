# Paper 6 Plan — Sync-Aware Joint Detection for SC-BSS (v1, 2026-09-13)

**Working title:** *Structure, Not Loss: Sync-Aware Joint Detection for
Single-Channel Blind Source Separation of Co-Frequency Signals*

**Code dir:** `paper6_sync_jd/` · **Manuscript dir:** `paper6/` (TODO) ·
**Log:** `paper6_sync_jd/EXPERIMENT_LOG.md` (real file, repo convention)

---

## 0. Context and positioning

Paper 5 (held from submission, 2026-09-13) established with controlled
experiments that, at SIR ≈ 0 dB, (i) waveform-metric gains do not reach
the bits, (ii) loss-level task orientation buys a sign-structured
fraction-of-a-point effect, and (iii) a naive joint soft-information head
fails at blind burst synchronisation / phase-entangled joint detection.
Its prescription: *explicit synchronisation, equalisation, and
joint-detection structure*.

**Paper 6 is that prescription, implemented and derived.** The creative
point is not another backbone — it is a receiver-structured separator in
which every module is derived from the factorisation of the joint
posterior (Section 2), with mathematical support (error-probability
bounds, identifiability, CRB) that paper 5 deliberately lacked.

Relationship to paper 5: paper 5 remains the measurement/protocol study
(submit later, possibly after paper 6 exists so they can cite each other;
or merge — decision deferred). Paper 6 reuses paper 5's benchmark,
`ser_comp.py` evaluation truth, and pretrained SlotSepNet checkpoints.

## 1. Problem setup (unchanged benchmark)

Same generator as paper 4/5: K ∈ {1,2,3} (K=4 zero-shot probe),
BPSK/QPSK/8PSK/16QAM, RRC 0.35, 3-tap multipath, AWGN, T=4096 @ 16 kHz,
carrier = 2000 Hz + U(0,5) Hz + ±5 Hz jitter, w_k ~ U(0.4,0.6) (SIR≈0),
train SNR [-5,20] dB, test grid {-10,...,20} dB, test seed 99999.

Key physical numbers for the sync design:

- Burst duration T_burst = 4096/16000 = **0.256 s** → DFT resolution
  ≈ 3.9 Hz. True carrier ∈ [1995, 2010] Hz (15 Hz span). Symbol rate
  1000 sym/s (16 sps); RRC bandwidth ≈ 1350 Hz → sources overlap fully.
- After down-conversion at the nominal 2000 Hz, residual offset
  Δf ∈ [-5, +10] Hz. M-th power spectral line sits at M·Δf ∈ [-40, 80]
  Hz worst case (8PSK) — far from Nyquist (8 kHz), no aliasing.
- Timing is fixed at the zero-delay symbol grid by construction (paper 5
  established this); timing recovery is out of scope.

## 2. Theory (paper §III–IV; each result must be simulation-validated)

### 2.1 Joint MAP detection and the separate-then-detect gap

Symbol-rate model after per-slot sync (Proposition 0 — model reduction):

  r_n = a_1 c_{1,n} + a_2 c_{2,n} + w_n,   w_n ~ CN(0, σ²)

with complex per-source gains a_k (post-equalisation). The joint MAP/ML
detector under known (a_1, a_2, σ²):

  (ĉ_1, ĉ_2) = argmin_{(c_i,c_j) ∈ C_{m1}×C_{m2}} |r_n − a_1 c_i − a_2 c_j|²

- **Thm 1 (pairwise error / union bound).**
  P_e,joint ≤ (1/(M_1 M_2)) Σ_{(c,c')≠} Q( |a_1 Δc_1 + a_2 Δc_2| / √(2σ²) ).
  The sum constellation a_1 C_1 + a_2 C_2 has minimum distance
  d_min(φ) = min |Δc_1 + e^{jφ}Δc_2|, φ = ∠(a_2/a_1). For PSK/QAM
  alphabets, d_min(φ) = 0 only on a measure-zero set of φ (the
  difference lattices are commensurate only at discrete rotations);
  for a.e. φ the joint detector operates near the single-user
  matched-filter bound even at SIR = 0 dB.
- **Thm 2 (separate-then-detect saturates).** Treating the interferer as
  Gaussian noise: P_e,sep → (const)·Q( d_min,1 |a_1| / √(2(σ²+|a_2|²)) ),
  which at SIR = |a_1|²/|a_2|² = 0 dB saturates at an SNR-independent
  error floor. Numerically: QPSK, SIR 0 dB → floor ≈ Q(1) ≈ 0.159 per
  axis; this is the analytic counterpart of paper 5's measured flatness
  (baseline SER 0.561 across SNR) and quantifies what separation losses
  must be overcome before bits move.
- **Validation:** plot Thm 1/2 bounds vs simulated SER of (a) joint ML on
  synthetic r_n, (b) separate detection, over SNR × SIR grids — must
  visually sandwich the paper-6 system's operating points.

### 2.2 Phase identifiability under co-frequency superposition

- **Prop 3 (gauge freedom).** y = e^{jφ_1} h_1 x_1 + e^{jφ_2} h_2 x_2:
  for any (θ_1, θ_2), the reparametrisation
  (e^{jθ_k}x_k, φ_k−θ_k) leaves y invariant → per-source phases are not
  identifiable from the mixture under continuous-symbol priors.
- **Prop 4 (discrete alphabets restore joint identifiability).** With
  x_k,n ∈ C_{m_k}, the constraint e^{jφ_k}·(rotated constellation) must
  land on a_k C_{m_k} reduces the gauge to the alphabets' rotational
  symmetry group (M-fold for M-PSK); (φ_1, φ_2) are jointly identifiable
  up to that finite ambiguity **only through the joint discrete
  structure** — i.e., sync and joint detection cannot be decoupled at
  K ≥ 2. This is the theorem-level version of paper 5's "frequency
  oracle fails at K=2" finding.

### 2.3 Frequency-offset CRB: why sync goes after separation

- Rife–Boorstyn CRB for a single tone in AWGN; extension to two
  superimposed tones: the Fisher matrix for (f_1, f_2) has off-diagonal
  term ∝ sinc(π Δf T_burst), so CRB inflates by
  ~1/(1−sinc²(π Δf T_burst)). With Δf ~ U(0,10) Hz and 1/T_burst ≈ 3.9 Hz,
  the median inflation factor is large (Δf ≲ 4 Hz → near-singular).
- **Prop 5 (informal/quantitative).** Blind per-source frequency
  estimation from the *mixture* is information-poor when |Δf| ≲ 1/T;
  after separation to per-slot residual SIR η, the CRB approaches the
  single-source bound scaled by (1+1/η). Consequence for architecture:
  **separate → per-slot sync → joint re-detection**, exactly the module
  order of Section 3.

### 2.4 (Discussion-level, optional) Soft-loss gradient sign

Gaussian-mixture model of the residual cloud; expected soft-CE gradient
points toward the correct grid cell at K=2 and is biased at K=3
(formalises paper 5's sign structure). Keep as analysis, not theorem.

## 3. Architecture — Receiver-Structured SlotSepNet (RS-SlotSepNet)

Backbone: paper 4/5 SlotSepNet **unchanged** (warm-start from paper-5
config (b)/(d) checkpoints). New modules, each tied to a §2 result:

### 3.1 BlindCarrierSync (per slot; replaces paper 5's oracle down-conversion)

Classical DSP, no learning (learning reserved for where structure fails):

1. Coarse down-conversion at the nominal 2000 Hz: x = ŝ·e^{−j2π·2000·t};
   residual Δf ∈ [−5, +10] Hz.
2. **Nonlinear spectral-line coarse frequency**: amplitude-normalise,
   raise to the M-th power of the modulation's symmetry order
   (BPSK 2, QPSK/16QAM 4, 8PSK 8), zero-padded FFT (×8), peak + parabolic
   interpolation → Δf̂ (sub-Hz accuracy expected at SNR ≥ 0 dB; the M-fold
   ambiguity lands on phase, not frequency).
3. **Decision-directed phase/frequency line fit**: correct with Δf̂,
   RRC matched filter, sample 0::16 grid, unit-power, hard-decide, fit
   per-symbol phase residual θ_n = ∠(z_n ĉ_n*) by LS line
   θ_n ≈ a + b·n (2 DD iterations); apply e^{−j(a+bn)}.
4. **Modulation-hypothesis selection**: run 3 for all four constellations,
   keep argmin of mean |z−ĉ|² (constellation-lock score) — blind to the
   true modulation; ablatable against oracle-mod.
Output: synced symbol stream z[256], estimates (Δf̂, a, b), lock score.

Training-time differentiable version: steps 1–4 run on **detached**
tensors to obtain (Δf̂, a, b); the correction is then applied to the
live tensor so gradients flow through the waveform — same
"estimate-detached / apply-with-gradient" convention as paper 5's
phase alignment (plan v2 O4).

### 3.2 DD equalizer (per slot, symbol rate)

Short complex FIR (L=11 taps, init = identity) adapted per burst by
decision-directed LMS (μ ~ 1e-2, freeze decisions each pass, 5–10
passes) — targets the 3-tap multipath ISI floor (16QAM clean floor
0.032). Differentiable version: unrolled fixed-iteration LMS or a
learned complex FIR trained end-to-end (ablation A-eq).

### 3.3 JointPairDetector (K=2; the headline module)

Input: synced unit-power streams z_1, z_2 [256], mod hypotheses, coupling
matrix A ∈ C^{2×2}. Joint log-likelihoods over ≤16×16 hypotheses:

  ℓ_n(i,j) = −‖ z_n − A [c_i; c_j] ‖² / σ²

- **A estimation (blind)**: EM from A=I — E-step: joint decisions;
  M-step: closed-form 2×2 complex LS refit of A on the decisions;
  2–3 rounds. Variants: oracle-A (upper bound), EM-A (blind), net-A
  (predicted by a small head from slot features — later).
- Prop 4 tie-in: A's columns absorb the per-source complex gains; the
  joint discrete structure is what makes them estimable at all.
- Output: joint decisions + joint soft CE for training (PIT-assigned
  slot pairs at K=2 only).

### 3.4 Training

Fine-tune from paper-5 checkpoints (warm start), loss =
paper-5's (b)/(d) recipe with the soft-SER term routed through the
**blind** sync front-end (§3.1 differentiable version) + a K=2 joint-CE
term (§3.3). Oracle information is used ONLY in labels/reference
decisions, never in the estimate path — this removes paper 5's
"oracle upper bound" caveat from the deployed pipeline.

## 4. Experiments

| ID | Question | Design | Success criterion |
|----|----------|--------|-------------------|
| E1 | Does blind sync approach oracle sync? | K=1 test cells; compare nominal proxy / oracle compensated rx / BlindCarrierSync per mod × SNR. CPU-feasible, no training. | Blind within ~1–2 SER points of oracle at SNR ≥ 0 dB; floors at high SNR (0.000/0.000/0.014/0.032) |
| E2 | Theory validation | §2.1 bounds + CRB numbers vs simulated operating points | Bounds sandwich simulated SER; CRB inflation curve matches sync-error measurements |
| E3 | End-to-end blind pipeline | Pretrained SlotSepNet slots → BlindCarrierSync → ser_comp scoring, K∈{1,2,3}, 5 seeds (reuse paper-5 ckpts) | Matches paper-5 oracle-scored SER/BER within small Δ → oracle assumption removed from eval pipeline |
| E4 | Joint detection at K=2 | JointPairDetector on E3's synced slots; A ∈ {oracle, EM}; vs waveform-route 0.5475 / baseline 0.5611 | EM-A joint SER significantly < 0.5475 (5 seeds, seed-level sign-flip test) |
| E5 | Ablations | −sync, −eq, −joint; mod-hypothesis vs oracle-mod | Each module's ΔSER quantified |

Statistics: same protocol as paper 5 (5 seeds 42–46, exact seed-level
sign-flip primary, cell-level MC descriptive, Bonferroni over
pre-declared contrasts). Evaluation truth stays `ser_comp.py`
(unchanged) for cross-paper comparability — joint-detector decisions are
scored against the same reference hard labels.

## 5. Risks and fallbacks

**Status 2026-09-13 (E1+E2 done locally, see `paper6_sync_jd/EXPERIMENT_LOG.md`):**E1 blind sync meets the criterion for BPSK/QPSK/8PSK at SNR ≥ 0 dB
(≤1.1 pts vs oracle); 16QAM lags +2–4 pts (weak-line bursts, R2
confirmed in reduced form). E2 validated the theory with three
corrections to this document: (i) QPSK sum constellation has **4** zero
contacts (k·90°), not 8; (ii) the Gaussian separate-detection floor
*underestimates* — the exact discrete-interferer floor (QPSK SIR 0:
SER ≈ 0.49) is the correct quote and matches MC to ≤0.003; (iii) the
sinc² CRB heuristic's null at Δf=1/T is an artefact of the unweighted
kernel (exact Fisher: 1.59 there; median inflation over our Δf law is
1.82×, p90 10.5×). Key headline validated: joint ML at SIR=0 tracks the
single-user AWGN bound for QPSK (SER 0.014 @ 15 dB) while separate
detection saturates at ~0.49 — the entire paper-6 opportunity in one
number. 16QAM×16QAM joint detection is hard (median d_min 0.16·d_su);
expect the K=2 gains to concentrate on PSK pairs.

**Status 2026-09-18 (E3 done on server, ~7 min full run):** oracle
reproduction is bit-exact (config (b): pooled 0.5027/0.2651 = paper 5's
published numbers). Blind pipeline vs oracle: pooled ΔSER +8.5 pts
(5/5 seeds), decomposed: K=1 gap +0.7–2.3 pts at SNR ≥ 0 dB (E1-level),
**K≥2 gap SNR-flat +8–11 pts** — the residual interferer's M-th-power
line pulls the per-slot carrier estimate 2–4 Hz off the true carrier at
every SNR (structured bias, not noise; crude SIC does not help:
0.6484 vs 0.6422). Corollary: K=2 blind slots (0.6422) ≈ K=2 blind
raw-mixture baseline (0.6427) — separation + independent sync buys
nothing. This is exactly the Prop-4 entanglement regime; E4 must do
**joint sync+detection** (ECM loop), not sync-then-detect.

**Status 2026-09-18 (E4 partial — headline result landed):** the ECM
joint sync+detector (`paper6_sync_jd/joint_sync_detect.py`) on the raw
K=2 mixture (V1, training-free, no separator) beats both references:
pooled SER **0.538** vs waveform route 0.5475 / mixture baseline 0.5611;
oracle-carrier bound (V3) 0.410 shows remaining sync headroom. Gains
concentrate where E2 predicted: PSK-only pairs V1 0.392 / V3 0.257;
16QAM-involving pairs hard (0.714/0.596). Non-monotone SNR curve (best
at 10 dB): noise-limited below, fading-ISI model-mismatch floor above —
motivates the §3.2 equaliser. Algorithm lessons (in module docstring):
EM must run to convergence at every ranked frequency candidate;
frequency moves need EM re-estimation (decision-frozen NLS can't leave
bad basins); gain-phase init grid must cover one symmetry sector
[0, 2π/M). V0 (naive chain) and V2 (slot-aided ECM) are implemented but
unrun — they need the paper-5 checkpoints that live only on the server,
which dropped mid-session (SSH timeout since ~15:05 local).

**Status 2026-09-18 late (E4 complete + V4 + E5 — experiment arc DONE):**
V0/V2 ran after the server returned (server cross-check matched local to
the 4th decimal): V0 0.6078, V2 0.5960 — V2 loses to mixture-ECM V1
because mask separation distorts the linear superposition model (slot
streams carry 1.6–1.84× the mixture's post-fit residual; true-stream
control 0.041 @ 20 dB quantifies the incentive to improve separation).
**V4** (`joint_isi.py`, ISI-aware ECM, L=5 centred taps): pooled 0.5299,
PSK-only @20 dB 0.241→0.217, gap closure grows with SNR (ISI-floor
signature); V1 = V4 with L=1 (built-in ablation). **E5**: blind mod-pair
selection FAILS under every selector tried (best: normalised soft
evidence; pair accuracy 0.16–0.36, end-to-end SER 0.867 vs known-mod
0.508, but 0.415 conditional on a correct pair — classifier is the
bottleneck) → known-modulation stays the operating assumption;
K=3 probe FAILS (+10–11 pts vs baseline; phantom-pair lock at the
sequential frequency init) → scope fixed at K∈{1,2}; ablations: NLS
polish ≈ 0, restarts ≈ +0.8 pp, tap dose-response L=1<3<5. Manuscript
`paper6/main.tex` complete at 40 pp (all E1–E5 written, builds clean,
no TODOs). Remaining decisions: venue; paper-5 disposition (standalone
IEEE Access vs merged narrative).

## 5.1 Original risk list (kept for the record)

- **R1: DD phase fit diverges at low SNR / high-order mod** (16QAM at
  −10 dB: decisions too noisy). Mitigation: restrict DD to lock-score
  gating, fall back to coarse sync only; report per-SNR where sync holds.
- **R2: M-th power line too weak for 16QAM** (non-constant modulus).
  Mitigation: coarse grid search over Δf maximising the 4th-power line
  OR the constellation-lock score directly (61-point grid, cheap).
- **R3: EM for A converges to swapped/degenerate solutions** at SIR≈0.
  Mitigation: init from A=I + per-slot power priors; constrain off-diag
  magnitude ≤ diag; oracle-A variant bounds the loss from A-error.
- **R4: joint gain small at K=2** (paper 5: residual interference is
  *structured* but slots may already absorb most of it). Fallback story:
  joint detector on the *mixture* symbol stream (no separation) as a
  second operating point — then the paper becomes "when to separate at
  all", still positive.
- **R5: nothing beats 0.5475** → paper 6 merges into paper 5 as its
  "we tried the structure too" section; measurement paper strengthened.

## 6. Compute and sequencing

1. E1 + E2: local CPU / light server, days. ← start here
2. E3: server, reuse paper-5 checkpoints, evaluation only.
3. E4: server, light training (EM-A is training-free; net-A later).
4. E5 + full 5-seed matrix: server, fine-tune runs (~100 epochs each,
   paper-5 cost scale).
5. Manuscript `paper6/` once E1–E4 numbers exist.

Server: `ssh -p 3009 vr301@172.16.65.85`, workdir `/data/experiment`,
venv `/data/experiment/venv_bss/` (NOT the system conda).
