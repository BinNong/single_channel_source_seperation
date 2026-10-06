# Response to Review 4 — paper6 (main(8).pdf), 2026-10-05

We thank the reviewer for the second-round reading and for acknowledging that the
previous revision was substantive. This round we address the two remaining P0
theory items, the statistical-reporting ambiguity, the wording qualifications, and
the strong-baseline request. All references are to the revised manuscript.

## 1. "tested loss-level modification" wording — FIXED

The fine-tuning control paragraph now reads: "the tested loss-level modification
does not remove the floor of Theorem 2 on this benchmark: within the tested
waveform-separator objective family, the remedy remains structural." The claim is
thus explicitly scoped to the tested objective family (soft constellation
cross-entropy, $\lambda_{\rm ser}\in\{0.1,1.0\}$), not to the space of all
receiver-aware learning objectives.

## 2. Proposition 2, Step 2 (amplitudes) — FIXED (P0)

The reviewer is correct: for $\kappa_1 \neq \kappa_2$ the quadratic system
$x+y=p$, $\kappa_1 x^2 + \kappa_2 y^2 + 4xy = q$ can have two distinct positive
unordered solutions, so Step 2 cannot claim unique amplitude identification.
Step 2 is rewritten as "**amplitudes: a finite candidate set**": eliminating $y$
leaves one quadratic in $x$, hence *at most two* positive unordered solutions —
the true pair and possibly a spurious one (for $\kappa_1=\kappa_2$ the two
solutions are exactly the source swap). The BPSK branch (via $\mathbb{E}[s^2]$)
is stated the same way.

Step 3 now carries the corresponding logic: the spurious phase root (equal-order
case) *and* any spurious amplitude candidate of Step 2 must additionally match a
further known mixture moment (e.g. $\mathbb{E}|s|^6$, whose constellation factors
are known and generically nonzero) — an extra algebraic equation in $(a_1,a_2)$
that holds only on a measure-zero set and is folded into the exceptional set of
Step 4. The logically invalid sentence "the spurious root is generically rejected
by Step 2's moduli" is removed.

## 3. "Finite-N identifiability follows from consistency" — FIXED (P0)

Removed. The proof now states: "The proposition concerns the *population* mixture
law; for finite bursts the empirical-moment inversion is consistent as
$N\to\infty$, not exact." The proposition statement itself now reads
"identifiable from the population mixture law up to $G$". Consistency is no
longer conflated with finite-sample identifiability, and the theorem carries no
finite-$N$ responsibility.

## 4. Proposition 2 scope — pinned, per the reviewer's suggestion

The proposition is now exactly "generic identifiability of the effective
coefficients from the population mixture law", with the equivalence group
$G=\mathbb{Z}_{M_1}\times\mathbb{Z}_{M_2}$ (resp. its $S_2$ extension), and
conditions: known constellations, i.i.d. zero-mean symbols, nonzero symmetry
moments, collision-free sum map, nondegenerate moment system, population law. No
finite-sample exact-recovery claim is made anywhere.

## 5. Strong published SC-BSS baseline — ADDED (P1): CNSE

We trained the published CNSE separator (Hou & Gao 2022, vendored implementation,
6{,}693{,}509 parameters — $26\times$ our SlotSepNet) on the benchmark's $K{=}2$
mixtures (same SNR range, PIT SI-SDR + MSE anchor, 5 seeds, 100 epochs;
`train_cnse_baseline.py`) and pushed it through the *identical* waveform route as
the E4 baseline — oracle front-end, symbol-wise and PSP/Viterbi arms, same
scoring, same 700 reference-grid cells (`eval_cnse_baseline.py`, 5 checkpoints,
7000 records):

| separator | SI-SDRi @ K=2 | pooled SER (psp) | pooled BER (psp) | PSK-only SER/BER |
|---|---|---|---|---|
| SlotSepNet (ours) | 3.2 dB | 0.5337 | 0.266 | 0.440 / 0.250 |
| CNSE (Hou & Gao 2022) | 6.0 dB | 0.4440 | 0.213 | 0.337 / 0.185 |

The outcome is the interesting one for the paper's story, and we report it
honestly:

1. **The waveform route is separator-quality-limited, not architecture-fixed.**
   Doubling SI-SDRi (3.2 → 6.0 dB) improves the route's pooled SER by 9 points
   (0.534 → 0.444). The floor of Theorem 2 is therefore confirmed as a property
   of the *linear-residual regime it models*, not of separation as such — the
   manuscript now says this explicitly, and the headline "tie" claim is rescoped
   to the paper's separator quality (see below).
2. **At matched oracle frequency information, the joint architecture still
   wins:** the oracle-frequency joint loop V3 (0.4106) beats the CNSE waveform
   route (0.4440) — and both use true carriers. The architecture gap is
   therefore not an artefact of our separator.
3. **The oracle sync is the load-bearing assumption.** The CNSE route's 0.444
   consumes true carriers. We measured the *blind* CNSE route (CNSE →
   BlindCarrierSync → symbol-wise/PSP detection, same cells and scoring) and the
   genie-free differential scores of all four arms (`eval_cnse_blind.py`,
   `results/cnse_blind.json`, 14000 records):
   - Blind CNSE-psp pays **+9.2 pooled SER points** over its oracle arm
     (0.5356 vs 0.4440) — squarely inside E3's +8–11 pts capture band measured
     with our separator. Interferer spectral-line capture persists at 36–45% of
     bursts even at SNR ≥ 5 dB despite the 6 dB cleaner slots: the E3 failure
     is confirmed **cross-architecture** — it is the per-slot sync stage, not
     the separator.
   - Blind-versus-blind, the joint loop again ties the strongest waveform route:
     V4 0.5310 vs blind CNSE-psp 0.5356.
   - Under genie-free differential scoring (PSK pairs), V4 (0.372 SER /
     0.244 BER) leads *every* waveform route: CNSE oracle-psp 0.438 / 0.283,
     CNSE blind-psp 0.442 / 0.286, SlotSepNet-psp 0.537 / 0.357.

The manuscript integrates all of this: the headline claim is rescoped from
"matches the strongest oracle-assisted waveform route we can build" to "ties the
strongest waveform route *at the paper's separator quality*", the E4 table gains
the CNSE row, and a "Cross-architecture check (CNSE)" paragraph in E4 reports
the three qualifications above. Net effect on the paper's message: the
waveform-to-bit gap is now demonstrated to be (i) separator-quality-limited in
its absolute position, (ii) architecture-driven at matched oracle information,
and (iii) oracle-synchronisation-driven in its blind deployment gap.

CTDCRN has no public implementation and remains citation-only.

## 6. E4 statistical reporting — FIXED (P1)

The "Statistical reporting" paragraph now states the exact sample units: each of
the five test grids holds $7\times100=700$ bursts; the V1–V4 contrast pools all
$3500$ paired bursts ($n{=}3500$ now printed inline); the headline waveform-route
contrast pairs the $700$ bursts of the reference grid (seed 99999) with the route
arm averaged over its five separator checkpoints per burst ($n{=}700$). The
CI/Wilcoxon coexistence is explained explicitly: the paired-bootstrap CI targets
the *mean* per-burst difference while the Wilcoxon signed-rank test targets
*symmetry of the paired differences about zero* — different estimands, so a
bootstrap interval containing zero can coexist with a significant Wilcoxon $p$
(the headline contrast: mean $+0.003$, CI $[-0.011,+0.018]$, $p=4.8\times10^{-4}$).

## 7. "disambiguating information" wording — FIXED

Introduction, contribution bullet, and conclusion now read: the disambiguating
information "is carried by the modulation-bearing waveform, but exploiting it
requires symbol-aware or decision-directed processing". This matches what the
known-symbols Fisher analysis actually proves and connects it to the joint loop's
decision-directed ("semi-pilot") mechanism. The abstract phrasing was already
qualified ("that non-data-aided synchronisation cannot reach").

## 8. Title — retained, with the reviewer's caveat accepted

We keep "Structure, Not Loss: ..." as a clearly labelled architectural slogan
whose precise, scoped version ("within the tested waveform-separator objective
family", "for linear-residual separate-then-detect receivers") is stated in the
abstract and Section III. The reviewer's alternative titles are appealing; we
judge the current one more discoverable, and the scoping language now carries the
precision the title trades away.

## 9. Items acknowledged with no further change

- Proposition 3 validation (Fig. 1 "Direct validation"), the
  MSE = Var + Bias² separation, V4's generalised-EM/monotone-refinement
  formulation, E5's selector/acquisition wording, and the decision not to expand
  blind modulation classification — all per the reviewer's assessment, unchanged.
- E5's known-modulation operating assumption remains explicit.

## Page budget

The Step-2/Step-3/Step-4 rewrite, the statistical-reporting expansion, and the
CNSE cross-architecture material (table row + E4 paragraph + abstract/intro/
discussion/conclusion sentences) were absorbed within the 13-page TSP target by
compression (\linespread 0.925 → 0.885, the three double-column figures at
0.68/0.57/0.57 \textwidth, minor sentence trims); the manuscript rebuilds
cleanly (xelatex, 13 pp, no unresolved references, no overfull boxes).
