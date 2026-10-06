"""Paper 6 — Proposition 2 (prop:discrete) numerical identifiability certificate.

Review-3 artifact: a constructive, numpy-only certificate for the generic
identifiability claim of Proposition prop:discrete (discrete alphabets:
finite symmetry Z_{M_1} x Z_{M_2}, plus S_2 for equal alphabets) of
paper6/main.tex.  For the noiseless symbol-rate mixture

    s = a1 c1 + a2 c2,      c_k i.i.d. uniform on a KNOWN alphabet C_k
    (BPSK/QPSK/8PSK/16QAM, E|c|^2 = 1; symmetry order M = 2/4/8/4),

the script follows the proof's moment-system argument step by step, with all
moments computed EXACTLY by enumeration over the finite product alphabet
(no Monte Carlo in the certificate itself):

  Step 1 (magnitudes).  p = E|s|^2 = |a1|^2 + |a2|^2.  If neither alphabet
    is BPSK (mu_2 = E[c^2] = 0): q = E|s|^4 = k1 |a1|^4 + k2 |a2|^4
    + 4 |a1|^2 |a2|^2, k_k = E|c_k|^4  ->  quadratic in x = |a1|^2.
    If BPSK is present: E[s^2] = a_B^2 (mu_2 of the partner alphabet is 0),
    resp. E[s^2] = a1^2 + a2^2 and q - |E s^2|^2 = 4 |a1|^2 |a2|^2 for
    BPSK+BPSK.
  Step 2 (phases), unequal nested orders (M_min | M_max; the benchmark's
    (2,4), (2,8), (4,8) pairs): the cross-terms of E[s^{M_min}] vanish
    (mu_j != 0 only for M_k | j), so E[s^{M_min}] = a_small^{M_min}
    mu_{M_min}^{(small)} separates the small-order source; E[s^{M_max}]
    then separates the other.
  Step 2 (phases), equal orders M: with u_k = a_k^M mu_M^{(k)},
    E[s^M] = u1 + u2 and E[s^{2M}] = alpha u1^2 + binom(2M,M) u1 u2
    + beta u2^2, alpha = mu_{2M}^{(1)} / mu_M^{(1)2} (beta likewise).
    Eliminating u1 gives ONE quadratic in u2: at most two roots.  For
    equal alphabets (alpha = beta) the two roots are exactly the swap.
  BPSK+BPSK: u_k = a_k^2 obey u1 + u2 = E[s^2]; substituting u2 = E[s^2] - u1
    into E[s^4] = u1^2 + u2^2 + 6 u1 u2 gives ONE quadratic in u1 whose two
    roots are exactly the swap (well-conditioned also at E[s^2] ~ 0, unlike
    the equivalent triangle construction).

Every candidate (a1~, a2~) from the constructive solver then passes three
filters, and each rejection is counted:

  F1 modulus constraint: |a_k~| implied by the phase-step powers must match
     a Step-1 magnitude solution.  This is the step that — per the proof —
     rejects the spurious quadratic root for the equal-order pair
     QPSK+16QAM; the script reports per-trial whether it actually does
     (and, if a spurious root ever passes F1, whether F2/F3 catch it).
  F2 higher-moment verification: the candidate must reproduce ALL raw
     moments E[s^m], m = 1..24, and E|s|^2/4/6 (moments NOT used by the
     solver), each with a tolerance relative to its own term scale.
  F3 distribution certificate: the candidate's full mixture multiset
     (support + multiplicities) must equal the truth's — exact
     distributional indistinguishability.

A trial succeeds iff every F3 survivor is group-equivalent to the truth
(Z_{M_1} x Z_{M_2}, plus S_2 for equal alphabets) and at least one survivor
exists (the truth's class is recovered).  Generic sampling: |a1| in
[0.7, 1.3], amplitude ratio in [0.5, 1.5], phases uniform.

Unit-ratio draws (2026-10-06, reviewer follow-up): draws with
|ratio - 1| < 0.02 were previously excluded as "tie = non-generic".  That
exclusion is REMOVED: the benchmark itself operates at SIR ~= 0 dB, and
for equal-kappa constellations the tie degeneracy of the Step-1 quadratic
is exactly the S_2 swap inside G (the double root is the correct unordered
solution, kept without a degeneracy flag).  For unequal-kappa pairs the
quadratic has two DISTINCT positive roots at the tie (the spurious one is
rejected by the higher-moment filters, as in the review-4 Prop-2 fix).
A dedicated `unit_ratio` certification block additionally evaluates every
pair at EXACT ratio 1 (200 draws, random phases/moduli).

Negative controls:
  (a) constellation-collision rotations (QPSK+QPSK, |a1|=|a2|=1, relative
      phase 90 deg — a zero contact of Thm 1; plus |a2|/|a1| = sqrt(2) at
      45 deg): non-injective symbol map (d_min = 0, support collapse),
      degenerate (double-root) moment system with sqrt(eps) ill-conditioning
      measured against moment perturbations, and an exhaustive 2-D grid
      search for NON-equivalent parameter pairs with an identical mixture
      pmf (MMD^2 with multiplicities).
  (b) continuous alphabet (c_k ~ CN(0,1)): the mixture is CN(0, |a1|^2 +
      |a2|^2), all raw moments E[s^m] vanish, and any phase rotation /
      power re-allocation leaves the distribution invariant — Prop 1's
      continuous gauge freedom, the contrast to Prop 2.

Output: results/theory_identifiability.json (+ printed summary).
Pure numpy (+ math/itertools).  Deterministic (seeded).  Local CPU, seconds.
Constellation tables are copied from signal_utils.CONSTELLATIONS (parity
asserted in __main__ when signal_utils is importable) so the certificate
stays torch/scipy-free.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')

MODS = ['BPSK', 'QPSK', '8PSK', '16QAM']
SYM_ORDER = {'BPSK': 2, 'QPSK': 4, '8PSK': 8, '16QAM': 4}
# Byte-for-byte the definitions of signal_utils.CONSTELLATIONS (vendored
# from paper1_cnn_se/utils.py); duplicated here to keep this script
# dependency-free.  Parity is asserted in __main__ if signal_utils imports.
CONSTELLATIONS = {
    'BPSK': np.array([-1 + 0j, 1 + 0j]),
    'QPSK': np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j]) / np.sqrt(2),
    '8PSK': np.exp(1j * (2 * np.pi * np.arange(8) / 8 + np.pi / 8)),
    '16QAM': (lambda: np.array([
        -3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,
        -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
         1 - 3j,  1 - 1j,  1 + 1j,  1 + 3j,
         3 - 3j,  3 - 1j,  3 + 1j,  3 + 3j,
    ]) / np.sqrt(10))(),
}

MMAX = 24                      # highest raw moment used by filter F2
TOL_EQ = 1e-6                  # group-equivalence tolerance (relative)
TOL_MAG = 1e-6                 # modulus-match tolerance (relative)
TOL_MOM = 1e-6                 # F2 moment-verification tolerance (relative)

MU = {m: np.array([np.mean(CONSTELLATIONS[m] ** k) for k in range(MMAX + 1)])
      for m in MODS}                                   # mu_k = E[c^k]
KAPPA = {m: float(np.mean(np.abs(CONSTELLATIONS[m]) ** 4)) for m in MODS}
MU2 = {m: complex(np.mean(CONSTELLATIONS[m] ** 2)) for m in MODS}


# ---------------------------------------------------------------------------
# exact moments of the mixture (enumeration over the product alphabet)
# ---------------------------------------------------------------------------
def mix_support(a1, a2, mod1, mod2):
    """All |C1|x|C2| mixture points (with multiplicities, flattened)."""
    return (a1 * CONSTELLATIONS[mod1][:, None]
            + a2 * CONSTELLATIONS[mod2][None, :]).ravel()


def mix_moments(a1, a2, mod1, mod2, mmax=MMAX):
    """Exact raw/absolute moments of s = a1 c1 + a2 c2 by enumeration."""
    S = mix_support(a1, a2, mod1, mod2)
    a2s = np.abs(S) ** 2
    return {'p': float(a2s.mean()), 'q': float((a2s ** 2).mean()),
            's6': float((a2s ** 3).mean()),
            'E': np.array([complex((S ** m).mean())
                           for m in range(mmax + 1)])}


def _sorted_points(S):
    S = np.asarray(S).ravel()
    key = np.lexsort((S.imag.round(9), S.real.round(9)))
    return S[key]


def multiset_equal(S1, S2, scale, tol=1e-6):
    """Exact equality of two mixture multisets (support + multiplicities)."""
    a, b = _sorted_points(S1), _sorted_points(S2)
    if a.shape != b.shape:
        return False
    return bool(np.all(np.abs(a - b) <= tol * max(scale, 1e-12)))


# ---------------------------------------------------------------------------
# constructive moment-system solver (the proof's root-finding procedure)
# ---------------------------------------------------------------------------
def solve_candidates(mod1, mod2, moms):
    """All candidate (a1~, a2~) produced by the proof's moment system.

    Returns (candidates, mag_pairs, flags).  candidates: list of dicts with
    complex a1/a2 and provenance 'kind'; mag_pairs: valid ordered
    (|a1|^2, |a2|^2) solutions of Step 1; flags: degeneracy markers.
    """
    M1, M2 = SYM_ORDER[mod1], SYM_ORDER[mod2]
    mu1, mu2 = MU[mod1], MU[mod2]
    p, q, E = moms['p'], moms['q'], moms['E']
    r2 = E[2]
    flags = []
    cands = []
    is_b = (mod1 == 'BPSK', mod2 == 'BPSK')

    # ---------------- Step 1: magnitudes ----------------
    if all(is_b):
        # q - |E s^2|^2 = 4 |a1|^2 |a2|^2 ; x + y = p  ->  quadratic
        xy = (q - abs(r2) ** 2) / 4.0
        disc = p * p - 4.0 * xy
        # BPSK+BPSK is an equal-kappa (same-alphabet) pair: at the tie the
        # double root x = y = p/2 IS the correct unordered solution (the
        # degeneracy is the S_2 swap inside G) — keep it, no flag.
        rt = math.sqrt(max(disc, 0.0))
        x, y = (p + rt) / 2.0, (p - rt) / 2.0
        mag_pairs = [(x, y)] if abs(x - y) <= 1e-9 * p else [(x, y), (y, x)]
    elif any(is_b):
        # E[s^2] = a_B^2 exactly (partner alphabet has mu_2 = 0)
        xb = abs(r2)
        xt = p - xb
        pair = [0.0, 0.0]
        pair[0 if is_b[0] else 1] = xb
        pair[1 if is_b[0] else 0] = xt
        mag_pairs = [tuple(pair)]
    else:
        k1, k2 = KAPPA[mod1], KAPPA[mod2]
        A = k1 + k2 - 4.0
        B = (4.0 - 2.0 * k2) * p
        Cq = k2 * p * p - q
        disc = B * B - 4.0 * A * Cq
        if disc <= 1e-9 * (B * B) and abs(k1 - k2) > 1e-12:
            # unequal-kappa tie: genuinely degenerate (not a group element)
            flags.append('tie_discriminant')
        # equal kappa (incl. same alphabet): the tie double root is the
        # correct unordered solution (the degeneracy is the S_2 swap for
        # equal alphabets) — no flag, proceed with rt = 0.
        rt = math.sqrt(max(disc, 0.0))
        mag_pairs = []
        for x in ((-B + rt) / (2 * A), (-B - rt) / (2 * A)):
            y = p - x
            if x >= -1e-9 and y >= -1e-9:
                mag_pairs.append((max(x, 0.0), max(y, 0.0)))
        mag_pairs = sorted({(round(a, 12), round(b, 12))
                            for a, b in mag_pairs})

    # ---------------- Step 2: phases ----------------
    if all(is_b):
        # u_k = a_k^2 obey u1 + u2 = r2 = E[s^2] and
        # E[s^4] = u1^2 + u2^2 + 6 u1 u2.  Substituting u2 = r2 - u1 gives
        # ONE quadratic: 4 u1^2 - 4 r2 u1 + (E[s^4] - r^2) = 0; its two
        # roots are exactly the S_2 swap (both kept; F1/F2/F3 vet them).
        # 2026-10-06: the direct quadratic replaces the earlier triangle
        # construction (|u_k| from Step 1 + acos geometry), which is
        # ill-conditioned as r2 -> 0 (near-cancellation a1^2 ~ -a2^2,
        # reachable at exact unit ratio): measured 4.7e-5 mirror residual
        # vs the 1.1e-5 tolerance on unit-ratio draw 115 (BPSK+BPSK),
        # costing 0/200 -> 1/200 failures there.
        t4 = E[4]
        rd = np.sqrt(2.0 * r2 ** 2 - t4 + 0j)
        seen = set()
        for sgn in (+1.0, -1.0):
            u1 = (r2 + sgn * rd) / 2.0
            u2 = r2 - u1
            if abs(u1) <= 1e-12 * p or abs(u2) <= 1e-12 * p:
                flags.append('degenerate_triangle')
                continue
            key = (round(u1.real, 9), round(u1.imag, 9))
            if key in seen:
                continue
            seen.add(key)
            cands.append({'a1': complex(np.sqrt(u1 + 0j)),
                          'a2': complex(np.sqrt(u2 + 0j)),
                          'kind': 'bpsk_bpsk'})
    elif any(is_b):
        # unequal nested orders (2, M): E[s^2] = a_B^2, then
        # E[s^M] = a_X^M mu_M^(X) + a_B^M separates the partner source.
        b = 0 if is_b[0] else 1
        t = 1 - b
        Mt = (M1, M2)[t]
        mut = (mu1, mu2)[t]
        atM = (E[Mt] - r2 ** (Mt // 2)) / mut[Mt]
        at = atM ** (1.0 / Mt)
        ab = np.sqrt(r2 + 0j)
        aa = [ab, at] if b == 0 else [at, ab]
        cands.append({'a1': complex(aa[0]), 'a2': complex(aa[1]),
                      'kind': 'bpsk_plus'})
    elif M1 != M2:
        # unequal nested orders (4, 8): E[s^4] = a_Q^4 mu_4^(QPSK); then
        # E[s^8] = a_Q^8 mu_8^(QPSK) + a_8^8 mu_8^(8PSK).
        s = 0 if M1 < M2 else 1
        big = 1 - s
        ms, mb = (M1, M2)[s], (M1, M2)[big]
        mus, mub = (mu1, mu2)[s], (mu1, mu2)[big]
        asM = E[ms] / mus[ms]
        abM = (E[mb] - asM ** (mb // ms) * mus[mb]) / mub[mb]
        aa = [None, None]
        aa[s] = asM ** (1.0 / ms)
        aa[big] = abM ** (1.0 / mb)
        cands.append({'a1': complex(aa[0]), 'a2': complex(aa[1]),
                      'kind': 'nested_order'})
    else:
        # equal orders: E[s^M] = u1 + u2,
        # E[s^{2M}] = alpha u1^2 + binom(2M,M) u1 u2 + beta u2^2
        # -> one quadratic in u2 (two roots; swap iff alpha == beta).
        M = M1
        S = E[M]
        alpha = mu1[2 * M] / mu1[M] ** 2
        beta = mu2[2 * M] / mu2[M] ** 2
        gam = float(math.comb(2 * M, M))
        E2M = E[2 * M]
        A = alpha + beta - gam
        B = S * (gam - 2.0 * alpha)
        Cq = alpha * S ** 2 - E2M
        if abs(A) <= 1e-12:
            flags.append('degenerate_quadratic')
        else:
            disc = B ** 2 - 4 * A * Cq
            if abs(disc) <= 1e-9 * max(abs(B) ** 2, abs(4 * A * Cq), 1e-30):
                flags.append('tie_discriminant')
            rd = np.sqrt(disc + 0j)
            for sgn in (+1.0, -1.0):
                u2 = (-B + sgn * rd) / (2 * A)
                u1 = S - u2
                cands.append({'a1': complex((u1 / mu1[M]) ** (1.0 / M)),
                              'a2': complex((u2 / mu2[M]) ** (1.0 / M)),
                              'kind': 'equal_order_root',
                              'u1': complex(u1), 'u2': complex(u2)})
    return cands, mag_pairs, flags


# ---------------------------------------------------------------------------
# filters + group equivalence
# ---------------------------------------------------------------------------
def _mag_match(m1, m2, mag_pairs, same_alphabet):
    tol = TOL_MAG * max(1.0, m1, m2)
    for (x, y) in mag_pairs:
        if abs(m1 - x) <= tol and abs(m2 - y) <= tol:
            return True
        if same_alphabet and abs(m1 - y) <= tol and abs(m2 - x) <= tol:
            return True
    return False


def filter_candidates(cands, mag_pairs, moms, mod1, mod2):
    """Apply F1 (modulus) then F2 (moments m<=24 + |s|^2/4/6, each with a
    tolerance relative to its own term scale).  Returns (survivors, counts).
    """
    same = (mod1 == mod2)
    counts = {'F1_modulus': 0, 'F2_moments': 0, 'F3_distribution': 0}
    surv = []
    for c in cands:
        m1, m2 = abs(c['a1']) ** 2, abs(c['a2']) ** 2
        if not _mag_match(m1, m2, mag_pairs, same):
            counts['F1_modulus'] += 1
            c['fate'] = 'F1_modulus'
            continue
        cm = mix_moments(c['a1'], c['a2'], mod1, mod2)
        term = abs(c['a1']) + abs(c['a2'])          # |s| <= term pointwise
        ok = (abs(cm['p'] - moms['p']) <= TOL_MOM * max(moms['p'], 1e-12) and
              abs(cm['q'] - moms['q']) <= TOL_MOM * max(moms['q'], 1e-12) and
              abs(cm['s6'] - moms['s6']) <= TOL_MOM * max(moms['s6'], 1e-12))
        if ok:
            for m in range(1, MMAX + 1):
                scale = max(abs(moms['E'][m]), term ** m, 1e-12)
                if abs(cm['E'][m] - moms['E'][m]) > TOL_MOM * scale:
                    ok = False
                    break
        if not ok:
            counts['F2_moments'] += 1
            c['fate'] = 'F2_moments'
            continue
        surv.append(c)
    return surv, counts


def equivalent(a1h, a2h, a1, a2, mod1, mod2):
    """Group-equivalence: a_k~ = a_k exp(2 pi i q_k / M_k), plus swap when
    the alphabets coincide.  Tested via the M_k-th powers."""

    def match(b1, b2):
        for bh, ah, Mk in ((b1, a1, SYM_ORDER[mod1]),
                           (b2, a2, SYM_ORDER[mod2])):
            if abs(bh ** Mk - ah ** Mk) > TOL_EQ * max(abs(ah) ** Mk, 1e-12):
                return False
        return True

    if match(a1h, a2h):
        return True
    return mod1 == mod2 and match(a2h, a1h)


def proof_identity_residuals(a1, a2, mod1, mod2, moms):
    """Residuals of the proof's key cross-term-vanishing identities,
    evaluated with the TRUE coefficients against the enumerated moments."""
    M1, M2 = SYM_ORDER[mod1], SYM_ORDER[mod2]
    mu1, mu2 = MU[mod1], MU[mod2]
    p, q, E = moms['p'], moms['q'], moms['E']
    x, y = abs(a1) ** 2, abs(a2) ** 2
    res = {}
    is_b = (mod1 == 'BPSK', mod2 == 'BPSK')
    if all(is_b):
        r2 = a1 ** 2 + a2 ** 2
        res['Es2'] = abs(E[2] - r2) / max(abs(r2), 1e-12)
        res['abs4'] = abs(q - (abs(r2) ** 2 + 4 * x * y)) / max(q, 1e-12)
        u1, u2 = a1 ** 2, a2 ** 2
        res['Es4'] = abs(E[4] - (u1 ** 2 + u2 ** 2 + 6 * u1 * u2)) \
            / max(abs(E[4]), 1e-12)
    elif any(is_b):
        b = 0 if is_b[0] else 1
        t = 1 - b
        ab, at = (a1, a2)[b], (a1, a2)[t]
        Mt = (M1, M2)[t]
        mut = (mu1, mu2)[t]
        res['Es2'] = abs(E[2] - ab ** 2) / max(abs(ab ** 2), 1e-12)
        sep = at ** Mt * mut[Mt] + ab ** Mt
        res['EsM_sep'] = abs(E[Mt] - sep) / max(abs(E[Mt]), 1e-12)
    elif M1 != M2:
        s = 0 if M1 < M2 else 1
        big = 1 - s
        ms, mb = (M1, M2)[s], (M1, M2)[big]
        mus, mub = (mu1, mu2)[s], (mu1, mu2)[big]
        a_s, a_b = (a1, a2)[s], (a1, a2)[big]
        res['EsM_small'] = abs(E[ms] - a_s ** ms * mus[ms]) \
            / max(abs(E[ms]), 1e-12)
        sep = a_s ** mb * mus[mb] + a_b ** mb * mub[mb]
        res['EsM_big'] = abs(E[mb] - sep) / max(abs(E[mb]), 1e-12)
    else:
        M = M1
        u1, u2 = a1 ** M * mu1[M], a2 ** M * mu2[M]
        alpha = mu1[2 * M] / mu1[M] ** 2
        beta = mu2[2 * M] / mu2[M] ** 2
        gam = float(math.comb(2 * M, M))
        res['EsM_sum'] = abs(E[M] - (u1 + u2)) / max(abs(E[M]), 1e-12)
        q2 = alpha * u1 ** 2 + gam * u1 * u2 + beta * u2 ** 2
        res['Es2M_quad'] = abs(E[2 * M] - q2) / max(abs(E[2 * M]), 1e-12)
        res['abs4'] = abs(q - (KAPPA[mod1] * x ** 2 + KAPPA[mod2] * y ** 2
                               + 4 * x * y)) / max(q, 1e-12)
    return {k: float(v) for k, v in res.items()}


def run_trial(a1, a2, mod1, mod2):
    """Full certificate pipeline for one generic (a1, a2)."""
    moms = mix_moments(a1, a2, mod1, mod2)
    cands, mag_pairs, flags = solve_candidates(mod1, mod2, moms)
    surv, rej = filter_candidates(cands, mag_pairs, moms, mod1, mod2)
    # F3: exact distribution certificate
    truth_pts = mix_support(a1, a2, mod1, mod2)
    scale = math.sqrt(max(moms['p'], 1e-12))
    final = []
    for c in surv:
        if multiset_equal(truth_pts, mix_support(c['a1'], c['a2'], mod1, mod2),
                          scale):
            final.append(c)
        else:
            c['fate'] = 'F3_distribution'
            rej['F3_distribution'] += 1
    equiv = [equivalent(c['a1'], c['a2'], a1, a2, mod1, mod2) for c in final]
    success = len(final) >= 1 and all(equiv)
    return {'success': success, 'n_candidates': len(cands),
            'n_survivors': len(final), 'rejections': rej, 'flags': flags,
            'candidates': cands,
            'residuals': proof_identity_residuals(a1, a2, mod1, mod2, moms),
            'equiv': equiv}


# ---------------------------------------------------------------------------
# benchmark sweep over the 10 constellation pairs
# ---------------------------------------------------------------------------
def draw_generic(rng, tie_tol=0.02):
    """Generic (a1, a2): |a1| in [0.7,1.3], ratio in [0.5,1.5], phases
    uniform.  Returns (a1, a2, n_excluded_draws).

    2026-10-06 (reviewer follow-up): unit-ratio draws are NO LONGER
    excluded — the benchmark operates at SIR ~= 0 dB, and for equal-kappa
    constellations the tie degeneracy is the S_2 swap inside G (handled in
    solve_candidates / equivalent).  tie_tol is retained only so that the
    returned exclusion counter stays comparable with the archived JSON
    (it is always 0 now)."""
    n_ex = 0
    r = rng.uniform(0.5, 1.5)
    m1 = rng.uniform(0.7, 1.3)
    ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
    return m1 * np.exp(1j * ph1), m1 * r * np.exp(1j * ph2), n_ex


def sweep_pairs(n_trials, seed):
    out = {}
    for idx, (mod1, mod2) in enumerate(
            itertools.combinations_with_replacement(MODS, 2)):
        rng = np.random.default_rng([seed, idx])
        n_ok, n_deg, n_ex = 0, 0, 0
        max_res = 0.0
        rej_tot = {'F1_modulus': 0, 'F2_moments': 0, 'F3_distribution': 0}
        n_cand_tot, n_surv_tot = 0, 0
        failures = []
        # equal-order unequal-alphabet (QPSK+16QAM): fate of the spurious
        # quadratic root under the modulus constraint
        spur = None
        if (mod1, mod2) == ('QPSK', '16QAM'):
            spur = {'n_trials': 0, 'n_spurious_roots': 0,
                    'rejected_F1': 0, 'passed_F1': 0,
                    'passed_F1_then_rejected_F2F3': 0,
                    'both_roots_pass_modulus_trials': 0}
        for t in range(n_trials):
            a1, a2, ex = draw_generic(rng)
            n_ex += ex
            r = run_trial(a1, a2, mod1, mod2)
            if r['flags']:
                n_deg += 1
                continue
            max_res = max(max_res, *r['residuals'].values())
            n_cand_tot += r['n_candidates']
            n_surv_tot += r['n_survivors']
            for k in rej_tot:
                rej_tot[k] += r['rejections'][k]
            if r['success']:
                n_ok += 1
            elif len(failures) < 5:
                failures.append({'trial': t, 'a1': [a1.real, a1.imag],
                                 'a2': [a2.real, a2.imag],
                                 'n_survivors': r['n_survivors'],
                                 'equiv': r['equiv']})
            if spur is not None:
                # identify the spurious root: the equal-order candidate whose
                # u2 is farthest from the truth's u2 = a2^4 mu_4^(16QAM)
                u2_true = a2 ** 4 * MU['16QAM'][4]
                roots = [c for c in r['candidates']
                         if c['kind'] == 'equal_order_root']
                if len(roots) == 2:
                    spur['n_trials'] += 1
                    d = [abs(c['u2'] - u2_true) for c in roots]
                    i_sp = int(np.argmax(d))
                    sp = roots[i_sp]
                    if abs(sp['u2'] - u2_true) > 1e-6 * max(abs(u2_true), 1):
                        spur['n_spurious_roots'] += 1
                        fate = sp.get('fate')
                        if fate == 'F1_modulus':
                            spur['rejected_F1'] += 1
                        else:
                            spur['passed_F1'] += 1
                            if fate in ('F2_moments', 'F3_distribution'):
                                spur['passed_F1_then_rejected_F2F3'] += 1
                            if roots[1 - i_sp].get('fate') != 'F1_modulus':
                                spur['both_roots_pass_modulus_trials'] += 1
        n_solved = n_trials - n_deg
        out[f'{mod1}+{mod2}'] = {
            'M': [SYM_ORDER[mod1], SYM_ORDER[mod2]],
            'n_trials': n_trials, 'n_solved': n_solved,
            'n_degenerate_flags': n_deg,
            'n_excluded_tie_draws': n_ex,
            'n_success': n_ok,
            'success_rate': n_ok / max(n_solved, 1),
            'max_proof_identity_residual': float(max_res),
            'mean_candidates': n_cand_tot / max(n_solved, 1),
            'mean_survivors': n_surv_tot / max(n_solved, 1),
            'rejections_total': rej_tot,
            'spurious_root_modulus_check_QPSK_16QAM': spur,
            'failures': failures}
    return out


def unit_ratio_check(n_trials, seed):
    """Dedicated certification at EXACT amplitude ratio 1 (the benchmark's
    SIR ~= 0 dB operating point): a2 = a1 * exp(j phi), |a1| ~ U(0.7,1.3),
    phases uniform.  Per pair: success rate over the full certificate
    pipeline (F1/F2/F3 + group equivalence incl. S_2 swap)."""
    out = {}
    for idx, (mod1, mod2) in enumerate(
            itertools.combinations_with_replacement(MODS, 2)):
        rng = np.random.default_rng([seed, 100 + idx])
        n_ok, n_deg, max_res = 0, 0, 0.0
        rej_tot = {'F1_modulus': 0, 'F2_moments': 0, 'F3_distribution': 0}
        failures = []
        for t in range(n_trials):
            m1 = rng.uniform(0.7, 1.3)
            ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
            a1 = m1 * np.exp(1j * ph1)
            a2 = m1 * np.exp(1j * ph2)              # EXACT unit ratio
            r = run_trial(a1, a2, mod1, mod2)
            if r['flags']:
                n_deg += 1
                continue
            max_res = max(max_res, *r['residuals'].values())
            for k in rej_tot:
                rej_tot[k] += r['rejections'][k]
            if r['success']:
                n_ok += 1
            elif len(failures) < 5:
                failures.append({'trial': t, 'a1': [a1.real, a1.imag],
                                 'a2': [a2.real, a2.imag],
                                 'n_survivors': r['n_survivors'],
                                 'equiv': r['equiv']})
        out[f'{mod1}+{mod2}'] = {
            'n_trials': n_trials, 'n_solved': n_trials - n_deg,
            'n_degenerate_flags': n_deg, 'n_success': n_ok,
            'success_rate': n_ok / max(n_trials - n_deg, 1),
            'max_proof_identity_residual': float(max_res),
            'rejections_total': rej_tot, 'failures': failures}
    return out
# ---------------------------------------------------------------------------
# negative control (a): constellation-collision rotations
# ---------------------------------------------------------------------------
def _ill_conditioning(a1, a2, mod1, mod2, rng, n_rep=50):
    """Median class-recovery error (in the M-th powers) when the moment
    inputs are perturbed at relative level eps.  sqrt(eps) growth at the
    tie (double root) vs linear growth at a generic point."""
    moms0 = mix_moments(a1, a2, mod1, mod2)
    M1, M2 = SYM_ORDER[mod1], SYM_ORDER[mod2]
    out = {}
    for eps in (1e-12, 1e-10, 1e-8, 1e-6):
        errs = []
        for _ in range(n_rep):
            mp = dict(moms0)
            sc = max(moms0['p'], 1e-12)
            mp['p'] = moms0['p'] + eps * sc * rng.standard_normal()
            mp['q'] = moms0['q'] + eps * sc ** 2 * rng.standard_normal()
            E = moms0['E'].copy()
            for m in range(1, 17):
                E[m] += eps * sc ** (m / 2) * (
                    rng.standard_normal() + 1j * rng.standard_normal())
            mp['E'] = E
            cands, _, _ = solve_candidates(mod1, mod2, mp)
            best = np.inf
            for c in cands:
                err = min(max(abs(c['a1'] ** M1 - a1 ** M1),
                              abs(c['a2'] ** M2 - a2 ** M2)),
                          max(abs(c['a2'] ** M1 - a1 ** M1),
                              abs(c['a1'] ** M2 - a2 ** M2))
                          if mod1 == mod2 else np.inf)
                best = min(best, err)
            errs.append(float(best))
        out[f'{eps:.0e}'] = float(np.median(errs))
    return out


def _pmf_search(a1, a2, mod1, mod2, sigma=0.5):
    """Grid search for NON-equivalent (a1~, a2~) with an identical mixture
    pmf.  Global-phase quotient a1~ = 1 (any pmf match can be rotated so
    that a1~ is real positive); grid over (r', phi') of a2~ = r' e^{j phi'};
    discrepancy = MMD^2 between the 16-point multisets (multiplicities
    included; 0 iff the multisets coincide)."""
    def kbatch(Y):                     # Y (n, P) -> (n,)
        d2 = np.abs(Y[:, :, None] - Y[:, None, :]) ** 2
        return np.exp(-d2 / (2 * sigma ** 2)).mean(axis=(1, 2))

    def kcross(X, Y):                  # X (P,), Y (n, P) -> (n,)
        d2 = np.abs(X[None, :, None] - Y[:, None, :]) ** 2
        return np.exp(-d2 / (2 * sigma ** 2)).mean(axis=(1, 2))

    # orbit intersected with a1~ = 1: a2~ = a2/a1 * z2 * conj(z1),
    # z_k a symmetry root; plus the swap if the alphabets coincide.
    orbit = []
    for q1 in range(SYM_ORDER[mod1]):
        for q2 in range(SYM_ORDER[mod2]):
            z = np.exp(2j * np.pi * q2 / SYM_ORDER[mod2]
                       - 2j * np.pi * q1 / SYM_ORDER[mod1])
            v = (a2 / a1) * z
            orbit.append((float(abs(v)), float(np.angle(v)) % (2 * np.pi)))
            if mod1 == mod2:
                v2 = (a1 / a2) * z
                orbit.append((float(abs(v2)),
                              float(np.angle(v2)) % (2 * np.pi)))
    r_grid = np.linspace(0.4, 2.5, 106)
    phi_grid = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    C1, C2 = CONSTELLATIONS[mod1], CONSTELLATIONS[mod2]
    X = mix_support(a1, a2, mod1, mod2)
    kxx = float(kbatch(X[None, :])[0])
    ephi = np.exp(1j * phi_grid)
    D = np.empty((len(r_grid), len(phi_grid)))
    for i, r in enumerate(r_grid):
        Y = (C1[None, :, None]
             + (r * ephi)[:, None, None] * C2[None, None, :])
        Y = Y.reshape(len(phi_grid), -1)             # (nphi, |C1|*|C2|)
        D[i] = kxx + kbatch(Y) - 2 * kcross(X, Y)
    mask = np.zeros(D.shape, bool)
    for ro, po in orbit:
        dphi = np.abs((phi_grid - po + np.pi) % (2 * np.pi) - np.pi)
        mask |= ((np.abs(r_grid - ro) <= 0.02 * max(ro, 1.0))[:, None]
                 & (dphi <= np.deg2rad(1.5))[None, :])
    i0 = np.unravel_index(np.argmin(D), D.shape)
    Dm = np.where(mask, np.inf, D)
    i1 = np.unravel_index(np.argmin(Dm), Dm.shape)
    exact = D < 1e-8          # exact multiset match (up to fp)
    return {'grid': {'r': [0.4, 2.5, 106], 'phi_deg': [0, 360, 720],
                     'mmd_sigma': sigma},
            'min_mmd2_overall': float(D[i0]),
            'argmin_overall': [float(r_grid[i0[0]]),
                               float(np.rad2deg(phi_grid[i0[1]]))],
            'n_exact_matches': int(exact.sum()),
            'exact_matches_all_on_orbit': bool(np.all(mask[exact])),
            'min_mmd2_outside_orbit': float(Dm[i1]),
            'argmin_outside_orbit': [float(r_grid[i1[0]]),
                                     float(np.rad2deg(phi_grid[i1[1]]))],
            'n_orbit_grid_points_masked': int(mask.sum())}


def control_collision(seed):
    rng = np.random.default_rng([seed, 999])
    res = {}
    configs = {'QPSK+QPSK_eqpow_phi90': (1.0 + 0j, np.exp(1j * np.pi / 2)),
               'QPSK+QPSK_sqrt2_phi45': (1.0 + 0j,
                                         np.sqrt(2) * np.exp(1j * np.pi / 4))}
    for name, (a1, a2) in configs.items():
        mod1 = mod2 = 'QPSK'
        S = mix_support(a1, a2, mod1, mod2)
        # (i) symbol-level non-injectivity (Thm-1 zero contact)
        uniq, counts = np.unique(np.round(S, 9), return_counts=True)
        dmin = float(min(abs(x - y) for x, y in itertools.combinations(S, 2)))
        # (ii) the certificate pipeline AT the collision
        trial = run_trial(a1, a2, mod1, mod2)
        res[name] = {
            'config': {'a1': [a1.real, a1.imag], 'a2': [a2.real, a2.imag]},
            'symbol_map': {'n_pairs': int(S.size),
                           'n_distinct_mixture_points': int(uniq.size),
                           'max_multiplicity': int(counts.max()),
                           'd_min': dmin},
            'excluded_by_legacy_tie_rule': bool(
                abs(abs(a2) / abs(a1) - 1.0) < 0.02),
            'unit_ratio_draws_included_since': '2026-10-06',
            'certificate_at_collision': {
                'success': trial['success'], 'flags': trial['flags'],
                'n_candidates': trial['n_candidates'],
                'n_survivors': trial['n_survivors']},
            'ill_conditioning_median_class_error':
                _ill_conditioning(a1, a2, mod1, mod2, rng),
            'pmf_search': _pmf_search(a1, a2, mod1, mod2),
        }
    # generic reference point for the ill-conditioning contrast
    res['generic_reference_QPSK+QPSK_r0.9_phi0.7'] = {
        'ill_conditioning_median_class_error': _ill_conditioning(
            1.0 + 0j, 0.9 * np.exp(1j * 0.7), 'QPSK', 'QPSK', rng)}
    return res


# ---------------------------------------------------------------------------
# negative control (b): continuous alphabet -> Prop-1 gauge freedom
# ---------------------------------------------------------------------------
def control_gaussian(seed):
    rng = np.random.default_rng([seed, 1000])
    a1, a2 = 1.0 * np.exp(1j * 0.3), 0.8 * np.exp(1j * 2.1)
    p = abs(a1) ** 2 + abs(a2) ** 2

    def gauss_moms(b1, b2):
        pc = abs(b1) ** 2 + abs(b2) ** 2
        return {'p': pc, 'q': 2 * pc ** 2,
                'E': np.array([0j] * (MMAX + 1))}   # circular: E[s^m] = 0

    truth = gauss_moms(a1, a2)
    # candidates: arbitrary phase rotations + power re-allocations
    n_cand = 40
    max_dev = 0.0
    n_equivalent = 0
    for _ in range(n_cand):
        t = rng.uniform(0.05, 0.95) * p
        th1, th2 = rng.uniform(0, 2 * np.pi, 2)
        b1 = np.sqrt(t) * np.exp(1j * th1)
        b2 = np.sqrt(max(p - t, 0.0)) * np.exp(1j * th2)
        cm = gauss_moms(b1, b2)
        dev = max(abs(cm['p'] - truth['p']), abs(cm['q'] - truth['q']),
                  float(np.max(np.abs(cm['E'] - truth['E']))))
        max_dev = max(max_dev, float(dev))
        # a finite-group relation to the truth would need |b_k| = |a_k|
        # AND b_k/a_k a root of unity — generically neither holds
        if abs(abs(b1) - abs(a1)) < 1e-9 and abs(abs(b2) - abs(a2)) < 1e-9:
            n_equivalent += 1
    # the moment phase-step is vacuous: E[s^m] = 0 for all m >= 1
    phase_info = float(np.max(np.abs(truth['E'][1:])))
    # empirical check: Monte Carlo samples of truth vs a rotated candidate
    n_mc = 200_000
    c1 = (rng.standard_normal(n_mc) + 1j * rng.standard_normal(n_mc)) \
        / np.sqrt(2)
    c2 = (rng.standard_normal(n_mc) + 1j * rng.standard_normal(n_mc)) \
        / np.sqrt(2)
    s_truth = a1 * c1 + a2 * c2
    th1, th2 = 0.83, 2.44
    s_rot = a1 * np.exp(1j * th1) * c1 + a2 * np.exp(1j * th2) * c2
    mc = {'Es2_truth': complex(np.mean(s_truth ** 2)),
          'Es2_rotated': complex(np.mean(s_rot ** 2)),
          'Es4_truth': complex(np.mean(s_truth ** 4)),
          'Es4_rotated': complex(np.mean(s_rot ** 4)),
          'abs2_truth': float(np.mean(np.abs(s_truth) ** 2)),
          'abs2_rotated': float(np.mean(np.abs(s_rot) ** 2)),
          'mc_se_Es2': float(np.std(s_truth ** 2) / np.sqrt(n_mc))}
    return {
        'model': 'c_k ~ CN(0,1) i.i.d.; s = a1 c1 + a2 c2 ~ CN(0, p), '
                 'p = |a1|^2 + |a2|^2',
        'truth': {'a1': [a1.real, a1.imag], 'a2': [a2.real, a2.imag],
                  'p': p},
        'n_candidates_tested': n_cand,
        'max_moment_deviation_of_candidates': max_dev,
        'n_candidates_with_truth_moduli': n_equivalent,
        'phase_moment_information_max_abs_Es^m_m>=1': phase_info,
        'interpretation': 'all raw moments E[s^m] vanish and the second/'
                          'fourth absolute moments depend only on the power '
                          'sum p: arbitrary per-source phase rotations and '
                          'any power re-allocation t -> (t, p-t) leave the '
                          'mixture distribution invariant — continuous gauge '
                          'freedom (Prop 1), in contrast to the finite group '
                          'of Prop 2',
        'monte_carlo_sanity': mc,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description='Prop 2 (prop:discrete) identifiability certificate')
    ap.add_argument('--n_trials', type=int, default=200,
                    help='generic (a1,a2) draws per constellation pair')
    ap.add_argument('--seed', type=int, default=20261004)
    ap.add_argument('--out', default=os.path.join(
        RESULTS_DIR, 'theory_identifiability.json'))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {'config': {'n_trials': args.n_trials, 'seed': args.seed,
                      'alphabets': MODS, 'symmetry_orders': SYM_ORDER,
                      'kappa_E|c|^4': KAPPA,
                      'mu2_E[c^2]': {m: [MU2[m].real, MU2[m].imag]
                                     for m in MODS},
                      'sampling': '|a1|~U(0.7,1.3), ratio~U(0.5,1.5) '
                                  '(unit ratio INCLUDED since 2026-10-06 — '
                                  'the equal-kappa tie degeneracy is the '
                                  'S_2 swap in G, not an exclusion), '
                                  'phases ~ U; separate unit_ratio block '
                                  'certifies EXACT ratio 1',
                      'mmax_verify': MMAX}}

    print('Part 1: 10 constellation pairs x '
          f'{args.n_trials} generic trials ...', flush=True)
    pairs = sweep_pairs(args.n_trials, args.seed)
    out['pairs'] = pairs

    print('Part 1b: exact unit-ratio certification (SIR = 0 dB) ...',
          flush=True)
    out['unit_ratio'] = unit_ratio_check(args.n_trials, args.seed)

    print('Part 2: negative control (a) — collision rotations ...',
          flush=True)
    out['control_collision'] = control_collision(args.seed)

    print('Part 3: negative control (b) — continuous (Gaussian) alphabet ...',
          flush=True)
    out['control_gaussian'] = control_gaussian(args.seed)

    all_ok = all(v['success_rate'] == 1.0 for v in pairs.values())
    ur = out['unit_ratio']
    ur_ok = all(v['success_rate'] == 1.0 for v in ur.values())
    out['summary'] = {
        'all_pairs_100pct': bool(all_ok),
        'pair_success_rates': {k: v['success_rate'] for k, v in
                               pairs.items()},
        'unit_ratio_all_pairs_100pct': bool(ur_ok),
        'unit_ratio_success_rates': {k: v['success_rate']
                                     for k, v in ur.items()},
        'max_proof_identity_residual_overall':
            max(v['max_proof_identity_residual'] for v in pairs.values()),
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # ---- printed summary ----
    print(f'\n{"pair":<14}{"M":<8}{"ok/solved":<12}{"rate":<7}'
          f'{"excl.tie":<9}{"degen":<6}{"maxResid":<10}{"rejF1/F2/F3"}')
    for k, v in pairs.items():
        r = v['rejections_total']
        print(f'{k:<14}{str(v["M"]):<8}'
              f'{v["n_success"]}/{v["n_solved"]:<10}'
              f'{v["success_rate"]:<7.3f}{v["n_excluded_tie_draws"]:<9}'
              f'{v["n_degenerate_flags"]:<6}'
              f'{v["max_proof_identity_residual"]:<10.2e}'
              f'{r["F1_modulus"]}/{r["F2_moments"]}/{r["F3_distribution"]}')
    print(f'\nEXACT unit-ratio certification (SIR = 0 dB, '
          f'{args.n_trials} draws/pair):')
    for k, v in ur.items():
        r = v['rejections_total']
        print(f'  {k:<13} ok {v["n_success"]}/{v["n_solved"]} '
              f'(rate {v["success_rate"]:.3f}, degen '
              f'{v["n_degenerate_flags"]}, maxResid '
              f'{v["max_proof_identity_residual"]:.2e}, rejF1/F2/F3 '
              f'{r["F1_modulus"]}/{r["F2_moments"]}/{r["F3_distribution"]})')
    print(f'  unit-ratio ALL PAIRS 100%: {ur_ok}')
    spur = pairs['QPSK+16QAM']['spurious_root_modulus_check_QPSK_16QAM']
    if spur:
        print(f'\nQPSK+16QAM spurious-root modulus check: '
              f'{spur["rejected_F1"]}/{spur["n_spurious_roots"]} spurious '
              f'roots rejected by F1 (modulus); passed F1: '
              f'{spur["passed_F1"]} (of which rejected later by F2/F3: '
              f'{spur["passed_F1_then_rejected_F2F3"]}); trials where both '
              f'roots satisfy the modulus constraint: '
              f'{spur["both_roots_pass_modulus_trials"]}')
    cc = out['control_collision']
    for name, v in cc.items():
        if 'symbol_map' not in v:
            continue
        sm = v['symbol_map']
        print(f'\ncollision {name}: distinct mixture points '
              f'{sm["n_distinct_mixture_points"]}/{sm["n_pairs"]}, '
              f'd_min={sm["d_min"]:.2e}, excluded-by-legacy-tie-rule='
              f'{v["excluded_by_legacy_tie_rule"]}, certificate success='
              f'{v["certificate_at_collision"]["success"]} '
              f'(flags {v["certificate_at_collision"]["flags"]})')
        print(f'  ill-conditioning (median |da^M| vs moment eps): '
              f'{v["ill_conditioning_median_class_error"]}')
        ps = v['pmf_search']
        print(f'  pmf search: min MMD^2 overall = '
              f'{ps["min_mmd2_overall"]:.3e} at (r,phi_deg)='
              f'{ps["argmin_overall"]}; exact matches (MMD^2<1e-8): '
              f'{ps["n_exact_matches"]}, all on group orbit: '
              f'{ps["exact_matches_all_on_orbit"]}; min MMD^2 outside '
              f'orbit = {ps["min_mmd2_outside_orbit"]:.3e} at '
              f'(r,phi_deg)={ps["argmin_outside_orbit"]} (grid-resolution '
              f'continuity artifact)')
    print(f'\ngeneric reference ill-conditioning: '
          f'{cc["generic_reference_QPSK+QPSK_r0.9_phi0.7"]["ill_conditioning_median_class_error"]}')
    cg = out['control_gaussian']
    print(f'\ngaussian control: max moment deviation over '
          f'{cg["n_candidates_tested"]} rotated/re-allocated candidates = '
          f'{cg["max_moment_deviation_of_candidates"]:.2e}; phase-moment '
          f'information = '
          f'{cg["phase_moment_information_max_abs_Es^m_m>=1"]:.2e}')
    print(f'\nALL PAIRS 100%: {all_ok}')
    print(f'saved {args.out}')


if __name__ == '__main__':
    # constellation parity with the vendored tables (if importable)
    try:
        import signal_utils as _su
        for _m in MODS:
            assert np.allclose(CONSTELLATIONS[_m], _su.CONSTELLATIONS[_m]), _m
        print('[parity] constellations == signal_utils.CONSTELLATIONS')
    except ImportError:
        print('[parity] signal_utils not importable; using local tables')
    # alphabet sanity: zero mean, unit power, symmetry-order moment structure
    for _m in MODS:
        _c = CONSTELLATIONS[_m]
        assert abs(np.mean(_c)) < 1e-15
        assert abs(np.mean(np.abs(_c) ** 2) - 1) < 1e-12
        _M = SYM_ORDER[_m]
        assert abs(np.mean(_c ** _M)) > 0.5          # mu_M != 0
        assert all(abs(MU[_m][k]) < 1e-12 for k in range(1, _M))
    main()
