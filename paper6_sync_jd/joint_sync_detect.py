"""Paper 6 — E4 core: ECM joint synchronisation + detection at K=2
(plan §3.3/§4-E4; the E3 lesson drives the design).

Model (after nominal-f0 down-conversion, RRC matched filter, 0::sps
symbol grid), per burst, N = 256 symbols:

    r_n = A [ e^{j2π Δf1 n Ts} c1_n ; e^{j2π Δf2 n Ts} c2_n ] + w_n

with r [N] (mixture route, P=1, A = [a1 a2]) or r [P, N] with P=2 slot
streams (slot-aided route, A a 2x2 coupling matrix).  Unknowns per
burst: residual carrier offsets Δf1, Δf2 ∈ [-5, +10] Hz, complex gains
A (constant over the burst), and the symbols.  Per-source modulations
are KNOWN (primary variant, like ser_comp's reference side).

Why joint: E3 measured that independent per-slot blind sync at K=2 is
biased 2-4 Hz by the interferer's spectral line (SNR-flat; SIC does not
help), so sync and detection must be JOINT (Prop 4).

Algorithm — SAGE-flavoured ECM with phase-grid restarts.  Hard-learned
design rules (2026-09-18, see EXPERIMENT_LOG):
  * EM must run to CONVERGENCE at every frequency candidate used for
    ranking: the likelihood landscape at SIR≈0 separates truth from
    phantom pairs only after ~5+ decide/gain rounds (measured: truth
    E=0.63 vs phantom E=1.74 converged, INVERTED at 2-4 rounds).
  * Frequency moves are scored with EM re-estimated at the candidate;
    decision-frozen NLS is only used for the final in-basin polish.
  * The gain-phase init grid needs only cover [0, 2π/M1) x [0, 2π/M2)
    (per-source phase is absorbed by the M-fold constellation symmetry),
    i.e. M1*M2 inits, batch-evaluated.
  * Symbol subsampling (stride 4) for the search stages: same burst
    window, same spin physics, 4x cheaper.
  * The M-th-power "extreme lines" idea for the two frequencies does NOT
    work (cross terms dominate at SIR 0; measured 2026-09-18).

Pipeline (ecm_joint, blind route):
  A. Coarse 1.5 Hz 2-D grid over [-5,10]² (d1 <= d2), each point scored
     by converged phase-grid EM on the stride-4 subsampled stream.
  B. Fine 2-D grid (±1.2 Hz, 0.2 Hz) around EACH of the top-4 coarse
     points, warm-started converged EM (stride 4).
  C. Full-rate converged phase-grid EM at the best fine point.
  D. Polish: alternate joint decisions / LS gains with a small joint
     2-D NLS (±0.15 Hz, 0.05 Hz) until the residual energy stalls.
Oracle / slot-init route (df_init given): stages A/B skipped.

__main__ smoke test: synthetic K=2 bursts; checks (i) ECM beats separate
detection at SNR 20 for QPSK pairs, (ii) ECM ≈ oracle-frequency ECM,
(iii) polish energy non-increasing, (iv) noiseless burst -> near-perfect.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Vendored paper6 modules (self-contained; no sys.path tricks).
import config as C                                     # noqa: E402 (paper6)
from joint_detect import constellation_np, MOD_TYPES   # noqa: E402
from signal_utils import rrc_filter                    # noqa: E402

_FS = C.SignalConfig.sample_rate
_SPS = C.SignalConfig.signal_length // C.SignalConfig.n_symbols  # 16
_TS = _SPS / _FS                              # symbol period (1 ms)
_NSYM = C.SignalConfig.n_symbols
_DF_LO, _DF_HI = C.SyncConfig.freq_search_lo, C.SyncConfig.freq_search_hi


# ---------------------------------------------------------------------------
# Front-end
# ---------------------------------------------------------------------------
def mixture_symbols(mix, tau_samples=0.0):
    """Nominal-f0 down-conversion + RRC matched filter + 0::sps grid +
    unit-power normalisation.  mix: [T] complex -> r [N] complex.

    tau_samples (2026-09-22, timing-offset sensitivity probe): fractional
    sampling-grid offset in samples (1 symbol = 16 samples).  tau=0
    (default) is bit-identical to the historical behaviour; tau != 0
    samples the matched-filter output at n*sps + tau via linear
    interpolation (eval_robustness_sweeps.py)."""
    T = len(mix)
    t = np.arange(T) / _FS
    rrc = rrc_filter(C.SignalConfig.num_taps, C.SignalConfig.roll_off, _SPS)
    x = np.asarray(mix, dtype=np.complex128) * np.exp(
        -1j * 2 * np.pi * C.SyncConfig.nominal_carrier * t)
    z = np.convolve(x, rrc, mode='same')
    if tau_samples:
        pos = np.arange(0, _NSYM * _SPS, _SPS, dtype=np.float64) + tau_samples
        idx = np.arange(T, dtype=np.float64)
        zr = np.interp(pos, idx, z.real)
        zi = np.interp(pos, idx, z.imag)
        z = zr + 1j * zi
    else:
        z = z[0::_SPS][:_NSYM]
    return z / (np.sqrt(np.mean(np.abs(z) ** 2)) + 1e-10)


# ---------------------------------------------------------------------------
# Batched EM machinery (G gain-phase inits evaluated together)
# ---------------------------------------------------------------------------
def _steer(df, n_idx):
    return np.exp(1j * 2 * np.pi * df * n_idx * _TS)


def _batch_decisions(Rs, Ag, U, const1, const2):
    """Ag [G,P,2], U [2,N'], Rs [P,N'] -> per-init joint decisions
    (dec1, dec2) [G,N'] and residual energies [G]."""
    G, P, _ = Ag.shape
    N = Rs.shape[1]
    M1, M2 = len(const1), len(const2)
    # pred[g, n, i, j, p] = Ag[g,p,0]*u1[n]*c1[i] + Ag[g,p,1]*u2[n]*c2[j]
    pred = (Ag[:, None, None, None, :, 0]
            * (U[0][None, :, None, None, None] * const1[None, None, :, None, None])
            + Ag[:, None, None, None, :, 1]
            * (U[1][None, :, None, None, None] * const2[None, None, None, :, None]))
    # [G, N, M1, M2, P]
    d2 = np.sum(np.abs(Rs.T[None, :, None, None, :] - pred) ** 2, axis=-1)
    h = np.argmin(d2.reshape(G, N, M1 * M2), axis=-1)          # [G,N]
    return h // M2, h % M2


def _batch_ls_gains(Rs, U, dec1, dec2, const1, const2):
    """Batched A = R D^H (D D^H)^{-1}.  dec1/dec2 [G,N'] -> A [G,P,2]."""
    G, N = dec1.shape
    D = np.stack([U[0][None, :] * const1[dec1],
                  U[1][None, :] * const2[dec2]], axis=1)       # [G,2,N]
    g00 = np.sum(np.abs(D[:, 0]) ** 2, axis=1)
    g11 = np.sum(np.abs(D[:, 1]) ** 2, axis=1)
    g01 = np.sum(D[:, 0] * np.conj(D[:, 1]), axis=1)
    det = g00 * g11 - np.abs(g01) ** 2 + 1e-12
    # R D^H : [G,P] per channel
    rd0 = (Rs[None, :, :] @ np.conj(D[:, 0])[..., None])[..., 0]   # [G,P]
    rd1 = (Rs[None, :, :] @ np.conj(D[:, 1])[..., None])[..., 0]
    # A = RD^H G^{-1};  G^{-1} = [[g11, -g01], [-conj(g01), g00]] / det
    a0 = (rd0 * g11[:, None] - rd1 * np.conj(g01)[:, None]) / det[:, None]
    a1 = (-rd0 * g01[:, None] + rd1 * g00[:, None]) / det[:, None]
    return np.stack([a0, a1], axis=-1)                         # [G,P,2]


def _batch_energy(Rs, Ag, U, dec1, dec2, const1, const2):
    G, N = dec1.shape
    D = np.stack([U[0][None, :] * const1[dec1],
                  U[1][None, :] * const2[dec2]], axis=1)       # [G,2,N]
    pred = Ag @ D                                              # [G,P,N]
    return np.sum(np.abs(Rs[None] - pred) ** 2, axis=(1, 2))   # [G]


# Unbatched wrappers (single (A) hypothesis — polish stage)
def _joint_decisions(R, A, U, const1, const2):
    d1, d2 = _batch_decisions(R, A[None], U, const1, const2)
    return d1[0], d2[0]


def _ls_gains(R, U, dec1, dec2, const1, const2):
    return _batch_ls_gains(R, U, dec1[None], dec2[None], const1, const2)[0]


def _res_energy(R, A, U, dec1, dec2, const1, const2):
    return float(_batch_energy(R, A[None], U, dec1[None], dec2[None],
                               const1, const2)[0])


def em_converged(R, const1, const2, df1, df2, stride=4,
                 warm=None, n_ph=4, tol=1e-5, max_rounds=30):
    """EM to convergence at fixed (df1, df2).  Cold start: batched
    n_ph x n_ph gain-phase inits covering [0, 2π/M1) x [0, 2π/M2) (the
    per-source gain phase is absorbed by the M-fold constellation
    symmetry, so only one symmetry sector needs covering).  Warm start:
    warm=(A [P,2], dec1, dec2).  Returns dict with decisions on the
    (subsampled) grid used."""
    Rs = R[:, ::stride]
    P = Rs.shape[0]
    n_idx = np.arange(Rs.shape[1]) * stride
    U = np.stack([_steer(df1, n_idx), _steer(df2, n_idx)], axis=0)
    M1, M2 = len(const1), len(const2)

    if warm is not None:
        A0, d1w, d2w = warm
        Ag = A0[None, :, :]                                    # [1,P,2]
    else:
        ph1 = np.exp(1j * 2 * np.pi * np.arange(n_ph) / (M1 * n_ph))
        ph2 = np.exp(1j * 2 * np.pi * np.arange(n_ph) / (M2 * n_ph))
        G1, G2 = np.meshgrid(ph1, ph2, indexing='ij')
        Ag = np.stack([np.broadcast_to(G1.ravel()[:, None], (n_ph * n_ph, P)),
                       np.broadcast_to(G2.ravel()[:, None], (n_ph * n_ph, P))],
                      axis=-1).astype(np.complex128)           # [G,P,2]

    E = None
    dec1 = dec2 = None
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        dec1, dec2 = _batch_decisions(Rs, Ag, U, const1, const2)
        Ag = _batch_ls_gains(Rs, U, dec1, dec2, const1, const2)
        E2 = _batch_energy(Rs, Ag, U, dec1, dec2, const1, const2)
        if E is not None and np.all(E - E2 < tol * np.maximum(E, 1e-12)):
            E = E2
            break
        E = E2
    EM_STATS['em_calls'] += 1
    EM_STATS['em_rounds'] += rounds
    g = int(np.argmin(E))
    return {'E': float(E[g]), 'A': Ag[g], 'dec1': dec1[g], 'dec2': dec2[g],
            'df1': df1, 'df2': df2, 'rounds': rounds}


def ecm_joint(r, const1, const2, df_init=None, coarse_step=1.5,
              coarse_stride=4, fine_hw=1.2, fine_step=0.2,
              nls_hw=0.15, nls_step=0.05, polish_rounds=6,
              n_coarse_keep=4):
    """Joint sync+detection of a K=2 burst (SAGE-flavoured ECM).

    r: [N] or [P, N] complex symbol-rate observation(s).
    const1/const2: per-source constellations (known modulation).
    df_init: optional (df1, df2) start; stages A/B skipped if given
      (oracle-frequency V3 arm; slot-init V2 arm).
    """
    R = np.asarray(r, dtype=np.complex128)
    if R.ndim == 1:
        R = R[None, :]
    N = R.shape[1]
    n_full = np.arange(N)

    if df_init is None:
        # A. coarse grid, converged EM on subsampled symbols
        grid = np.arange(_DF_LO, _DF_HI + 1e-9, coarse_step)
        scored = []
        for d1 in grid:
            for d2 in grid:
                if d2 < d1:
                    continue
                res = em_converged(R, const1, const2, d1, d2,
                                   stride=coarse_stride)
                scored.append((res['E'], d1, d2))
        scored.sort(key=lambda t: t[0])
        # B. fine grid around each of the top-K, warm-started
        best = None
        for _E, d1, d2 in scored[:n_coarse_keep]:
            warm0 = em_converged(R, const1, const2, d1, d2,
                                 stride=coarse_stride)
            warm = (warm0['A'], warm0['dec1'], warm0['dec2'])
            for g1 in np.arange(d1 - fine_hw, d1 + fine_hw + 1e-9,
                                fine_step):
                for g2 in np.arange(d2 - fine_hw, d2 + fine_hw + 1e-9,
                                    fine_step):
                    res = em_converged(R, const1, const2, g1, g2,
                                       stride=coarse_stride, warm=warm)
                    if best is None or res['E'] < best['E']:
                        best = res
        f1, f2 = best['df1'], best['df2']
    else:
        f1, f2 = float(df_init[0]), float(df_init[1])

    # C. full-rate converged phase-grid EM
    res = em_converged(R, const1, const2, f1, f2, stride=1)
    dec1, dec2, A, E = res['dec1'], res['dec2'], res['A'], res['E']

    # D. polish: joint decisions / LS gains / joint 2-D NLS
    hist = [E]
    for _ in range(polish_rounds):
        bestn = None
        for g1 in np.arange(f1 - nls_hw, f1 + nls_hw + 1e-9, nls_step):
            for g2 in np.arange(f2 - nls_hw, f2 + nls_hw + 1e-9, nls_step):
                U = np.stack([_steer(g1, n_full), _steer(g2, n_full)])
                Af = _ls_gains(R, U, dec1, dec2, const1, const2)
                Ef = _res_energy(R, Af, U, dec1, dec2, const1, const2)
                EM_STATS['nls_evals'] += 1
                if bestn is None or Ef < bestn[0]:
                    bestn = (Ef, g1, g2, Af)
        En, g1, g2, A = bestn
        f1, f2 = g1, g2
        U = np.stack([_steer(f1, n_full), _steer(f2, n_full)])
        d1n, d2n = _joint_decisions(R, A, U, const1, const2)
        En2 = _res_energy(R, A, U, d1n, d2n, const1, const2)
        dec1, dec2 = d1n, d2n
        hist.append(En2)
        if E - En2 < 1e-4 * max(E, 1e-12):
            E = En2
            break
        E = En2
    return {'dec1': dec1, 'dec2': dec2, 'A': A, 'df1': f1, 'df2': f2,
            'res_energy': E, 'rounds': len(hist) - 1, 'hist': hist}


def separate_detect(r, const1, const2, df1, df2):
    """V0-style separate per-source detection on the (single) stream after
    de-spinning with the given frequencies."""
    z1 = r * np.conj(_steer(df1, np.arange(len(r))))
    z2 = r * np.conj(_steer(df2, np.arange(len(r))))
    d1 = np.argmin(np.abs(z1[:, None] - const1[None, :]) ** 2, axis=1)
    d2 = np.argmin(np.abs(z2[:, None] - const2[None, :]) ** 2, axis=1)
    return d1, d2


# EM-iteration accounting (complexity reporting): incremented by
# em_converged / ecm_joint; read + reset by evaluation drivers.
EM_STATS = {'em_calls': 0, 'em_rounds': 0, 'nls_evals': 0}


def marginal_decisions(R, A, df1, df2, const1, const2, sigma2):
    """Per-stream MARGINAL-MAP decisions from a converged ECM fit.

    The joint ECM rule minimises the PAIRWISE error; the Bayesian rule
    for per-SOURCE symbol error is marginal MAP,
        c1_n = argmax_{c1} (1/M2) sum_{c2} p(r_n | c1, c2)
    with the fitted (A, df) and noise level sigma2.  At high SNR the two
    rules coincide (the posterior concentrates on the joint mode); at
    low SNR marginal MAP is the per-source optimum.  R [P,N], A [P,2].
    Returns (m1, m2) decision index arrays [N]."""
    R = np.asarray(R, dtype=np.complex128)
    if R.ndim == 1:
        R = R[None, :]
    P, N = R.shape
    n_full = np.arange(N)
    U = np.stack([_steer(df1, n_full), _steer(df2, n_full)])
    M1, M2 = len(const1), len(const2)
    # pred[p, n, i, j] = A[p,0] u1_n c1_i + A[p,1] u2_n c2_j
    # pred[p, n, i, j] = A[p,0] u1_n c1_i + A[p,1] u2_n c2_j
    pred = (A[:, None, None, 0:1]
            * (U[0][None, :, None, None] * const1[None, None, :, None])
            + A[:, None, None, 1:2]
            * (U[1][None, :, None, None] * const2[None, None, None, :]))
    # pred [P, N, M1, M2]
    diff = R[:, :, None, None] - pred                  # [P, N, M1, M2]
    d2 = np.sum(np.abs(diff) ** 2, axis=0)             # [N, M1, M2]
    ll = -d2 / max(sigma2, 1e-12)
    ll -= ll.max(axis=(1, 2), keepdims=True)
    lik = np.exp(ll)
    m1 = np.argmax(lik.mean(axis=2), axis=1)
    m2 = np.argmax(lik.mean(axis=1), axis=1)
    return m1.astype(np.int64), m2.astype(np.int64)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    import torch
    import ser_comp
    from data_generator import generate_vark_mixture

    print("Testing paper6 joint_sync_detect (ECM) ...")
    np.random.seed(C.SEED)

    def rot_ser(dec_syms, const, ref_lab, m_order):
        best = 1.0
        for k in range(m_order):
            dv = dec_syms * np.exp(-1j * 2 * np.pi * k / m_order)
            d = np.argmin(np.abs(dv[:, None] - const[None, :]), axis=1)
            best = min(best, float(np.mean(d != ref_lab)))
        return best

    def pit(dec1, dec2, c1, c2, ref1, ref2, m1o, m2o, same):
        s_dir = 0.5 * (rot_ser(c1[dec1], c1, ref1, m1o)
                       + rot_ser(c2[dec2], c2, ref2, m2o))
        if same:
            s_sw = 0.5 * (rot_ser(c1[dec2], c1, ref1, m1o)
                          + rot_ser(c2[dec1], c2, ref2, m2o))
            return min(s_dir, s_sw)
        return s_dir

    def gen(mods, snr):
        return generate_vark_mixture(
            C.SignalConfig.signal_length, _FS, float(snr), mods, k=2,
            carrier_base=C.SignalConfig.carrier_base,
            freq_gap_range=C.SignalConfig.freq_gap_range,
            n_symbols=_NSYM, roll_off=C.SignalConfig.roll_off,
            num_taps=C.SignalConfig.num_taps,
            apply_fading=C.SignalConfig.apply_fading,
            fading_taps=C.SignalConfig.fading_taps, return_carriers=True)

    for mods, tag in [(['QPSK', 'QPSK'], 'QPSKxQPSK'),
                      (['8PSK', 'QPSK'], '8PSKxQPSK')]:
        s_ecm, s_sep, s_ora, monot = [], [], [], True
        t0 = time.time()
        for trial in range(3):
            mix, srcs, midx, cars = gen(mods, 20)
            c1, c2 = constellation_np(midx[0]), constellation_np(midx[1])
            r = mixture_symbols(mix)
            ref1 = ser_comp.ref_labels_only(
                torch.from_numpy(np.asarray(srcs[0], dtype=np.complex64)),
                mod_type=mods[0], carrier_freq=cars[0])
            ref2 = ser_comp.ref_labels_only(
                torch.from_numpy(np.asarray(srcs[1], dtype=np.complex64)),
                mod_type=mods[1], carrier_freq=cars[1])
            m1o = C.SyncConfig.sym_order[mods[0]]
            m2o = C.SyncConfig.sym_order[mods[1]]
            same = mods[0] == mods[1]

            res = ecm_joint(r, c1, c2)
            # polish-stage energy should be non-increasing up to the
            # sub-percent NLS/LS refit wiggle at the stall point
            h = np.asarray(res['hist'])
            monot &= bool(np.all(np.diff(h) <= 0.01 * np.maximum(h[:-1], 1e-12)))
            s_ecm.append(pit(res['dec1'], res['dec2'], c1, c2,
                             ref1, ref2, m1o, m2o, same))
            d1, d2 = separate_detect(r, c1, c2, res['df1'], res['df2'])
            s_sep.append(pit(d1, d2, c1, c2, ref1, ref2, m1o, m2o, same))
            res_o = ecm_joint(r, c1, c2,
                              df_init=(cars[0] - C.SyncConfig.nominal_carrier,
                                       cars[1] - C.SyncConfig.nominal_carrier))
            s_ora.append(pit(res_o['dec1'], res_o['dec2'], c1, c2,
                             ref1, ref2, m1o, m2o, same))
        dt = (time.time() - t0) / 3
        print(f"  {tag}: ECM={np.mean(s_ecm):.4f}  separate={np.mean(s_sep):.4f}"
              f"  oracle-df={np.mean(s_ora):.4f}  monotone-E={monot}  "
              f"{dt:.1f}s/burst")
        assert monot, "residual energy must not increase"
        assert np.mean(s_ecm) < np.mean(s_sep), "ECM must beat separate"

    # noiseless sanity: clean QPSK pair at SNR 40 -> near-perfect
    mix, srcs, midx, cars = gen(['QPSK', 'QPSK'], 40)
    c1 = c2 = constellation_np(midx[0])
    r = mixture_symbols(mix)
    res = ecm_joint(r, c1, c2)
    ref1 = ser_comp.ref_labels_only(
        torch.from_numpy(np.asarray(srcs[0], dtype=np.complex64)),
        mod_type='QPSK', carrier_freq=cars[0])
    ref2 = ser_comp.ref_labels_only(
        torch.from_numpy(np.asarray(srcs[1], dtype=np.complex64)),
        mod_type='QPSK', carrier_freq=cars[1])
    s = pit(res['dec1'], res['dec2'], c1, c2, ref1, ref2, 4, 4, True)
    print(f"  noiseless QPSKxQPSK: ECM SER={s:.4f}")
    assert s < 0.05

    print("\njoint_sync_detect smoke test passed!")
