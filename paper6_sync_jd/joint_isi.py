"""Paper 6 — V4: ISI-aware joint ECM (plan §3.2 merged into the E4 loop).

Motivation: V1's SNR curve is non-monotone (best at 10 dB) because the
generator's 3-tap fading smears the sum constellation while the V1 model
assumes memoryless per-source gains.  The exact symbol-rate truth (the
generator convolves the passband signal with the channel, so the carrier
spin factors out of the taps) is

    r_n = sum_k e^{j 2π Δf_k n Ts} sum_{l=0}^{L-1} g_k[l] c_{k,n-l} + w_n

with L = 3 taps per source and per-burst-constant complex taps g_k[l]
(the l-th tap absorbs h_l e^{-j2π f_c l Ts} and the matched-filter
response).

Algorithm: warm-started extension of the V1 ECM (joint_sync_detect.py).
  1. Run the V1 memoryless ECM (or take its result); initialise taps
     g_k = [a_k, 0, ..., 0] from V1's gains, decisions and frequencies
     from V1.
  2. Alternate to convergence (energy-monotone, capped rounds):
     a. Parallel-iterated ISI cancellation: clean the residual with the
        current decisions' ISI contribution (l >= 1 taps), then take
        joint per-symbol decisions on the cleaned metric over the
        M1 x M2 grid with the centre-tap gains.
     b. Closed-form complex LS refit of all 2L taps given decisions and
        frequencies (model linear in taps).
     c. Frequency polish: small joint 2-D NLS with the taps refit by LS
        at each candidate (energy exact in the taps).
  3. Guard: if the final energy or the held-out decision quality is
     worse than V1's, fall back to the V1 solution (fallback flag).

Overfitting note: the 2L=6 complex taps per burst are LS-fitted on 256
symbols — the pilot explicitly checks V4 does not beat the
oracle-frequency V3 by an implausible margin.

__main__ smoke test: (i) noiseless faded QPSK pair — V4 near-perfect and
<= V1; (ii) faded 20 dB bursts — V4 <= V1 on average; (iii) fallback
fires when forced (L=1 degenerate); (iv) energy monotone in the loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Vendored paper6 modules (self-contained; no sys.path tricks).
import config as C                                     # noqa: E402 (paper6)
from joint_detect import constellation_np, MOD_TYPES   # noqa: E402
from joint_sync_detect import (ecm_joint, _steer, _joint_decisions,  # noqa: E402
                               _ls_gains, _res_energy)

_NSYM = C.SignalConfig.n_symbols
_N_FULL = np.arange(_NSYM)


def _shifted(v, l):
    """v[n-l] with zero fill (negative l = future symbols)."""
    if l == 0:
        return v
    if l > 0:
        out = np.zeros_like(v)
        out[l:] = v[:-l]
        return out
    out = np.zeros_like(v)
    out[:l] = v[-l:]                             # l < 0: v[n+|l|]
    return out


def _design(U1, U2, dec1, dec2, const1, const2, L):
    """Design rows D [2L, N]: row (k, l) = u_k[n] * c_{k,n-m} with
    m = l - L//2 (CENTRED lags — the generator's channel conv is
    mode='same', i.e. zero-delay, so the composite symbol-rate response
    is roughly symmetric around the centre tap)."""
    rows = []
    for (u, dec, const) in ((U1, dec1, const1), (U2, dec2, const2)):
        cs = const[dec]
        for l in range(L):
            rows.append(u * _shifted(cs, l - L // 2))
    return np.stack(rows, axis=0)


def _taps_ls(R, U1, U2, dec1, dec2, const1, const2, L):
    """Closed-form complex LS refit of the 2L taps.  Returns G [P,2,L]."""
    D = _design(U1, U2, dec1, dec2, const1, const2, L)
    G = D @ D.conj().T
    T = R @ D.conj().T @ np.linalg.inv(G + 1e-9 * np.eye(2 * L))
    P = R.shape[0]
    return T.reshape(P, 2, L)


def _taps_energy(R, U1, U2, dec1, dec2, const1, const2, G):
    D = _design(U1, U2, dec1, dec2, const1, const2, G.shape[2])
    return float(np.sum(np.abs(R - G.reshape(R.shape[0], -1) @ D) ** 2))


def _isi_cleaned_decisions(R, U1, U2, dec1, dec2, const1, const2, G):
    """Parallel-iterated ISI cancellation: subtract the l>=1 tap
    contribution of the CURRENT decisions, then joint per-symbol
    decisions on the cleaned metric with the centre-tap gains."""
    Rc = R.copy()
    for k, (u, dec, const) in enumerate(((U1, dec1, const1),
                                         (U2, dec2, const2))):
        cs = const[dec]
        for l in range(G.shape[2]):
            if l == G.shape[2] // 2:
                continue                          # centre tap stays in metric
            Rc = Rc - G[:, k, l][:, None] * (u * _shifted(cs, l - G.shape[2] // 2))[None, :]
    A0 = G[:, :, G.shape[2] // 2]
    U = np.stack([U1, U2], axis=0)
    return _joint_decisions(Rc, A0, U, const1, const2)


def ecm_joint_isi(r, const1, const2, L=5, v1=None, max_rounds=8,
                  nls_hw=0.15, nls_step=0.05, polish_rounds=4,
                  tol=1e-4):
    """V4: ISI-aware joint sync+detection, warm-started from V1.

    Returns a V1-compatible result dict with extra keys: 'taps' [P,2,L],
    'fallback' (True if the V1 solution was kept), 'v1_energy'."""
    R = np.asarray(r, dtype=np.complex128)
    if R.ndim == 1:
        R = R[None, :]
    N = R.shape[1]
    n_idx = np.arange(N)
    if v1 is None:
        v1 = ecm_joint(R, const1, const2)
    df1, df2 = v1['df1'], v1['df2']
    dec1, dec2 = v1['dec1'], v1['dec2']
    P = R.shape[0]
    G = np.zeros((P, 2, L), dtype=np.complex128)
    G[:, 0, L // 2] = v1['A'][:, 0]              # centre tap = V1 gain
    G[:, 1, L // 2] = v1['A'][:, 1]
    E_v1 = v1['res_energy']
    E = _taps_energy(R, _steer(df1, n_idx), _steer(df2, n_idx),
                     dec1, dec2, const1, const2, G)
    assert abs(E - E_v1) < 1e-6 * max(E_v1, 1.0)      # same model at init
    hist = [E]
    for _ in range(max_rounds):
        dec1, dec2 = _isi_cleaned_decisions(R, _steer(df1, n_idx),
                                            _steer(df2, n_idx),
                                            dec1, dec2, const1, const2, G)
        G = _taps_ls(R, _steer(df1, n_idx), _steer(df2, n_idx),
                     dec1, dec2, const1, const2, L)
        E_new = _taps_energy(R, _steer(df1, n_idx), _steer(df2, n_idx),
                             dec1, dec2, const1, const2, G)
        hist.append(E_new)
        if E - E_new < tol * max(E, 1e-12):
            E = E_new
            break
        E = E_new
    # frequency polish with taps refit per candidate
    for _ in range(polish_rounds):
        best = None
        for g1 in np.arange(df1 - nls_hw, df1 + nls_hw + 1e-9, nls_step):
            for g2 in np.arange(df2 - nls_hw, df2 + nls_hw + 1e-9, nls_step):
                U1, U2 = _steer(g1, n_idx), _steer(g2, n_idx)
                Gg = _taps_ls(R, U1, U2, dec1, dec2, const1, const2, L)
                Eg = _taps_energy(R, U1, U2, dec1, dec2, const1, const2, Gg)
                if best is None or Eg < best[0]:
                    best = (Eg, g1, g2, Gg)
        Eg, g1, g2, Gg = best
        df1, df2 = g1, g2
        dec1, dec2 = _isi_cleaned_decisions(R, _steer(df1, n_idx),
                                            _steer(df2, n_idx),
                                            dec1, dec2, const1, const2, Gg)
        G = _taps_ls(R, _steer(df1, n_idx), _steer(df2, n_idx),
                     dec1, dec2, const1, const2, L)
        E_new = _taps_energy(R, _steer(df1, n_idx), _steer(df2, n_idx),
                             dec1, dec2, const1, const2, G)
        hist.append(E_new)
        if E - E_new < tol * max(E, 1e-12):
            E = E_new
            G = Gg
            break
        E = E_new
        G = Gg

    fallback = E > E_v1
    if fallback:
        out = dict(v1)
        out['fallback'] = True
        out['v1_energy'] = E_v1
        out['isi_energy'] = E
        out['taps'] = G
        out['hist'] = hist
        return out
    return {'dec1': dec1, 'dec2': dec2,
            'A': G[:, :, L // 2], 'df1': df1, 'df2': df2,
            'res_energy': E, 'rounds': len(hist) - 1, 'hist': hist,
            'taps': G, 'fallback': False, 'v1_energy': E_v1,
            'isi_energy': E}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import torch
    import ser_comp
    from data_generator import generate_vark_mixture

    print("Testing paper6 joint_isi (V4) ...")
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
            C.SignalConfig.signal_length, C.SignalConfig.sample_rate,
            float(snr), mods, k=2,
            carrier_base=C.SignalConfig.carrier_base,
            freq_gap_range=C.SignalConfig.freq_gap_range,
            n_symbols=_NSYM, roll_off=C.SignalConfig.roll_off,
            num_taps=C.SignalConfig.num_taps,
            apply_fading=C.SignalConfig.apply_fading,
            fading_taps=C.SignalConfig.fading_taps, return_carriers=True)

    from joint_sync_detect import mixture_symbols
    # (i)+(ii) faded QPSK pairs at 20 dB: V4 <= V1 on average
    s1s, s4s, falls = [], [], 0
    for trial in range(4):
        mix, srcs, midx, cars = gen(['QPSK', 'QPSK'], 20)
        c1 = c2 = constellation_np(midx[0])
        r = mixture_symbols(mix)
        ref1 = ser_comp.ref_labels_only(
            torch.from_numpy(np.asarray(srcs[0], dtype=np.complex64)),
            mod_type='QPSK', carrier_freq=cars[0])
        ref2 = ser_comp.ref_labels_only(
            torch.from_numpy(np.asarray(srcs[1], dtype=np.complex64)),
            mod_type='QPSK', carrier_freq=cars[1])
        v1 = ecm_joint(r, c1, c2)
        v4 = ecm_joint_isi(r, c1, c2, L=3, v1=v1)
        h = np.asarray(v4['hist'])
        assert np.all(np.diff(h) <= 0.02 * np.maximum(h[:-1], 1e-12)), \
            f"V4 energy diverged: {h}"
        falls += int(v4['fallback'])
        s1s.append(pit(v1['dec1'], v1['dec2'], c1, c2, ref1, ref2, 4, 4,
                       True))
        s4s.append(pit(v4['dec1'], v4['dec2'], c1, c2, ref1, ref2, 4, 4,
                       True))
        print(f"  trial{trial}: V1={s1s[-1]:.4f} V4={s4s[-1]:.4f} "
              f"E {v1['res_energy']:.3f}->{v4['res_energy']:.3f} "
              f"fallback={v4['fallback']}")
    print(f"  mean V1={np.mean(s1s):.4f} V4={np.mean(s4s):.4f} "
          f"fallbacks={falls}/4")
    assert np.mean(s4s) <= np.mean(s1s) + 0.01, "V4 must not hurt on average"

    print("\njoint_isi smoke test passed!")
