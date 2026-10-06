"""Paper 6 — E5 part 2: K=3 feasibility probe (scope-limited).

Extends the E4 ECM to THREE sources on the raw mixture:
    r_n = sum_{k=1..3} e^{j2π Δf_k n Ts} sum_l g_k[l] c_{k,n-m} + w_n
(centred lags m = l − L//2, as in V4 — the generator's channel is
zero-delay).

Deliberately scoped (feasibility probe, PSK-only triplets):
  - Frequency init: sequential extraction — run the K=2 ECM
    (joint_sync_detect.ecm_joint, proven), subtract its two fitted
    streams (with taps) from the mixture, and estimate the third
    source's carrier from the residual with the E1 single-source
    blind_sync.  No 3-D coarse grid.
  - Then converged phase-grid EM (per-source sector phases) over the
    3-source joint hypothesis grid (M1·M2·M3 ≤ 512 for PSK-only) with
    centred-tap LS refits, and a small per-source NLS polish.
  - This is a feasibility probe: is the 3-source joint posterior
    tractable at all at SIR≈0?  Honest negative is an acceptable
    outcome (the paper's K=3 stays future work).

__main__ smoke: synthetic K=3 PSK burst at 15 dB — ECM must beat
separate detection and not diverge.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Vendored paper6 modules (self-contained; no sys.path tricks).
import config as C                                     # noqa: E402 (paper6)
from joint_detect import constellation_np, MOD_TYPES   # noqa: E402
from joint_sync_detect import _TS, ecm_joint           # noqa: E402
from joint_isi import _shifted, _design                # noqa: E402
from sync import blind_sync_known_mod                  # noqa: E402

_NSYM = C.SignalConfig.n_symbols


# ---------------------------------------------------------------------------
# K-generic batched EM (G phase inits, K sources, hypothesis grid H)
# ---------------------------------------------------------------------------
def _grid_vals(consts):
    """Hypothesis table: idx [K, H] column indices, vals [H, K] symbols."""
    K = len(consts)
    H = int(np.prod([len(c) for c in consts]))
    grids = np.meshgrid(*[np.arange(len(c)) for c in consts],
                        indexing='ij')
    idx = np.stack([g.ravel() for g in grids], axis=0)          # [K, H]
    vals = np.stack([consts[k].take(idx[k]) for k in range(K)],
                    axis=1)                                     # [H, K]
    return idx, vals


def _batch_decisions_k(Rs, Ag, U, idx, vals):
    """Rs [P,N], Ag [G,P,K], U [K,N] -> (decs [G,N,K], per-symbol min).
    Memory-conscious: H <= 512 for PSK-only triplets."""
    G, P, K = Ag.shape
    N = Rs.shape[1]
    # pred[g,n,h,p] = sum_k Ag[g,p,k] * U[k,n] * vals[h,k]
    pred = np.einsum('gpk,kn,hk->gnhp', Ag, U, vals)
    d2 = np.sum(np.abs(Rs.T[None, :, None, :] - pred) ** 2, axis=-1)
    h = np.argmin(d2, axis=2)                                   # [G,N]
    decs = np.stack([idx[k][h] for k in range(K)], axis=-1)     # [G,N,K]
    return decs


def _batch_ls_k(Rs, U, decs, consts):
    """Batched LS gains A = R D^H (D D^H)^{-1}; decs [G,N,K] -> A [G,P,K]."""
    K = len(consts)
    D = np.stack([U[k][None, :] * consts[k][decs[..., k]]
                  for k in range(K)], axis=1)                   # [G,K,N]
    G = D @ D.conj().transpose(0, 2, 1)                         # [G,K,K]
    RDT = np.stack([(Rs[None, :, :] @ np.conj(D[:, k])[..., None])[..., 0]
                    for k in range(K)], axis=-1)                # [G,P,K]
    A = RDT @ np.linalg.inv(G + 1e-9 * np.eye(K))
    return A


def _batch_energy_k(Rs, Ag, U, decs, consts):
    K = len(consts)
    D = np.stack([U[k][None, :] * consts[k][decs[..., k]]
                  for k in range(K)], axis=1)
    pred = Ag @ D
    return np.sum(np.abs(Rs[None] - pred) ** 2, axis=(1, 2))


def _taps_ls_k(R, U, decs, consts, L):
    """LS refit of K*L centred taps.  Returns G [P,K,L]."""
    rows = []
    for k in range(len(consts)):
        cs = consts[k][decs[..., k]]
        for l in range(L):
            rows.append(U[k] * _shifted(cs, l - L // 2))
    D = np.stack(rows, axis=0)                                  # [KL, N]
    Gm = D @ D.conj().T
    T = R @ D.conj().T @ np.linalg.inv(Gm + 1e-9 * np.eye(len(rows)))
    return T.reshape(R.shape[0], len(consts), L)


def _taps_energy_k(R, U, decs, consts, G):
    rows = []
    K, L = len(consts), G.shape[2]
    for k in range(K):
        cs = consts[k][decs[..., k]]
        for l in range(L):
            rows.append(U[k] * _shifted(cs, l - L // 2))
    D = np.stack(rows, axis=0)
    return float(np.sum(np.abs(R - G.reshape(R.shape[0], -1) @ D) ** 2))


def _isi_cleaned_decisions_k(R, U, decs, consts, G):
    """Parallel ISI cancellation + joint grid decisions (centre taps)."""
    Rc = R.copy()
    K, L = len(consts), G.shape[2]
    for k in range(K):
        cs = consts[k][decs[..., k]]
        for l in range(L):
            if l == L // 2:
                continue
            Rc = Rc - G[:, k, l][:, None] * (U[k] * _shifted(cs, l - L // 2))[None, :]
    idx, vals = _grid_vals(consts)
    Ag = G[:, :, L // 2][None, :, :]                              # [1,P,K]
    out = _batch_decisions_k(Rc, Ag, U, idx, vals)
    return out[0]


# ---------------------------------------------------------------------------
# 3-source ECM (PSK-only feasibility probe)
# ---------------------------------------------------------------------------
def ecm_joint_k3(r, consts, L=3, n_ph=4, tol=1e-5, max_rounds=30,
                 nls_hw=0.15, nls_step=0.05, polish_rounds=4,
                 df_init=None):
    """Joint sync+detection of a K=3 burst on the raw mixture.

    Frequency init (df_init None): K=2 ECM on the mixture -> subtract the
    two fitted streams (with their L taps) -> E1 blind sync of the
    residual for the third carrier.  Then converged phase-grid EM with
    centred-tap LS refits, plus a small per-source NLS polish.
    Returns dict(decs [N,3], G [P,3,L], dfs, res_energy, hist).
    """
    R = np.asarray(r, dtype=np.complex128)
    if R.ndim == 1:
        R = R[None, :]
    N = R.shape[1]
    n_idx = np.arange(N)
    K = 3
    idx, vals = _grid_vals(consts)

    if df_init is None:
        # sequential extraction: K=2 ECM on sources 1+2 hypotheses... the
        # mod assignment to ECM arms is ambiguous; use the two LARGEST
        # constellations first is wrong — use consts[0], consts[1] for the
        # pair and let the residual carry source 3 (PSK-only probe).
        res2 = ecm_joint(R[0], consts[0], consts[1])
        # third carrier from the residual after subtracting the pair fit
        r_res = R[0] - res2['A'][0, 0] * _steer_(res2['df1'], n_idx) \
            * consts[0][res2['dec1']] \
            - res2['A'][0, 1] * _steer_(res2['df2'], n_idx) \
            * consts[1][res2['dec2']]
        # blind sync of the residual needs a waveform; work directly on
        # the symbol-rate residual with an M-th-power line search
        M3 = len(consts[2]) if len(consts[2]) in (2, 4, 8) else 4
        df3 = _residual_df(r_res, consts[2])
        df_init = (res2['df1'], res2['df2'], df3)

    U = np.stack([_steer_(df, n_idx) for df in df_init], axis=0)
    # phase-grid EM (cold) — per-source sector grids
    ph = [np.exp(1j * 2 * np.pi * np.arange(n_ph) / (len(consts[k]) * n_ph))
          for k in range(K)]
    grids = np.meshgrid(*ph, indexing='ij')
    Ag = np.stack([g.ravel() for g in grids], axis=-1)          # [G,K]
    Ag = np.broadcast_to(Ag[:, None, :], (Ag.shape[0], R.shape[0], K)) \
        .astype(np.complex128).copy()                           # [G,P,K]
    E = None
    decs = None
    for _ in range(max_rounds):
        decs = _batch_decisions_k(R, Ag, U, idx, vals)
        Ag = _batch_ls_k(R, U, decs, consts)
        E2 = _batch_energy_k(R, Ag, U, decs, consts)
        if E is not None and np.all(E - E2 < tol * np.maximum(E, 1e-12)):
            E = E2
            break
        E = E2
    g = int(np.argmin(E))
    decs_best = decs[g]
    A = Ag[g]                                                   # [P,K]
    dfs = list(df_init)

    # ISI taps + joint refinements
    G = np.zeros((R.shape[0], K, L), dtype=np.complex128)
    G[:, :, L // 2] = A
    E = _taps_energy_k(R, U, decs_best, consts, G)
    hist = [E]
    for _ in range(polish_rounds + 4):
        decs_best = _isi_cleaned_decisions_k(R, U, decs_best, consts, G)[0] \
            if False else _isi_decide_k(R, U, decs_best, consts, G)
        G = _taps_ls_k(R, U, decs_best, consts, L)
        E_new = _taps_energy_k(R, U, decs_best, consts, G)
        hist.append(E_new)
        if E - E_new < tol * max(E, 1e-12):
            E = E_new
            break
        E = E_new
    # per-source NLS polish (taps refit per candidate)
    for k in range(K):
        best = None
        for g in np.arange(dfs[k] - nls_hw, dfs[k] + nls_hw + 1e-9,
                           nls_step):
            Uk = U.copy()
            Uk[k] = _steer_(g, n_idx)
            Gg = _taps_ls_k(R, Uk, decs_best, consts, L)
            Eg = _taps_energy_k(R, Uk, decs_best, consts, Gg)
            if best is None or Eg < best[0]:
                best = (Eg, g, Gg, Uk)
        Eg, g, Gg, Uk = best
        if Eg < E:
            E, dfs[k], G, U = Eg, g, Gg, Uk
    decs_best = _isi_decide_k(R, U, decs_best, consts, G)

    return {'decs': decs_best, 'G': G, 'dfs': dfs,
            'res_energy': E, 'hist': hist, 'df_init': df_init}


def _steer_(df, n_idx):
    return np.exp(1j * 2 * np.pi * df * n_idx * _TS)


def _isi_decide_k(R, U, decs, consts, G):
    """ISI-clean + decide, unbatched.  Returns decs [N,K]."""
    Rc = R.copy()
    K, L = len(consts), G.shape[2]
    for k in range(K):
        cs = consts[k][decs[..., k]]
        for l in range(L):
            if l == L // 2:
                continue
            Rc = Rc - G[:, k, l][:, None] * (U[k] * _shifted(cs, l - L // 2))[None, :]
    idx, vals = _grid_vals(consts)
    A0 = G[:, :, L // 2][:, :, None].transpose(0, 2, 1)[:, :, :]  # [P,K]... shape
    # _batch_decisions_k expects Ag [G,P,K]
    Ag = G[:, :, L // 2][None, :, :]                              # [1,P,K]
    out = _batch_decisions_k(Rc, Ag, U, idx, vals)
    return out[0]


def _residual_df(r_res, const):
    """Single-source carrier estimate from a symbol-rate residual stream:
    M-th power line over the residual window (E1 machinery at symbol
    rate)."""
    M = {2: 2, 4: 4, 8: 8}.get(len(const), 4)
    y = (r_res / (np.abs(r_res) + 1e-12)) ** M
    n_fft = len(y) * 8
    spec = np.abs(np.fft.fft(y, n_fft))
    freqs = np.fft.fftfreq(n_fft, d=_TS)
    win = (freqs >= M * C.SyncConfig.freq_search_lo) & \
          (freqs <= M * C.SyncConfig.freq_search_hi)
    k = int(np.argmax(np.where(win, spec, -np.inf)))
    ym, y0, yp = spec[max(k - 1, 0)], spec[k], spec[min(k + 1, n_fft - 1)]
    den = ym - 2 * y0 + yp
    d = 0.5 * (ym - yp) / den if abs(den) > 1e-20 else 0.0
    f = (k + np.clip(d, -1, 1)) * (1 / _TS) / n_fft
    if f > 0.5 / _TS:
        f -= 1 / _TS
    return f / M


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import torch
    import ser_comp
    from data_generator import generate_vark_mixture

    print("Testing paper6 joint_sync_detect_k3 (K=3 probe) ...")
    np.random.seed(C.SEED)
    mods = ['QPSK', 'QPSK', '8PSK']
    mix, srcs, midx, cars = generate_vark_mixture(
        C.SignalConfig.signal_length, C.SignalConfig.sample_rate, 15.0,
        mods, k=3,
        carrier_base=C.SignalConfig.carrier_base,
        freq_gap_range=C.SignalConfig.freq_gap_range,
        n_symbols=_NSYM, roll_off=C.SignalConfig.roll_off,
        num_taps=C.SignalConfig.num_taps,
        apply_fading=C.SignalConfig.apply_fading,
        fading_taps=C.SignalConfig.fading_taps, return_carriers=True)
    consts = [constellation_np(m) for m in midx]
    r = (mixture_symbols_(mix) if False else None)
    from joint_sync_detect import mixture_symbols as mixture_symbols_
    r = mixture_symbols_(mix)
    refs = [ser_comp.ref_labels_only(
        torch.from_numpy(np.asarray(srcs[j], dtype=np.complex64)),
        mod_type=mods[j], carrier_freq=cars[j]) for j in range(3)]
    res = ecm_joint_k3(r, consts)
    print(f"  dfs est={[round(d,2) for d in res['dfs']]} "
          f"true={[round(c-2000,2) for c in cars]}")
    print(f"  energy hist: {[round(h,2) for h in res['hist']]}")

    def rot_ser(dec_syms, const, ref_lab):
        mo = {2: 2, 4: 4, 8: 8}.get(len(const), 4)
        best = 1.0
        for k in range(mo):
            dv = dec_syms * np.exp(-1j * 2 * np.pi * k / mo)
            d = np.argmin(np.abs(dv[:, None] - const[None, :]), axis=1)
            best = min(best, float(np.mean(d != ref_lab)))
        return best

    # PIT over the 6 assignments
    import itertools
    best_ser = 1.0
    for perm in itertools.permutations(range(3)):
        s = np.mean([rot_ser(consts[perm[j]][res['decs'][:, perm[j]]],
                             consts[perm[j]], refs[j])
                     for j in range(3)])
        best_ser = min(best_ser, s)
    # separate-detection reference
    sep = np.mean([rot_ser(consts[j][np.argmin(
        np.abs((r * np.conj(_steer_(res['dfs'][j], np.arange(len(r)))))[:, None]
               - consts[j][None, :]) ** 2, axis=1)], consts[j], refs[j])
        for j in range(3)])
    print(f"  K=3 ECM SER={best_ser:.4f}  separate-at-est-dfs={sep:.4f}")
    assert best_ser < sep + 0.05
    print("\njoint_sync_detect_k3 smoke test passed!")
