"""Paper 6 — Reviewer-2 (i): PSP/Viterbi sequence-detection upgrade of the
symbol-wise "waveform route" (separate -> sync -> detect) at K=2.

The E4 headline compares V4 against paper 5's waveform route, whose
detector is symbol-wise nearest-point demodulation after the oracle
front-end (true-carrier down-conversion -> RRC matched filter -> 0::sps
symbol grid -> unit-power normalisation).  A fairer symbol-wise baseline
is upgraded here to per-source sequence detection over the 3-tap fading
ISI channel (L=3 taps):

  init     : memoryless hard decisions, LS-fit of g[0..2] on the design
             matrix [c_n, c_{n-1}, c_{n-2}];
  iterate  : Viterbi (state = previous 2 symbol indices; branch metric
             |z_n - sum_l g[l] c_{n-l}|^2; M^2 states — 16 for QPSK,
             256 for 16QAM; vectorised over states in numpy) followed by
             an LS refit of g on the Viterbi decisions.  3 rounds.

Both arms (symbol-wise / PSP) are scored IDENTICALLY to the other E4
variants: eval_joint_k2.score_decisions (PIT over slot<->source
assignments + per-source M-fold rotation resolution) against
eval_joint_k2._ref_labels.  The symbol-wise arm should reproduce the
paper's waveform-route pooled SER (~0.5534 on test seed 99999); a
deviation is reported, not hidden.

Slots: occupancy-top-2 of the 5 MSE-anchored SlotSepNet checkpoints,
Hungarian-matched to the 2 true sources on SI-SDR (same as E4's V2 path).

Usage:
    python eval_psp_baseline.py --n_per_cell 3      # smoke
    python eval_psp_baseline.py --n_per_cell 100    # full
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# Vendored paper6 modules (self-contained; no sys.path tricks).
import config as C                                          # noqa: E402
from data_generator import (CommBSSVarKTestDataset,         # noqa: E402
                            MOD_TYPES)
from signal_utils import rrc_filter                         # noqa: E402
from joint_detect import constellation_np                   # noqa: E402
from eval_joint_k2 import score_decisions, _ref_labels      # noqa: E402
from eval_blind_pipeline import (build_slot_from_ckpt,      # noqa: E402
                                 _si_sdr_np)

CKPT_PATTERN = 'slot_h64_l4_k13_bs16_lr0.001_mse_s{seed}_best.pt'
L_TAPS = 3            # ISI channel memory of the PSP model (fading_taps=3)
PSP_ROUNDS = 3        # Viterbi -> LS refit iterations after the init fit


# ---------------------------------------------------------------------------
# Oracle front-end (per matched slot waveform; true carrier of its source)
# ---------------------------------------------------------------------------
def oracle_frontend(wave_np, carrier):
    """True-carrier down-conversion -> RRC MF -> 0::sps grid -> unit power.
    Mirrors ser_comp._demod_pair_labels' estimate side minus the phase
    alignment (the M-fold rotation resolution in scoring absorbs the
    constant phase, exactly as in E1/E3/E4)."""
    T = len(wave_np)
    fs = C.SignalConfig.sample_rate
    n_sym = C.SignalConfig.n_symbols
    sps = T // n_sym
    t = np.arange(T) / fs
    bb = np.asarray(wave_np, dtype=np.complex128) \
        * np.exp(-1j * 2 * np.pi * carrier * t)
    rrc = rrc_filter(C.SignalConfig.num_taps, C.SignalConfig.roll_off, sps)
    mf = np.convolve(bb, rrc, mode='same')
    z = mf[0::sps][:n_sym]
    return z / (np.sqrt(np.mean(np.abs(z) ** 2)) + 1e-10)


# ---------------------------------------------------------------------------
# PSP / Viterbi sequence detection over an L-tap ISI channel
# ---------------------------------------------------------------------------
def _ls_fit_g(z, c, L=L_TAPS):
    """Complex LS fit of g[0..L-1] in z_n ~ sum_l g[l] c_{n-l}, rows n >= L-1.
    z, c: [N] complex (c = decided constellation points)."""
    N = len(z)
    X = np.stack([c[L - 1 - l:N - l] for l in range(L)], axis=1)  # [N-L+1, L]
    y = z[L - 1:]
    G = X.conj().T @ X
    return np.linalg.solve(G + 1e-9 * np.eye(L), X.conj().T @ y)


def _viterbi(z, const, g, L=L_TAPS):
    """ML sequence detection: state = previous L-1 symbol indices.

    Branch metric |z_n - sum_l g[l] c_{n-l}|^2, trellis vectorised over
    the M^(L-1) states.  Returns decision indices [N].
    """
    M = len(const)
    N = len(z)
    # pred[a, b, c] = g0*const[c] + g1*const[b] + g2*const[a]  (L=3)
    pred = (g[0] * const[None, None, :]
            + g[1] * const[None, :, None]
            + g[2] * const[:, None, None])          # [M, M, M]
    # trellis[a, b] = best cost up to symbol n with (c_{n-1}, c_n) = (a, b)
    # init at n = 1 (c_{-1} treated as absent: g2 term dropped)
    trellis = (np.abs(z[0] - g[0] * const[:, None]) ** 2
               + np.abs(z[1] - g[0] * const[None, :]
                        - g[1] * const[:, None]) ** 2)          # [a, b]
    bp = np.zeros((N, M, M), dtype=np.int16)       # bp[n, b, c] = best a
    for n in range(2, N):
        metric = np.abs(z[n] - pred) ** 2                    # [a, b, c]
        cand = trellis[:, :, None] + metric                  # [a, b, c]
        bp[n] = np.argmin(cand, axis=0)
        trellis = np.min(cand, axis=0)                       # [b, c]
    # backtrack: state at step n is (c_{n-1}, c_n) = (b, c); its best
    # predecessor at step n-1 is (a, b) with a = bp[n, b, c].
    dec = np.zeros(N, dtype=np.int64)
    a, b = np.unravel_index(np.argmin(trellis), (M, M))
    dec[N - 2], dec[N - 1] = a, b
    for n in range(N - 1, 1, -1):
        dec[n - 2] = int(bp[n, dec[n - 1], dec[n]])
    return dec


def psp_decisions(z, const, L=L_TAPS, rounds=PSP_ROUNDS):
    """PSP: memoryless init -> LS fit -> `rounds` x (Viterbi -> LS refit).
    Returns decision indices [N]."""
    dec = np.argmin(np.abs(z[:, None] - const[None, :]), axis=1)
    g = _ls_fit_g(z, const[dec], L)
    for _ in range(rounds):
        dec = _viterbi(z, const, g, L)
        g = _ls_fit_g(z, const[dec], L)
    return dec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description='PSP/Viterbi waveform-route baseline at K=2')
    p.add_argument('--n_per_cell', type=int, default=100)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--snr_points', type=float, nargs='+',
                   default=list(C.SignalConfig.snr_test_points))
    p.add_argument('--ckpt_dir', type=str, default=C.CHECKPOINT_DIR)
    p.add_argument('--test_seed', type=int, default=C.DataConfig.test_seed)
    p.add_argument('--out', type=str,
                   default=os.path.join(C.RESULTS_DIR, 'psp_baseline.json'))
    args = p.parse_args()
    device = C.DEVICE

    print(f"Building test grid (seed {args.test_seed}) ...", flush=True)
    ds = CommBSSVarKTestDataset(
        n_per_cell=args.n_per_cell,
        snr_points=list(C.SignalConfig.snr_test_points),
        mod_types=MOD_TYPES, k_lo=1, k_hi=3, k_extrap=4, k_slots=4,
        signal_length=C.SignalConfig.signal_length,
        sample_rate=C.SignalConfig.sample_rate,
        carrier_base=C.SignalConfig.carrier_base,
        freq_gap_range=C.SignalConfig.freq_gap_range,
        seed=args.test_seed, return_carriers=True)
    cells = []
    for i in range(len(ds)):
        s = ds.samples[i]
        if s['k'] == 2 and float(s['snr']) in set(args.snr_points):
            cells.append(i)
    print(f"  {len(cells)} K=2 cells", flush=True)

    # per-cell shared data (independent of the checkpoint)
    cell_data = []
    for i in cells:
        s = ds.samples[i]
        srcs = [s['sources'].numpy()[j, 0] for j in range(2)]
        mods_idx = [int(m) for m in s['mods'].numpy()[:2]]
        cars = [float(c) for c in s['carriers'][:2]]
        refs = [_ref_labels(srcs[j], mods_idx[j], cars[j]) for j in range(2)]
        cell_data.append({'snr': float(s['snr']),
                          'mix': s['mixture'].numpy()[0],
                          'srcs': srcs, 'mods_idx': mods_idx,
                          'cars': cars, 'refs': refs})
    consts = [constellation_np(m) for m in range(len(MOD_TYPES))]

    records = []
    for seed in args.seeds:
        ckpt = os.path.join(args.ckpt_dir,
                            CKPT_PATTERN.format(seed=seed))
        model, _a = build_slot_from_ckpt(ckpt, device)
        t0 = time.time()
        # batched forwards (batch 16), then per-cell detection
        B = 16
        for c0 in range(0, len(cell_data), B):
            chunk = cell_data[c0:c0 + B]
            mix_b = torch.stack([
                torch.from_numpy(cd['mix']).view(1, -1)
                for cd in chunk]).to(torch.complex64).to(device)
            with torch.no_grad():
                slots, occ_logits, _ = model(mix_b)
            slots_np = slots.cpu().numpy()
            occ_np = torch.sigmoid(occ_logits).cpu().numpy()
            for bi, cd in enumerate(chunk):
                top2 = np.argsort(-occ_np[bi])[:2]   # oracle count (E4 V2)
                waves = [slots_np[bi, j] for j in top2]
                # Hungarian match to the true sources on SI-SDR (V2 path)
                cost = np.full((2, 2), 1e6)
                for i, s in enumerate(cd['srcs']):
                    for j, w in enumerate(waves):
                        cost[i, j] = -_si_sdr_np(w, s)
                rows, cols = linear_sum_assignment(cost)
                # slot j (waves index) serves source i = rows position
                slot_of_src = {int(i): waves[j]
                               for i, j in zip(rows, cols)}
                decs = {'sym': [None, None], 'psp': [None, None]}
                for i in range(2):
                    z = oracle_frontend(slot_of_src[i], cd['cars'][i])
                    const = consts[cd['mods_idx'][i]]
                    decs['sym'][i] = np.argmin(
                        np.abs(z[:, None] - const[None, :]), axis=1)
                    decs['psp'][i] = psp_decisions(z, const)
                same = cd['mods_idx'][0] == cd['mods_idx'][1]
                mods_names = [MOD_TYPES[m] for m in cd['mods_idx']]
                for var in ('sym', 'psp'):
                    ser1, ber1, ser2, ber2 = score_decisions(
                        decs[var][0], decs[var][1],
                        consts[cd['mods_idx'][0]], consts[cd['mods_idx'][1]],
                        cd['refs'][0], cd['refs'][1], mods_names,
                        allow_swap=same)
                    records.append({'variant': var, 'seed': seed,
                                    'snr': cd['snr'],
                                    'mods': cd['mods_idx'],
                                    'ser1': ser1, 'ser2': ser2,
                                    'ber1': ber1, 'ber2': ber2,
                                    'ser': 0.5 * (ser1 + ser2),
                                    'ber': 0.5 * (ber1 + ber2)})
            done = min(c0 + B, len(cell_data))
            if done % 100 < B or done == len(cell_data):
                el = time.time() - t0
                print(f"  s{seed}: {done}/{len(cell_data)} cells "
                      f"({el / done:.2f}s/cell)", flush=True)
        print(f"seed {seed} done in {time.time() - t0:.0f}s", flush=True)
        _save(args.out, records, args)

    _save(args.out, records, args)
    _report(records)


def _save(out_path, records, args):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'n_per_cell': args.n_per_cell, 'seeds': args.seeds,
                   'snr_points': args.snr_points,
                   'test_seed': args.test_seed,
                   'L_taps': L_TAPS, 'psp_rounds': PSP_ROUNDS,
                   'records': records}, f)
    print(f"saved {out_path} ({len(records)} records)", flush=True)


def _agg(records, key='ser'):
    return float(np.mean([r[key] for r in records])) if records \
        else float('nan')


def _report(records):
    print("\n=== PSP baseline summary (SER; per-source average) ===")
    snrs = sorted({r['snr'] for r in records})
    print(f"{'variant':>8s} | " + ' | '.join(f"{s:>6g}" for s in snrs)
          + " | pooled")
    for var in ('sym', 'psp'):
        row = [_agg([r for r in records if r['variant'] == var
                     and r['snr'] == s]) for s in snrs]
        pooled = _agg([r for r in records if r['variant'] == var])
        print(f"{var:>8s} | " + ' | '.join(f"{v:6.4f}" for v in row)
              + f" | {pooled:.4f}")
    sym = _agg([r for r in records if r['variant'] == 'sym'])
    print(f"\nsymbol-wise pooled SER = {sym:.4f} "
          "(paper waveform-route reference: ~0.5534 on seed 99999)")
    psk = [r for r in records if r['variant'] == 'psp'
           and all(MOD_TYPES[m] != '16QAM' for m in r['mods'])]
    if psk:
        print(f"PSK-only @ PSP pooled: {_agg(psk):.4f} (n={len(psk)})")


if __name__ == '__main__':
    main()
