"""Paper 6 — experiment E1 (plan §4): does blind sync approach oracle sync?

K=1 test cells (no separation — isolates the synchroniser).  For each SNR
on the test grid and n_per_cell mixtures, three receivers demodulate the
SAME noisy mixture and are scored against the SAME reference labels
(ser_comp.ref_labels_only on the clean source, at the TRUE carrier):

  (a) nominal proxy : down-convert at the nominal 2000 Hz, matched filter,
      no carrier/phase recovery at all (expected broken; reference point).
  (b) oracle        : ser_comp.compute_ser_compensated — oracle down-
      conversion at the true carrier + oracle phase alignment to the
      clean source (paper 5's evaluation receiver).
  (c) blind         : sync.blind_sync_known_mod (BlindCarrierSync with the
      true modulation hypothesis) — decisions vs the reference labels.

Blind-carrier-sync caveat (see sync.py docstring): the M-th-power + DD
pipeline is blind to the constellation's M-fold rotation symmetry, so (c)
is scored as min over the M symmetry rotations of the decision SER
(genie ambiguity resolution — the standard convention; a real system uses
differential coding).  Receivers (a) and (b) need no such resolution:
(b) phase-aligns to the reference by construction, (a) is far above any
ambiguity floor.

Usage:
    python probe_blind_sync_k1.py [--n_per_cell 20] [--snr_points -10 0 20]

Output: results/e1_blind_sync_k1.json + a printed per-modulation x SNR
SER table for (b) and (c), against the clean-floor targets
0.000 / 0.000 / 0.014 / 0.032 (BPSK / QPSK / 8PSK / 16QAM).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Vendored paper6 modules (self-contained; no sys.path tricks).
import config as C                                          # noqa: E402
from data_generator import (generate_vark_mixture, MOD_TYPES,  # noqa: E402
                            IDX_TO_MOD)
import ser_comp                                             # noqa: E402
from signal_utils import rrc_filter                         # noqa: E402
from sync import blind_sync_known_mod, constellation_np     # noqa: E402

CLEAN_FLOOR = {'BPSK': 0.000, 'QPSK': 0.000, '8PSK': 0.014, '16QAM': 0.032}


def make_data(n, snr, seed):
    """K=1 samples at a fixed SNR: (mix[T], src[T], mod int, carrier).

    Copies paper5 probe_sync_head.make_data's RNG pattern: the global RNG
    state is stashed/restored so generation does not perturb other streams,
    and all draws inside come from the seeded global generator (the
    RandomState rng is unused here since the SNR is fixed per cell).
    """
    data = []
    st = np.random.get_state()
    np.random.seed(seed)
    try:
        for _ in range(n):
            mix, srcs, midx, cars = generate_vark_mixture(
                C.SignalConfig.signal_length, C.SignalConfig.sample_rate,
                float(snr), MOD_TYPES, k=1,
                carrier_base=C.SignalConfig.carrier_base,
                freq_gap_range=C.SignalConfig.freq_gap_range,
                n_symbols=C.SignalConfig.n_symbols,
                roll_off=C.SignalConfig.roll_off,
                num_taps=C.SignalConfig.num_taps,
                apply_fading=C.SignalConfig.apply_fading,
                fading_taps=C.SignalConfig.fading_taps,
                return_carriers=True)
            data.append((mix, srcs[0], midx[0], cars[0]))
    finally:
        np.random.set_state(st)
    return data


def nominal_proxy_decisions(mix, mod_idx):
    """Receiver (a): nominal down-conversion at 2000 Hz, matched filter,
    0::sps grid, unit-power, min-distance decisions.  NO sync of any kind
    (no frequency estimate, no phase alignment)."""
    T = len(mix)
    n_symbols = C.SignalConfig.n_symbols
    sps = T // n_symbols
    t = np.arange(T) / C.SignalConfig.sample_rate
    x_bb = mix * np.exp(-1j * 2 * np.pi * C.SyncConfig.nominal_carrier * t)
    rrc = rrc_filter(C.SignalConfig.num_taps, C.SignalConfig.roll_off, sps)
    mf = np.convolve(x_bb, rrc, mode='same')
    z = mf[0::sps][:min(n_symbols, T // sps)]
    z = z / (np.sqrt(np.mean(np.abs(z) ** 2)) + 1e-10)
    const = constellation_np(mod_idx)
    return np.argmin(np.abs(z[:, None] - const[None, :]), axis=1)


def ser_min_rotation(z, mod_idx, ref_labels):
    """SER of the blind receiver's synced symbols vs reference labels,
    minimised over the M-fold constellation-symmetry rotations (genie
    ambiguity resolution; see module docstring)."""
    const = constellation_np(mod_idx)
    m_order = C.SyncConfig.sym_order[MOD_TYPES[mod_idx]]
    best = 1.0
    for k in range(m_order):
        zr = z * np.exp(-1j * 2 * np.pi * k / m_order)
        d = np.argmin(np.abs(zr[:, None] - const[None, :]), axis=1)
        best = min(best, float(np.mean(d != ref_labels)))
    return best


def main():
    p = argparse.ArgumentParser(description='E1: blind vs oracle sync, K=1')
    p.add_argument('--n_per_cell', type=int, default=20)
    p.add_argument('--snr_points', type=float, nargs='+',
                   default=list(C.SignalConfig.snr_test_points))
    p.add_argument('--out', type=str,
                   default=os.path.join(C.RESULTS_DIR, 'e1_blind_sync_k1.json'))
    p.add_argument('--seed_base', type=int, default=C.DataConfig.test_seed,
                   help='base seed for the per-SNR draws (test_seed + cell '
                        'index); use a second base for an independent grid')
    args = p.parse_args()

    # cell -> list of SERs: table[mod][snr][receiver] = [...]
    table: dict[str, dict] = {
        m: {float(s): {'nominal': [], 'oracle': [], 'blind': []}
            for s in args.snr_points}
        for m in MOD_TYPES}

    for ci, snr in enumerate(args.snr_points):
        data = make_data(args.n_per_cell, float(snr),
                         seed=args.seed_base + ci)
        for mix, src, midx, car in data:
            mod_name = IDX_TO_MOD[midx]
            src_t = torch.from_numpy(np.asarray(src, dtype=np.complex64))
            mix_t = torch.from_numpy(np.asarray(mix, dtype=np.complex64))
            ref_labels = ser_comp.ref_labels_only(
                src_t, mod_type=mod_name,
                sample_rate=C.SignalConfig.sample_rate,
                n_symbols=C.SignalConfig.n_symbols,
                roll_off=C.SignalConfig.roll_off,
                num_taps=C.SignalConfig.num_taps,
                carrier_freq=float(car))
            cell = table[mod_name][float(snr)]

            # (a) nominal proxy, no sync
            dec_a = nominal_proxy_decisions(mix, midx)
            cell['nominal'].append(float(np.mean(dec_a != ref_labels)))

            # (b) oracle compensated receiver
            cell['oracle'].append(ser_comp.compute_ser_compensated(
                mix_t, src_t, mod_type=mod_name,
                sample_rate=C.SignalConfig.sample_rate,
                n_symbols=C.SignalConfig.n_symbols,
                roll_off=C.SignalConfig.roll_off,
                num_taps=C.SignalConfig.num_taps,
                carrier_freq=float(car)))

            # (c) blind sync (true-mod hypothesis), genie rotation resolution
            res = blind_sync_known_mod(np.asarray(mix, dtype=np.complex128),
                                       midx)
            cell['blind'].append(ser_min_rotation(res['z'], midx, ref_labels))
        print(f"SNR {snr:>5.1f} dB done ({args.n_per_cell} mixtures)")

    # ---- report ------------------------------------------------------------
    def agg(mod, snr, rx):
        v = table[mod][float(snr)][rx]
        return float(np.mean(v)) if v else float('nan')

    hdr = (f"{'SNR':>6s} | " +
           ' | '.join(f"{m + ' o/b':>17s}" for m in MOD_TYPES) +
           " | clean-floor targets "
           + '/'.join(f"{CLEAN_FLOOR[m]:.3f}" for m in MOD_TYPES))
    print('\n' + hdr)
    print('-' * len(hdr))
    for snr in args.snr_points:
        row = f"{snr:>6.1f} | "
        row += ' | '.join(
            f"{agg(m, snr, 'oracle'):8.4f}/{agg(m, snr, 'blind'):8.4f}"
            for m in MOD_TYPES)
        print(row)

    print("\nnominal proxy (no sync) overall per SNR: "
          + ', '.join(
              f"{snr:g} dB: {np.mean([x for m in MOD_TYPES for x in table[m][float(snr)]['nominal']]):.4f}"
              for snr in args.snr_points))

    out = {
        'experiment': 'E1_blind_sync_k1',
        'n_per_cell': args.n_per_cell,
        'snr_points': [float(s) for s in args.snr_points],
        'seed_base': args.seed_base,
        'receivers': ['nominal', 'oracle', 'blind'],
        'blind_note': 'blind SER is min over the M-fold symmetry rotations '
                      '(genie ambiguity resolution; M-th-power sync is blind '
                      'to constellation rotations)',
        'clean_floor_targets': CLEAN_FLOOR,
        'table': {m: {f"{snr:g}": {rx: agg(m, snr, rx)
                                   for rx in ('nominal', 'oracle', 'blind')}
                      for snr in args.snr_points}
                  for m in MOD_TYPES},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == '__main__':
    main()
