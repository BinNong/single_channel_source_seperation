"""Paper 6 — E5 part 2 driver: K=3 feasibility probe (scope-limited).

PSK-only triplets (mod restriction declared — this is a feasibility
probe), K=3 cells of the deterministic grid (seed 99999), SNR {5, 15} dB,
n=25/cell.  Per cell:
  - ECM-K3 (joint_sync_detect_k3.ecm_joint_k3): sequential-extraction
    frequency init + 3-source converged EM + centred-tap ISI (L=3).
  - Oracle mixture baseline per source (ser_comp.compute_ser_compensated
    on the raw mixture vs each source — the paper-5 baseline receiver),
    computed on the same cells.

Scoring: PIT over the 6 assignments, per-source M-fold rotation
resolution (E1 convention).  Success = any significant movement below
the baseline; failure = honest negative.

Usage: python eval_joint_k3.py [--n_per_cell 25]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Vendored paper6 modules (self-contained; no sys.path tricks).
import config as C                                          # noqa: E402
from data_generator import CommBSSVarKTestDataset, MOD_TYPES  # noqa: E402
import ser_comp                                             # noqa: E402
from joint_detect import constellation_np                    # noqa: E402
from joint_sync_detect import mixture_symbols                # noqa: E402
from joint_sync_detect_k3 import ecm_joint_k3                # noqa: E402


def rot_ser(dec_syms, const, ref_lab):
    m_order = {2: 2, 4: 4, 8: 8, 16: 4}[len(const)]
    best = 1.0
    for k in range(m_order):
        dv = dec_syms * np.exp(-1j * 2 * np.pi * k / m_order)
        d = np.argmin(np.abs(dv[:, None] - const[None, :]), axis=1)
        best = min(best, float(np.mean(d != ref_lab)))
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_per_cell', type=int, default=25)
    p.add_argument('--snr_points', type=float, nargs='+', default=[5, 15])
    args = p.parse_args()

    ds = CommBSSVarKTestDataset(
        n_per_cell=args.n_per_cell,
        snr_points=list(C.SignalConfig.snr_test_points),
        mod_types=MOD_TYPES, k_lo=1, k_hi=3, k_extrap=4, k_slots=4,
        signal_length=C.SignalConfig.signal_length,
        sample_rate=C.SignalConfig.sample_rate,
        carrier_base=C.SignalConfig.carrier_base,
        freq_gap_range=C.SignalConfig.freq_gap_range,
        seed=C.DataConfig.test_seed, return_carriers=True)

    recs = []
    done = 0
    t0 = time.time()
    for i in range(len(ds)):
        s = ds.samples[i]
        if s['k'] != 3 or float(s['snr']) not in set(args.snr_points):
            continue
        mods_idx = [int(m) for m in s['mods'].numpy()[:3]]
        if any(MOD_TYPES[m] == '16QAM' for m in mods_idx):
            continue                                    # PSK-only probe
        mix_np = s['mixture'].numpy()[0]
        srcs = [s['sources'].numpy()[j, 0] for j in range(3)]
        cars = [float(c) for c in s['carriers'][:3]]
        consts = [constellation_np(m) for m in mods_idx]
        refs = [ser_comp.ref_labels_only(
            torch.from_numpy(np.asarray(srcs[j], dtype=np.complex64)),
            mod_type=MOD_TYPES[mods_idx[j]], carrier_freq=cars[j])
            for j in range(3)]

        # oracle mixture baseline per source
        base = []
        for j in range(3):
            ser_j, _ = ser_comp.compute_ser_ber_compensated(
                torch.from_numpy(np.asarray(mix_np, dtype=np.complex64)),
                torch.from_numpy(np.asarray(srcs[j], dtype=np.complex64)),
                mod_type=MOD_TYPES[mods_idx[j]],
                sample_rate=C.SignalConfig.sample_rate,
                n_symbols=C.SignalConfig.n_symbols,
                roll_off=C.SignalConfig.roll_off,
                num_taps=C.SignalConfig.num_taps,
                carrier_freq=cars[j])
            base.append(ser_j)

        # ECM-K3
        r = mixture_symbols(mix_np)
        res = ecm_joint_k3(r, consts)
        best_ser = 1.0
        for perm in itertools.permutations(range(3)):
            sv = np.mean([rot_ser(consts[perm[j]][res['decs'][:, perm[j]]],
                                  consts[perm[j]], refs[j])
                          for j in range(3)])
            best_ser = min(best_ser, float(sv))
        recs.append({'snr': float(s['snr']), 'mods': mods_idx,
                     'ser_ecm_k3': best_ser,
                     'ser_baseline': float(np.mean(base)),
                     'res_energy': res['res_energy'],
                     'df_est': [float(d) for d in res['dfs']],
                     'df_true': [float(c - C.SyncConfig.nominal_carrier)
                                 for c in cars]})
        done += 1
        if done % 5 == 0:
            print(f"  {done} cells ({(time.time()-t0)/done:.1f}s/cell)",
                  flush=True)

    print("\n=== E5 K=3 probe (PSK-only triplets) ===")
    for snr in args.snr_points:
        rs = [r for r in recs if r['snr'] == snr]
        if not rs:
            continue
        e = np.mean([r['ser_ecm_k3'] for r in rs])
        b = np.mean([r['ser_baseline'] for r in rs])
        print(f"  SNR {snr:>5g}: ECM-K3={e:.4f}  baseline={b:.4f}  "
              f"delta={e - b:+.4f}  (n={len(rs)})")
    path = os.path.join(C.RESULTS_DIR, 'e5_k3_probe.json')
    with open(path, 'w') as f:
        json.dump({'n_per_cell': args.n_per_cell,
                   'snr_points': args.snr_points, 'records': recs},
                  f, indent=2)
    print(f"saved {path}")


if __name__ == '__main__':
    main()
