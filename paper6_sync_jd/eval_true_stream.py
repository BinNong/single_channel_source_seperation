"""Paper 6 — true-stream control for E4 (reviewer-requested artifact).

The E4 discussion cites a positive control that was previously only logged
in EXPERIMENT_LOG.md (2026-09-18), not scripted:

  * "the SAME 2-channel ECM on the two TRUE source streams gives SER 0.041
    (vs mixture ECM 0.121 on those bursts)" — the joint loop wins big IF
    separation is perfect;
  * "V4-oracle-df with L=5 on QPSK pairs @20 dB: 0.106 vs 0.164
    memoryless-oracle (true-stream control 0.041)".

This script fixes the protocol and records all arms on the SAME K=2 test
cells (deterministic grid, seed 99999 — the paper grid).  Arms per burst
(r = mixture_symbols front-end; R_ts = per-source front-end streams
stacked, i.e. a PERFECT separator's output, no learned separator):

  v1_mix   memoryless ECM, blind frequency search, raw mixture   (E4 V1)
  v4_mix   ISI-aware ECM (L=5), blind, raw mixture               (E4 V4)
  v3_mix   memoryless ECM, ORACLE df (no frequency search), mixture (V3)
  v4o_mix  ISI-aware ECM (L=5) warm-started from v3_mix, mixture
           ("V4-oracle-df" of the log entry)
  v1_ts    memoryless ECM, blind frequency search, TRUE streams
  v4_ts    ISI-aware ECM (L=5), blind, TRUE streams
  v3_ts    memoryless ECM, ORACLE df, TRUE streams

Scoring is identical to E4 (eval_joint_k2.score_decisions: PIT over the
source assignment when the modulations match + per-source M-fold rotation
resolution against ser_comp reference labels).

Note on the historical numbers: the log does not record which burst subset
the 0.041/0.121 pair was measured on (the surrounding diagnosis used 40
bursts at 20 dB).  This artifact therefore reports the full per-SNR and
pooled tables plus the QPSK-pairs@20 dB subset (the 0.106/0.164 context);
the JSON 'paper_claims' block lists the historical values side by side
with the closest-matching subsets computed here.

Usage:
    python eval_true_stream.py --n_per_cell 2 --snr_points 20   # smoke
    python eval_true_stream.py --n_per_cell 100                 # full grid
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import config as C                                          # noqa: E402
from data_generator import (CommBSSVarKTestDataset,         # noqa: E402
                            MOD_TYPES)
from joint_detect import constellation_np                   # noqa: E402
from joint_sync_detect import ecm_joint, mixture_symbols    # noqa: E402
from joint_isi import ecm_joint_isi                         # noqa: E402
from eval_joint_k2 import score_decisions, _ref_labels      # noqa: E402

ARMS = ['v1_mix', 'v4_mix', 'v3_mix', 'v4o_mix',
        'v1_ts', 'v4_ts', 'v3_ts']

PAPER_CLAIMS = {
    'v1_ts (ECM on TRUE streams)': 0.041,
    'v1_mix on the same bursts (mixture ECM)': 0.121,
    'v4o_mix, QPSK pairs @20 dB (V4-oracle-df, L=5)': 0.106,
    'v3_mix, QPSK pairs @20 dB (memoryless oracle)': 0.164,
}


def run_cell(cd, arms):
    """All arms on one K=2 cell.  cd: dict with mix/srcs/mods/cars/refs."""
    c1 = constellation_np(cd['mods_idx'][0])
    c2 = constellation_np(cd['mods_idx'][1])
    same = cd['mods_idx'][0] == cd['mods_idx'][1]
    mods_names = [MOD_TYPES[m] for m in cd['mods_idx']]
    df_true = (cd['cars'][0] - C.SyncConfig.nominal_carrier,
               cd['cars'][1] - C.SyncConfig.nominal_carrier)

    res = {}
    r_m = mixture_symbols(cd['mix'])
    if 'v1_mix' in arms or 'v4_mix' in arms:
        v1 = ecm_joint(r_m, c1, c2)
        res['v1_mix'] = v1
        if 'v4_mix' in arms:
            res['v4_mix'] = ecm_joint_isi(r_m, c1, c2, L=5, v1=v1)
    if 'v3_mix' in arms or 'v4o_mix' in arms:
        v3 = ecm_joint(r_m, c1, c2, df_init=df_true, nls_hw=0.0,
                       nls_step=1.0)
        res['v3_mix'] = v3
        if 'v4o_mix' in arms:
            res['v4o_mix'] = ecm_joint_isi(r_m, c1, c2, L=5, v1=v3)
    if any(a in arms for a in ('v1_ts', 'v4_ts', 'v3_ts')):
        R_ts = np.stack([mixture_symbols(np.asarray(s, dtype=np.complex128))
                         for s in cd['srcs']], axis=0)
        if 'v1_ts' in arms or 'v4_ts' in arms:
            v1t = ecm_joint(R_ts, c1, c2)
            res['v1_ts'] = v1t
            if 'v4_ts' in arms:
                res['v4_ts'] = ecm_joint_isi(R_ts, c1, c2, L=5, v1=v1t)
        if 'v3_ts' in arms:
            res['v3_ts'] = ecm_joint(R_ts, c1, c2, df_init=df_true,
                                     nls_hw=0.0, nls_step=1.0)

    out = {}
    for arm, r in res.items():
        ser1, ber1, ser2, ber2 = score_decisions(
            r['dec1'], r['dec2'], c1, c2, cd['refs'][0], cd['refs'][1],
            mods_names, allow_swap=same)
        out[arm] = {'ser': 0.5 * (ser1 + ser2), 'ber': 0.5 * (ber1 + ber2),
                    'res_energy': r['res_energy'],
                    'fallback': bool(r.get('fallback', False))}
    return out


def _subset(records, snr=None, qpsk_only=False, psk_only=False):
    rs = records
    if snr is not None:
        rs = [r for r in rs if r['snr'] == snr]
    if qpsk_only:
        rs = [r for r in rs if all(MOD_TYPES[m] == 'QPSK' for m in r['mods'])]
    if psk_only:
        rs = [r for r in rs if all(MOD_TYPES[m] != '16QAM' for m in r['mods'])]
    return rs


def _arm_means(records, key='ser'):
    out = {}
    for arm in ARMS:
        v = [r['arms'][arm][key] for r in records if arm in r['arms']]
        out[arm] = float(np.mean(v)) if v else None
    return out


def aggregate(records):
    snrs = sorted({r['snr'] for r in records})
    agg = {
        'per_snr': {f'{s:g}': {'n': len(_subset(records, snr=s)),
                               **_arm_means(_subset(records, snr=s))}
                    for s in snrs},
        'pooled': {'n': len(records), **_arm_means(records)},
        'snr20_only': {'n': len(_subset(records, snr=20.0)),
                       **_arm_means(_subset(records, snr=20.0))},
        'qpsk_pairs_snr20': {
            'n': len(_subset(records, snr=20.0, qpsk_only=True)),
            **_arm_means(_subset(records, snr=20.0, qpsk_only=True))},
        'psk_only_pooled': {
            'n': len(_subset(records, psk_only=True)),
            **_arm_means(_subset(records, psk_only=True))},
        'v4_fallbacks': {arm: sum(int(r['arms'][arm]['fallback'])
                                  for r in records if arm in r['arms'])
                         for arm in ('v4_mix', 'v4o_mix', 'v4_ts')},
    }
    return agg


def main():
    p = argparse.ArgumentParser(description='E4 true-stream control')
    p.add_argument('--n_per_cell', type=int, default=100)
    p.add_argument('--snr_points', type=float, nargs='+',
                   default=list(C.SignalConfig.snr_test_points))
    p.add_argument('--arms', type=str, nargs='+', default=ARMS,
                   choices=ARMS)
    p.add_argument('--test_seed', type=int, default=C.DataConfig.test_seed)
    p.add_argument('--qpsk20', type=int, default=0, metavar='N',
                   help='dedicated mode: N fresh QPSKxQPSK bursts at 20 dB '
                        '(generate_vark_mixture stream, seed --test_seed) — '
                        'the grid has only ~6 QPSK pairs per 100 cells, too '
                        'thin for the 0.106/0.164/0.041/0.121 comparisons')
    p.add_argument('--out', type=str, default=os.path.join(
        C.RESULTS_DIR, 'true_stream_control.json'))
    args = p.parse_args()

    if args.qpsk20:
        return main_qpsk20(args)

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
    cells = [i for i in range(len(ds))
             if ds.samples[i]['k'] == 2
             and float(ds.samples[i]['snr']) in set(args.snr_points)]
    print(f"  {len(cells)} K=2 cells, arms={args.arms}", flush=True)

    records = []
    t0 = time.time()
    for ci, i in enumerate(cells):
        s = ds.samples[i]
        cd = {
            'snr': float(s['snr']),
            'mix': s['mixture'].numpy()[0],
            'srcs': [s['sources'].numpy()[j, 0] for j in range(2)],
            'mods_idx': [int(m) for m in s['mods'].numpy()[:2]],
            'cars': [float(c) for c in s['carriers'][:2]],
        }
        cd['refs'] = [_ref_labels(cd['srcs'][j], cd['mods_idx'][j],
                                  cd['cars'][j]) for j in range(2)]
        arm_out = run_cell(cd, args.arms)
        records.append({'snr': cd['snr'], 'mods': cd['mods_idx'],
                        'arms': arm_out})
        if (ci + 1) % 10 == 0 or ci == len(cells) - 1:
            print(f"  {ci + 1}/{len(cells)} "
                  f"({(time.time() - t0) / (ci + 1):.2f}s/cell)",
                  flush=True)
        if (ci + 1) % 50 == 0:
            _save(args.out, records, args)

    _save(args.out, records, args)
    agg = aggregate(records)
    print("\n=== true-stream control (SER) ===")
    hdr = f"{'subset':>22s} | " + ' | '.join(f"{a:>8s}" for a in args.arms)
    print(hdr)
    print('-' * len(hdr))
    for sub in ('pooled', 'snr20_only', 'qpsk_pairs_snr20'):
        row = agg[sub]
        print(f"{sub:>22s} | " + ' | '.join(
            f"{row[a]:8.4f}" if row[a] is not None else f"{'—':>8s}"
            for a in args.arms) + f"   (n={row['n']})")
    print("\nper-SNR (pooled over pairs):")
    for s, row in agg['per_snr'].items():
        print(f"  SNR {s:>4s} (n={row['n']:3d}): " + '  '.join(
            f"{a}={row[a]:.4f}" for a in args.arms if row[a] is not None))
    print(f"\nV4 fallbacks to V1: {agg['v4_fallbacks']}")
    print("\nhistorical log values for comparison:")
    for k, v in PAPER_CLAIMS.items():
        print(f"  {k}: {v}")


def main_qpsk20(args):
    """Dedicated QPSKxQPSK @20 dB run: N fresh bursts from the standard
    generator (K=2, mods fixed to QPSK, weights/carriers/noise drawn under
    np.random.seed(test_seed)).  Same arms and scoring as the grid mode."""
    from data_generator import generate_vark_mixture
    np.random.seed(args.test_seed)
    records = []
    t0 = time.time()
    for b in range(args.qpsk20):
        mix, srcs, midx, cars = generate_vark_mixture(
            C.SignalConfig.signal_length, C.SignalConfig.sample_rate, 20.0,
            MOD_TYPES, k=2, carrier_base=C.SignalConfig.carrier_base,
            freq_gap_range=C.SignalConfig.freq_gap_range,
            n_symbols=C.SignalConfig.n_symbols,
            roll_off=C.SignalConfig.roll_off, num_taps=C.SignalConfig.num_taps,
            apply_fading=C.SignalConfig.apply_fading,
            fading_taps=C.SignalConfig.fading_taps, mods=['QPSK', 'QPSK'],
            return_carriers=True)
        cd = {'snr': 20.0, 'mix': mix, 'srcs': list(srcs),
              'mods_idx': [int(m) for m in midx],
              'cars': [float(c) for c in cars]}
        cd['refs'] = [_ref_labels(cd['srcs'][j], cd['mods_idx'][j],
                                  cd['cars'][j]) for j in range(2)]
        records.append({'snr': 20.0, 'mods': cd['mods_idx'],
                        'arms': run_cell(cd, args.arms)})
        if (b + 1) % 10 == 0 or b == args.qpsk20 - 1:
            print(f"  {b + 1}/{args.qpsk20} "
                  f"({(time.time() - t0) / (b + 1):.2f}s/burst)",
                  flush=True)
        if (b + 1) % 25 == 0:
            _save(args.out, records, args)
    _save(args.out, records, args)
    means = _arm_means(records)
    print("\n=== QPSKxQPSK @20 dB dedicated (SER) ===")
    for a in args.arms:
        print(f"  {a}: {means[a]:.4f}")
    print("historical log values: v4o_mix 0.106, v3_mix 0.164, "
          "v1_ts 0.041, v1_mix 0.121")


def _save(out_path, records, args):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    agg = aggregate(records) if records else {}
    with open(out_path, 'w') as f:
        json.dump({'experiment': 'E4 true-stream control',
                   'test_seed': args.test_seed,
                   'n_per_cell': args.n_per_cell,
                   'snr_points': args.snr_points,
                   'arms': args.arms,
                   'arm_definitions': {
                       'v1_mix': 'memoryless ECM, blind, raw mixture',
                       'v4_mix': 'ISI-aware ECM L=5, blind, raw mixture',
                       'v3_mix': 'memoryless ECM, oracle df, raw mixture',
                       'v4o_mix': 'ISI-aware ECM L=5 warm-started from '
                                  'v3_mix, raw mixture',
                       'v1_ts': 'memoryless ECM, blind, TRUE source streams',
                       'v4_ts': 'ISI-aware ECM L=5, blind, TRUE streams',
                       'v3_ts': 'memoryless ECM, oracle df, TRUE streams'},
                   'paper_claims_from_log_2026_09_18': PAPER_CLAIMS,
                   'aggregate': agg,
                   'records': records}, f)
    print(f"saved {out_path} ({len(records)} records)", flush=True)


if __name__ == '__main__':
    main()
