"""Aggregate the reviewer-2 experiments for the WCL letter v2.

1. lambda_ser sweep (MC3): results/lambda_sweep/ (lser0p01/0p1/10,
   3 seeds) joined with the existing results/s2 points (lambda 0 = mse,
   lambda 1 = ser_mse, 5 seeds).  Reports pooled SER/BER/SI-SDRi per
   lambda and the per-K paired deltas vs lambda=0.

2. Multi-test-seed robustness (MC5): results/testseed_robustness/
   (baseline + paper4 honest pool + ser_mse on seeds 100001-100003)
   plus results/testseed_robustness_s2b/ (the S2 (b) mse supplement —
   run separately because of a checkpoint-basename collision with
   paper4's slot mse files).  Reports S1 flatness per test grid and
   the S2 fair-additive per-K sign per test grid.

Numbers from this script feed paper5/letter.tex v2 (Sections II-IV);
see EXPERIMENT_LOG.md 2026-10-04.
"""

import glob
import json

import numpy as np

S2 = 'results/s2'
LS = 'results/lambda_sweep'
TS = 'results/testseed_robustness'
TSB = 'results/testseed_robustness_s2b'


def load(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        seed = int(f.split('_s')[-1].split('.')[0].split('_')[0])
        out[seed] = json.load(open(f))
    return out


def lambda_sweep():
    points = {
        0.0:  load(f'{S2}/eval_slot_h64_l4_k13_bs16_lr0.001_mse_s4[2-6].json'),
        0.01: load(f'{LS}/eval_slot_h64_l4_k13_bs16_lr0.001_lser0p01_s4[2-6].json'),
        0.1:  load(f'{LS}/eval_slot_h64_l4_k13_bs16_lr0.001_lser0p1_s4[2-6].json'),
        1.0:  load(f'{S2}/eval_slot_h64_l4_k13_bs16_lr0.001_ser_mse_s4[2-6].json'),
        10.0: load(f'{LS}/eval_slot_h64_l4_k13_bs16_lr0.001_lser10_s4[2-6].json'),
    }
    print('=== lambda_ser sweep (pooled) ===')
    print('lambda  n  SER(mean±std)    BER(mean±std)    SI-SDRi')
    for lam, ds in points.items():
        ser = [d['separation']['ser_comp'] for d in ds.values()]
        ber = [d['separation']['ber_comp'] for d in ds.values()]
        si = [d['separation']['si_sdri'] for d in ds.values()]
        print(f'{lam:<7} {len(ds)}  {np.mean(ser):.4f}±{np.std(ser):.4f}  '
              f'{np.mean(ber):.4f}±{np.std(ber):.4f}  +{np.mean(si):.2f}±{np.std(si):.2f}')
    print('\nper-K paired delta vs lambda=0 (pp):')
    base = points[0.0]
    for lam in (0.01, 0.1, 1.0, 10.0):
        ds = points[lam]
        for k in ('1', '2', '3'):
            dser = [(base[s]['separation']['per_k'][k]['ser_comp']
                     - ds[s]['separation']['per_k'][k]['ser_comp']) * 100
                    for s in ds]
            dber = [(base[s]['separation']['per_k'][k]['ber_comp']
                     - ds[s]['separation']['per_k'][k]['ber_comp']) * 100
                    for s in ds]
            print(f'  lambda={lam:<5} K={k}: dSER {np.mean(dser):+.3f}±{np.std(dser):.3f}  '
                  f'dBER {np.mean(dber):+.3f}±{np.std(dber):.3f}  '
                  f'({sum(1 for x in dber if x > 0)}/{len(dber)} BER pos)')


def testseed_robustness():
    print('\n=== S1 flatness per test grid (honest pool, 10 runs) ===')
    for ts in (100001, 100002, 100003):
        base = json.load(open(f'{TS}/baseline_ts{ts}.json'))
        sers, bers, sis = [], [], []
        for f in glob.glob(f'{TS}/eval_*_mse_s4[2-6]_ts{ts}.json'):
            d = json.load(open(f))
            sers.append(d['separation']['ser_comp'])
            bers.append(d['separation']['ber_comp'])
            sis.append(d['separation']['si_sdri'])
        print(f'ts{ts}: mixture SER {base["overall"]:.4f} BER {base["overall_ber"]:.4f} | '
              f'sep SER {np.mean(sers):.4f}±{np.std(sers):.4f} '
              f'BER {np.mean(bers):.4f}±{np.std(bers):.4f} '
              f'SI-SDRi +{np.mean(sis):.2f}')
    print('\n=== S2 fair-additive per-K sign per test grid (5 seeds) ===')
    for ts in (100001, 100002, 100003):
        b = load(f'{TSB}/eval_slot_h64_l4_k13_bs16_lr0.001_mse_s4[2-6]_ts{ts}.json')
        t = load(f'{TS}/eval_slot_h64_l4_k13_bs16_lr0.001_ser_mse_s4[2-6]_ts{ts}.json')
        for k in ('1', '2', '3'):
            dser = [(b[s]['separation']['per_k'][k]['ser_comp']
                     - t[s]['separation']['per_k'][k]['ser_comp']) * 100
                    for s in sorted(set(b) & set(t))]
            dber = [(b[s]['separation']['per_k'][k]['ber_comp']
                     - t[s]['separation']['per_k'][k]['ber_comp']) * 100
                    for s in sorted(set(b) & set(t))]
            print(f'ts{ts} K={k}: dSER {np.mean(dser):+.3f}±{np.std(dser):.3f} '
                  f'({sum(1 for x in dser if x > 0)}/5 pos)  '
                  f'dBER {np.mean(dber):+.3f}±{np.std(dber):.3f} '
                  f'({sum(1 for x in dber if x > 0)}/5 pos)')


if __name__ == '__main__':
    lambda_sweep()
    testseed_robustness()
