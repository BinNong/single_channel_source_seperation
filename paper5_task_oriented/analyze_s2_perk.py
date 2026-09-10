"""Paper 5 — S2 per-K paired permutation test (review follow-up, 2026-09-08).

The pooled S2 analysis (summary_s2.json / summary_s2_cells_perm.json)
mixes K=1/2/3 cells.  An external review noted that the per-K breakdown
(already present in every eval JSON under separation.per_k but never
aggregated) might hide sign-flipped effects that cancel in the pool:
at K=2 the soft-SER term could HELP while at K=3 it HURTS.

This script recomputes the four pre-declared config contrasts per K at
SEED level: 5 paired seed differences per (contrast, metric, K), exact
two-sided sign-flip permutation over all 2^5=32 assignments (smallest
attainable p = 2/32 = 0.0625).  Delta convention matches
analyze_s2_cells.py: d = metric(b) - metric(a), positive = first-named
config (a) better.  A pooled-delta reconstruction check (pairs 1:2:3
weighted) is printed for sanity against summary_s2.json.

Inputs : results/s2/eval_*.json  (via analyze_s2.load)
Outputs: results/s2/summary_s2_perk.json  +  stdout table
"""
from __future__ import annotations

import json
import os
from itertools import product

import numpy as np

from analyze_s2 import RES, load

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = [('ser', 'sisdr'), ('ser', 'mse'), ('ser_mse', 'mse'),
         ('ser_mse', 'sisdr')]
METRICS = (('SER', 'ser_comp'), ('BER', 'ber_comp'))
KS = (1, 2, 3)
N_SEEDS = 5


def perk_diffs(runs, a, b, key, k):
    """Paired per-seed differences b - a at source count k."""
    seeds = sorted(set(runs[a]) & set(runs[b]))
    return np.asarray(
        [runs[b][s]['separation']['per_k'][str(k)][key]
         - runs[a][s]['separation']['per_k'][str(k)][key]
         for s in seeds], float)


def exact_signflip_p(d):
    """Exact two-sided sign-flip permutation p of mean(d)=0 (2^n enum)."""
    obs = abs(d.mean())
    n = len(d)
    worst = 0
    for signs in product((1, -1), repeat=n):
        t = abs(np.dot(signs, d) / n)
        if t >= obs - 1e-15:
            worst += 1
    return worst / (2 ** n)


def main():
    runs = load()
    out = {'design': 'per-K seed-level paired diffs (n=5), exact sign-flip '
                     'over 2^5=32; delta positive = first config better '
                     '(lower SER/BER)'}
    hdr = f'{"contrast":>18s} {"metric":>4s} {"K":>2s} {"delta":>10s} ' \
          f'{"sign+":>6s} {"exact_p":>8s}'
    print(hdr)
    for a, b in PAIRS:
        for metric, key in METRICS:
            rec = {}
            for k in KS:
                d = perk_diffs(runs, a, b, key, k)
                p = exact_signflip_p(d)
                npos = int((d > 0).sum())
                rec[k] = {'delta': float(d.mean()), 'n_pos': npos,
                          'exact_p': p,
                          'per_seed': [float(x) for x in d]}
                print(f'{a + " vs " + b:>18s} {metric:>4s} {k:>2d} '
                      f'{d.mean():+10.5f} {npos:>3d}/5 {p:>8.4f}')
            # pooled reconstruction: pairs per mixture scale as 1:2:3
            w = np.asarray([1., 2., 3.]) / 6.
            pooled = sum(w[i] * rec[k]['delta'] for i, k in enumerate(KS))
            rec['pooled_recon'] = float(pooled)
            out[f'{a}_vs_{b}_{key}'] = rec
            print(f'{"":>18s} {metric:>4s} pooled(1:2:3) {pooled:+10.5f}')
    path = os.path.join(RES, 'summary_s2_perk.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {path}')


if __name__ == '__main__':
    main()
