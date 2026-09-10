"""Paper 5 — S2 cell-level paired permutation test (review follow-up, 2026-09-08).

The seed-level exact sign-flip test (n=5) is power-capped at p=0.0625
(2/2^5).  This script reruns the same four config contrasts at CELL level:
7 SNR cells x 5 shared seeds = 35 paired differences per contrast.  Runs
with the same seed consume identical data streams and are evaluated on the
same deterministic test set, so (seed, SNR) cells are legitimately paired.

Under the sharp null (no effect in any cell) each paired difference is
symmetric about 0, so sign-flipping is valid per pair.  2^35 is too large
to enumerate; we use 1,000,000 Monte-Carlo sign assignments (rng seed 0),
p = (1 + #{|T*| >= |Tobs|}) / (1 + N).

Inputs : results/s2/eval_*.json  (via analyze_s2.load)
Outputs: results/s2/summary_s2_cells_perm.json  +  stdout table
"""
from __future__ import annotations

import json
import os

import numpy as np

from analyze_s2 import CONFIGS, RES, SNRS, load

HERE = os.path.dirname(os.path.abspath(__file__))
N_FLIPS = 1_000_000
PAIRS = [('ser', 'sisdr'), ('ser', 'mse'), ('ser_mse', 'mse'),
         ('ser_mse', 'sisdr')]
METRICS = (('SER', 'ser_comp'), ('BER', 'ber_comp'))


def cell_diffs(runs, a, b, key):
    """Paired (seed, SNR)-cell differences b - a (positive = a better)."""
    seeds = sorted(set(runs[a]) & set(runs[b]))
    d = []
    for s in seeds:
        for snr in SNRS:
            da = runs[a][s]['separation']['per_snr'][str(snr)][key]
            db = runs[b][s]['separation']['per_snr'][str(snr)][key]
            d.append(db - da)
    return np.asarray(d, float)


def mc_signflip_p(d, n_flips=N_FLIPS, seed=0):
    """Two-sided MC sign-flip permutation p of mean(d)=0."""
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    count = 0
    # chunked to bound memory: [chunk, n] sign matrices
    chunk = 100_000
    done = 0
    while done < n_flips:
        m = min(chunk, n_flips - done)
        signs = rng.integers(0, 2, size=(m, len(d))) * 2 - 1
        t = np.abs((signs * d).mean(axis=1))
        count += int((t >= obs - 1e-15).sum())
        done += m
    return (1 + count) / (1 + n_flips)


def main():
    runs = load()
    out = {'design': '7 SNR cells x 5 seeds = 35 paired diffs per contrast; '
                     'MC sign-flip, 1e6 flips, rng seed 0; '
                     'delta positive = first config better (lower)'}
    print(f'{"contrast":>22s} {"metric":>4s} {"delta":>10s} {"mc_p":>10s}')
    for a, b in PAIRS:
        for metric, key in METRICS:
            d = cell_diffs(runs, a, b, key)
            p = mc_signflip_p(d)
            print(f'{a + " vs " + b:>22s} {metric:>4s} {d.mean():+10.5f} '
                  f'{p:10.6f}')
            out[f'{a}_vs_{b}_{key}'] = {
                'delta': float(d.mean()), 'n_pairs': int(len(d)),
                'mc_p': p, 'n_flips': N_FLIPS}
    path = os.path.join(RES, 'summary_s2_cells_perm.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {path}')


if __name__ == '__main__':
    main()
