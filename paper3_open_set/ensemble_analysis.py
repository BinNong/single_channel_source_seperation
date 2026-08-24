"""SNR-adaptive ensemble analysis for OOD scores.

Estimates the benefit of routing between OOD scorers by per-sample SNR:
  - routed   : energy for snr <= threshold, prototype otherwise
               (rule fixed a priori from the baseline per-SNR profile)
  - oracle   : best of the three scorers per SNR bin (upper bound, post-hoc)

AUROC is computed per SNR bin (fair: both pools at the same SNR) and then
averaged across bins weighted by n_known * n_unknown pairs, so the result
is comparable to a pooled aggregate AUROC without mixing score scales.

Usage:
    python ensemble_analysis.py "results/*_lc0.1*_ood_scores.npz"
    python ensemble_analysis.py file1.npz file2.npz --threshold 0
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from open_set_metrics import auroc

METHODS = ['energy', 'prototype', 'vos']


def analyze_file(path: str, threshold: float) -> dict:
    d = np.load(path)
    k_snr, u_snr = d['known_snr'], d['unknown_snr']
    scores = {m: (d[f'{m}_score_known'], d[f'{m}_score_unknown'])
              for m in METHODS}

    snrs = sorted(set(k_snr.tolist()) & set(u_snr.tolist()))
    per_snr = {}
    for s in snrs:
        mk, mu = k_snr == s, u_snr == s
        if mk.sum() < 5 or mu.sum() < 5:
            continue
        aucs = {m: auroc(sk[mk], su[mu]) for m, (sk, su) in scores.items()}
        rule_m = 'energy' if s <= threshold else 'prototype'
        per_snr[s] = {
            'n_pairs': int(mk.sum()) * int(mu.sum()),
            'rule_method': rule_m,
            'routed': aucs[rule_m],
            'oracle': max(aucs.values()),
            **aucs,
        }

    def wavg(key):
        num = sum(v[key] * v['n_pairs'] for v in per_snr.values())
        den = sum(v['n_pairs'] for v in per_snr.values())
        return num / den

    return {'per_snr': per_snr,
            'wavg': {k: wavg(k) for k in ['routed', 'oracle'] + METHODS}}


def noisy_routed_wavg(d: dict, threshold: float, noise_std: float,
                      rng: np.random.Generator) -> float:
    """Weighted-avg per-SNR AUROC of the routed scorer when the routing
    decision uses a noisy SNR estimate (est = true + N(0, noise_std)).

    The metric bins by TRUE SNR (fair known/unknown comparison); only the
    method choice per sample is perturbed — this mirrors a deployed system
    routing on an SNR estimator's output.
    """
    k_snr, u_snr = d['known_snr'], d['unknown_snr']
    e_k, e_u = d['energy_score_known'], d['energy_score_unknown']
    p_k, p_u = d['prototype_score_known'], d['prototype_score_unknown']

    k_est = k_snr + rng.normal(0, noise_std, size=k_snr.shape)
    u_est = u_snr + rng.normal(0, noise_std, size=u_snr.shape)
    k_routed = np.where(k_est <= threshold, e_k, p_k)
    u_routed = np.where(u_est <= threshold, e_u, p_u)

    num, den = 0.0, 0.0
    for s in sorted(set(k_snr.tolist()) & set(u_snr.tolist())):
        mk, mu = k_snr == s, u_snr == s
        if mk.sum() < 5 or mu.sum() < 5:
            continue
        w = int(mk.sum()) * int(mu.sum())
        num += auroc(k_routed[mk], u_routed[mu]) * w
        den += w
    return num / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument('patterns', nargs='+', help='npz files or glob patterns')
    p.add_argument('--threshold', type=float, default=0.0,
                   help='SNR at or below which energy is used (default 0)')
    p.add_argument('--snr_noise', type=float, nargs='*', default=[],
                   help='SNR-estimation noise levels (std, dB) for routing '
                        'sensitivity analysis, e.g. --snr_noise 3 6')
    p.add_argument('--trials', type=int, default=50,
                   help='Monte-Carlo trials per noise level (default 50)')
    p.add_argument('--mc_seed', type=int, default=0)
    args = p.parse_args()

    files = []
    for pat in args.patterns:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        sys.exit('no npz files matched')

    all_wavg = []
    for f in files:
        r = analyze_file(f, args.threshold)
        all_wavg.append(r['wavg'])
        print(f"\n== {os.path.basename(f)} ==")
        print(f"  {'SNR':>5} | {'energy':>7} {'proto':>7} {'vos':>7} | "
              f"{'routed':>7} ({'rule'}) | {'oracle':>7}")
        for s, v in sorted(r['per_snr'].items()):
            print(f"  {s:>5.0f} | {v['energy']:>7.3f} {v['prototype']:>7.3f} "
                  f"{v['vos']:>7.3f} | {v['routed']:>7.3f} ({v['rule_method'][:5]}) | "
                  f"{v['oracle']:>7.3f}")
        w = r['wavg']
        print(f"  weighted-avg: routed={w['routed']:.3f}  oracle={w['oracle']:.3f}  "
              f"energy={w['energy']:.3f}  proto={w['prototype']:.3f}  vos={w['vos']:.3f}")

    print(f"\n== Across {len(files)} file(s) (mean ± std of weighted-avg) ==")
    for k in ['routed', 'oracle'] + METHODS:
        v = [w[k] for w in all_wavg]
        print(f"  {k:<10}: {np.mean(v):.3f} ± {np.std(v):.3f}")

    # ---- SNR-estimation noise sensitivity (routed scorer only) ----
    for noise in args.snr_noise:
        rng = np.random.default_rng(args.mc_seed)
        per_file = []
        for f in files:
            d = np.load(f)
            trials = [noisy_routed_wavg(d, args.threshold, noise, rng)
                      for _ in range(args.trials)]
            per_file.append(trials)
        # Per-file trial means, then mean ± std across files
        file_means = [np.mean(t) for t in per_file]
        trial_spread = np.mean([np.std(t) for t in per_file])
        print(f"\n== Routed AUROC with SNR-est noise σ={noise:g} dB "
              f"({args.trials} MC trials/file) ==")
        for f, m in zip(files, file_means):
            print(f"  {os.path.basename(f)}: {m:.3f}")
        print(f"  across files: {np.mean(file_means):.3f} ± "
              f"{np.std(file_means):.3f}  (avg MC spread {trial_spread:.3f})")


if __name__ == '__main__':
    main()
