"""
Paper 3 — Extended OOD-baseline comparison.

Adds the standard reference baselines that reviewers expect to see next to
Energy / Prototype / VOS:

  - Mahalanobis : min class-conditional Mahalanobis distance in embedding
                  space (shared covariance, shrinkage 0.1), fit on the same
                  known pool the prototypes are computed from (Lee et al.,
                  NeurIPS 2018).  Higher = more OOD.
  - MSP         : -max softmax(logits) (Hendrycks & Gimpel, ICLR 2017).
                  Requires known_logits/unknown_logits in the npz (re-run
                  evaluate.py after the 2026-08 logits-dump edit).
  - ODIN        : -max softmax(f(x~)/T) on the perturbed input (Liang et
                  al., ICLR 2018).  Loaded from the separate
                  `<run>_odin_eps*_T*.npz` produced by odin_dump.py
                  (gradient-based, runs on GPU).

For every method: pooled AUROC (the "trap" view) and per-SNR AUROC with
the weighted average used by the SNR-routed ensemble analysis.  The
routing rule itself is unchanged (energy <= 0 dB, prototype >= 5 dB,
fixed a priori); the new baselines enter only as additional single
scorers and in the post-hoc oracle bound.

Usage:
    python ood_baselines.py "results/*alpha1.0_seed4*_best_ood_scores.npz"
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from open_set_metrics import auroc

SNR_ROUTE_THRESHOLD = 0.0  # a-priori rule: energy <= 0 dB, prototype >= 5 dB


def mahalanobis_scores(ref_emb, ref_mods, query_emb, n_classes=4, shrink=0.1):
    """Min class-conditional Mahalanobis distance with shared covariance."""
    mus, covs = [], []
    for k in range(n_classes):
        z = ref_emb[ref_mods == k]
        mus.append(z.mean(axis=0))
        covs.append(np.cov(z.T))
    mus = np.asarray(mus)
    cov = np.mean(covs, axis=0)
    cov = (1 - shrink) * cov + shrink * np.eye(cov.shape[0]) * \
        np.trace(cov) / cov.shape[0]
    prec = np.linalg.inv(cov)
    d = query_emb[:, None, :] - mus[None, :, :]
    return np.einsum('nkc,cd,nkd->nk', d, prec, d).min(axis=1)


def msp_scores(logits):
    """Negative max softmax probability.  Higher = more OOD."""
    logits = np.asarray(logits, dtype=np.float64)
    z = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
    return -p.max(axis=-1)


def collect_scores(path):
    """Returns {method: (score_known, score_unknown)} for all available
    methods, plus the SNR arrays.  Missing artefacts are skipped with a
    note so the script works both before and after the remote re-runs."""
    d = np.load(path)
    scores = {
        'energy':    (d['energy_score_known'], d['energy_score_unknown']),
        'prototype': (d['prototype_score_known'], d['prototype_score_unknown']),
        'vos':       (d['vos_score_known'], d['vos_score_unknown']),
        'mahalanobis': (
            mahalanobis_scores(d['known_emb'], d['known_mods'], d['known_emb']),
            mahalanobis_scores(d['known_emb'], d['known_mods'], d['unknown_emb']),
        ),
    }
    if 'known_logits' in d.files:
        scores['msp'] = (msp_scores(d['known_logits']),
                         msp_scores(d['unknown_logits']))
    odin = sorted(glob.glob(path.replace('_ood_scores.npz',
                                         '_odin_eps*_T*.npz')))
    if odin:
        o = np.load(odin[-1])
        scores['odin'] = (o['odin_score_known'], o['odin_score_unknown'])
    return scores, d['known_snr'], d['unknown_snr']


def analyze(path):
    scores, k_snr, u_snr = collect_scores(path)
    methods = list(scores.keys())
    pooled = {m: auroc(sk, su) for m, (sk, su) in scores.items()}

    per_snr = {}
    for s in sorted(set(k_snr.tolist()) & set(u_snr.tolist())):
        mk, mu = k_snr == s, u_snr == s
        if mk.sum() < 5 or mu.sum() < 5:
            continue
        aucs = {m: auroc(sk[mk], su[mu]) for m, (sk, su) in scores.items()}
        rule_m = 'energy' if s <= SNR_ROUTE_THRESHOLD else 'prototype'
        per_snr[s] = {'n_pairs': int(mk.sum()) * int(mu.sum()),
                      'routed': aucs[rule_m],
                      'oracle': max(aucs.values()), **aucs}

    def wavg(key):
        num = sum(v[key] * v['n_pairs'] for v in per_snr.values())
        den = sum(v['n_pairs'] for v in per_snr.values())
        return num / den

    keys = ['routed', 'oracle'] + methods
    return {'pooled': pooled, 'per_snr': per_snr,
            'wavg': {k: wavg(k) for k in keys}, 'methods': methods}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('patterns', nargs='+')
    args = p.parse_args()
    files = []
    for pat in args.patterns:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        sys.exit('no npz files matched')

    all_pooled, all_wavg = [], []
    methods = None
    for f in files:
        r = analyze(f)
        methods = r['methods']
        all_pooled.append(r['pooled'])
        all_wavg.append(r['wavg'])
        print(f"\n== {os.path.basename(f)} ==")
        hdr = ''.join(f'{m:>12}' for m in methods)
        print(f"  {'SNR':>5} |{hdr} | {'routed':>7} | {'oracle':>7}")
        for s, v in sorted(r['per_snr'].items()):
            row = ''.join(f'{v[m]:>12.3f}' for m in methods)
            print(f"  {s:>5.0f} |{row} | {v['routed']:>7.3f} | {v['oracle']:>7.3f}")
        print('  pooled  : ' + '  '.join(f"{m}={r['pooled'][m]:.3f}"
                                         for m in methods))
        print('  wavg    : ' + '  '.join(f"{m}={r['wavg'][m]:.3f}" for m in methods)
              + f"   routed={r['wavg']['routed']:.3f}  oracle={r['wavg']['oracle']:.3f}")

    print(f"\n== Across {len(files)} file(s) (mean ± std) ==")
    print('  pooled:')
    for m in methods:
        v = [p[m] for p in all_pooled]
        print(f"    {m:<12}: {np.mean(v):.3f} ± {np.std(v):.3f}")
    print('  weighted-avg per-SNR:')
    for k in ['routed', 'oracle'] + methods:
        v = [w[k] for w in all_wavg]
        print(f"    {k:<12}: {np.mean(v):.3f} ± {np.std(v):.3f}")


if __name__ == '__main__':
    main()
