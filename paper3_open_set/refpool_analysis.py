"""
Paper 3 — Re-fit OOD reference statistics on the held-out reference pool.

Companion analysis to refpool_dump.py: re-fits the geometry-based
scorers (Prototype, VOS, Mahalanobis) on the held-out reference pool
(seed 88888) instead of the test kk pool, rescores the test pools, and
recomputes per-SNR AUROC + the SNR-routed ensemble.  Logit-based
scorers (Energy, MSP, ODIN) need no reference and are unchanged.

Usage:
    python refpool_analysis.py "results/*seed4[2-6]_best_ood_scores.npz"
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from open_set_metrics import auroc
from ood_scores import compute_prototypes, prototype_score, vos_score
from ood_baselines import mahalanobis_scores

SNR_ROUTE_THRESHOLD = 0.0


def analyze(scores_path: str) -> dict:
    d = np.load(scores_path)
    ref_path = scores_path.replace('_ood_scores.npz', '_refpool.npz')
    r = np.load(ref_path)
    ref_emb, ref_mods = r['ref_emb'], r['ref_mods']

    k_emb, u_emb = d['known_emb'], d['unknown_emb']
    k_snr, u_snr = d['known_snr'], d['unknown_snr']

    proto = compute_prototypes(ref_emb, ref_mods, 4)
    scores = {
        'energy':    (d['energy_score_known'], d['energy_score_unknown']),
        'prototype': (prototype_score(k_emb, proto),
                      prototype_score(u_emb, proto)),
        'vos':       (vos_score(k_emb, proto, seed=0),
                      vos_score(u_emb, proto, seed=0)),
        'mahalanobis': (
            mahalanobis_scores(ref_emb, ref_mods, k_emb),
            mahalanobis_scores(ref_emb, ref_mods, u_emb)),
        'msp': None,
        'odin': None,
    }
    if 'known_logits' in d.files:
        from ood_baselines import msp_scores
        scores['msp'] = (msp_scores(d['known_logits']),
                         msp_scores(d['unknown_logits']))
    odin = sorted(glob.glob(scores_path.replace('_ood_scores.npz',
                                                '_odin_eps*_T*.npz')))
    if odin:
        o = np.load(odin[-1])
        scores['odin'] = (o['odin_score_known'], o['odin_score_unknown'])
    scores = {k: v for k, v in scores.items() if v is not None}

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

    keys = ['routed', 'oracle'] + list(scores)
    return {'per_snr': per_snr, 'wavg': {k: wavg(k) for k in keys},
            'methods': list(scores)}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('patterns', nargs='+')
    args = p.parse_args()
    files = []
    for pat in args.patterns:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        sys.exit('no npz files matched')

    all_wavg, methods = [], None
    for f in files:
        r = analyze(f)
        methods = r['methods']
        all_wavg.append(r['wavg'])
        print(f"\n== {os.path.basename(f)} (reference = held-out pool) ==")
        hdr = ''.join(f'{m:>12}' for m in methods)
        print(f"  {'SNR':>5} |{hdr} | {'routed':>7} | {'oracle':>7}")
        for s, v in sorted(r['per_snr'].items()):
            row = ''.join(f'{v[m]:>12.3f}' for m in methods)
            print(f"  {s:>5.0f} |{row} | {v['routed']:>7.3f} | {v['oracle']:>7.3f}")

    print(f"\n== Across {len(files)} file(s): weighted-avg per-SNR AUROC "
          f"(mean ± std), reference = HELD-OUT pool ==")
    for k in ['routed', 'oracle'] + methods:
        v = [w[k] for w in all_wavg]
        print(f"  {k:<12}: {np.mean(v):.3f} ± {np.std(v):.3f}")


if __name__ == '__main__':
    main()
