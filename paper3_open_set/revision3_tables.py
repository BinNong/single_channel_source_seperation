"""
Paper 3 — Revision 3 tables (review2.md additions).

Adds, on top of revision2_tables.py (which it imports and re-uses):

  S7  Consistent Table-I statistics: concatenated pooled AUROC paired with
      its own concatenated bootstrap CI (the v2 table mistakenly paired the
      seed-mean point estimate with the concatenated CI), plus
      Delta-AUROC = AUROC - 0.5 and a label-permutation test vs chance
      (two-sided, B=10000) per scorer.
  S8  Null-oracle calibration: (a) data-based null — within each SNR bin the
      known/unknown labels of the CONCATENATED pool are permuted jointly
      across all six scorers (preserving cross-scorer correlation), the
      per-bin max (oracle) and the a-priori routed rule are recomputed,
      B=2000 per seed, giving the null distribution of the 5-seed mean;
      (b) iid-Gaussian chance-scorer simulation (6 scorers, 7 bins,
      384+384 per bin, per-bin max), B=10000.
  S9  Protocol decomposition from the existing dumps: the unknown pool is
      positionally [U(ku) 0:1344 | U(uu) 1344:2688]; reports K(kk) vs U(ku)
      and K(kk) vs U(uu) wavg AUROC for all six scorers + routed.
      (The same-mixture K(ku) vs U(ku) comparison requires re-inference of
      the ku known slots and is produced server-side by
      eval_protocol_split.py; this script's two comparisons must combine to
      reproduce the revision2 pooled numbers — asserted.)

Standalone analysis script — modifies nothing, writes
results/revision3_tables.json.

Usage:
    ../.venv_verify/bin/python revision3_tables.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import rankdata

from revision2_tables import (BINS, METHODS, RESULTS, SEEDS, auroc_rank,
                              auroc_rows, load_run, route_method, skey)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(RESULTS, 'revision3_tables.json')
REV2_JSON = os.path.join(RESULTS, 'revision2_tables.json')

B_PERM = 10000
B_NULL = 2000
B_SIM = 10000


def r4(x):
    if isinstance(x, dict):
        return {k: r4(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r4(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return round(float(x), 6)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def perm_test_pvalue(sk, su, B=B_PERM, seed=0, chunk=500):
    """Two-sided permutation test of pooled AUROC vs 0.5 (label shuffle)."""
    rng = np.random.RandomState(seed)
    sk = np.asarray(sk, dtype=np.float64).ravel()
    su = np.asarray(su, dtype=np.float64).ravel()
    nk = len(sk)
    comb = np.concatenate([sk, su])
    n = len(comb)
    obs = auroc_rank(sk, su)
    null = np.empty(B)
    for lo in range(0, B, chunk):
        hi = min(lo + chunk, B)
        nb = hi - lo
        idx = np.argsort(rng.rand(nb, n), axis=1)  # nb random permutations
        r = rankdata(comb[idx], axis=1)[:, nk:]
        nu = n - nk
        null[lo:hi] = (r.sum(axis=1) - nu * (nu + 1) / 2.0) / (nk * nu)
    p = (1 + int((np.abs(null - 0.5) >= abs(obs - 0.5)).sum())) / (B + 1)
    return obs, float(p), null


def main():
    out = {'method': {
        's7_table1': 'concatenated pooled AUROC (13440 known + 13440 unknown,'
                     ' all 5 seeds) paired with its own concatenated'
                     ' bootstrap 95% CI (reused from revision2_tables.json'
                     ' s3, B=2000); Delta = AUROC - 0.5; permutation test'
                     f' vs chance: label shuffle, two-sided, B={B_PERM}.',
        's8_null_oracle': '(data) per seed, per SNR bin: joint known/unknown'
                          ' label permutation of the concatenated 768-entry'
                          ' pool, SAME partition across all six scorers'
                          ' (cross-scorer correlation preserved); per-bin'
                          ' max = oracle, energy<=0/prototype>0 = routed;'
                          f' wavg across 7 bins; B={B_NULL} per seed; the'
                          ' null distribution of the 5-seed mean pairs draws'
                          ' by index. (sim) iid Gaussian chance scorers, 6'
                          ' scorers x 7 bins x (384+384), per-bin max,'
                          f' uniform wavg, B={B_SIM}.',
        's9_protocol_split': 'unknown pool positions [0:1344]=ku OOD slot,'
                             ' [1344:2688]=uu both slots (evaluate.py pool'
                             ' order); per-bin AUROC + uniform wavg; sanity'
                             ' asserts the two segments combined reproduce'
                             ' the revision2 s1 wavg numbers.',
    }}

    # ------------------------------------------------------------------
    print('== Loading corrected runs (refpool fitting) ==')
    runs = {}
    for seed in SEEDS:
        scores, k_snr, u_snr, extras = load_run(seed, 'refpool')
        runs[seed] = (scores, k_snr, u_snr, extras)
    print('  loaded seeds', SEEDS)

    rev2 = json.load(open(REV2_JSON))
    concat_auroc = rev2['s3_bootstrap']['auroc_concatenated']
    concat_ci = rev2['s3_bootstrap']['pooled_ci_concatenated']

    # ==================================================================
    print('\n== S7: consistent Table I + Delta-AUROC + permutation tests ==')
    s7 = {}
    for m in METHODS:
        sk = np.concatenate([runs[s][0][m][0] for s in SEEDS])
        su = np.concatenate([runs[s][0][m][1] for s in SEEDS])
        obs, p, _ = perm_test_pvalue(sk, su, seed=hash(m) % 2**31)
        ci95 = concat_ci[m]
        assert abs(obs - concat_auroc[m]) < 2e-4, (m, obs, concat_auroc[m])
        s7[m] = {'auroc_concat': obs, 'ci95_concat': ci95,
                 'delta': obs - 0.5,
                 'delta_ci95': [ci95[0] - 0.5, ci95[1] - 0.5],
                 'perm_p_two_sided': p}
        print(f"  {m:<12} AUROC={obs:.4f} CI=[{ci95[0]:.4f},{ci95[1]:.4f}] "
              f"Delta={obs - 0.5:+.4f} p={p:.2e}")
    out['s7_table1_consistent'] = s7

    # ==================================================================
    print('\n== S8a: data-based null-oracle (bin-wise joint label perm) ==')
    rng = np.random.RandomState(0)
    oracle_draws = np.empty((len(SEEDS), B_NULL))
    routed_draws = np.empty((len(SEEDS), B_NULL))
    for si, seed in enumerate(SEEDS):
        scores, k_snr, u_snr, _ = runs[seed]
        per_bin_oracle = np.zeros((B_NULL, len(BINS)))
        per_bin_routed = np.zeros((B_NULL, len(BINS)))
        for bi, s in enumerate(BINS):
            mk, mu = k_snr == s, u_snr == s
            SK = np.stack([scores[m][0][mk] for m in METHODS])  # (6, nk)
            SU = np.stack([scores[m][1][mu] for m in METHODS])  # (6, nu)
            nk, nu = SK.shape[1], SU.shape[1]
            comb = np.concatenate([SK, SU], axis=1)             # (6, n)
            n = nk + nu
            idx = np.argsort(rng.rand(B_NULL, n), axis=1)       # (B, n)
            # apply same partition to all 6 scorers: (6,B,n) -> (B*6, n)
            perm = comb[:, idx].transpose(1, 0, 2).reshape(B_NULL * 6, n)
            r = rankdata(perm, axis=1)
            r = r.reshape(B_NULL, 6, n)[:, :, nk:]
            aucs = (r.sum(axis=2) - nu * (nu + 1) / 2.0) / (nk * nu)  # (B,6)
            per_bin_oracle[:, bi] = aucs.max(axis=1)
            per_bin_routed[:, bi] = aucs[:, METHODS.index(route_method(s))]
        oracle_draws[si] = per_bin_oracle.mean(axis=1)
        routed_draws[si] = per_bin_routed.mean(axis=1)
        print(f'  seed {seed}: null oracle wavg '
              f'{oracle_draws[si].mean():.4f}±{oracle_draws[si].std():.4f}  '
              f'null routed {routed_draws[si].mean():.4f}±'
              f'{routed_draws[si].std():.4f}')
    oracle_null_mean = oracle_draws.mean(axis=0)
    routed_null_mean = routed_draws.mean(axis=0)
    obs_oracle = rev2['s1_corrected']['across_seeds']['wavg']['oracle']['mean']
    obs_routed = rev2['s1_corrected']['across_seeds']['wavg']['routed']['mean']
    s8a = {
        'oracle_null': {
            'mean': float(oracle_null_mean.mean()),
            'std': float(oracle_null_mean.std()),
            'ci95': [float(np.percentile(oracle_null_mean, 2.5)),
                     float(np.percentile(oracle_null_mean, 97.5))],
            'p_obs_ge': float((1 + (oracle_null_mean >= obs_oracle).sum())
                              / (B_NULL + 1)),
        },
        'routed_null': {
            'mean': float(routed_null_mean.mean()),
            'std': float(routed_null_mean.std()),
            'ci95': [float(np.percentile(routed_null_mean, 2.5)),
                     float(np.percentile(routed_null_mean, 97.5))],
            'p_obs_ge': float((1 + (routed_null_mean >= obs_routed).sum())
                              / (B_NULL + 1)),
        },
        'observed_oracle': obs_oracle, 'observed_routed': obs_routed,
    }
    print(f"  5-seed-mean null: oracle {s8a['oracle_null']['mean']:.4f} "
          f"CI95={s8a['oracle_null']['ci95']}  P(null>={obs_oracle:.4f})="
          f"{s8a['oracle_null']['p_obs_ge']:.4f}")
    print(f"  5-seed-mean null: routed {s8a['routed_null']['mean']:.4f} "
          f"CI95={s8a['routed_null']['ci95']}  P(null>={obs_routed:.4f})="
          f"{s8a['routed_null']['p_obs_ge']:.4f}")

    # ==================================================================
    print('\n== S8b: iid-Gaussian chance-scorer null simulation ==')
    rng2 = np.random.RandomState(1)
    nk = nu = 384
    sim_oracle = np.empty(B_SIM)
    chunk = 1000
    for lo in range(0, B_SIM, chunk):
        nb = min(chunk, B_SIM - lo)
        # (nb, 7 bins, 6 scorers, nk+nu)
        x = rng2.randn(nb, len(BINS), len(METHODS), nk + nu)
        r = rankdata(x.reshape(nb * len(BINS) * len(METHODS), nk + nu),
                     axis=1)
        r = r.reshape(nb, len(BINS), len(METHODS), nk + nu)[..., nk:]
        aucs = (r.sum(axis=3) - nu * (nu + 1) / 2.0) / (nk * nu)  # (nb,7,6)
        sim_oracle[lo:lo + nb] = aucs.max(axis=2).mean(axis=1)
    s8b = {
        'mean': float(sim_oracle.mean()), 'std': float(sim_oracle.std()),
        'ci95': [float(np.percentile(sim_oracle, 2.5)),
                 float(np.percentile(sim_oracle, 97.5))],
        'p_obs_ge': float((1 + (sim_oracle >= obs_oracle).sum()) / (B_SIM + 1)),
        'note': 'single-trial null; the 5-seed MEAN null has std/sqrt(5).',
    }
    s8b['five_seed_mean_std'] = float(sim_oracle.std() / np.sqrt(5))
    s8b['p_obs_ge_5seed_mean'] = float(
        (1 + (np.array([sim_oracle[i::5].mean() for i in
                        range(0, B_SIM - B_SIM % 5, 5)]) >= obs_oracle).sum())
        / (len(range(0, B_SIM - B_SIM % 5, 5)) + 1)) if B_SIM >= 5 else None
    print(f"  iid sim: oracle null {s8b['mean']:.4f}±{s8b['std']:.4f} "
          f"CI95={s8b['ci95']}  P(null>={obs_oracle:.4f})="
          f"{s8b['p_obs_ge']:.4f}")
    out['s8_null_oracle'] = {'data_based': s8a, 'iid_simulation': s8b}

    # ==================================================================
    print('\n== S9: protocol decomposition (K(kk) vs U(ku) / U(uu)) ==')
    s9 = {}
    for m in METHODS + ['routed']:
        s9[m] = {'kk_vs_ku': [], 'kk_vs_uu': []}
    for seed in SEEDS:
        scores, k_snr, u_snr, _ = runs[seed]
        u_ku = np.zeros(len(u_snr), dtype=bool)
        u_ku[:1344] = True
        u_uu = ~u_ku
        # positional sanity: 192 ku entries and 192 uu entries per bin
        for s in BINS:
            assert int(((u_snr == s) & u_ku).sum()) == 192, (seed, s)
            assert int(((u_snr == s) & u_uu).sum()) == 192, (seed, s)
        for m in METHODS:
            for tag, umask in [('kk_vs_ku', u_ku), ('kk_vs_uu', u_uu)]:
                num = den = 0.0
                for s in BINS:
                    mk = k_snr == s
                    mu = (u_snr == s) & umask
                    w = int(mk.sum()) * int(mu.sum())
                    num += w * auroc_rank(scores[m][0][mk], scores[m][1][mu])
                    den += w
                s9[m][tag].append(num / den)
        # routed per comparison
        for tag, umask in [('kk_vs_ku', u_ku), ('kk_vs_uu', u_uu)]:
            num = den = 0.0
            for s in BINS:
                mk = k_snr == s
                mu = (u_snr == s) & umask
                m = route_method(s)
                w = int(mk.sum()) * int(mu.sum())
                num += w * auroc_rank(scores[m][0][mk], scores[m][1][mu])
                den += w
            s9['routed'][tag].append(num / den)
    # sanity: combined segments reproduce revision2 s1 wavg
    s1w = rev2['s1_corrected']['across_seeds']['wavg']
    for m in METHODS + ['routed']:
        for tag in ['kk_vs_ku', 'kk_vs_uu']:
            v = s9[m][tag]
            s9[m][tag] = {'per_seed': v, 'mean': float(np.mean(v)),
                          'std': float(np.std(v))}
        comb = [(a + b) / 2 for a, b in zip(s9[m]['kk_vs_ku']['per_seed'],
                                            s9[m]['kk_vs_uu']['per_seed'])]
        assert abs(np.mean(comb) - s1w[m]['mean']) < 5e-3, \
            (m, np.mean(comb), s1w[m]['mean'])
    print('  sanity: ku/uu segments combined reproduce revision2 s1 wavg: OK')
    print(f"  {'scorer':<12}{'K vs U(ku)':>18}{'K vs U(uu)':>18}")
    for m in METHODS + ['routed']:
        print(f"  {m:<12}{s9[m]['kk_vs_ku']['mean']:>10.4f}±"
              f"{s9[m]['kk_vs_ku']['std']:<6.4f}"
              f"{s9[m]['kk_vs_uu']['mean']:>10.4f}±"
              f"{s9[m]['kk_vs_uu']['std']:<6.4f}")
    out['s9_protocol_split'] = s9

    with open(OUT_JSON, 'w') as f:
        json.dump(r4(out), f, indent=1)
    print(f'\nWrote {OUT_JSON}')


if __name__ == '__main__':
    main()
