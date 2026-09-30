"""
Paper 3 — Revision 4 tables: all headline statistics on the
TRUTH-ANCHORED dumps (second label-bug fix, 2026-09-30).

evaluate._collect_predictions PIT-anchors waveforms/embeddings/logits to
the true sources but also swapped the per-source labels; the *_ta.npz
dumps (eval_truth_anchor_dump.py, server, test seed 99999, 192/192/96
per bin) keep content anchored and labels truth-anchored. This script
re-runs the revision2/revision3 analyses on those dumps:

  T1  master table (per-bin / pooled / wavg, 5 seeds, refpool fit)
  T2  pooled bootstrap CIs (concatenated) + Delta-AUROC + permutation
      tests vs chance (B=10000)
  T3  per-unknown-class wavg
  T4  permutation ablation of known-pool SNR labels (pi_repeat now
      isolates bug 1 alone — pool membership is clean)
  T5  bin-wise label-permutation null for the post-hoc oracle (B=2000)
  T6  protocol decomposition by unknown-pool position (ku | uu)

Writes results/revision4_tables_ta.json.

Usage:
    ../.venv_verify/bin/python revision4_tables_ta.py
"""

from __future__ import annotations

import json
import os

import numpy as np

import revision2_tables as r2
import revision3_tables as r3

RESULTS = r2.RESULTS
OUT_JSON = os.path.join(RESULTS, 'revision4_tables_ta.json')
SEEDS, BINS, METHODS = r2.SEEDS, r2.BINS, r2.METHODS
auroc_rank, auroc_rows = r2.auroc_rank, r2.auroc_rows
route_method, skey, ci = r2.route_method, r2.skey, r2.ci

# sanity targets: fully-TA pipeline (TA dumps + TA refpool fit) — routed/
# prototype from the server's dump_ta_scores sanity (2026-09-30); others
# from truth_anchor.json (swapped-refpool flavor), tolerance 0.02.
TA_TARGET = {'energy': 0.4314, 'msp': 0.3936, 'odin': 0.4354,
             'mahalanobis': 0.5320, 'prototype': 0.5178, 'vos': 0.5323,
             'routed': 0.5077}
TA_TOL = 0.02


def main():
    r2.DUMP_SUFFIX = '_ta'
    r2.REFPOOL_SUFFIX = '_ta'
    out = {'method': {
        'dumps': 'TRUTH-ANCHORED *_ta.npz (test seed 99999, 192/192/96 '
                 'per bin): content PIT-anchored to true sources, labels '
                 'truth-anchored (NOT swapped); refpool fitting as in '
                 'revision2 (seed 88888, kk).',
        'note': 'same statistics as revision2_tables.py s1/s3/s4/s5 and '
                'revision3_tables.py s7/s8a/s9, recomputed on the TA dumps.',
    }}

    # ------------------------------------------------------------------
    print('== T1: master table (TA dumps) ==')
    runs = {}
    results = []
    for seed in SEEDS:
        scores, k_snr, u_snr, extras = r2.load_run(seed, 'refpool')
        runs[seed] = (scores, k_snr, u_snr, extras)
        results.append(r2.eval_master(scores, k_snr, u_snr))
    r2.print_master('TA/refpool', results)
    across = r2.across_seeds(results)
    r2.print_across(across)
    for m, tgt in TA_TARGET.items():
        got = across['wavg'][m]['mean']
        status = 'OK' if abs(got - tgt) < TA_TOL else 'MISMATCH'
        print(f'  sanity {m:<12} TA-dump wavg {got:.4f} vs '
              f'target {tgt:.4f}: {status}')
    # per-bin known modulation audit
    d0 = np.load(os.path.join(RESULTS, r2.BASE.format(42)
                              + '_ood_scores_ta.npz'))
    km, ks = d0['known_mods'], d0['known_snr']
    for s in BINS:
        counts = [int(((ks == s) & (km == c)).sum()) for c in range(4)]
        assert counts == [96, 96, 96, 96], (s, counts)
    print('  per-bin known-mod audit 96/96/96/96: OK')
    out['t1_master'] = {'per_seed': {str(s): r for s, r in zip(SEEDS, results)},
                        'across_seeds': across}

    # ------------------------------------------------------------------
    print('\n== T2: bootstrap CIs + Delta + permutation tests (TA) ==')
    t2 = {}
    for m in METHODS:
        sk = np.concatenate([runs[s][0][m][0] for s in SEEDS])
        su = np.concatenate([runs[s][0][m][1] for s in SEEDS])
        draws = r2.boot_pooled(sk, su, chunk=250)
        obs, p, _ = r3.perm_test_pvalue(sk, su, seed=hash(m) % 2**31)
        t2[m] = {'auroc_concat': obs, 'ci95_concat': ci(draws),
                 'delta': obs - 0.5,
                 'delta_ci95': [ci(draws)[0] - 0.5, ci(draws)[1] - 0.5],
                 'perm_p_two_sided': p}
        c = ci(draws)
        print(f"  {m:<12} AUROC={obs:.4f} CI=[{c[0]:.4f},{c[1]:.4f}] "
              f"Delta={obs - 0.5:+.4f} p={p:.2e}")
    out['t2_bootstrap'] = t2

    # ------------------------------------------------------------------
    print('\n== T3: per-unknown-class wavg (TA) ==')
    t3 = {m: {} for m in METHODS}
    for m in METHODS:
        for c, cname in r2.CLASS_NAMES.items():
            wavgs = []
            for seed in SEEDS:
                scores, k_snr, u_snr, extras = runs[seed]
                mods = extras['unknown_mods']
                num = den = 0.0
                for s in BINS:
                    mk = k_snr == s
                    mu = (u_snr == s) & (mods == c)
                    w = int(mk.sum()) * int(mu.sum())
                    num += w * auroc_rank(scores[m][0][mk], scores[m][1][mu])
                    den += w
                wavgs.append(num / den)
            t3[m][cname] = {'mean': float(np.mean(wavgs)),
                            'std': float(np.std(wavgs))}
    for cname in r2.CLASS_NAMES.values():
        print(f"  {cname:<12}" + ''.join(f"{t3[m][cname]['mean']:>8.4f}"
                                         for m in METHODS))
    out['t3_per_class'] = t3

    # ------------------------------------------------------------------
    print('\n== T4: permutation ablation on TA dumps (bug-1 isolation) ==')
    rng_perm = np.random.RandomState(0)
    t4 = {}
    per_seed_res = {}
    for seed in SEEDS:
        scores, k_snr, u_snr, extras = runs[seed]
        scores_st = {
            'energy': (extras['stored_energy_known'],
                       extras['stored_energy_unknown']),
            'prototype': (extras['stored_prototype_known'],
                          extras['stored_prototype_unknown']),
        }
        labels0 = k_snr.copy()
        perms = {
            'pi_identity': labels0,
            'pi_repeat': np.repeat(extras['snr_kk'], 2).astype(np.float64),
            'pi_shiftHalf': np.roll(labels0, len(labels0) // 2),
            'pi_reverse': labels0[::-1],
        }
        perms['pi_random'] = [rng_perm.permutation(labels0)
                              for _ in range(10)]

        def wavg_with(lab):
            res = {}
            for m in ['energy', 'prototype']:
                num = den = 0.0
                for s in BINS:
                    mk, mu = lab == s, u_snr == s
                    w = int(mk.sum()) * int(mu.sum())
                    num += w * auroc_rank(scores_st[m][0][mk],
                                          scores_st[m][1][mu])
                    den += w
                res[m] = num / den
            num = den = 0.0
            for s in BINS:
                mk, mu = lab == s, u_snr == s
                m = route_method(s)
                w = int(mk.sum()) * int(mu.sum())
                num += w * auroc_rank(scores_st[m][0][mk],
                                      scores_st[m][1][mu])
                den += w
            res['routed'] = num / den
            return res

        seed_res = {name: wavg_with(p) for name, p in perms.items()
                    if name != 'pi_random'}
        rnd = [wavg_with(p) for p in perms['pi_random']]
        seed_res['pi_random'] = {m: float(np.mean([r[m] for r in rnd]))
                                 for m in ['energy', 'prototype', 'routed']}
        per_seed_res[str(seed)] = seed_res
    for name in ['pi_identity', 'pi_repeat', 'pi_shiftHalf', 'pi_reverse',
                 'pi_random']:
        row = {}
        for m in ['energy', 'prototype', 'routed']:
            v = [per_seed_res[str(s)][name][m] for s in SEEDS]
            row[m] = {'mean': float(np.mean(v)), 'std': float(np.std(v))}
        t4[name] = row
        print(f"  {name:<14}{row['energy']['mean']:>10.4f}"
              f"{row['prototype']['mean']:>10.4f}{row['routed']['mean']:>10.4f}")
    t4['per_seed'] = per_seed_res
    out['t4_permutation'] = t4

    # ------------------------------------------------------------------
    print('\n== T5: null-oracle (bin-wise joint label perm, TA) ==')
    rng = np.random.RandomState(0)
    from scipy.stats import rankdata
    B_NULL = 2000
    oracle_draws = np.empty((len(SEEDS), B_NULL))
    for si, seed in enumerate(SEEDS):
        scores, k_snr, u_snr, _ = runs[seed]
        per_bin_oracle = np.zeros((B_NULL, len(BINS)))
        for bi, s in enumerate(BINS):
            mk, mu = k_snr == s, u_snr == s
            SK = np.stack([scores[m][0][mk] for m in METHODS])
            SU = np.stack([scores[m][1][mu] for m in METHODS])
            nk, nu = SK.shape[1], SU.shape[1]
            comb = np.concatenate([SK, SU], axis=1)
            n = nk + nu
            idx = np.argsort(rng.rand(B_NULL, n), axis=1)
            perm = comb[:, idx].transpose(1, 0, 2).reshape(B_NULL * 6, n)
            r = rankdata(perm, axis=1).reshape(B_NULL, 6, n)[:, :, nk:]
            aucs = (r.sum(axis=2) - nu * (nu + 1) / 2.0) / (nk * nu)
            per_bin_oracle[:, bi] = aucs.max(axis=1)
        oracle_draws[si] = per_bin_oracle.mean(axis=1)
    null_mean = oracle_draws.mean(axis=0)
    obs_oracle = across['wavg']['oracle']['mean']
    t5 = {'oracle_null_mean': float(null_mean.mean()),
          'oracle_null_ci95': [float(np.percentile(null_mean, 2.5)),
                               float(np.percentile(null_mean, 97.5))],
          'observed_oracle': float(obs_oracle),
          'p_obs_ge': float((1 + (null_mean >= obs_oracle).sum())
                            / (B_NULL + 1))}
    print(f"  null oracle {t5['oracle_null_mean']:.4f} "
          f"CI95={t5['oracle_null_ci95']}, observed {obs_oracle:.4f}, "
          f"P(null>=obs)={t5['p_obs_ge']:.4f}")
    out['t5_null_oracle'] = t5

    # ------------------------------------------------------------------
    print('\n== T6: protocol decomposition by position (TA) ==')
    t6 = {m: {'kk_vs_ku': [], 'kk_vs_uu': []} for m in METHODS + ['routed']}
    for seed in SEEDS:
        scores, k_snr, u_snr, _ = runs[seed]
        u_ku = np.zeros(len(u_snr), dtype=bool)
        u_ku[:1344] = True
        for s in BINS:
            assert int(((u_snr == s) & u_ku).sum()) == 192, (seed, s)
            assert int(((u_snr == s) & ~u_ku).sum()) == 192, (seed, s)
        for m in METHODS + ['routed']:
            for tag, umask in [('kk_vs_ku', u_ku), ('kk_vs_uu', ~u_ku)]:
                num = den = 0.0
                for s in BINS:
                    mk = k_snr == s
                    mu = (u_snr == s) & umask
                    mm = route_method(s) if m == 'routed' else m
                    w = int(mk.sum()) * int(mu.sum())
                    num += w * auroc_rank(scores[mm][0][mk],
                                          scores[mm][1][mu])
                    den += w
                t6[m][tag].append(num / den)
    for m in METHODS + ['routed']:
        for tag in ['kk_vs_ku', 'kk_vs_uu']:
            v = t6[m][tag]
            t6[m][tag] = {'mean': float(np.mean(v)),
                          'std': float(np.std(v))}
        print(f"  {m:<12} ku {t6[m]['kk_vs_ku']['mean']:.4f}±"
              f"{t6[m]['kk_vs_ku']['std']:.4f}   uu "
              f"{t6[m]['kk_vs_uu']['mean']:.4f}±"
              f"{t6[m]['kk_vs_uu']['std']:.4f}")
    out['t6_protocol_split'] = t6

    with open(OUT_JSON, 'w') as f:
        json.dump(r3.r4(out), f, indent=1)
    print(f'\nWrote {OUT_JSON}')


if __name__ == '__main__':
    main()
