"""
Paper 3 — Revision 2 tables: six-scorer master tables (corrected vs archived
buggy SNR labels), bootstrap confidence intervals, per-unknown-class AUROC,
SNR-label permutation ablation, and operating-point summary.

Standalone analysis script — modifies nothing, reads only:

  corrected dumps : results/openset_..._seed{42..46}_best_ood_scores.npz
                    results/openset_..._seed{42..46}_best_odin_eps0.005_T1000.npz
                    results/openset_..._seed{42..46}_best_refpool.npz
  archived buggy  : results/archive_buggy_snr_labels/ (same filenames)
  operating point : results/routed_threshold_metrics.json

and writes results/revision2_tables.json.

Label-layout facts (verified 2026-09-23):
  * Score arrays are TILE-stacked: entries [0:1344] = slot 1 of every
    mixture, entries [1344:2688] = slot 2.  Corrected known_snr equals
    np.tile(snr_kk, 2); the archived (buggy) known_snr equals
    np.repeat(snr_kk, 2).  unknown_snr is IDENTICAL in both dumps (the
    unknown pool follows its own per-mixture SNR schedule, not snr_kk).
  * Per SNR bin: 384 known and 384 unknown entries (192 mixtures x 2 slots);
    each unknown class contributes 96 per bin.

Usage:
    ../.venv_verify/bin/python revision2_tables.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from open_set_metrics import auroc as auroc_pairwise          # noqa: E402
from ood_scores import (compute_prototypes, energy_score,     # noqa: E402
                        prototype_score, vos_score)
from ood_baselines import mahalanobis_scores, msp_scores      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
ARCHIVE = os.path.join(RESULTS, 'archive_buggy_snr_labels')
OUT_JSON = os.path.join(RESULTS, 'revision2_tables.json')

SEEDS = [42, 43, 44, 45, 46]
BINS = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
METHODS = ['energy', 'msp', 'odin', 'mahalanobis', 'prototype', 'vos']
CLASS_NAMES = {4: '64QAM', 5: 'pi/4-DQPSK', 6: 'MSK', 7: 'OFDM-QPSK'}
B_BOOT = 2000

BASE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed{}_best'
ODIN_SUFFIX = '_odin_eps0.005_T1000.npz'
# Truth-anchored dumps (second label-bug fix, 2026-09-30) carry the '_ta'
# suffix before the extension: set DUMP_SUFFIX='_ta' (before calling
# load_run) to analyse them. REFPOOL_SUFFIX='_ta' additionally fits the
# reference scorers on the truth-anchored reference pool (the original
# refpool's per-slot labels were also ~37% swapped).
DUMP_SUFFIX = ''
REFPOOL_SUFFIX = ''


def skey(s):
    return str(int(s))


def route_method(s):
    return 'energy' if s <= 0 else 'prototype'


# ----------------------------------------------------------------------------
# Fast rank-based AUROC (Mann-Whitney, average ranks for ties)
# ----------------------------------------------------------------------------
def auroc_rank(sk, su):
    sk = np.asarray(sk, dtype=np.float64).ravel()
    su = np.asarray(su, dtype=np.float64).ravel()
    nk, nu = len(sk), len(su)
    if nk == 0 or nu == 0:
        return float('nan')
    r = rankdata(np.concatenate([sk, su]))
    return float((r[nk:].sum() - nu * (nu + 1) / 2.0) / (nk * nu))


def auroc_rows(comb, nk):
    """Vectorised AUROC for rows of [known | unknown] concatenated scores."""
    nu = comb.shape[1] - nk
    r = rankdata(comb, axis=1)[:, nk:]
    return (r.sum(axis=1) - nu * (nu + 1) / 2.0) / (nk * nu)


# ----------------------------------------------------------------------------
# Data loading / scorer computation
# ----------------------------------------------------------------------------
def load_run(seed, fit):
    """Returns (scores, k_snr, u_snr, extras).

    fit='refpool'  : corrected dump; prototype/vos/mahalanobis fit on the
                     disjoint reference pool (seed 88888, kk protocol).
    fit='testpool' : archived buggy dump; reference scorers fit on the
                     archived known test pool (the original artefact
                     pipeline); labels are the stored buggy ones.
    """
    folder = RESULTS if fit == 'refpool' else ARCHIVE
    sfx = DUMP_SUFFIX if fit == 'refpool' else ''
    d = np.load(os.path.join(folder, BASE.format(seed)
                             + '_ood_scores' + sfx + '.npz'))
    o = np.load(os.path.join(folder, BASE.format(seed)
                             + ODIN_SUFFIX.replace('.npz', sfx + '.npz')))

    if fit == 'refpool':
        rp = np.load(os.path.join(RESULTS, BASE.format(seed) + '_refpool'
                                  + REFPOOL_SUFFIX + '.npz'))
        ref_emb, ref_mods = rp['ref_emb'], rp['ref_mods']
    else:
        ref_emb, ref_mods = d['known_emb'], d['known_mods']

    protos = compute_prototypes(ref_emb, ref_mods, 4)
    scores = {
        'energy': (energy_score(d['known_logits']),
                   energy_score(d['unknown_logits'])),
        'msp': (msp_scores(d['known_logits']),
                msp_scores(d['unknown_logits'])),
        'odin': (o['odin_score_known'].astype(np.float64),
                 o['odin_score_unknown'].astype(np.float64)),
        'mahalanobis': (mahalanobis_scores(ref_emb, ref_mods, d['known_emb'],
                                           n_classes=4, shrink=0.1),
                        mahalanobis_scores(ref_emb, ref_mods, d['unknown_emb'],
                                           n_classes=4, shrink=0.1)),
        'prototype': (prototype_score(d['known_emb'], protos),
                      prototype_score(d['unknown_emb'], protos)),
        'vos': (vos_score(d['known_emb'], protos, alpha=2.0, n_per_class=100,
                          seed=0),
                vos_score(d['unknown_emb'], protos, alpha=2.0,
                          n_per_class=100, seed=0)),
    }
    extras = {'unknown_mods': d['unknown_mods'], 'snr_kk': d['snr_kk'],
              'stored_energy_known': d['energy_score_known'],
              'stored_prototype_known': d['prototype_score_known'],
              'stored_energy_unknown': d['energy_score_unknown'],
              'stored_prototype_unknown': d['prototype_score_unknown']}
    return scores, d['known_snr'].astype(np.float64), \
        d['unknown_snr'].astype(np.float64), extras


# ----------------------------------------------------------------------------
# Per-seed master table evaluation
# ----------------------------------------------------------------------------
def eval_master(scores, k_snr, u_snr):
    per_bin = {}
    for s in BINS:
        mk, mu = k_snr == s, u_snr == s
        nk, nu = int(mk.sum()), int(mu.sum())
        aucs = {m: auroc_rank(scores[m][0][mk], scores[m][1][mu])
                for m in METHODS}
        per_bin[skey(s)] = {
            'n_known': nk, 'n_unknown': nu, 'n_pairs': nk * nu,
            **aucs,
            'routed': aucs[route_method(s)],
            'oracle': max(aucs.values()),
        }
    pooled = {m: auroc_rank(scores[m][0], scores[m][1]) for m in METHODS}

    def wavg(key):
        num = sum(v[key] * v['n_pairs'] for v in per_bin.values())
        den = sum(v['n_pairs'] for v in per_bin.values())
        return num / den

    w = {k: wavg(k) for k in METHODS + ['routed', 'oracle']}
    return {'per_bin': per_bin, 'pooled': pooled, 'wavg': w}


def across_seeds(per_seed):
    out = {'pooled': {}, 'wavg': {}}
    for m in METHODS:
        v = [r['pooled'][m] for r in per_seed]
        out['pooled'][m] = {'mean': float(np.mean(v)),
                            'std': float(np.std(v))}
    for k in METHODS + ['routed', 'oracle']:
        v = [r['wavg'][k] for r in per_seed]
        out['wavg'][k] = {'mean': float(np.mean(v)), 'std': float(np.std(v))}
    return out


def print_master(tag, per_seed):
    for seed, r in zip(SEEDS, per_seed):
        print(f"\n== {tag} seed {seed} ==")
        hdr = ''.join(f'{m:>12}' for m in METHODS)
        print(f"  {'SNR':>5} |{hdr} | {'routed':>7} | {'oracle':>7}")
        for s in BINS:
            v = r['per_bin'][skey(s)]
            row = ''.join(f'{v[m]:>12.3f}' for m in METHODS)
            print(f"  {s:>5.0f} |{row} | {v['routed']:>7.3f} | "
                  f"{v['oracle']:>7.3f}")
        print('  pooled : ' + '  '.join(f"{m}={r['pooled'][m]:.3f}"
                                        for m in METHODS))
        print('  wavg   : ' + '  '.join(f"{m}={r['wavg'][m]:.3f}"
                                        for m in METHODS)
              + f"   routed={r['wavg']['routed']:.3f}  "
                f"oracle={r['wavg']['oracle']:.3f}")


def print_across(across, keys_pooled=None):
    print('  pooled:')
    for m in METHODS:
        v = across['pooled'][m]
        print(f"    {m:<12}: {v['mean']:.4f} ± {v['std']:.4f}")
    print('  weighted-avg per-SNR:')
    for k in ['routed', 'oracle'] + METHODS:
        v = across['wavg'][k]
        print(f"    {k:<12}: {v['mean']:.4f} ± {v['std']:.4f}")


# ----------------------------------------------------------------------------
# Bootstrap helpers
# ----------------------------------------------------------------------------
def boot_pooled(sk, su, B=B_BOOT, seed=0, chunk=500):
    """Percentile bootstrap of the pooled AUROC: known and unknown index
    sets resampled independently with replacement (sizes preserved)."""
    rng = np.random.RandomState(seed)
    sk = np.asarray(sk, dtype=np.float64)
    su = np.asarray(su, dtype=np.float64)
    nk, nu = len(sk), len(su)
    out = np.empty(B)
    for lo in range(0, B, chunk):
        hi = min(lo + chunk, B)
        nb = hi - lo
        ik = rng.randint(0, nk, (nb, nk))
        iu = rng.randint(0, nu, (nb, nu))
        comb = np.concatenate([sk[ik], su[iu]], axis=1)
        out[lo:hi] = auroc_rows(comb, nk)
    return out


def boot_pooled_pair(sk1, su1, sk2, su2, B=B_BOOT, seed=0, chunk=500):
    """Paired bootstrap of pooled-AUROC difference (score1 - score2) on
    identically resampled indices."""
    rng = np.random.RandomState(seed)
    nk, nu = len(sk1), len(su1)
    out = np.empty(B)
    for lo in range(0, B, chunk):
        hi = min(lo + chunk, B)
        nb = hi - lo
        ik = rng.randint(0, nk, (nb, nk))
        iu = rng.randint(0, nu, (nb, nu))
        a1 = auroc_rows(np.concatenate([sk1[ik], su1[iu]], axis=1), nk)
        a2 = auroc_rows(np.concatenate([sk2[ik], su2[iu]], axis=1), nk)
        out[lo:hi] = a1 - a2
    return out


def boot_paired_wavg(scores, k_snr, u_snr, B=B_BOOT, seed=0):
    """Paired per-bin bootstrap of wavg(oracle) - wavg(routed).

    Per draw: within each SNR bin resample known and unknown indices
    independently with replacement; compute all six per-bin AUROCs;
    oracle = per-bin max over scorers, routed = a-priori rule; difference
    of the pair-weighted averages.
    """
    rng = np.random.RandomState(seed)
    bins_data = []
    for s in BINS:
        mk, mu = k_snr == s, u_snr == s
        SK = np.stack([scores[m][0][mk] for m in METHODS])   # (6, nk)
        SU = np.stack([scores[m][1][mu] for m in METHODS])   # (6, nu)
        bins_data.append((s, SK, SU))
    diffs = np.empty(B)
    oracle_w = np.empty(B)
    routed_w = np.empty(B)
    for b in range(B):
        num_o = num_r = den = 0.0
        for s, SK, SU in bins_data:
            nk, nu = SK.shape[1], SU.shape[1]
            ik = rng.randint(0, nk, nk)
            iu = rng.randint(0, nu, nu)
            comb = np.concatenate([SK[:, ik], SU[:, iu]], axis=1)  # (6, n)
            aucs = auroc_rows(comb, nk)                            # (6,)
            w = nk * nu
            num_o += w * aucs.max()
            num_r += w * aucs[METHODS.index(route_method(s))]
            den += w
        oracle_w[b] = num_o / den
        routed_w[b] = num_r / den
        diffs[b] = oracle_w[b] - routed_w[b]
    return diffs, oracle_w, routed_w


def ci(draws):
    return [float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5))]


# ----------------------------------------------------------------------------
# JSON rounding
# ----------------------------------------------------------------------------
def r4(x):
    if isinstance(x, dict):
        return {k: r4(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r4(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return round(float(x), 4)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


# ----------------------------------------------------------------------------
def main():
    out = {'method': {
        'auroc': 'Mann-Whitney rank AUROC with average ranks for ties '
                 '(scipy.stats.rankdata); verified identical to '
                 'open_set_metrics.auroc pairwise formulation.',
        's1_fitting': 'CORRECTED dumps; prototype/vos/mahalanobis fit on the '
                      'disjoint reference pool (refpool npz, seed 88888, kk '
                      'protocol); energy/msp from test-pool logits; odin '
                      'precomputed (eps=0.005, T=1000). Labels = corrected '
                      'known_snr/unknown_snr (tile-stacked).',
        's2_fitting': 'ARCHIVED buggy dumps; prototype/vos/mahalanobis fit on '
                      'the archived known TEST pool (the original artefact '
                      'pipeline); labels = stored buggy known_snr '
                      '(np.repeat(snr_kk,2) interleaved) + stored unknown_snr '
                      '(identical to corrected).',
        'wavg': 'per-SNR-bin AUROC weighted by n_known*n_unknown per bin '
                '(uniform 384*384 here).',
        'routed': 'a-priori rule: energy if snr<=0 dB else prototype.',
        'oracle': 'post-hoc per-bin max over the six scorers.',
        'bootstrap': f'pooled-AUROC percentile bootstrap, B={B_BOOT}, known '
                     'and unknown index sets resampled independently with '
                     'replacement (sizes preserved), rng seed 0 per '
                     '(seed, scorer); 2.5/97.5 percentiles. (a) seed-averaged '
                     'CI = mean of the five per-seed bounds; (b) concatenated '
                     'CI = bootstrap on all-seeds-concatenated score arrays '
                     '(13440 known + 13440 unknown).',
        'paired_bootstrap': 'oracle-routed difference: (per_bin_wavg) paired '
                            'per-bin resampling, per draw all six per-bin '
                            'AUROCs recomputed, oracle=per-bin max, '
                            'routed=rule, pair-weighted wavg difference; '
                            '(pooled_composite) pooled-AUROC difference of '
                            'per-sample composite scores (oracle composite = '
                            'post-hoc bin-winning scorer per sample, routed '
                            'composite = rule scorer per sample) on '
                            'identically resampled indices. '
                            f'B={B_BOOT}, rng seed 0 per seed; the '
                            'five-seed-mean CI averages per-draw differences '
                            'across seeds before percentiling.',
        'permutation_ablation': 'corrected STORED energy/prototype scores; '
                                'permutation applied to the KNOWN-pool SNR '
                                'label array only (unknown labels exact); '
                                'routed derived per bin (energy<=0, '
                                'prototype>0). pi_repeat reconstructs the '
                                'original bug np.repeat(snr_kk,2). pi_random: '
                                '10 uniform permutations, rng seed 0.',
    }}

    # ------------------------------------------------------------------
    # Consistency checks + audit
    # ------------------------------------------------------------------
    print('== Consistency checks ==')
    s0, k0, u0, x0 = load_run(42, 'refpool')
    d0 = np.load(os.path.join(RESULTS, BASE.format(42) + '_ood_scores.npz'))
    assert np.allclose(s0['energy'][0], d0['energy_score_known']), \
        'energy recomputation mismatch'
    print('  energy(logits) == stored energy_score_known: OK')
    # scorer wiring: fitting on the TEST pool must reproduce the stored
    # prototype/vos scores of the corrected dump
    p_test = compute_prototypes(d0['known_emb'], d0['known_mods'], 4)
    assert np.allclose(prototype_score(d0['known_emb'], p_test),
                       d0['prototype_score_known']), 'prototype wiring mismatch'
    assert np.allclose(vos_score(d0['known_emb'], p_test, alpha=2.0,
                                 n_per_class=100, seed=0),
                       d0['vos_score_known']), 'vos wiring mismatch'
    print('  prototype/vos wiring reproduces stored scores (test-pool fit): OK')

    a_pair = auroc_pairwise(s0['energy'][0][k0 == -10], s0['energy'][1][u0 == -10])
    a_rank = auroc_rank(s0['energy'][0][k0 == -10], s0['energy'][1][u0 == -10])
    assert abs(a_pair - a_rank) < 1e-12, (a_pair, a_rank)
    print(f'  rank-AUROC == pairwise AUROC ({a_rank:.6f}): OK')

    print('  per-bin counts (corrected, seed 42):')
    for s in BINS:
        print(f'    {s:>5.0f} dB: known={(k0 == s).sum()} '
              f'unknown={(u0 == s).sum()}')

    # ==================================================================
    # Section 1 — corrected, refpool-fitted master table
    # ==================================================================
    print('\n' + '=' * 78)
    print('SECTION 1 — six-scorer master table, CORRECTED, refpool-fitted')
    print('=' * 78)
    s1_runs = []
    s1_results = []
    for seed in SEEDS:
        scores, k_snr, u_snr, extras = load_run(seed, 'refpool')
        s1_runs.append((scores, k_snr, u_snr, extras))
        s1_results.append(eval_master(scores, k_snr, u_snr))
    print_master('corrected/refpool', s1_results)
    s1_across = across_seeds(s1_results)
    print(f"\n== Across {len(SEEDS)} seeds (mean ± std), corrected/refpool ==")
    print_across(s1_across)

    routed_mean = s1_across['wavg']['routed']['mean']
    if not (0.498 - 0.02 <= routed_mean <= 0.498 + 0.02):
        print(f'\n!! SANITY CHECK FAILED: corrected routed wavg mean = '
              f'{routed_mean:.4f}, expected ~0.498 ± 0.02')
        print('   per-bin audit (per seed, n_known per bin):')
        for seed, (_, k_snr, u_snr, _) in zip(SEEDS, s1_runs):
            print(f'   seed {seed}: known='
                  f'{[int((k_snr == s).sum()) for s in BINS]} unknown='
                  f'{[int((u_snr == s).sum()) for s in BINS]}')
        sys.exit('STOP: sanity check failed — investigate label/order '
                 'assumptions before trusting any output.')
    print(f'\n  SANITY OK: corrected routed wavg mean = {routed_mean:.4f} '
          f'(expected ~0.498 ± 0.02)')

    out['s1_corrected'] = {
        'per_seed': {str(seed): r for seed, r in zip(SEEDS, s1_results)},
        'across_seeds': s1_across,
        'sanity_routed_wavg_mean': routed_mean,
    }

    # ==================================================================
    # Section 2 — archived buggy dumps
    # ==================================================================
    print('\n' + '=' * 78)
    print('SECTION 2 — six-scorer master table, ARCHIVED BUGGY dumps')
    print('=' * 78)
    s2_results = []
    for seed in SEEDS:
        scores, k_snr, u_snr, _ = load_run(seed, 'testpool')
        s2_results.append(eval_master(scores, k_snr, u_snr))
    print_master('archived-buggy', s2_results)
    s2_across = across_seeds(s2_results)
    print(f"\n== Across {len(SEEDS)} seeds (mean ± std), archived buggy ==")
    print_across(s2_across)

    buggy_routed_mean = s2_across['wavg']['routed']['mean']
    if not (0.625 - 0.031 <= buggy_routed_mean <= 0.625 + 0.031):
        print(f'\n!! SANITY CHECK FAILED: buggy routed wavg mean = '
              f'{buggy_routed_mean:.4f}, expected ~0.625 ± 0.031')
        sys.exit('STOP: could not reproduce the archived artefact.')
    print(f'\n  SANITY OK: buggy routed wavg mean = {buggy_routed_mean:.4f} '
          f'(reproduces the 0.625 ± 0.031 artefact)')

    out['s2_buggy'] = {
        'per_seed': {str(seed): r for seed, r in zip(SEEDS, s2_results)},
        'across_seeds': s2_across,
        'sanity_routed_wavg_mean': buggy_routed_mean,
    }

    # ==================================================================
    # Section 3 — bootstrap CIs (corrected, refpool-fitted)
    # ==================================================================
    print('\n' + '=' * 78)
    print(f'SECTION 3 — bootstrap 95% CIs (B={B_BOOT}, corrected/refpool)')
    print('=' * 78)
    boot = {'pooled_ci_per_seed': {m: {} for m in METHODS},
            'pooled_ci_seed_averaged': {},
            'pooled_ci_concatenated': {},
            'auroc_concatenated': {}}
    concat = {m: ([], []) for m in METHODS}
    for seed, (scores, k_snr, u_snr, _) in zip(SEEDS, s1_runs):
        for m in METHODS:
            draws = boot_pooled(scores[m][0], scores[m][1])
            boot['pooled_ci_per_seed'][m][str(seed)] = ci(draws)
            concat[m][0].append(scores[m][0])
            concat[m][1].append(scores[m][1])
    for m in METHODS:
        los = [boot['pooled_ci_per_seed'][m][str(s)][0] for s in SEEDS]
        his = [boot['pooled_ci_per_seed'][m][str(s)][1] for s in SEEDS]
        boot['pooled_ci_seed_averaged'][m] = [float(np.mean(los)),
                                              float(np.mean(his))]
        sk_all = np.concatenate(concat[m][0])
        su_all = np.concatenate(concat[m][1])
        draws = boot_pooled(sk_all, su_all, chunk=250)
        boot['pooled_ci_concatenated'][m] = ci(draws)
        boot['auroc_concatenated'][m] = auroc_rank(sk_all, su_all)

    print('\n  pooled AUROC 95% CI per scorer:')
    print(f"  {'scorer':<12} {'seed-avg CI':>24} {'concatenated CI':>24} "
          f"{'concat AUROC':>13}")
    for m in METHODS:
        a = boot['pooled_ci_seed_averaged'][m]
        c = boot['pooled_ci_concatenated'][m]
        print(f"  {m:<12} [{a[0]:.4f}, {a[1]:.4f}]     "
              f"[{c[0]:.4f}, {c[1]:.4f}]     {boot['auroc_concatenated'][m]:.4f}")

    # Paired bootstrap: oracle - routed (corrected/refpool)
    print('\n  paired bootstrap, oracle - routed (corrected/refpool):')
    paired = {'per_bin_wavg': {}, 'pooled_composite': {}}
    wavg_draws, comp_draws = [], []
    for seed, (scores, k_snr, u_snr, _) in zip(SEEDS, s1_runs):
        diffs, _, _ = boot_paired_wavg(scores, k_snr, u_snr)
        paired['per_bin_wavg'][str(seed)] = {
            'mean_diff': float(diffs.mean()), 'ci95': ci(diffs)}
        wavg_draws.append(diffs)

        # composite scores: oracle = post-hoc bin winner, routed = rule
        per_bin = eval_master(scores, k_snr, u_snr)['per_bin']
        win = {s: max(METHODS, key=lambda m: per_bin[skey(s)][m])
               for s in BINS}
        routed_sk = np.empty_like(k_snr)
        routed_su = np.empty_like(u_snr)
        oracle_sk = np.empty_like(k_snr)
        oracle_su = np.empty_like(u_snr)
        for s in BINS:
            mk, mu = k_snr == s, u_snr == s
            rm, wm = route_method(s), win[s]
            routed_sk[mk] = scores[rm][0][mk]
            routed_su[mu] = scores[rm][1][mu]
            oracle_sk[mk] = scores[wm][0][mk]
            oracle_su[mu] = scores[wm][1][mu]
        d2 = boot_pooled_pair(oracle_sk, oracle_su, routed_sk, routed_su)
        paired['pooled_composite'][str(seed)] = {
            'mean_diff': float(d2.mean()), 'ci95': ci(d2)}
        comp_draws.append(d2)
        print(f"    seed {seed}: per-bin wavg diff "
              f"{diffs.mean():.4f} [{ci(diffs)[0]:.4f}, {ci(diffs)[1]:.4f}]  | "
              f"pooled-composite diff {d2.mean():.4f} "
              f"[{ci(d2)[0]:.4f}, {ci(d2)[1]:.4f}]")
    wavg_mean_draws = np.mean(wavg_draws, axis=0)
    comp_mean_draws = np.mean(comp_draws, axis=0)
    paired['per_bin_wavg']['five_seed_mean'] = {
        'mean_diff': float(wavg_mean_draws.mean()),
        'ci95': ci(wavg_mean_draws)}
    paired['pooled_composite']['five_seed_mean'] = {
        'mean_diff': float(comp_mean_draws.mean()),
        'ci95': ci(comp_mean_draws)}
    print(f"    5-seed mean: per-bin wavg diff "
          f"{wavg_mean_draws.mean():.4f} "
          f"[{ci(wavg_mean_draws)[0]:.4f}, {ci(wavg_mean_draws)[1]:.4f}]  | "
          f"pooled-composite diff {comp_mean_draws.mean():.4f} "
          f"[{ci(comp_mean_draws)[0]:.4f}, {ci(comp_mean_draws)[1]:.4f}]")
    boot['paired_oracle_minus_routed'] = paired
    out['s3_bootstrap'] = boot

    # ==================================================================
    # Section 4 — per-unknown-class AUROC (corrected, refpool-fitted)
    # ==================================================================
    print('\n' + '=' * 78)
    print('SECTION 4 — per-unknown-class wavg AUROC (corrected/refpool)')
    print('=' * 78)
    s4 = {m: {} for m in METHODS}
    for m in METHODS:
        for c, cname in CLASS_NAMES.items():
            wavgs = []
            for seed, (scores, k_snr, u_snr, extras) in zip(SEEDS, s1_runs):
                mods = extras['unknown_mods']
                num = den = 0.0
                for s in BINS:
                    mk = k_snr == s
                    mu = (u_snr == s) & (mods == c)
                    w = int(mk.sum()) * int(mu.sum())
                    num += w * auroc_rank(scores[m][0][mk], scores[m][1][mu])
                    den += w
                wavgs.append(num / den)
            s4[m][cname] = {'wavg_per_seed': wavgs,
                            'mean': float(np.mean(wavgs)),
                            'std': float(np.std(wavgs))}
    hdr = ''.join(f'{m:>13}' for m in METHODS)
    print(f"  {'class':<12}{hdr}")
    for c, cname in CLASS_NAMES.items():
        row = ''.join(f"{s4[m][cname]['mean']:>7.4f}±{s4[m][cname]['std']:.3f}"
                      for m in METHODS)
        print(f"  {cname:<12}{row}")
    near = {m: float(np.mean([s4[m]['64QAM']['mean'],
                              s4[m]['pi/4-DQPSK']['mean']])) for m in METHODS}
    far = {m: float(np.mean([s4[m]['MSK']['mean'],
                             s4[m]['OFDM-QPSK']['mean']])) for m in METHODS}
    print(f"  {'near(4,5)':<12}" + ''.join(f'{near[m]:>13.4f}'
                                           for m in METHODS))
    print(f"  {'far(6,7)':<12}" + ''.join(f'{far[m]:>13.4f}'
                                          for m in METHODS))
    s4['near_far'] = {'near_64QAM_DQPSK': near, 'far_MSK_OFDM': far}
    out['s4_per_class'] = s4

    # ==================================================================
    # Section 5 — permutation ablation (corrected dumps, stored scores)
    # ==================================================================
    print('\n' + '=' * 78)
    print('SECTION 5 — permutation ablation of KNOWN-pool SNR labels '
          '(corrected dumps)')
    print('=' * 78)
    N_RANDOM = 10
    rng_perm = np.random.RandomState(0)
    s5 = {}
    s5_per_seed = {}
    for seed, (_, k_snr, u_snr, extras) in zip(SEEDS, s1_runs):
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
            'pi_shift1': np.roll(labels0, 1),
            'pi_shiftHalf': np.roll(labels0, len(labels0) // 2),
            'pi_reverse': labels0[::-1],
        }
        perms['pi_random'] = [rng_perm.permutation(labels0)
                              for _ in range(N_RANDOM)]

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
            pbr = []
            for s in BINS:
                mk, mu = lab == s, u_snr == s
                m = route_method(s)
                w = int(mk.sum()) * int(mu.sum())
                a = auroc_rank(scores_st[m][0][mk], scores_st[m][1][mu])
                num += w * a
                den += w
                pbr.append(a)
            res['routed'] = num / den
            res['per_bin_routed'] = pbr
            return res

        seed_res = {}
        for name in ['pi_identity', 'pi_repeat', 'pi_shift1', 'pi_shiftHalf',
                     'pi_reverse']:
            seed_res[name] = wavg_with(perms[name])
        rnd = [wavg_with(p) for p in perms['pi_random']]
        seed_res['pi_random'] = {
            m: {'mean': float(np.mean([r[m] for r in rnd])),
                'std': float(np.std([r[m] for r in rnd]))}
            for m in ['energy', 'prototype', 'routed']}
        seed_res['pi_random']['per_bin_routed'] = np.mean(
            [r['per_bin_routed'] for r in rnd], axis=0).tolist()
        s5_per_seed[str(seed)] = seed_res

        # cross-check pi_repeat against the Section-2 routed wavg
        rep = seed_res['pi_repeat']['routed']
        s2w = out['s2_buggy']['per_seed'][str(seed)]['wavg']['routed']
        assert abs(rep - s2w) < 1e-9, (seed, rep, s2w)
    print('  cross-check: pi_repeat routed wavg == Section-2 buggy routed '
          'wavg for every seed: OK')

    print(f"\n  {'permutation':<14}{'energy':>10}{'prototype':>10}"
          f"{'routed':>10}   (5-seed mean of wavg AUROC)")
    s5_summary = {}
    for name in ['pi_identity', 'pi_repeat', 'pi_shift1', 'pi_shiftHalf',
                 'pi_reverse']:
        row = {}
        for m in ['energy', 'prototype', 'routed']:
            v = [s5_per_seed[str(s)][name][m] for s in SEEDS]
            row[m] = {'mean': float(np.mean(v)), 'std': float(np.std(v))}
        row['per_bin_routed'] = np.mean(
            [s5_per_seed[str(s)][name]['per_bin_routed'] for s in SEEDS],
            axis=0).tolist()
        s5_summary[name] = row
        print(f"  {name:<14}{row['energy']['mean']:>10.4f}"
              f"{row['prototype']['mean']:>10.4f}{row['routed']['mean']:>10.4f}")
    row = {}
    for m in ['energy', 'prototype', 'routed']:
        v = [s5_per_seed[str(s)]['pi_random'][m]['mean'] for s in SEEDS]
        row[m] = {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                  'within_seed_std': float(np.mean(
                      [s5_per_seed[str(s)]['pi_random'][m]['std']
                       for s in SEEDS]))}
    row['per_bin_routed'] = np.mean(
        [s5_per_seed[str(s)]['pi_random']['per_bin_routed'] for s in SEEDS],
        axis=0).tolist()
    s5_summary['pi_random'] = row
    print(f"  {'pi_random':<14}{row['energy']['mean']:>10.4f}"
          f"{row['prototype']['mean']:>10.4f}{row['routed']['mean']:>10.4f}"
          f"   (mean over 10 perms; within-seed std "
          f"{row['routed']['within_seed_std']:.4f})")
    print('\n  per-bin routed AUROC (5-seed mean; bins '
          + ' '.join(f'{int(s)}' for s in BINS) + '):')
    for name in ['pi_identity', 'pi_repeat', 'pi_shift1', 'pi_shiftHalf',
                 'pi_reverse', 'pi_random']:
        pbr = ' '.join(f'{v:.3f}' for v in s5_summary[name]['per_bin_routed'])
        print(f"    {name:<14}{pbr}")
    s5_summary['note'] = (
        'pi_shiftHalf is an exact no-op (the tile-stacked label array has '
        'period N/2); pi_shift1 moves only the block-boundary entries. '
        'Single-scorer wavg under pi_random recovers the POOLED "trap view" '
        '(bin comparison = random known subset vs unknown@bin, which averages '
        'to the pooled AUROC). The SNR-ROUTED score is inflated by ANY '
        'permutation that decorrelates known labels from true SNR — random '
        'ones included (pi_random 0.6405, pi_reverse 0.7653, pi_repeat '
        '0.6253) — because mismatched per-bin comparisons expose the '
        'SNR-slope confound (energy AUROC(known-pool vs unknown@bin) '
        'decreases with bin, prototype increases) and the fixed a-priori '
        'routing rule (energy<=0, prototype>0) harvests the rising edge of '
        'each gradient. Structured permutations aligned with the block '
        'layout (reverse/repeat) manufacture the LARGEST routed inflation; '
        'the per-bin routed profiles above show the gradient directly. '
        'Verified with an independent permutation seed and the pairwise '
        'open_set_metrics.auroc implementation.')
    s5_summary['per_seed'] = s5_per_seed
    s5_summary['n_random'] = N_RANDOM
    out['s5_permutation'] = s5_summary

    # ==================================================================
    # Section 6 — operating-point summary (reformat existing file)
    # ==================================================================
    print('\n' + '=' * 78)
    print('SECTION 6 — operating point (from routed_threshold_metrics.json)')
    print('=' * 78)
    rt = json.load(open(os.path.join(RESULTS, 'routed_threshold_metrics.json')))
    op = {}
    for var in ['A', 'B']:
        p = rt['across_seeds'][var]['routed']['pooled']
        op[var] = {'pooled': {k: p[k] for k in
                              ['det', 'frr', 'fpr95', 'oscr', 'joint_acc']},
                   'tau': rt['across_seeds'][var]['routed']['tau']}
        print(f"  variant {var}: tau={op[var]['tau']['mean']:.4f}±"
              f"{op[var]['tau']['std']:.4f}  "
              f"Det@tau={p['det']['mean']:.4f}±{p['det']['std']:.4f}  "
              f"FRR={p['frr']['mean']:.4f}±{p['frr']['std']:.4f}  "
              f"FPR95={p['fpr95']['mean']:.4f}±{p['fpr95']['std']:.4f}  "
              f"OSCR={p['oscr']['mean']:.4f}±{p['oscr']['std']:.4f}  "
              f"joint_acc={p['joint_acc']['mean']:.4f}±"
              f"{p['joint_acc']['std']:.4f}")
    per_snr_A = {}
    for s in BINS:
        e = rt['across_seeds']['A']['routed']['per_snr'][skey(s)]
        per_snr_A[skey(s)] = {'det': e['det'], 'frr': e['frr']}
    op['A']['per_snr'] = per_snr_A
    print('  variant A per-SNR:')
    print(f"    {'SNR':>5} {'Det@tau':>18} {'FRR':>18}")
    for s in BINS:
        e = per_snr_A[skey(s)]
        print(f"    {s:>5.0f} {e['det']['mean']:>9.4f}±{e['det']['std']:<8.4f}"
              f" {e['frr']['mean']:>9.4f}±{e['frr']['std']:<8.4f}")
    op['source'] = 'results/routed_threshold_metrics.json (variants A/B, ' \
                   'scorer routed; reformatted, not recomputed)'
    out['s6_operating_point'] = op

    # ==================================================================
    # FOR MANUSCRIPT block
    # ==================================================================
    print('\n' + '=' * 78)
    print('FOR MANUSCRIPT')
    print('=' * 78)
    print('\n(i) Corrected six-scorer table (refpool-fitted), 5-seed mean ± std:')
    print(f"  {'scorer':<12}{'wavg mean±std':>22}{'boot95 CI (concat)':>22}"
          f"{'pooled mean':>13}")
    for m in METHODS:
        w = s1_across['wavg'][m]
        c = boot['pooled_ci_concatenated'][m]
        print(f"  {m:<12}{w['mean']:>10.4f}±{w['std']:<10.4f}"
              f"[{c[0]:.4f}, {c[1]:.4f}]     {s1_across['pooled'][m]['mean']:>12.4f}")
    print('\n(ii) Buggy six-scorer wavg, 5-seed mean ± std:')
    for m in METHODS:
        w = s2_across['wavg'][m]
        print(f"  {m:<12}{w['mean']:.4f} ± {w['std']:.4f}")
    print('\n(iii) Routed / oracle wavg:')
    print(f"  corrected: routed {s1_across['wavg']['routed']['mean']:.4f} ± "
          f"{s1_across['wavg']['routed']['std']:.4f}   oracle "
          f"{s1_across['wavg']['oracle']['mean']:.4f} ± "
          f"{s1_across['wavg']['oracle']['std']:.4f}")
    print(f"  buggy    : routed {s2_across['wavg']['routed']['mean']:.4f} ± "
          f"{s2_across['wavg']['routed']['std']:.4f}   oracle "
          f"{s2_across['wavg']['oracle']['mean']:.4f} ± "
          f"{s2_across['wavg']['oracle']['std']:.4f}")
    p5 = paired['per_bin_wavg']['five_seed_mean']
    c5 = paired['pooled_composite']['five_seed_mean']
    print(f"  paired bootstrap oracle-routed (corrected, 5-seed mean): "
          f"per-bin wavg diff {p5['mean_diff']:.4f} "
          f"[{p5['ci95'][0]:.4f}, {p5['ci95'][1]:.4f}] ; "
          f"pooled-composite diff {c5['mean_diff']:.4f} "
          f"[{c5['ci95'][0]:.4f}, {c5['ci95'][1]:.4f}]")
    print('\n(iv) Per-unknown-class wavg AUROC (corrected/refpool), '
          '5-seed mean ± std:')
    hdr = ''.join(f'{m:>13}' for m in METHODS)
    print(f"  {'class':<12}{hdr}")
    for c, cname in CLASS_NAMES.items():
        row = ''.join(f"{s4[m][cname]['mean']:>7.4f}±"
                      f"{s4[m][cname]['std']:.3f}" for m in METHODS)
        print(f"  {cname:<12}{row}")
    print(f"  {'near(4,5)':<12}" + ''.join(f'{near[m]:>13.4f}'
                                           for m in METHODS))
    print(f"  {'far(6,7)':<12}" + ''.join(f'{far[m]:>13.4f}'
                                          for m in METHODS))
    print('\n(v) Permutation ablation (5-seed mean wavg AUROC):')
    print(f"  {'permutation':<14}{'energy':>10}{'prototype':>10}"
          f"{'routed':>10}")
    for name in ['pi_identity', 'pi_repeat', 'pi_shift1', 'pi_shiftHalf',
                 'pi_reverse', 'pi_random']:
        row = s5_summary[name]
        print(f"  {name:<14}{row['energy']['mean']:>10.4f}"
              f"{row['prototype']['mean']:>10.4f}{row['routed']['mean']:>10.4f}")
    print('  per-bin routed AUROC (bins '
          + ' '.join(f'{int(s)}' for s in BINS) + '):')
    for name in ['pi_identity', 'pi_repeat', 'pi_reverse', 'pi_random']:
        pbr = ' '.join(f'{v:.3f}' for v in s5_summary[name]['per_bin_routed'])
        print(f"    {name:<14}{pbr}")
    print('  NOTE: random perms do NOT save the routed score — any '
          'label/SNR decorrelation exposes the SNR-slope confound and the '
          'a-priori routing rule harvests it (see s5 note in JSON).')
    print('\n(vi) Operating point (routed_threshold_metrics.json):')
    for var in ['A', 'B']:
        p = op[var]['pooled']
        print(f"  variant {var}: Det@tau={p['det']['mean']:.4f}±"
              f"{p['det']['std']:.4f}  FRR={p['frr']['mean']:.4f}±"
              f"{p['frr']['std']:.4f}  FPR95={p['fpr95']['mean']:.4f}±"
              f"{p['fpr95']['std']:.4f}  OSCR={p['oscr']['mean']:.4f}±"
              f"{p['oscr']['std']:.4f}  joint_acc="
              f"{p['joint_acc']['mean']:.4f}±{p['joint_acc']['std']:.4f}")
    print('  variant A per-SNR Det@tau / FRR (mean±std):')
    for s in BINS:
        e = op['A']['per_snr'][skey(s)]
        print(f"    {int(s):>4} dB: {e['det']['mean']:.4f}±"
              f"{e['det']['std']:.4f} / {e['frr']['mean']:.4f}±"
              f"{e['frr']['std']:.4f}")

    # ==================================================================
    with open(OUT_JSON, 'w') as f:
        json.dump(r4(out), f, indent=1)
    print(f'\nWrote {OUT_JSON}')


if __name__ == '__main__':
    main()
