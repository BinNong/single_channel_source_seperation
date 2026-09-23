"""
Paper 3 (revision) — Routed open-set detector under a SINGLE global rejection threshold.

Reviewer-motivated evaluation: the original headline (SNR-routed ensemble scored by
per-SNR-bin AUROC, pair-weighted average = 0.625 +/- 0.031 over 5 seeds) permits
SNR-dependent operating points.  This script instead fixes ONE global rejection
threshold tau, calibrated WITHOUT any unknown-class data at the 5% false-rejection-rate
(FRR) operating point on knowns, and reports operational metrics:

  Det@tau    : unknown detection (true rejection) rate at tau
  FRR@tau    : known false-rejection rate at tau (calibration target: 5%)
  FPR95      : fpr_at_95_tpr from open_set_metrics.py (standard convention; the
               threshold there is anchored on unknowns — unlike tau)
  OSCR       : area under the open-set classification-rate curve (threshold-free);
               uses the top-1 correctness from the stored classifier logits
  JointAcc   : (known accepted AND top-1 correctly classified
                + unknown rejected) / N   at tau

Routing rule (identical to ensemble_analysis.py):
  score(x) = energy(x) if SNR(x) <= 0 dB else prototype(x).

Three calibration variants are reported:
  A (split-half, raw scores): knowns split 50/50 stratified by SNR; tau = 95th
     percentile of the calibration half's scores; metrics on the evaluation half
     + all unknowns; 20 random splits, mean +/- std.  Uses the stored test-pool
     scores (whose prototype/VOS reference statistics were fit on the test kk
     pool — transductive; variant B removes this).
  B (held-out reference pool — HEADLINE): prototypes re-fit on the DISJOINT
     reference pool (seed 88888, refpool_dump.py; test protocols use seed 99999);
     test prototype/VOS scores recomputed against it; tau = 95th percentile of
     routed scores on the reference pool itself; evaluation on the full test
     pools.  The reference dump contains embeddings but NO logits, so low-SNR
     reference samples (which route to energy) cannot be scored there; because
     the observed energy and prototype score ranges are globally disjoint
     (all energy scores < -1.3 < all prototype scores), those samples provably
     sit below any candidate tau and are counted as below-threshold (verified
     per seed by an assertion).  Energy ALONE cannot be calibrated on the
     reference pool (no logits) and is therefore omitted from variant B.
  C (split-half, rank-normalized): raw energy (negative) and prototype
     (positive) scores live on disjoint scales, so a single raw tau on the
     routed score degenerates — it can only bite in the prototype region.
     Variant C maps each scorer through its calibration-known empirical CDF
     (a monotone, unknown-free calibration), routes on the normalized scores,
     and sets tau at their 95th percentile.  For single scorers the decisions
     are identical to variant A (monotone invariance); only the routed row
     changes.  This is the scale-fair routed-vs-single comparison at one
     operating point.

Usage:
    cd paper3_open_set && python routed_threshold_eval.py

Outputs:
    results/routed_threshold_metrics.txt   (human-readable; protocol + tables)
    results/routed_threshold_metrics.json  (machine-readable; full detail)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

from open_set_metrics import auroc, fpr_at_95_tpr, oscr
from ood_scores import compute_prototypes, prototype_score, vos_score

RESULTS_DIR = os.path.join(_HERE, 'results')
SEEDS = [42, 43, 44, 45, 46]
BASE_TEMPLATE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed%d_best_ood_scores.npz'
REFPOOL_TEMPLATE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed%d_best_refpool.npz'

SNR_ROUTE_THRESHOLD = 0.0   # energy at SNR <= 0 dB, prototype at SNR >= 5 dB
TAU_QUANTILE = 95.0         # 5% false-rejection-rate operating point on knowns
N_SPLITS = 20               # random split-half repetitions (variants A and C)
SPLIT_SEED = 20260920       # fixed split RNG seed (same splits for every scorer)
NUM_KNOWN_CLASSES = 4
VOS_ALPHA = 2.0             # mirror evaluate.py / refpool_analysis.py
VOS_N_SYNTHETIC = 100
VOS_SEED = 0

METHODS = ['energy', 'prototype', 'vos']
ALL_METHODS = ['routed'] + METHODS
METRIC_KEYS = ['det', 'frr', 'fpr95', 'oscr', 'joint_acc', 'auroc']
PAPER_ROUTED_AUROC = (0.625, 0.031)   # ensemble headline to sanity-check against


# ----------------------------------------------------------------------------
# Loading / routing
# ----------------------------------------------------------------------------
def load_seed(seed: int) -> dict:
    """Load one seed's score dump and reconstruct the routed score per sample."""
    d = np.load(os.path.join(RESULTS_DIR, BASE_TEMPLATE % seed))
    out = {
        'k_snr': d['known_snr'].astype(np.float64),
        'u_snr': d['unknown_snr'].astype(np.float64),
        'correct_k': d['known_logits'].argmax(axis=1) == d['known_mods'],
        'known_emb': d['known_emb'],
        'unknown_emb': d['unknown_emb'],
        'scores': {m: (d[f'{m}_score_known'].astype(np.float64),
                       d[f'{m}_score_unknown'].astype(np.float64))
                   for m in METHODS},
    }
    e_k, e_u = out['scores']['energy']
    p_k, p_u = out['scores']['prototype']
    out['scores']['routed'] = (
        np.where(out['k_snr'] <= SNR_ROUTE_THRESHOLD, e_k, p_k),
        np.where(out['u_snr'] <= SNR_ROUTE_THRESHOLD, e_u, p_u))
    return out


def snr_bins(data: dict) -> list:
    return sorted(set(data['k_snr'].tolist()) & set(data['u_snr'].tolist()))


# ----------------------------------------------------------------------------
# Metrics at a fixed threshold, per SNR bin + pair-weighted pooled
# ----------------------------------------------------------------------------
def bin_metrics(s_k, s_u, k_snr, u_snr, correct_k, tau, bins) -> dict:
    per_snr = {}
    for b in bins:
        mk, mu = k_snr == b, u_snr == b
        sk, su, ck = s_k[mk], s_u[mu], correct_k[mk]
        if mk.sum() < 5 or mu.sum() < 5:
            continue
        per_snr[b] = {
            'n_pairs': int(mk.sum()) * int(mu.sum()),
            'det': float((su > tau).mean()),
            'frr': float((sk > tau).mean()),
            'fpr95': fpr_at_95_tpr(sk, su),
            'oscr': oscr(ck, sk, su),
            'joint_acc': float((((sk <= tau) & ck).sum() + (su > tau).sum())
                               / (mk.sum() + mu.sum())),
            'auroc': auroc(sk, su),
        }
    return per_snr


def pair_weighted(per_snr: dict) -> dict:
    """Pool per-bin metrics with the paper's convention: weight = n_known*n_unknown."""
    den = sum(v['n_pairs'] for v in per_snr.values())
    return {k: float(sum(v[k] * v['n_pairs'] for v in per_snr.values()) / den)
            for k in METRIC_KEYS}


# ----------------------------------------------------------------------------
# Variant A / C — split-half calibration on the test known pool
# ----------------------------------------------------------------------------
def stratified_half_split(k_snr: np.ndarray, bins: list,
                          rng: np.random.Generator):
    cal_idx, ev_idx = [], []
    for b in bins:
        idx = np.flatnonzero(k_snr == b)
        perm = rng.permutation(idx)
        half = len(idx) // 2
        cal_idx.append(perm[:half])
        ev_idx.append(perm[half:])
    return np.concatenate(cal_idx), np.concatenate(ev_idx)


def run_split_half(data: dict, bins: list, rank_norm: bool) -> dict:
    """20 stratified split-half repetitions.

    rank_norm=False -> variant A (raw scores, all four methods).
    rank_norm=True  -> variant C (routed on per-scorer known-ECDF-normalized
                       scores; single scorers are decision-identical to A and
                       are not recomputed).
    """
    rng = np.random.default_rng(SPLIT_SEED)
    methods = ['routed'] if rank_norm else ALL_METHODS
    per_split = {m: [] for m in methods}

    for _ in range(N_SPLITS):
        cal_idx, ev_idx = stratified_half_split(data['k_snr'], bins, rng)
        if rank_norm:
            ecdf = {}
            for m in METHODS:
                s_cal = np.sort(data['scores'][m][0][cal_idx])
                n = len(s_cal)
                ecdf[m] = (lambda x, s=s_cal, n=n:
                           np.searchsorted(s, x, side='right') / n)
            k_lo = data['k_snr'] <= SNR_ROUTE_THRESHOLD
            u_lo = data['u_snr'] <= SNR_ROUTE_THRESHOLD
            s_k = np.where(k_lo, ecdf['energy'](data['scores']['energy'][0]),
                           ecdf['prototype'](data['scores']['prototype'][0]))
            s_u = np.where(u_lo, ecdf['energy'](data['scores']['energy'][1]),
                           ecdf['prototype'](data['scores']['prototype'][1]))
            scorer = {'routed': (s_k, s_u)}
        else:
            scorer = {m: data['scores'][m] for m in methods}

        for m, (s_k, s_u) in scorer.items():
            tau = float(np.percentile(s_k[cal_idx], TAU_QUANTILE))
            per_snr = bin_metrics(s_k[ev_idx], s_u,
                                  data['k_snr'][ev_idx], data['u_snr'],
                                  data['correct_k'][ev_idx], tau, bins)
            per_split[m].append({'tau': tau, 'per_snr': per_snr,
                                 'pooled': pair_weighted(per_snr)})

    # Per-seed summary: mean/std across splits
    out = {}
    for m, lst in per_split.items():
        all_bins = sorted({b for x in lst for b in x['per_snr']})
        out[m] = {
            'tau_mean': float(np.mean([x['tau'] for x in lst])),
            'tau_std': float(np.std([x['tau'] for x in lst])),
            'pooled_mean': {k: float(np.mean([x['pooled'][k] for x in lst]))
                            for k in METRIC_KEYS},
            'pooled_split_std': {k: float(np.std([x['pooled'][k] for x in lst]))
                                 for k in METRIC_KEYS},
            'per_snr_mean': {b: {k: float(np.mean([x['per_snr'][b][k]
                                                   for x in lst]))
                                 for k in METRIC_KEYS} for b in all_bins},
        }
    return out


# ----------------------------------------------------------------------------
# Variant B — calibration on the held-out reference pool (seed 88888)
# ----------------------------------------------------------------------------
def run_refpool(data: dict, seed: int, bins: list) -> dict:
    ref = np.load(os.path.join(RESULTS_DIR, REFPOOL_TEMPLATE % seed))
    ref_emb, ref_mods = ref['ref_emb'], ref['ref_mods']
    ref_snr = ref['ref_snr'].astype(np.float64)

    # Reference statistics fit ONLY on the held-out reference pool.
    proto_ref = compute_prototypes(ref_emb, ref_mods, NUM_KNOWN_CLASSES)
    scores = {
        'prototype': (prototype_score(data['known_emb'], proto_ref),
                      prototype_score(data['unknown_emb'], proto_ref)),
        'vos': (vos_score(data['known_emb'], proto_ref,
                          VOS_ALPHA, VOS_N_SYNTHETIC, seed=VOS_SEED),
                vos_score(data['unknown_emb'], proto_ref,
                          VOS_ALPHA, VOS_N_SYNTHETIC, seed=VOS_SEED)),
    }
    e_k, e_u = data['scores']['energy']     # logit-based; no reference needed
    p_k, p_u = scores['prototype']
    scores['routed'] = (np.where(data['k_snr'] <= SNR_ROUTE_THRESHOLD, e_k, p_k),
                        np.where(data['u_snr'] <= SNR_ROUTE_THRESHOLD, e_u, p_u))

    # Calibration scores on the reference pool itself.  The dump has no
    # logits, so low-SNR reference samples (routed to energy) cannot be
    # scored; the observed score ranges are globally disjoint (all energy
    # < -1.3 < all prototype scores), so they sit below any candidate tau
    # and are entered as -inf (asserted below).
    hi = ref_snr > SNR_ROUTE_THRESHOLD
    ref_proto = prototype_score(ref_emb, proto_ref)
    calib = {
        'prototype': ref_proto,
        'vos': vos_score(ref_emb, proto_ref, VOS_ALPHA, VOS_N_SYNTHETIC,
                         seed=VOS_SEED),
        'routed': np.concatenate([np.full(int((~hi).sum()), -np.inf),
                                  ref_proto[hi]]),
    }
    e_max = float(max(e_k.max(), e_u.max()))

    out = {'_energy_score_max_observed': e_max}
    for m in ['routed', 'prototype', 'vos']:
        tau = float(np.percentile(calib[m], TAU_QUANTILE))
        # Validates the -inf fill for low-SNR reference samples: tau (a
        # prototype-scale value) exceeds every observed energy score.
        assert tau > e_max, (f'seed {seed}: tau {tau:.3f} not above max '
                             f'energy score {e_max:.3f}; -inf fill invalid')
        s_k, s_u = scores[m]
        per_snr = bin_metrics(s_k, s_u, data['k_snr'], data['u_snr'],
                              data['correct_k'], tau, bins)
        out[m] = {'tau_mean': tau, 'tau_std': 0.0,
                  'pooled_mean': pair_weighted(per_snr),
                  'pooled_split_std': {k: 0.0 for k in METRIC_KEYS},
                  'per_snr_mean': {b: v for b, v in per_snr.items()}}
    return out   # energy omitted: no logits in the reference dump


# ----------------------------------------------------------------------------
# Aggregation across seeds
# ----------------------------------------------------------------------------
def agg_across_seeds(per_seed: dict, methods: list) -> dict:
    out = {}
    for m in methods:
        if any(m not in per_seed[s] for s in per_seed):
            continue
        seeds = sorted(per_seed)
        taus = [per_seed[s][m]['tau_mean'] for s in seeds]
        pooled = {k: {'mean': float(np.mean([per_seed[s][m]['pooled_mean'][k]
                                             for s in seeds])),
                      'std': float(np.std([per_seed[s][m]['pooled_mean'][k]
                                           for s in seeds]))}
                  for k in METRIC_KEYS}
        all_bins = sorted({b for s in seeds
                           for b in per_seed[s][m]['per_snr_mean']})
        per_snr = {b: {k: {'mean': float(np.mean(
                               [per_seed[s][m]['per_snr_mean'][b][k]
                                for s in seeds
                                if b in per_seed[s][m]['per_snr_mean']])),
                           'std': float(np.std(
                               [per_seed[s][m]['per_snr_mean'][b][k]
                                for s in seeds
                                if b in per_seed[s][m]['per_snr_mean']]))}
                       for k in METRIC_KEYS} for b in all_bins}
        split_std = {k: float(np.mean([per_seed[s][m]['pooled_split_std'][k]
                                       for s in seeds])) for k in METRIC_KEYS}
        out[m] = {'tau': {'mean': float(np.mean(taus)),
                          'std': float(np.std(taus))},
                  'pooled': pooled, 'per_snr': per_snr,
                  'mean_split_std': split_std}
    return out


# ----------------------------------------------------------------------------
# Text report
# ----------------------------------------------------------------------------
def _pm(x, prec=3):
    return f"{x['mean']:.{prec}f} ± {x['std']:.{prec}f}"


def _pooled_table(lines, agg, methods, tau_prec=3):
    hdr = (f"  {'method':<10} {'tau':>17} {'Det@tau':>15} {'FRR@tau':>15} "
           f"{'FPR95':>15} {'OSCR':>15} {'JointAcc':>15} {'AUROC':>15}")
    lines.append(hdr)
    lines.append('  ' + '-' * (len(hdr) - 2))
    for m in methods:
        if m not in agg:
            lines.append(f"  {m:<10} {'n/a (no refpool logits)':>17}")
            continue
        a, p = agg[m], agg[m]['pooled']
        lines.append(
            f"  {m:<10} {_pm(a['tau'], tau_prec):>17} {_pm(p['det']):>15} "
            f"{_pm(p['frr']):>15} {_pm(p['fpr95']):>15} {_pm(p['oscr']):>15} "
            f"{_pm(p['joint_acc']):>15} {_pm(p['auroc']):>15}")


def _per_snr_table(lines, agg, methods, metric, title):
    lines.append(f"  {title}")
    hdr = f"  {'SNR':>6}" + ''.join(f"{m:>12}" for m in methods)
    lines.append(hdr)
    lines.append('  ' + '-' * (len(hdr) - 2))
    bins = sorted({b for m in methods if m in agg
                   for b in agg[m]['per_snr']})
    for b in bins:
        row = f"  {b:>6.0f}"
        for m in methods:
            if m in agg and b in agg[m]['per_snr']:
                row += f"{agg[m]['per_snr'][b][metric]['mean']:>12.3f}"
            else:
                row += f"{'—':>12}"
        lines.append(row)


def write_txt(path, sanity, aggs, split_std_notes):
    L = []
    L.append(__doc__.strip())
    L.append('')
    L.append('=' * 78)
    L.append('SANITY CHECK — routed per-SNR AUROC, pair-weighted (paper headline: '
             f"{PAPER_ROUTED_AUROC[0]:.3f} ± {PAPER_ROUTED_AUROC[1]:.3f})")
    L.append('=' * 78)
    L.append(f"  per seed: "
             f"{[round(x, 4) for x in sanity['per_seed']]}")
    L.append(f"  mean ± std: {sanity['mean']:.3f} ± {sanity['std']:.3f}  "
             f"({'MATCHES' if abs(sanity['mean'] - PAPER_ROUTED_AUROC[0]) < 0.01 else 'MISMATCH!'})")
    L.append('')

    for key, title, methods in [
            ('A', 'VARIANT A — split-half calibration on test knowns, raw scores '
                  f'({N_SPLITS} splits; std across 5 seeds)', ALL_METHODS),
            ('B', 'VARIANT B — HEADLINE: calibration on the held-out reference '
                  'pool (seed 88888), raw scores', ALL_METHODS),
            ('C', 'VARIANT C — split-half, rank-normalized (scale-fair '
                  'single-operating-point comparison)', ALL_METHODS)]:
        L.append('=' * 78)
        L.append(title)
        L.append('=' * 78)
        agg = aggs[key]
        show = [m for m in methods if m in agg] + [m for m in methods
                                                   if m not in agg]
        _pooled_table(L, agg, show)
        if key in split_std_notes:
            n = split_std_notes[key]
            L.append(f"  (mean split-level std for routed: Det ±{n['det']:.3f}, "
                     f"FRR ±{n['frr']:.3f}, JointAcc ±{n['joint_acc']:.3f})")
        if key == 'C':
            L.append('  (single-scorer rows repeated from variant A: rank '
                     'normalization leaves their decisions unchanged)')
        L.append('')
        _per_snr_table(L, agg, show, 'det', 'Per-SNR Det@tau (mean across seeds):')
        L.append('')
        _per_snr_table(L, agg, show, 'frr', 'Per-SNR FRR@tau (mean across seeds):')
        L.append('')
        _per_snr_table(L, agg, show, 'fpr95', 'Per-SNR FPR95 (mean across seeds):')
        L.append('')
        _per_snr_table(L, agg, show, 'oscr', 'Per-SNR OSCR (mean across seeds):')
        L.append('')
        _per_snr_table(L, agg, show, 'joint_acc',
                       'Per-SNR JointAcc@tau (mean across seeds):')
        L.append('')

    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    data = {s: load_seed(s) for s in SEEDS}
    bins = snr_bins(data[SEEDS[0]])
    print(f"Loaded {len(SEEDS)} seeds; SNR bins: {[int(b) for b in bins]}; "
          f"known/unknown per seed: "
          f"{len(data[SEEDS[0]]['k_snr'])}/{len(data[SEEDS[0]]['u_snr'])}")

    # Sanity: routed AUROC must reproduce the paper's ensemble headline.
    sanity_per_seed = []
    for s in SEEDS:
        s_k, s_u = data[s]['scores']['routed']
        per_snr = bin_metrics(s_k, s_u, data[s]['k_snr'], data[s]['u_snr'],
                              data[s]['correct_k'], tau=np.inf, bins=bins)
        sanity_per_seed.append(pair_weighted(per_snr)['auroc'])
    sanity = {'per_seed': [float(x) for x in sanity_per_seed],
              'mean': float(np.mean(sanity_per_seed)),
              'std': float(np.std(sanity_per_seed))}
    print(f"Sanity — routed wavg AUROC: {sanity['mean']:.3f} ± "
          f"{sanity['std']:.3f} (paper: {PAPER_ROUTED_AUROC[0]} ± "
          f"{PAPER_ROUTED_AUROC[1]})")

    # Variants
    per_seed_A, per_seed_B, per_seed_C = {}, {}, {}
    for s in SEEDS:
        per_seed_A[s] = run_split_half(data[s], bins, rank_norm=False)
        per_seed_C[s] = run_split_half(data[s], bins, rank_norm=True)
        per_seed_B[s] = run_refpool(data[s], seed=s, bins=bins)
        print(f"seed {s}: A/B/C done "
              f"(B tau_routed={per_seed_B[s]['routed']['tau_mean']:.3f})")

    agg_A = agg_across_seeds(per_seed_A, ALL_METHODS)
    agg_C = agg_across_seeds(per_seed_C, ['routed'])
    # Variant C's single scorers are decision-identical to variant A.
    for m in METHODS:
        agg_C[m] = agg_A[m]
    agg_B = agg_across_seeds(per_seed_B, ALL_METHODS)  # skips missing 'energy'

    aggs = {'A': agg_A, 'B': agg_B, 'C': agg_C}
    split_std_notes = {k: aggs[k]['routed']['mean_split_std']
                       for k in ('A', 'C')}

    txt_path = os.path.join(RESULTS_DIR, 'routed_threshold_metrics.txt')
    write_txt(txt_path, sanity, aggs, split_std_notes)

    def _jsonable(obj):
        if isinstance(obj, dict):
            return {(f"{k:g}" if isinstance(k, float) else str(k)):
                    _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    json_doc = {
        'protocol': {
            'routing': 'score(x) = energy(x) if snr <= 0 dB else prototype(x)',
            'tau': f'{TAU_QUANTILE}th percentile of known-pool scores '
                   '(5% FRR operating point; no unknown data used for tau)',
            'n_splits': N_SPLITS, 'split_seed': SPLIT_SEED,
            'seeds': SEEDS, 'snr_bins': [float(b) for b in bins],
            'pooling': 'pair-weighted across SNR bins (n_known*n_unknown), '
                       'same convention as the paper',
            'variants': {
                'A': 'split-half 50/50 stratified by SNR, raw stored scores',
                'B': 'headline: prototypes refit on held-out reference pool '
                     '(seed 88888), tau calibrated on the reference pool; '
                     'energy not calibratable there (no logits) — omitted',
                'C': 'split-half, per-scorer known-ECDF rank normalization '
                     'before routing; single scorers decision-identical to A',
            },
        },
        'sanity_routed_auroc': sanity,
        'across_seeds': _jsonable(aggs),
        'per_seed': _jsonable({'A': per_seed_A, 'B': per_seed_B,
                               'C': per_seed_C}),
    }
    json_path = os.path.join(RESULTS_DIR, 'routed_threshold_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(json_doc, f, indent=1)
    print(f"Wrote {txt_path}\nWrote {json_path}")


if __name__ == '__main__':
    main()
