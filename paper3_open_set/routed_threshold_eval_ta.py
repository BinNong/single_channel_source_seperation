"""
Paper 3 — Truth-anchored operating point (routed_threshold_metrics_ta).

Identical protocol to routed_threshold_eval.py (variant A split-half,
variant B refpool-refit headline, variant C rank-normalized; same routing
rule energy<=0/prototype>0 dB, tau = 95th percentile of knowns, 20 splits,
split seed 20260920) but consuming the TRUTH-ANCHORED dumps produced by
dump_ta_scores.py:

  results/{base}_ood_scores_ta.npz   (pool membership truth-anchored;
                                      stored prototype/vos fit on the TA
                                      known pool with TA labels)
  results/{base}_refpool_ta.npz      (variant B refit pool, TA labels)

All metric/calibration logic is reused from routed_threshold_eval (imported,
not copied); only the file paths change.  Runs on CPU (dumps only, no model).

Sanity: per-seed routed wavg AUROC (tau=inf, stored TA scores) is reported;
self-consistency reference = the same quantity recomputed directly from the
TA dumps (identical by construction).  For context the original
(swapped-label) routed_threshold_metrics.json sanity mean was 0.4980.

Usage:  ../.venv_verify/bin/python routed_threshold_eval_ta.py
Outputs: results/routed_threshold_metrics_ta.{json,txt}
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import routed_threshold_eval as rte

RESULTS_DIR = rte.RESULTS_DIR
SEEDS = rte.SEEDS
BASE_TA = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed%d_best_ood_scores_ta.npz'
REFPOOL_TA = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed%d_best_refpool_ta.npz'


def load_seed_ta(seed: int) -> dict:
    d = np.load(os.path.join(RESULTS_DIR, BASE_TA % seed))
    out = {
        'k_snr': d['known_snr'].astype(np.float64),
        'u_snr': d['unknown_snr'].astype(np.float64),
        'correct_k': d['known_logits'].argmax(axis=1) == d['known_mods'],
        'known_emb': d['known_emb'],
        'unknown_emb': d['unknown_emb'],
        'scores': {m: (d[f'{m}_score_known'].astype(np.float64),
                       d[f'{m}_score_unknown'].astype(np.float64))
                   for m in rte.METHODS},
    }
    e_k, e_u = out['scores']['energy']
    p_k, p_u = out['scores']['prototype']
    out['scores']['routed'] = (
        np.where(out['k_snr'] <= rte.SNR_ROUTE_THRESHOLD, e_k, p_k),
        np.where(out['u_snr'] <= rte.SNR_ROUTE_THRESHOLD, e_u, p_u))
    return out


def main():
    # point the reused variant-B refit at the TA refpool dumps
    rte.REFPOOL_TEMPLATE = REFPOOL_TA

    data = {s: load_seed_ta(s) for s in SEEDS}
    bins = rte.snr_bins(data[SEEDS[0]])
    print(f"Loaded {len(SEEDS)} TA seeds; bins {[int(b) for b in bins]}; "
          f"pool sizes {len(data[SEEDS[0]]['k_snr'])}/"
          f"{len(data[SEEDS[0]]['u_snr'])}")

    sanity_per_seed = []
    for s in SEEDS:
        s_k, s_u = data[s]['scores']['routed']
        per_snr = rte.bin_metrics(s_k, s_u, data[s]['k_snr'], data[s]['u_snr'],
                                  data[s]['correct_k'], tau=np.inf, bins=bins)
        sanity_per_seed.append(rte.pair_weighted(per_snr)['auroc'])
    sanity = {'per_seed': [float(x) for x in sanity_per_seed],
              'mean': float(np.mean(sanity_per_seed)),
              'std': float(np.std(sanity_per_seed))}
    print(f"Sanity — TA routed wavg AUROC (stored TA scores): "
          f"{sanity['mean']:.4f} ± {sanity['std']:.4f} "
          f"(swapped-label equivalent was 0.4980 ± 0.0201; refpool-fit TA "
          f"headline 0.5106 ± 0.0198)")

    per_seed_A, per_seed_B, per_seed_C = {}, {}, {}
    for s in SEEDS:
        per_seed_A[s] = rte.run_split_half(data[s], bins, rank_norm=False)
        per_seed_C[s] = rte.run_split_half(data[s], bins, rank_norm=True)
        per_seed_B[s] = rte.run_refpool(data[s], seed=s, bins=bins)
        print(f"seed {s}: A/B/C done "
              f"(B tau_routed={per_seed_B[s]['routed']['tau_mean']:.3f})")

    agg_A = rte.agg_across_seeds(per_seed_A, rte.ALL_METHODS)
    agg_C = rte.agg_across_seeds(per_seed_C, ['routed'])
    for m in rte.METHODS:
        agg_C[m] = agg_A[m]
    agg_B = rte.agg_across_seeds(per_seed_B, rte.ALL_METHODS)

    aggs = {'A': agg_A, 'B': agg_B, 'C': agg_C}
    split_std_notes = {k: aggs[k]['routed']['mean_split_std'] for k in ('A', 'C')}

    txt_path = os.path.join(RESULTS_DIR, 'routed_threshold_metrics_ta.txt')
    rte.write_txt(txt_path, sanity, aggs, split_std_notes)

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
            'truth_anchored': True,
            'source_dumps': 'ood_scores_ta.npz / refpool_ta.npz '
                            '(dump_ta_scores.py; test seed 99999, '
                            '192/192/96 per bin)',
            'routing': 'score(x) = energy(x) if snr <= 0 dB else prototype(x)',
            'tau': f'{rte.TAU_QUANTILE}th percentile of known-pool scores '
                   '(5% FRR operating point; no unknown data used for tau)',
            'n_splits': rte.N_SPLITS, 'split_seed': rte.SPLIT_SEED,
            'seeds': SEEDS, 'snr_bins': [float(b) for b in bins],
            'pooling': 'pair-weighted across SNR bins (n_known*n_unknown)',
            'variants': {
                'A': 'split-half 50/50 stratified by SNR, stored TA scores',
                'B': 'headline: prototypes refit on TA held-out reference '
                     'pool (seed 88888, TA labels), tau calibrated there; '
                     'energy not calibratable (no logits) — omitted',
                'C': 'split-half, per-scorer known-ECDF rank normalization; '
                     'single scorers decision-identical to A',
            },
        },
        'sanity_routed_auroc': sanity,
        'across_seeds': _jsonable(aggs),
        'per_seed': _jsonable({'A': per_seed_A, 'B': per_seed_B,
                               'C': per_seed_C}),
    }
    json_path = os.path.join(RESULTS_DIR, 'routed_threshold_metrics_ta.json')
    with open(json_path, 'w') as f:
        json.dump(json_doc, f, indent=1)
    print(f"Wrote {txt_path}\nWrote {json_path}")


if __name__ == '__main__':
    main()
