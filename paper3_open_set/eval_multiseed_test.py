"""
Paper 3 — Review-2 experiment C: multiple independent test seeds
(reviewer Major Concern 4: a single fixed test realization (seed 99999)
cannot separate training-run variability from test-generation variability).

For each test seed in {99999 (validation), 100001, 100002, 100003} the full
test protocol is generated EXACTLY as evaluate.py does (kk/ku n_per_snr=200,
uu n_per_snr=100, SNR grid {-10,-5,0,5,10,15,20}, CommBSSOpenSetTestDataset
with seed=test_seed), and every model checkpoint (seeds 42-46) is evaluated:
six scorers + routed wavg AUROC, refpool-fitted (seed 88888, unchanged —
the reference library is independent of the test realization), corrected
tile-stacked SNR labels.

Pipeline validation: test seed 99999 is run FIRST and its across-seed wavg
must reproduce the corrected baselines (revision2_tables.json s1:
energy 0.4755, msp 0.4306, odin 0.4798, mahalanobis 0.5203,
prototype 0.5055, vos 0.5056, routed 0.4984) within +-0.002; the script
aborts before the new test seeds if validation fails.

Per group (model_seed x test_seed): per-bin AUROC, pooled AUROC, pair-
weighted wavg for six scorers + routed.  Summaries: per test seed across
model seeds; per model seed across the NEW test seeds; grand summary over
the 15 new-test-seed groups.

Outputs: results/multiseed_test.json (smoke: multiseed_test_smoke.json).

Usage:
  python eval_multiseed_test.py --smoke
  python eval_multiseed_test.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import config as C
from data_generator_extended import CommBSSOpenSetTestDataset
from evaluate import _build_model_from_ckpt, _collect_predictions
from odin_dump import collect_odin
from ood_baselines import mahalanobis_scores, msp_scores
from ood_scores import (
    compute_prototypes, energy_score, prototype_score, vos_score,
)
from open_set_metrics import auroc

METHODS = ['energy', 'msp', 'odin', 'mahalanobis', 'prototype', 'vos']
BINS = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
BASE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed{}_best'
ODIN_EPS, ODIN_T = 0.005, 1000.0
VALIDATION_TEST_SEED = 99999
NEW_TEST_SEEDS = [100001, 100002, 100003]
BASELINE_WAVG = {'energy': 0.4755, 'msp': 0.4306, 'odin': 0.4798,
                 'mahalanobis': 0.5203, 'prototype': 0.5055, 'vos': 0.5056,
                 'routed': 0.4984}
SANITY_TOL = 0.002


def build_loaders(test_seed, n_per_snr, n_per_snr_uu, batch_size):
    """evaluate.py main()'s construction, with the test seed parameterised."""
    common = dict(n_per_snr=n_per_snr, snr_points=C.SNR_TEST_POINTS,
                  signal_length=C.SIGNAL_LENGTH, sample_rate=C.SAMPLE_RATE,
                  seed=test_seed)
    ds = {
        'kk': CommBSSOpenSetTestDataset(protocol='kk', **common),
        'ku': CommBSSOpenSetTestDataset(protocol='ku', **common),
        'uu': CommBSSOpenSetTestDataset(
            n_per_snr=n_per_snr_uu, snr_points=C.SNR_TEST_POINTS,
            signal_length=C.SIGNAL_LENGTH, sample_rate=C.SAMPLE_RATE,
            seed=test_seed, protocol='uu', carrier_freq_2=2005.0),
    }
    return {p: DataLoader(d, batch_size=batch_size, num_workers=2)
            for p, d in ds.items()}, {p: len(d) for p, d in ds.items()}


def _vos(emb, protos, chunk=1024):
    return np.concatenate([
        vos_score(emb[i:i + chunk], protos, alpha=C.VOS_ALPHA,
                  n_per_class=C.VOS_N_SYNTHETIC, seed=0)
        for i in range(0, len(emb), chunk)])


def eval_group(model, loaders, ref_emb, ref_mods, device):
    """One (model, test-seed) group: returns wavg/pooled/per_bin for the six
    scorers + routed, pools built exactly like revision2_tables s1."""
    pred, odin = {}, {}
    for p in ('kk', 'ku', 'uu'):
        pred[p] = _collect_predictions(model, loaders[p], device)
        odin[p] = collect_odin(model, loaders[p], device, ODIN_T, ODIN_EPS)

    protos = compute_prototypes(ref_emb, ref_mods, C.NUM_KNOWN_CLASSES)

    def slot_scores(pr, od, slot):
        emb, logits = pr[f'emb_{slot}'], pr[f'logits_{slot}']
        return {
            'energy': energy_score(logits),
            'msp': msp_scores(logits),
            'odin': np.asarray(od[f'odin_{slot}'], dtype=np.float64),
            'mahalanobis': mahalanobis_scores(ref_emb, ref_mods, emb,
                                              n_classes=C.NUM_KNOWN_CLASSES,
                                              shrink=0.1),
            'prototype': prototype_score(emb, protos),
            'vos': _vos(emb, protos),
        }

    kk1, kk2 = slot_scores(pred['kk'], odin['kk'], '1'), slot_scores(pred['kk'], odin['kk'], '2')
    ku1, ku2 = slot_scores(pred['ku'], odin['ku'], '1'), slot_scores(pred['ku'], odin['ku'], '2')
    uu1, uu2 = slot_scores(pred['uu'], odin['uu'], '1'), slot_scores(pred['uu'], odin['uu'], '2')

    m1, m2 = pred['ku']['is_ood_1'], pred['ku']['is_ood_2']
    k_scores = {m: np.concatenate([kk1[m], kk2[m]]) for m in METHODS}
    k_snr = np.tile(pred['kk']['snr'], 2)   # slot1 block then slot2 block
    u_scores = {m: np.concatenate([ku1[m][m1], ku2[m][m2], uu1[m], uu2[m]])
                for m in METHODS}
    u_snr = np.concatenate([pred['ku']['snr'][m1], pred['ku']['snr'][m2],
                            pred['uu']['snr'], pred['uu']['snr']])

    per_bin = {}
    for s in BINS:
        mk, mu = k_snr == s, u_snr == s
        nk, nu = int(mk.sum()), int(mu.sum())
        aucs = {m: auroc(k_scores[m][mk], u_scores[m][mu]) for m in METHODS}
        per_bin[str(int(s))] = {'n_known': nk, 'n_unknown': nu, **aucs,
                                'routed': aucs['energy' if s <= 0 else 'prototype']}

    def wavg(m):
        num = sum(v[m] * v['n_known'] * v['n_unknown'] for v in per_bin.values())
        den = sum(v['n_known'] * v['n_unknown'] for v in per_bin.values())
        return num / den if den else float('nan')

    return {'per_bin': per_bin,
            'wavg': {m: wavg(m) for m in METHODS + ['routed']},
            'pooled': {m: auroc(k_scores[m], u_scores[m]) for m in METHODS}}


def _ms(vals):
    v = np.asarray(vals, dtype=np.float64)
    return {'mean': float(v.mean()), 'std': float(v.std()), 'n': int(v.size)}


def _round(obj):
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 4)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def get_args():
    p = argparse.ArgumentParser(description='Multi-test-seed robustness')
    p.add_argument('--seeds', type=str, default='42,43,44,45,46',
                   help='model (training) seeds')
    p.add_argument('--test_seeds', type=str,
                   default=','.join(str(s) for s in NEW_TEST_SEEDS))
    p.add_argument('--n_per_snr', type=int, default=200)
    p.add_argument('--n_per_snr_uu', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--skip_validation', action='store_true',
                   help='do not run the test-seed-99999 validation pass first '
                        '(NOT recommended; for debugging only)')
    p.add_argument('--smoke', action='store_true',
                   help='1 model seed, test seed 99999 + first new seed, '
                        'n_per_snr=8; writes multiseed_test_smoke.json; '
                        'baseline sanity is skipped (tiny cells).')
    p.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    return p.parse_args()


def main():
    args = get_args()
    model_seeds = [int(s) for s in args.seeds.split(',')]
    test_seeds = [int(s) for s in args.test_seeds.split(',')]
    n_per_snr, n_uu = args.n_per_snr, args.n_per_snr_uu
    if args.smoke:
        model_seeds = model_seeds[:1]
        test_seeds = test_seeds[:1]
        n_per_snr, n_uu = 8, 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[eval_multiseed_test] model_seeds={model_seeds} '
          f'test_seeds={test_seeds} n_per_snr={n_per_snr} device={device}')

    # refpool npz is independent of the test seed — load once per model seed
    refpools = {sd: np.load(os.path.join(C.RESULTS_DIR,
                                         BASE.format(sd) + '_refpool.npz'))
                for sd in model_seeds}
    models = {sd: _build_model_from_ckpt(
        os.path.join(C.CHECKPOINT_DIR, BASE.format(sd) + '.pt'), device)
        for sd in model_seeds}

    per_group = {}
    sanity = {'checked': False}

    def run_test_seed(ts):
        loaders, sizes = build_loaders(ts, n_per_snr, n_uu, args.batch_size)
        print(f'\n--- test seed {ts}: sizes {sizes} ---')
        for sd in model_seeds:
            r = eval_group(models[sd], loaders, refpools[sd]['ref_emb'],
                           refpools[sd]['ref_mods'], device)
            key = f'ts{ts}_sd{sd}'
            per_group[key] = {'test_seed': ts, 'model_seed': sd, **r}
            print(f'  model seed {sd}: ' + '  '.join(
                f'{m}={r["wavg"][m]:.4f}' for m in METHODS)
                + f'  routed={r["wavg"]["routed"]:.4f}')
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # ---- 1. validation on the canonical test seed ----
    if not args.skip_validation:
        run_test_seed(VALIDATION_TEST_SEED)
        if not args.smoke:
            means = {m: float(np.mean([per_group[f'ts{VALIDATION_TEST_SEED}_sd{sd}']['wavg'][m]
                                       for sd in model_seeds]))
                     for m in METHODS + ['routed']}
            devs = {m: abs(means[m] - BASELINE_WAVG[m]) for m in BASELINE_WAVG}
            passed = all(d <= SANITY_TOL for d in devs.values())
            sanity = {'checked': True, 'tol': SANITY_TOL,
                      'baseline': BASELINE_WAVG, 'validation_means': means,
                      'abs_deviation': devs, 'passed': bool(passed)}
            print('\n=== SANITY: test seed 99999 vs corrected baselines ===')
            for m in BASELINE_WAVG:
                print(f'  {m:>11}: {means[m]:.4f} vs {BASELINE_WAVG[m]:.4f} '
                      f'(|dev|={devs[m]:.4f})')
            print(f'  PASSED: {passed}')
            if not passed:
                sys.exit('STOP: validation on test seed 99999 failed to '
                         'reproduce the corrected baselines — investigate '
                         'before trusting the new test seeds.')

    # ---- 2. new test seeds ----
    for ts in test_seeds:
        run_test_seed(ts)

    # ---- summaries ----
    keys = list(per_group.keys())
    per_ts = {}
    for ts in sorted({per_group[k]['test_seed'] for k in keys}):
        gk = [k for k in keys if per_group[k]['test_seed'] == ts]
        per_ts[str(ts)] = {
            m: _ms([per_group[k]['wavg'][m] for k in gk])
            for m in METHODS + ['routed']}
    new_keys = [k for k in keys if per_group[k]['test_seed'] != VALIDATION_TEST_SEED]
    per_ms = {}
    for sd in model_seeds:
        gk = [k for k in new_keys if per_group[k]['model_seed'] == sd]
        if gk:
            per_ms[str(sd)] = {
                m: _ms([per_group[k]['wavg'][m] for k in gk])
                for m in METHODS + ['routed']}
    grand = {m: _ms([per_group[k]['wavg'][m] for k in new_keys])
             for m in METHODS + ['routed']} if new_keys else {}

    out = {
        'experiment': 'multi_test_seed_robustness (review-2 Major Concern 4)',
        'config': {'model_seeds': model_seeds,
                   'validation_test_seed': VALIDATION_TEST_SEED,
                   'new_test_seeds': test_seeds,
                   'n_per_snr': n_per_snr, 'n_per_snr_uu': n_uu,
                   'snr_bins': BINS, 'smoke': bool(args.smoke),
                   'scorer_fit': 'refpool npz (kk, seed 88888); odin recomputed '
                                 f'per group (eps={ODIN_EPS}, T={ODIN_T:g})',
                   'labels': 'corrected tile-stacked known_snr; unknown pool = '
                             'ku OOD side + uu both slots'},
        'per_group': per_group,
        'summary_per_test_seed': per_ts,
        'summary_per_model_seed_new_test_seeds': per_ms,
        'grand_summary_new_test_seeds': grand,
        'sanity': sanity,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    fname = 'multiseed_test_smoke.json' if args.smoke else 'multiseed_test.json'
    fpath = os.path.join(args.out_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(_round(out), f, indent=1)
    print(f'\nSaved {fpath}')

    if grand:
        print('\n=== Grand summary over new-test-seed groups (wavg AUROC) ===')
        for m in METHODS + ['routed']:
            print(f'  {m:>11}: {grand[m]["mean"]:.4f} ± {grand[m]["std"]:.4f}')


if __name__ == '__main__':
    main()
