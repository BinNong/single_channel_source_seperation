"""
Paper 3 — Review-2 experiment C, truth-anchored variant.

Identical design to eval_multiseed_test.py (test seeds 100001/100002/100003
plus validation on 99999; five model seeds; six scorers + routed; refpool
fit) but with TRUTH-ANCHORED pool membership: the ku OOD/known sides are
selected by the dataset's true per-source ood flags, not the label-swapped
is_ood_1/2 returned by evaluate._collect_predictions (see
eval_truth_anchor.py header for the bug analysis; kk/uu pools are identical
either way, only ku side-selection changes).

Validation: on test seed 99999 the truth-anchored combined wavg must match
eval_truth_anchor.py's combined numbers (routed 0.5106, mahalanobis 0.5320,
prototype 0.5326, vos 0.5323, energy 0.4314, msp 0.3936, odin 0.4354)
within +-0.002 before the new test seeds are run.

Output: results/multiseed_test_ta.json (smoke: multiseed_test_ta_smoke.json).
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
from open_set_metrics import auroc
from eval_protocol_split import (
    METHODS, BINS, BASE, ODIN_EPS, ODIN_T, score_slots, _ms, _round,
)
from eval_truth_anchor import truth_labels

NEW_TEST_SEEDS = [100001, 100002, 100003]
VALIDATION_TEST_SEED = 99999
TA_BASELINE_WAVG = {'energy': 0.4314, 'msp': 0.3936, 'odin': 0.4354,
                    'mahalanobis': 0.5320, 'prototype': 0.5326,
                    'vos': 0.5323, 'routed': 0.5106}
SANITY_TOL = 0.002


def build_loaders_with_datasets(test_seed, n_per_snr, n_per_snr_uu, batch_size):
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
    return ({p: DataLoader(d, batch_size=batch_size, num_workers=2)
             for p, d in ds.items()}, ds)


def eval_group_ta(model, loaders, truth, ref_emb, ref_mods, device):
    """One (model, test-seed) group with truth-anchored pool membership."""
    pred, odin, sc = {}, {}, {}
    for p in ('kk', 'ku', 'uu'):
        pred[p] = _collect_predictions(model, loaders[p], device)
        odin[p] = collect_odin(model, loaders[p], device, ODIN_T, ODIN_EPS)
        sc[p] = score_slots(pred[p], odin[p], ref_emb, ref_mods)
        assert len(truth[p]['mod1']) == len(pred[p]['snr'])

    to1, to2 = truth['ku']['ood1'], truth['ku']['ood2']
    k_scores = {m: np.concatenate([sc['kk']['1'][m], sc['kk']['2'][m]])
                for m in METHODS}
    k_snr = np.tile(pred['kk']['snr'], 2)
    u_scores = {m: np.concatenate([sc['ku']['1'][m][to1], sc['ku']['2'][m][to2],
                                   sc['uu']['1'][m], sc['uu']['2'][m]])
                for m in METHODS}
    u_snr = np.concatenate([pred['ku']['snr'][to1], pred['ku']['snr'][to2],
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


def get_args():
    p = argparse.ArgumentParser(description='Multi-test-seed, truth-anchored')
    p.add_argument('--seeds', type=str, default='42,43,44,45,46')
    p.add_argument('--test_seeds', type=str,
                   default=','.join(str(s) for s in NEW_TEST_SEEDS))
    p.add_argument('--n_per_snr', type=int, default=200)
    p.add_argument('--n_per_snr_uu', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--smoke', action='store_true',
                   help='1 model seed, 99999 + first new seed, n=8')
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
    print(f'[eval_multiseed_ta] model_seeds={model_seeds} test_seeds={test_seeds} '
          f'n={n_per_snr}/{n_uu} device={device}')

    refpools = {sd: np.load(os.path.join(C.RESULTS_DIR,
                                         BASE.format(sd) + '_refpool.npz'))
                for sd in model_seeds}
    models = {sd: _build_model_from_ckpt(
        os.path.join(C.CHECKPOINT_DIR, BASE.format(sd) + '.pt'), device)
        for sd in model_seeds}

    per_group = {}
    sanity = {'checked': False}

    def run_test_seed(ts):
        loaders, ds = build_loaders_with_datasets(ts, n_per_snr, n_uu,
                                                  args.batch_size)
        truth = {p: truth_labels(d) for p, d in ds.items()}
        print(f'--- test seed {ts} ---')
        for sd in model_seeds:
            r = eval_group_ta(models[sd], loaders, truth,
                              refpools[sd]['ref_emb'], refpools[sd]['ref_mods'],
                              device)
            per_group[f'ts{ts}_sd{sd}'] = {'test_seed': ts, 'model_seed': sd, **r}
            print(f'  sd{sd}: ' + '  '.join(f'{m}={r["wavg"][m]:.4f}' for m in METHODS)
                  + f'  routed={r["wavg"]["routed"]:.4f}')
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # validation on the canonical test seed first
    run_test_seed(VALIDATION_TEST_SEED)
    if not args.smoke:
        means = {m: float(np.mean([per_group[f'ts{VALIDATION_TEST_SEED}_sd{sd}']['wavg'][m]
                                   for sd in model_seeds]))
                 for m in METHODS + ['routed']}
        devs = {m: abs(means[m] - TA_BASELINE_WAVG[m]) for m in TA_BASELINE_WAVG}
        passed = all(d <= SANITY_TOL for d in devs.values())
        sanity = {'checked': True, 'tol': SANITY_TOL,
                  'baseline_truth_anchored_99999': TA_BASELINE_WAVG,
                  'validation_means': means, 'abs_deviation': devs,
                  'passed': bool(passed)}
        print('\n=== SANITY: ts99999 truth-anchored vs eval_truth_anchor ===')
        for m in TA_BASELINE_WAVG:
            print(f'  {m:>11}: {means[m]:.4f} vs {TA_BASELINE_WAVG[m]:.4f} '
                  f'(|dev|={devs[m]:.4f})')
        print(f'  PASSED: {passed}')
        if not passed:
            sys.exit('STOP: truth-anchored validation on 99999 failed.')

    for ts in test_seeds:
        run_test_seed(ts)

    keys = list(per_group.keys())
    per_ts = {}
    for ts in sorted({per_group[k]['test_seed'] for k in keys}):
        gk = [k for k in keys if per_group[k]['test_seed'] == ts]
        per_ts[str(ts)] = {m: _ms([per_group[k]['wavg'][m] for k in gk])
                           for m in METHODS + ['routed']}
    new_keys = [k for k in keys if per_group[k]['test_seed'] != VALIDATION_TEST_SEED]
    per_ms = {}
    for sd in model_seeds:
        gk = [k for k in new_keys if per_group[k]['model_seed'] == sd]
        if gk:
            per_ms[str(sd)] = {m: _ms([per_group[k]['wavg'][m] for k in gk])
                               for m in METHODS + ['routed']}
    grand = {m: _ms([per_group[k]['wavg'][m] for k in new_keys])
             for m in METHODS + ['routed']} if new_keys else {}

    out = {
        'experiment': 'multi_test_seed_robustness, TRUTH-ANCHORED labels',
        'config': {'model_seeds': model_seeds,
                   'validation_test_seed': VALIDATION_TEST_SEED,
                   'new_test_seeds': test_seeds,
                   'n_per_snr': n_per_snr, 'n_per_snr_uu': n_uu,
                   'smoke': bool(args.smoke),
                   'scorer_fit': 'refpool npz (kk, seed 88888); odin recomputed',
                   'labels': 'truth-anchored pool membership (dataset ood '
                             'flags); see eval_truth_anchor.py'},
        'per_group': per_group,
        'summary_per_test_seed': per_ts,
        'summary_per_model_seed_new_test_seeds': per_ms,
        'grand_summary_new_test_seeds': grand,
        'sanity': sanity,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    fname = 'multiseed_test_ta_smoke.json' if args.smoke else 'multiseed_test_ta.json'
    fpath = os.path.join(args.out_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(_round(out), f, indent=1)
    print(f'\nSaved {fpath}')
    if grand:
        print('\n=== Grand summary over new-test-seed groups (TA wavg) ===')
        for m in METHODS + ['routed']:
            print(f'  {m:>11}: {grand[m]["mean"]:.4f} ± {grand[m]["std"]:.4f}')


if __name__ == '__main__':
    main()
