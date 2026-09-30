"""
Paper 3 — WP2a: no-SE backbone, truth-anchored re-evaluation.

The no-SE cross-backbone control (3 checkpoints, seeds 42-44,
`openset_cse_h32_l4_bs16_lr0.001_alpha1.0_nose_seed{42,43,44}_best.pt`)
predates the PIT label-swap fix.  This script recomputes the standard test
protocol (seed 99999, 192/192/96) with TA pool membership and TA-refpool
scorer fits: the per-checkpoint reference pool (kk, seed 88888) is rebuilt
with truth labels and saved as `results/{run}_refpool_ta.npz`.

Six scorers (energy/msp/odin/mahalanobis/prototype/vos; odin recomputed per
slot) + routed wavg AUROC, per-bin + pooled.  Sanity: TA closed-set cls acc
must clearly exceed the label-swapped one (main-model reference: 0.42 ->
0.52+).

Output: results/nose_ta.json (smoke: nose_ta_smoke.json).
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
from ta_common import (
    BINS, METHODS6, ODIN_EPS, ODIN_T, _build_model_from_ckpt,
    _collect_predictions, collect_odin, truth_labels, slot_scores, pools_ta,
    combine_pools, wavg_table, cls_acc, ms, round4,
)

NOSE_BASE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_nose_seed{}_best'


def build_loaders(n_per_snr, n_per_snr_uu, batch_size):
    common = dict(n_per_snr=n_per_snr, snr_points=C.SNR_TEST_POINTS,
                  signal_length=C.SIGNAL_LENGTH, sample_rate=C.SAMPLE_RATE,
                  seed=99999)
    ds = {
        'kk': CommBSSOpenSetTestDataset(protocol='kk', **common),
        'ku': CommBSSOpenSetTestDataset(protocol='ku', **common),
        'uu': CommBSSOpenSetTestDataset(
            n_per_snr=n_per_snr_uu, snr_points=C.SNR_TEST_POINTS,
            signal_length=C.SIGNAL_LENGTH, sample_rate=C.SAMPLE_RATE,
            seed=99999, protocol='uu', carrier_freq_2=2005.0),
        'ref': CommBSSOpenSetTestDataset(
            n_per_snr=200, snr_points=C.SNR_TEST_POINTS,
            signal_length=C.SIGNAL_LENGTH, sample_rate=C.SAMPLE_RATE,
            seed=88888, protocol='kk'),
    }
    return ({p: DataLoader(d, batch_size=batch_size, num_workers=2)
             for p, d in ds.items()}, ds)


def main():
    ap = argparse.ArgumentParser(description='no-SE backbone, TA re-eval')
    ap.add_argument('--seeds', type=str, default='42,43,44')
    ap.add_argument('--n_per_snr', type=int, default=192)
    ap.add_argument('--n_per_snr_uu', type=int, default=96)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    if args.smoke:
        seeds = seeds[:1]
        args.n_per_snr, args.n_per_snr_uu = 8, 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[eval_nose_ta] seeds={seeds} n={args.n_per_snr}/'
          f'{args.n_per_snr_uu} device={device}')

    loaders, ds = build_loaders(args.n_per_snr, args.n_per_snr_uu,
                                args.batch_size)
    tr = {p: truth_labels(d) for p, d in ds.items()}

    per_seed = {}
    for seed in seeds:
        base = NOSE_BASE.format(seed)
        print(f'\n=== no-SE seed {seed} ===')
        model = _build_model_from_ckpt(
            os.path.join(C.CHECKPOINT_DIR, base + '.pt'), device)

        # TA refpool for this checkpoint (saved for reproducibility)
        pred_ref = _collect_predictions(model, loaders['ref'], device)
        ref_emb = np.concatenate([pred_ref['emb_1'], pred_ref['emb_2']])
        ref_mods = np.concatenate([tr['ref']['mod1'], tr['ref']['mod2']])
        ref_snr = np.tile(pred_ref['snr'], 2)
        if not args.smoke:
            np.savez(os.path.join(args.out_dir, base + '_refpool_ta.npz'),
                     ref_emb=ref_emb.astype(np.float32),
                     ref_mods=ref_mods.astype(np.int64),
                     ref_snr=ref_snr.astype(np.float32))

        pred, odin, sc = {}, {}, {}
        for p in ('kk', 'ku', 'uu'):
            pred[p] = _collect_predictions(model, loaders[p], device)
            odin[p] = collect_odin(model, loaders[p], device, ODIN_T, ODIN_EPS)
            sc[p] = slot_scores(pred[p], odin[p], ref_emb, ref_mods, METHODS6)
        acc_stored, acc_truth = cls_acc(pred['kk'], tr['kk'])
        print(f'  kk cls acc: stored={acc_stored:.4f}  TA={acc_truth:.4f}')

        pools = pools_ta(sc, pred, tr, METHODS6)
        U_all = combine_pools(pools['U_ku'], pools['U_uu'], methods=METHODS6)
        comb = wavg_table(pools['K_kk'], U_all, METHODS6)
        per_seed[str(seed)] = {'cls_acc_stored': acc_stored,
                               'cls_acc_ta': acc_truth,
                               'combined': comb}
        print('  combined wavg: ' + '  '.join(
            f'{m}={comb["wavg"][m]:.4f}' for m in METHODS6)
            + f'  routed={comb["wavg"]["routed"]:.4f}')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    across = {
        'wavg': {m: ms([per_seed[s]['combined']['wavg'][m] for s in per_seed])
                 for m in METHODS6 + ['routed']},
        'pooled': {m: ms([per_seed[s]['combined']['pooled'][m] for s in per_seed])
                   for m in METHODS6},
        'per_bin': {str(int(s)): {
            m: ms([per_seed[sd]['combined']['per_bin'][str(int(s))][m]
                   for sd in per_seed]) for m in METHODS6 + ['routed']}
            for s in BINS},
        'cls_acc_stored': ms([per_seed[s]['cls_acc_stored'] for s in per_seed]),
        'cls_acc_ta': ms([per_seed[s]['cls_acc_ta'] for s in per_seed]),
    }

    out = {
        'experiment': 'no-SE cross-backbone control, truth-anchored (WP2a)',
        'config': {'seeds': seeds, 'checkpoints': NOSE_BASE,
                   'n_per_snr': args.n_per_snr, 'n_per_snr_uu': args.n_per_snr_uu,
                   'test_seed': 99999, 'refpool_seed': 88888,
                   'smoke': bool(args.smoke),
                   'pools': 'truth-anchored; scorer fits on per-checkpoint TA '
                            'refpool (saved next to this file)'},
        'per_seed': per_seed,
        'across_seeds': across,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    fname = 'nose_ta_smoke.json' if args.smoke else 'nose_ta.json'
    fpath = os.path.join(args.out_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(round4(out), f, indent=1)
    print(f'\nSaved {fpath}')


if __name__ == '__main__':
    main()
