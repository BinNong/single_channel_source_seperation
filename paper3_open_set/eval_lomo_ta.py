"""
Paper 3 — WP2b: LOMO, truth-anchored re-evaluation.

Re-runs the eval_lomo.py protocol (4 leave-one-modulation-out splits x
seeds 42-44 = 12 runs) with truth-anchored pool membership and TA-refpool
scorer fits:

  (a) pseudo-OOD validation (seed 77777, unknown = held-out mod only),
      TA pools, prototype/vos/mahalanobis fit on a split-specific TA
      reference pool (kk of the 3 trained mods, seed 88888, TA labels) —
      the routing boundary is selected EXCLUSIVELY on this validation
      split, as before (eval_lomo.select_boundary, unchanged rule);
  (b) test (seed 99999, unknown = held-out mod + the 4 unknown classes),
      TA pools, routed wavg AUROC at the boundary from (a).

Scorers: energy / prototype / vos (as eval_lomo.py) + mahalanobis.
n_per_snr=200 throughout (the original LOMO protocol).

Sanity per run: TA closed-set cls acc on the test kk set must clearly
exceed the label-swapped one.

Output: results/lomo_ta.json (smoke: lomo_ta_smoke.json).
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
from data_generator_extended import (
    CommBSSOpenSetTestDataset, MOD_KNOWN, MOD_UNKNOWN, MOD_TO_IDX,
)
from eval_lomo import _build_model, select_boundary
from ta_common import (
    ODIN_EPS, ODIN_T, _collect_predictions, truth_labels, slot_scores,
    pools_ta, wavg_table, ms, round4,
)

VAL_SEED = 77777
TEST_SEED = 99999
REF_SEED = 88888
LOMO_BASE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_km{}_seed{}_best.pt'
METHODS = ['energy', 'mahalanobis', 'prototype', 'vos']


def build_set(known_mods, unknown_pool, seed, n_per_snr, batch_size):
    common = dict(n_per_snr=n_per_snr, snr_points=C.SNR_TEST_POINTS,
                  signal_length=C.SIGNAL_LENGTH, sample_rate=C.SAMPLE_RATE,
                  seed=seed, mod_known_pool=list(known_mods),
                  mod_unknown_pool=list(unknown_pool))
    ds = {p: CommBSSOpenSetTestDataset(protocol=p, **common) for p in ('kk', 'ku')}
    return ({p: DataLoader(d, batch_size=batch_size, num_workers=2)
             for p, d in ds.items()}, ds)


def build_refpool(known_mods, batch_size):
    ds = CommBSSOpenSetTestDataset(
        n_per_snr=200, snr_points=C.SNR_TEST_POINTS,
        signal_length=C.SIGNAL_LENGTH, sample_rate=C.SAMPLE_RATE,
        seed=REF_SEED, protocol='kk', mod_known_pool=list(known_mods))
    return DataLoader(ds, batch_size=batch_size, num_workers=2), ds


def collect_split(model, loaders, ds, ref_emb, ref_mods_local, device):
    """TA pools + scores for one kk+ku split (no odin — matches LOMO)."""
    pred, sc, tr = {}, {}, {}
    for p in ('kk', 'ku'):
        pred[p] = _collect_predictions(model, loaders[p], device)
        tr[p] = truth_labels(ds[p])
        sc[p] = slot_scores(pred[p], None, ref_emb, ref_mods_local, METHODS,
                            num_known=len(np.unique(ref_mods_local)))
    pools = pools_ta(sc, pred, tr, METHODS)
    return pools, pred, tr


def main():
    ap = argparse.ArgumentParser(description='LOMO, truth-anchored')
    ap.add_argument('--seeds', type=str, default='42,43,44')
    ap.add_argument('--n_per_snr', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--smoke', action='store_true',
                    help='1 split x 1 seed, n_per_snr=8')
    ap.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    held_outs = list(MOD_KNOWN)
    if args.smoke:
        seeds, held_outs = seeds[:1], held_outs[:1]
        args.n_per_snr = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[eval_lomo_ta] seeds={seeds} held_outs={held_outs} '
          f'n={args.n_per_snr} device={device}')

    runs = []
    for held_out in held_outs:
        known_mods = [m for m in MOD_KNOWN if m != held_out]
        lut = {MOD_TO_IDX[m]: i for i, m in enumerate(known_mods)}
        unknown_pool_test = [held_out] + list(MOD_UNKNOWN)
        km_tag = '-'.join(known_mods)

        print(f'\n##### split: known={known_mods} held_out={held_out} #####')
        # datasets are shared across the 3 seeds of this split
        val_loaders, val_ds = build_set(known_mods, [held_out], VAL_SEED,
                                        args.n_per_snr, args.batch_size)
        test_loaders, test_ds = build_set(known_mods, unknown_pool_test,
                                          TEST_SEED, args.n_per_snr,
                                          args.batch_size)
        ref_loader, ref_ds = build_refpool(known_mods, args.batch_size)
        tr_ref = truth_labels(ref_ds)

        for seed in seeds:
            ckpt = os.path.join(C.CHECKPOINT_DIR,
                                LOMO_BASE.format(km_tag, seed))
            print(f'\n=== run km={km_tag} seed={seed} ===')
            model, _a = _build_model(ckpt, device, len(known_mods))

            # split-specific TA refpool (local labels)
            pred_ref = _collect_predictions(model, ref_loader, device)
            ref_emb = np.concatenate([pred_ref['emb_1'], pred_ref['emb_2']])
            ref_mods_local = np.array(
                [lut[int(v)] for v in np.concatenate([tr_ref['mod1'],
                                                      tr_ref['mod2']])],
                dtype=np.int64)

            # (a) validation -> boundary
            pools_val, _pv, _tv = collect_split(model, val_loaders, val_ds,
                                                ref_emb, ref_mods_local,
                                                device)
            val_tbl = wavg_table(pools_val['K_kk'], pools_val['U_ku'], METHODS)
            val_e = {int(s): v['energy']
                     for s, v in val_tbl['per_bin'].items()}
            val_p = {int(s): v['prototype']
                     for s, v in val_tbl['per_bin'].items()}
            boundary, reason = select_boundary(val_e, val_p, fallback=0.0)
            print(f'  validation boundary = {boundary:g} dB ({reason})')

            # (b) test
            pools_test, pred_test, tr_test = collect_split(
                model, test_loaders, test_ds, ref_emb, ref_mods_local, device)
            test_tbl = wavg_table(pools_test['K_kk'], pools_test['U_ku'],
                                  METHODS, boundary=boundary)
            acc_stored = float(np.mean(np.concatenate([
                pred_test['kk']['logits_1'].argmax(-1) == np.array(
                    [lut.get(int(v), -1) for v in pred_test['kk']['mod1_idx']]),
                pred_test['kk']['logits_2'].argmax(-1) == np.array(
                    [lut.get(int(v), -1) for v in pred_test['kk']['mod2_idx']])])))
            acc_ta = float(np.mean(np.concatenate([
                pred_test['kk']['logits_1'].argmax(-1) == np.array(
                    [lut[int(v)] for v in tr_test['kk']['mod1']]),
                pred_test['kk']['logits_2'].argmax(-1) == np.array(
                    [lut[int(v)] for v in tr_test['kk']['mod2']])])))

            run = {
                'known_mods': known_mods, 'held_out': held_out, 'seed': seed,
                'boundary_db': boundary, 'boundary_reason': reason,
                'cls_acc_stored': acc_stored, 'cls_acc_ta': acc_ta,
                'val_per_bin': val_tbl['per_bin'],
                'test_wavg': test_tbl['wavg'], 'test_pooled': test_tbl['pooled'],
                'test_per_bin': test_tbl['per_bin'],
            }
            runs.append(run)
            print(f'  cls acc stored={acc_stored:.4f} TA={acc_ta:.4f}')
            print('  test wavg: ' + '  '.join(
                f'{m}={test_tbl["wavg"][m]:.4f}' for m in METHODS)
                + f'  routed={test_tbl["wavg"]["routed"]:.4f}')
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # boundary distribution + across-run summary
    bounds = [r['boundary_db'] for r in runs]
    u, cnt = np.unique(bounds, return_counts=True)
    boundary_dist = {f'{b:g}': int(c) for b, c in zip(u, cnt)}
    summary = {
        'boundary_distribution': boundary_dist,
        'routed_wavg': ms([r['test_wavg']['routed'] for r in runs]),
        **{f'{m}_wavg': ms([r['test_wavg'][m] for r in runs]) for m in METHODS},
        'cls_acc_stored': ms([r['cls_acc_stored'] for r in runs]),
        'cls_acc_ta': ms([r['cls_acc_ta'] for r in runs]),
        'per_split_routed': {
            ho: ms([r['test_wavg']['routed'] for r in runs
                    if r['held_out'] == ho]) for ho in held_outs},
    }
    out = {
        'experiment': 'LOMO, truth-anchored (WP2b)',
        'config': {'seeds': seeds, 'held_outs': held_outs,
                   'n_per_snr': args.n_per_snr, 'val_seed': VAL_SEED,
                   'test_seed': TEST_SEED, 'refpool_seed': REF_SEED,
                   'smoke': bool(args.smoke),
                   'pools': 'truth-anchored; scorers fit on split-specific TA '
                            'refpool (kk of the 3 trained mods, seed 88888)',
                   'boundary': 'select_boundary on the TA validation split '
                               'only (unchanged rule from eval_lomo.py)'},
        'runs': runs,
        'summary': summary,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    fname = 'lomo_ta_smoke.json' if args.smoke else 'lomo_ta.json'
    fpath = os.path.join(args.out_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(round4(out), f, indent=1)
    print(f'\nSaved {fpath}')
    print('boundary distribution:', boundary_dist)


if __name__ == '__main__':
    main()
