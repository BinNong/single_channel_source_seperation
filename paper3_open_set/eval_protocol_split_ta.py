"""
Paper 3 — WP1: protocol split, FULLY truth-anchored (TA pools + TA-refpool
scorer fits), letter-standard protocol: test seed 99999, 192/192/96 mixtures
per bin (n_per_snr=192, n_per_snr_uu=96 — identical samples to 200/100).

Contrasts (six scorers + routed, per-bin AUROC + pair-weighted wavg +
pooled, 5 seeds):
  kkK_vs_kuU : K(kk both slots) vs U(ku true-OOD slot)
  kkK_vs_uuU : K(kk both slots) vs U(uu both slots)
  kuK_vs_kuU : K(ku true-known slot) vs U(ku true-OOD slot)  (same mixture)
  combined   : K(kk) vs U(ku + uu)  — the headline pool

Sanity: the combined contrast must match the fully-TA headline recomputed
in dump_ta_scores.py (results/ta_dumps_sanity.json, TA-refpool fit column)
within +-0.01 per seed.

Output: results/protocol_split_ta.json (smoke: protocol_split_ta_smoke.json).
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
    BASE, BINS, METHODS6, ODIN_EPS, ODIN_T, _build_model_from_ckpt,
    _collect_predictions, collect_odin, truth_labels, slot_scores, pools_ta,
    combine_pools, wavg_table, cls_acc, ms, round4,
)

SANITY_TOL = 0.01


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
    }
    return ({p: DataLoader(d, batch_size=batch_size, num_workers=2)
             for p, d in ds.items()}, ds)


def main():
    ap = argparse.ArgumentParser(description='Protocol split, fully TA')
    ap.add_argument('--seeds', type=str, default='42,43,44,45,46')
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
    print(f'[eval_protocol_split_ta] seeds={seeds} n={args.n_per_snr}/'
          f'{args.n_per_snr_uu} device={device}')

    loaders, ds = build_loaders(args.n_per_snr, args.n_per_snr_uu,
                                args.batch_size)
    tr = {p: truth_labels(d) for p, d in ds.items()}

    # fully-TA headline per seed (dump_ta_scores sanity, TA-refpool fit)
    target = None
    sp = os.path.join(C.RESULTS_DIR, 'ta_dumps_sanity.json')
    if os.path.exists(sp) and not args.smoke:
        sd = json.load(open(sp))
        target = {s: sd[s]['d']['ta_refpool'] for s in sd}

    per_seed, sanity = {}, {}
    for seed in seeds:
        base = BASE.format(seed)
        print(f'\n=== seed {seed} ===')
        model = _build_model_from_ckpt(
            os.path.join(C.CHECKPOINT_DIR, base + '.pt'), device)
        rp = np.load(os.path.join(C.RESULTS_DIR, base + '_refpool_ta.npz'))
        ref_emb, ref_mods = rp['ref_emb'], rp['ref_mods']

        pred, odin, sc = {}, {}, {}
        for p in ('kk', 'ku', 'uu'):
            pred[p] = _collect_predictions(model, loaders[p], device)
            odin[p] = collect_odin(model, loaders[p], device, ODIN_T, ODIN_EPS)
            sc[p] = slot_scores(pred[p], odin[p], ref_emb, ref_mods, METHODS6)
        acc_stored, acc_truth = cls_acc(pred['kk'], tr['kk'])
        print(f'  kk cls acc: stored={acc_stored:.4f}  TA={acc_truth:.4f}')

        pools = pools_ta(sc, pred, tr, METHODS6)
        U_all = combine_pools(pools['U_ku'], pools['U_uu'], methods=METHODS6)
        comps = {
            'kkK_vs_kuU': wavg_table(pools['K_kk'], pools['U_ku'], METHODS6),
            'kkK_vs_uuU': wavg_table(pools['K_kk'], pools['U_uu'], METHODS6),
            'kuK_vs_kuU': wavg_table(pools['K_ku'], pools['U_ku'], METHODS6),
            'combined': wavg_table(pools['K_kk'], U_all, METHODS6),
        }
        per_seed[str(seed)] = {'cls_acc_stored': acc_stored,
                               'cls_acc_ta': acc_truth,
                               'comparisons': comps}
        print('  combined wavg: ' + '  '.join(
            f'{m}={comps["combined"]["wavg"][m]:.4f}' for m in METHODS6)
            + f'  routed={comps["combined"]["wavg"]["routed"]:.4f}')

        if target is not None:
            devs = {m: abs(comps['combined']['wavg'][m] - target[str(seed)][m])
                    for m in METHODS6 + ['routed']}
            sanity[str(seed)] = {'max_abs_dev': max(devs.values()),
                                 'passed': bool(max(devs.values()) <= SANITY_TOL),
                                 'devs': devs}
            print(f'  sanity vs ta_dumps_sanity (TA-refpool): '
                  f'max|dev|={max(devs.values()):.5f} '
                  f'-> {"OK" if sanity[str(seed)]["passed"] else "FAIL"}')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    comp_names = ['kkK_vs_kuU', 'kkK_vs_uuU', 'kuK_vs_kuU', 'combined']
    across = {}
    for cname in comp_names:
        agg = {'wavg': {}, 'pooled': {}, 'per_bin': {}}
        for m in METHODS6 + ['routed']:
            agg['wavg'][m] = ms([per_seed[s]['comparisons'][cname]['wavg'][m]
                                 for s in per_seed])
        for m in METHODS6:
            agg['pooled'][m] = ms([per_seed[s]['comparisons'][cname]['pooled'][m]
                                   for s in per_seed])
        for s in BINS:
            key = str(int(s))
            agg['per_bin'][key] = {
                m: ms([per_seed[sd]['comparisons'][cname]['per_bin'][key][m]
                       for sd in per_seed])
                for m in METHODS6 + ['routed']}
        across[cname] = agg

    out = {
        'experiment': 'protocol_split, FULLY truth-anchored (WP1)',
        'config': {'seeds': seeds, 'n_per_snr': args.n_per_snr,
                   'n_per_snr_uu': args.n_per_snr_uu, 'test_seed': 99999,
                   'smoke': bool(args.smoke),
                   'pools': 'truth-anchored membership (dataset ood flags)',
                   'scorer_fit': 'TA refpool npz (kk, seed 88888, TA labels)',
                   'odin': f'recomputed per slot (eps={ODIN_EPS}, T={ODIN_T:g})'},
        'cls_acc': {'stored': ms([per_seed[s]['cls_acc_stored'] for s in per_seed]),
                    'ta': ms([per_seed[s]['cls_acc_ta'] for s in per_seed])},
        'per_seed': per_seed,
        'across_seeds': across,
        'sanity_vs_ta_dumps': sanity,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    fname = 'protocol_split_ta_smoke.json' if args.smoke else 'protocol_split_ta.json'
    fpath = os.path.join(args.out_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(round4(out), f, indent=1)
    print(f'\nSaved {fpath}')


if __name__ == '__main__':
    main()
