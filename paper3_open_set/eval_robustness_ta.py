"""
Paper 3 — WP2c: robustness battery, truth-anchored re-evaluation.

All sub-batteries recomputed with truth-anchored pool membership and
TA-refpool scorer fits (the letter's standardized corrected pipeline).
Scorers: energy / prototype / vos / mahalanobis + routed (a-priori 0 dB
boundary).  ODIN/MSP are omitted here to keep the battery's runtime
bounded (the original R1-6 battery used energy/prototype/vos only);
six-scorer TA coverage exists in dump_ta_scores.py / protocol_split_ta.json.

Sections
--------
  sir_timing_cfo : eval_robustness.py conditions {baseline, sir ±6/±3/0 dB,
                   timing, cfo25}, main checkpoints seeds 42-44, kk+ku
                   protocols (faithful to the original battery), test seed
                   99999.  Baseline cross-checks against
                   protocol_split_ta.json kkK_vs_kuU per seed.
  carrier_gap    : carrier_freq_2 = 2000 + gap Hz, gap in {10, 50, 100, 500},
                   main checkpoints seeds 42-46, full kk+ku+uu pools
                   (evaluate.py --carrier_gap protocol).
  center_loss    : lambda_c=0.1 checkpoints, seeds 42-46, full pools.
  embed_dim      : embed_dim {16,32,128} checkpoints, seeds 42-44, full pools
                   (emb64 = main model; TA reference numbers from
                   ta_dumps_sanity.json, not recomputed here).
  snr_sigma_sim  : Gaussian routing-noise simulation (sigma = 1/3/6 dB,
                   50 MC trials, seed 0) on the TA standard dumps
                   (noisy_routed_wavg consumes the corrected tile-stacked
                   known_snr directly); GT routed included for reference.

Sanity: TA closed-set cls acc on each section's kk set (main model:
~0.52+, clearly above the ~0.42 label-swapped value).

Output: results/robustness_ta.json (smoke: robustness_ta_smoke.json).
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
from ensemble_analysis import noisy_routed_wavg
from ta_common import (
    BASE, BINS, ODIN_EPS, ODIN_T, _build_model_from_ckpt,
    _collect_predictions, truth_labels, slot_scores, pools_ta, combine_pools,
    wavg_table, cls_acc, ms, round4,
)

METHODS = ['energy', 'mahalanobis', 'prototype', 'vos']
MAIN_SEEDS = [42, 43, 44, 45, 46]
ROB_SEEDS = [42, 43, 44]           # matches rerun_robustness.sh
EMB_SEEDS = [42, 43, 44]           # matches run_embed_dim_ablation.sh
CONDITIONS = {
    'baseline': {},
    'sir_-6dB': {'sir_db': -6.0},
    'sir_-3dB': {'sir_db': -3.0},
    'sir_0dB':  {'sir_db': 0.0},
    'sir_+3dB': {'sir_db': 3.0},
    'sir_+6dB': {'sir_db': 6.0},
    'timing':   {'timing_offset_s2': True},
    'cfo25':    {'carrier_jitter_hz': 25.0},
}
GAPS = [10, 50, 100, 500]
LC_BASE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_lc0.1_seed{}_lc01_best'
EMB_BASE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed{}_emb{}_best'
SIGMA_LEVELS = (1.0, 3.0, 6.0)


def make_loaders(proto_sizes, batch_size, **gen_kwargs):
    """proto_sizes: {'kk': n, 'ku': n, 'uu': n_or_None}; extra gen kwargs
    (carrier_freq_2 / sir_db / timing_offset_s2 / carrier_jitter_hz) are
    forwarded to every dataset."""
    ds = {}
    for p, n in proto_sizes.items():
        if n is None:
            continue
        ds[p] = CommBSSOpenSetTestDataset(
            n_per_snr=n, snr_points=C.SNR_TEST_POINTS,
            signal_length=C.SIGNAL_LENGTH, sample_rate=C.SAMPLE_RATE,
            seed=99999, protocol=p, **gen_kwargs)
    return ({p: DataLoader(d, batch_size=batch_size, num_workers=2)
             for p, d in ds.items()}, ds)


def collect_score(model, loaders, ds, ref_emb, ref_mods, device):
    pred, sc, tr = {}, {}, {}
    for p in loaders:
        pred[p] = _collect_predictions(model, loaders[p], device)
        tr[p] = truth_labels(ds[p])
        sc[p] = slot_scores(pred[p], None, ref_emb, ref_mods, METHODS,
                            num_known=C.NUM_KNOWN_CLASSES)
    return pred, sc, tr


def eval_full_pool(pred, sc, tr):
    pools = pools_ta(sc, pred, tr, METHODS)
    U = combine_pools(pools['U_ku'], pools['U_uu'], methods=METHODS) \
        if 'U_uu' in pools else pools['U_ku']
    return wavg_table(pools['K_kk'], U, METHODS)


def build_ta_refpool(model, device, batch_size, save_as=None,
                     n_per_snr=200):
    ds = CommBSSOpenSetTestDataset(
        n_per_snr=n_per_snr, snr_points=C.SNR_TEST_POINTS,
        signal_length=C.SIGNAL_LENGTH, sample_rate=C.SAMPLE_RATE,
        seed=88888, protocol='kk')
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2)
    pred = _collect_predictions(model, loader, device)
    tr = truth_labels(ds)
    ref_emb = np.concatenate([pred['emb_1'], pred['emb_2']])
    ref_mods = np.concatenate([tr['mod1'], tr['mod2']])
    ref_snr = np.tile(pred['snr'], 2)
    if save_as:
        np.savez(save_as, ref_emb=ref_emb.astype(np.float32),
                 ref_mods=ref_mods.astype(np.int64),
                 ref_snr=ref_snr.astype(np.float32))
    return ref_emb, ref_mods


def load_main_refpool_ta(seed):
    rp = np.load(os.path.join(C.RESULTS_DIR, BASE.format(seed) + '_refpool_ta.npz'))
    return rp['ref_emb'], rp['ref_mods']


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------
def sec_sir_timing_cfo(args, device):
    out = {}
    for seed in ROB_SEEDS:
        print(f'\n--- sir_timing_cfo: seed {seed} ---')
        model = _build_model_from_ckpt(
            os.path.join(C.CHECKPOINT_DIR, BASE.format(seed) + '.pt'), device)
        ref_emb, ref_mods = load_main_refpool_ta(seed)
        per_cond = {}
        for cname, kw in CONDITIONS.items():
            loaders, ds = make_loaders(
                {'kk': args.n_per_snr, 'ku': args.n_per_snr, 'uu': None},
                args.batch_size, **kw)
            pred, sc, tr = collect_score(model, loaders, ds, ref_emb,
                                         ref_mods, device)
            tbl = eval_full_pool(pred, sc, tr)
            acc_s, acc_t = cls_acc(pred['kk'], tr['kk'])
            per_cond[cname] = {'wavg': tbl['wavg'], 'pooled': tbl['pooled'],
                               'per_bin': tbl['per_bin'],
                               'cls_acc_stored': acc_s, 'cls_acc_ta': acc_t}
            print(f'  {cname:<9} routed={tbl["wavg"]["routed"]:.4f} '
                  f'proto={tbl["wavg"]["prototype"]:.4f} '
                  f'maha={tbl["wavg"]["mahalanobis"]:.4f} '
                  f'(cls {acc_s:.3f}->{acc_t:.3f})')
        out[str(seed)] = per_cond
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    agg = {cname: {m: ms([out[str(s)][cname]['wavg'][m] for s in ROB_SEEDS])
                   for m in METHODS + ['routed']} for cname in CONDITIONS}
    return {'per_seed': out, 'across_seeds': agg}


def sec_carrier_gap(args, device):
    out = {}
    for gap in GAPS:
        cf2 = 2000.0 + gap
        loaders, ds = make_loaders(
            {'kk': args.n_per_snr, 'ku': args.n_per_snr,
             'uu': args.n_per_snr_uu}, args.batch_size, carrier_freq_2=cf2)
        per_seed = {}
        for seed in MAIN_SEEDS:
            print(f'--- carrier_gap {gap} Hz: seed {seed} ---')
            model = _build_model_from_ckpt(
                os.path.join(C.CHECKPOINT_DIR, BASE.format(seed) + '.pt'),
                device)
            ref_emb, ref_mods = load_main_refpool_ta(seed)
            pred, sc, tr = collect_score(model, loaders, ds, ref_emb,
                                         ref_mods, device)
            tbl = eval_full_pool(pred, sc, tr)
            acc_s, acc_t = cls_acc(pred['kk'], tr['kk'])
            per_seed[str(seed)] = {'wavg': tbl['wavg'], 'pooled': tbl['pooled'],
                                   'cls_acc_stored': acc_s, 'cls_acc_ta': acc_t}
            print(f'  routed={tbl["wavg"]["routed"]:.4f} '
                  f'proto={tbl["wavg"]["prototype"]:.4f}')
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        out[str(gap)] = {
            'per_seed': per_seed,
            'across_seeds': {m: ms([per_seed[str(s)]['wavg'][m]
                                    for s in MAIN_SEEDS])
                             for m in METHODS + ['routed']},
        }
    return out


def _eval_variant_ckpts(args, device, ckpt_bases, save_refpools, tag):
    loaders, ds = make_loaders(
        {'kk': args.n_per_snr, 'ku': args.n_per_snr, 'uu': args.n_per_snr_uu},
        args.batch_size)
    per_ckpt = {}
    for base in ckpt_bases:
        print(f'--- {tag}: {base} ---')
        model = _build_model_from_ckpt(
            os.path.join(C.CHECKPOINT_DIR, base + '.pt'), device)
        save_as = os.path.join(args.out_dir, base + '_refpool_ta.npz') \
            if (save_refpools and not args.smoke) else None
        ref_emb, ref_mods = build_ta_refpool(
            model, device, args.batch_size, save_as=save_as,
            n_per_snr=(8 if args.smoke else 200))
        pred, sc, tr = collect_score(model, loaders, ds, ref_emb, ref_mods,
                                     device)
        tbl = eval_full_pool(pred, sc, tr)
        acc_s, acc_t = cls_acc(pred['kk'], tr['kk'])
        per_ckpt[base] = {'wavg': tbl['wavg'], 'pooled': tbl['pooled'],
                          'cls_acc_stored': acc_s, 'cls_acc_ta': acc_t}
        print(f'  routed={tbl["wavg"]["routed"]:.4f} '
              f'proto={tbl["wavg"]["prototype"]:.4f} '
              f'maha={tbl["wavg"]["mahalanobis"]:.4f} '
              f'(cls {acc_s:.3f}->{acc_t:.3f})')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return per_ckpt


def sec_sigma_sim(args):
    """Gaussian routing-noise sim on the TA standard dumps (CPU, no model)."""
    rng = np.random.default_rng(0)
    out = {}
    for sigma in SIGMA_LEVELS:
        per_seed = []
        for seed in MAIN_SEEDS:
            f = os.path.join(C.RESULTS_DIR,
                             BASE.format(seed) + '_ood_scores_ta.npz')
            d = np.load(f)
            trials = [noisy_routed_wavg(d, 0.0, sigma, rng)
                      for _ in range(50)]
            per_seed.append(float(np.mean(trials)))
        out[f'{sigma:g}'] = {'per_seed': per_seed,
                             'mean': float(np.mean(per_seed)),
                             'std': float(np.std(per_seed))}
    # GT routed on the TA dumps (stored scores, 0 dB boundary)
    gt = []
    for seed in MAIN_SEEDS:
        d = np.load(os.path.join(C.RESULTS_DIR,
                                 BASE.format(seed) + '_ood_scores_ta.npz'))
        K = {'scores': {'energy': d['energy_score_known'],
                        'prototype': d['prototype_score_known']},
             'snr': d['known_snr']}
        U = {'scores': {'energy': d['energy_score_unknown'],
                        'prototype': d['prototype_score_unknown']},
             'snr': d['unknown_snr']}
        gt.append(wavg_table(K, U, ['energy', 'prototype'])['wavg']['routed'])
    out['gt_routed'] = {'per_seed': gt, 'mean': float(np.mean(gt)),
                        'std': float(np.std(gt))}
    return out


def main():
    ap = argparse.ArgumentParser(description='Robustness battery, TA')
    ap.add_argument('--sections', type=str, nargs='*',
                    default=['sir_timing_cfo', 'carrier_gap', 'center_loss',
                             'embed_dim', 'snr_sigma_sim'])
    ap.add_argument('--n_per_snr', type=int, default=192)
    ap.add_argument('--n_per_snr_uu', type=int, default=96)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--smoke', action='store_true',
                    help='seeds 42 only, baseline gap-10 dim-16 subsets, n=8')
    ap.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.smoke:
        args.n_per_snr, args.n_per_snr_uu = 8, 8
    print(f'[eval_robustness_ta] sections={args.sections} '
          f'n={args.n_per_snr}/{args.n_per_snr_uu} device={device}')

    out = {'experiment': 'robustness battery, truth-anchored (WP2c)',
           'config': {'n_per_snr': args.n_per_snr,
                      'n_per_snr_uu': args.n_per_snr_uu, 'test_seed': 99999,
                      'smoke': bool(args.smoke),
                      'scorers': METHODS,
                      'pools': 'truth-anchored; TA-refpool fits '
                               '(seed 88888, TA labels)',
                      'note': 'odin/msp omitted (battery parity with the '
                              'original R1-6 set + runtime); six-scorer TA '
                              'coverage in dump_ta_scores.py'}}

    global ROB_SEEDS, MAIN_SEEDS, EMB_SEEDS, GAPS, CONDITIONS
    if args.smoke:
        ROB_SEEDS, MAIN_SEEDS, EMB_SEEDS = [42], [42], [42]
        GAPS = [10]
        CONDITIONS = {'baseline': {}, 'sir_+6dB': {'sir_db': 6.0}}

    if 'sir_timing_cfo' in args.sections:
        print('\n########## section: sir_timing_cfo ##########')
        out['sir_timing_cfo'] = sec_sir_timing_cfo(args, device)
    if 'carrier_gap' in args.sections:
        print('\n########## section: carrier_gap ##########')
        out['carrier_gap'] = sec_carrier_gap(args, device)
    if 'center_loss' in args.sections:
        print('\n########## section: center_loss ##########')
        bases = [LC_BASE.format(s) for s in MAIN_SEEDS]
        out['center_loss'] = _eval_variant_ckpts(args, device, bases, True,
                                                 'center_loss')
    if 'embed_dim' in args.sections:
        print('\n########## section: embed_dim ##########')
        bases = [EMB_BASE.format(s, d) for d in (16, 32, 128)
                 for s in EMB_SEEDS]
        out['embed_dim'] = _eval_variant_ckpts(args, device, bases, True,
                                               'embed_dim')
    if 'snr_sigma_sim' in args.sections:
        print('\n########## section: snr_sigma_sim ##########')
        out['snr_sigma_sim'] = sec_sigma_sim(args)
        print('  GT routed (TA dumps):', out['snr_sigma_sim']['gt_routed'])

    os.makedirs(args.out_dir, exist_ok=True)
    fname = 'robustness_ta_smoke.json' if args.smoke else 'robustness_ta.json'
    fpath = os.path.join(args.out_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(round4(out), f, indent=1)
    print(f'\nSaved {fpath}')


if __name__ == '__main__':
    main()
