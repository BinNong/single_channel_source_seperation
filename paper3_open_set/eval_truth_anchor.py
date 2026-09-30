"""
Paper 3 — CRITICAL follow-up (found during review-2 experiment B
cross-checks): the PIT label anchoring in evaluate._collect_predictions.

_collect_predictions aligns waveforms/embeddings/logits to the true sources
(s1_best = estimate of src1 etc.) but ALSO swaps the per-source LABELS
(mod1_best = where(swap, mod2, mod1), ood1_best = where(swap, ood2, ood1)).
Since the content is anchored to the truth, the labels must be anchored to
the truth too (the src1-aligned slot's true label is ALWAYS mod1/ood1).
Swapping the labels along with the content mislabels exactly the swapped
samples: on ku mixtures a swapped sample puts the KNOWN source's embedding
into the unknown pool and vice versa, attenuating every AUROC toward 0.5.

Scope of the contamination (label swap only matters when the two sources
differ in the label):
  * kk protocol: both sources known  -> UNAFFECTED (known pool exact)
  * uu protocol: both sources unknown -> UNAFFECTED (uu pool exact)
  * ku protocol: one known + one unknown -> CONTAMINATED at swapped samples
  * closed-set cls_acc: depressed (compares slot content to the OTHER
    source's class)
  * refpool npz (kk): UNAFFECTED
  * the 2026-09-21 repeat/tile SNR-label fix: UNAFFECTED (SNR labels are
    per-mixture, identical for both sources)

This script measures the swap rate and the truth-anchored closed-set
accuracy, then recomputes the FULL corrected headline (six scorers + routed,
wavg + pooled, test seed 99999, refpool-fitted — same fitting as
revision2_tables s1) and the protocol split (kkK vs kuU / kkK vs uuU /
kuK vs kuU same-mixture) with truth-anchored labels.

Truth labels come straight from the dataset objects (CommBSSOpenSetTestDataset
.samples, same order as the sequential DataLoader); embeddings/logits/ODIN
from _collect_predictions / odin_dump.collect_odin are already
content-aligned, so only the pool membership changes.

Output: results/truth_anchor.json (smoke: truth_anchor_smoke.json).

Usage:
  python eval_truth_anchor.py --smoke
  python eval_truth_anchor.py
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
from data_generator_extended import CommBSSOpenSetTestDataset, IDX_TO_MOD
from evaluate import _build_model_from_ckpt, _collect_predictions
from odin_dump import collect_odin
from ood_scores import compute_prototypes
from open_set_metrics import auroc
from eval_protocol_split import (
    METHODS, BINS, BASE, ODIN_EPS, ODIN_T, score_slots, eval_comparison,
    _ms, _round,
)


def build_loaders_with_datasets(n_per_snr, n_per_snr_uu, batch_size):
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
    loaders = {p: DataLoader(d, batch_size=batch_size, num_workers=2)
               for p, d in ds.items()}
    return loaders, ds


def truth_labels(ds):
    """Per-sample true labels in dataset (== sequential loader) order."""
    return {
        'mod1': np.asarray([s['mod1_idx'] for s in ds.samples], dtype=np.int64),
        'mod2': np.asarray([s['mod2_idx'] for s in ds.samples], dtype=np.int64),
        'ood1': np.asarray([s['mod1_is_ood'] for s in ds.samples], dtype=bool),
        'ood2': np.asarray([s['mod2_is_ood'] for s in ds.samples], dtype=bool),
    }


def get_args():
    p = argparse.ArgumentParser(description='Truth-anchored label correction')
    p.add_argument('--seeds', type=str, default='42,43,44,45,46')
    p.add_argument('--n_per_snr', type=int, default=200)
    p.add_argument('--n_per_snr_uu', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--smoke', action='store_true',
                   help='1 seed, n_per_snr=8; writes truth_anchor_smoke.json')
    p.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    return p.parse_args()


def main():
    args = get_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    n_per_snr, n_uu = args.n_per_snr, args.n_per_snr_uu
    if args.smoke:
        seeds, n_per_snr, n_uu = seeds[:1], 8, 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[eval_truth_anchor] seeds={seeds} n={n_per_snr}/{n_uu} device={device}')

    loaders, ds = build_loaders_with_datasets(n_per_snr, n_uu, args.batch_size)
    truth = {p: truth_labels(d) for p, d in ds.items()}

    per_seed = {}
    for seed in seeds:
        ckpt = os.path.join(C.CHECKPOINT_DIR, BASE.format(seed) + '.pt')
        print(f'\n=== seed {seed} ===')
        model = _build_model_from_ckpt(ckpt, device)
        rp = np.load(os.path.join(C.RESULTS_DIR, BASE.format(seed) + '_refpool.npz'))
        ref_emb, ref_mods = rp['ref_emb'], rp['ref_mods']

        pred, odin, sc = {}, {}, {}
        for p in ('kk', 'ku', 'uu'):
            pred[p] = _collect_predictions(model, loaders[p], device)
            odin[p] = collect_odin(model, loaders[p], device, ODIN_T, ODIN_EPS)
            sc[p] = score_slots(pred[p], odin[p], ref_emb, ref_mods)
            # dataset order == loader order (sequential); lengths must match
            assert len(truth[p]['mod1']) == len(pred[p]['snr'])

        # ---- swap rate + label diagnostics (ku: ood flags differ per side) --
        swap_ku = (pred['ku']['is_ood_1'] != truth['ku']['ood1'])
        swap_kk = (pred['kk']['mod1_idx'] != truth['kk']['mod1'])  # only visible
        # when mod1 != mod2; restricted below
        diff_mod = truth['kk']['mod1'] != truth['kk']['mod2']
        swap_kk_rate = float(swap_kk[diff_mod].mean()) if diff_mod.any() else float('nan')

        # ---- closed-set accuracy: stored (label-swap) vs truth-anchored ----
        pr1 = pred['kk']['logits_1'].argmax(-1)
        pr2 = pred['kk']['logits_2'].argmax(-1)
        acc_stored = float(np.mean(np.concatenate([
            pr1 == pred['kk']['mod1_idx'], pr2 == pred['kk']['mod2_idx']])))
        acc_truth = float(np.mean(np.concatenate([
            pr1 == truth['kk']['mod1'], pr2 == truth['kk']['mod2']])))

        # ---- truth-anchored pools ----
        snr_kk = np.tile(pred['kk']['snr'], 2)
        snr_uu = np.tile(pred['uu']['snr'], 2)
        to1, to2 = truth['ku']['ood1'], truth['ku']['ood2']
        K_kk = {m: np.concatenate([sc['kk']['1'][m], sc['kk']['2'][m]]) for m in METHODS}
        U_uu = {m: np.concatenate([sc['uu']['1'][m], sc['uu']['2'][m]]) for m in METHODS}
        U_ku = {m: np.concatenate([sc['ku']['1'][m][to1], sc['ku']['2'][m][to2]])
                for m in METHODS}
        K_ku = {m: np.concatenate([sc['ku']['1'][m][~to1], sc['ku']['2'][m][~to2]])
                for m in METHODS}
        snr_uku = np.concatenate([pred['ku']['snr'][to1], pred['ku']['snr'][to2]])
        snr_kku = np.concatenate([pred['ku']['snr'][~to1], pred['ku']['snr'][~to2]])
        U_all = {m: np.concatenate([U_ku[m], U_uu[m]]) for m in METHODS}
        snr_uall = np.concatenate([snr_uku, snr_uu])

        comps = {
            'kkK_vs_kuU': eval_comparison(K_kk, snr_kk, U_ku, snr_uku),
            'kkK_vs_uuU': eval_comparison(K_kk, snr_kk, U_uu, snr_uu),
            'kuK_vs_kuU': eval_comparison(K_ku, snr_kku, U_ku, snr_uku),
            'combined': eval_comparison(K_kk, snr_kk, U_all, snr_uall),
        }
        per_seed[str(seed)] = {
            'swap_rate_ku': float(swap_ku.mean()),
            'swap_rate_kk_diffmod': swap_kk_rate,
            'cls_acc_stored_labels': acc_stored,
            'cls_acc_truth_anchored': acc_truth,
            'comparisons': comps,
        }
        print(f'  swap rate: ku={swap_ku.mean():.3f} kk(diff-mod)={swap_kk_rate:.3f}')
        print(f'  cls_acc: stored={acc_stored:.4f}  truth-anchored={acc_truth:.4f}')
        print('  truth-anchored combined wavg: ' + '  '.join(
            f'{m}={comps["combined"]["wavg"][m]:.4f}' for m in METHODS)
            + f'  routed={comps["combined"]["wavg"]["routed"]:.4f}')

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---- across-seed aggregation ----
    comp_names = ['kkK_vs_kuU', 'kkK_vs_uuU', 'kuK_vs_kuU', 'combined']
    across = {}
    for cname in comp_names:
        agg = {'wavg': {}, 'pooled': {}, 'per_bin': {}}
        for m in METHODS + ['routed']:
            agg['wavg'][m] = _ms([per_seed[s]['comparisons'][cname]['wavg'][m]
                                  for s in per_seed])
        for m in METHODS:
            agg['pooled'][m] = _ms([per_seed[s]['comparisons'][cname]['pooled'][m]
                                    for s in per_seed])
        for s in BINS:
            key = str(int(s))
            agg['per_bin'][key] = {
                m: _ms([per_seed[sd]['comparisons'][cname]['per_bin'][key][m]
                        for sd in per_seed])
                for m in METHODS + ['routed']}
        across[cname] = agg
    diag = {
        'swap_rate_ku': _ms([per_seed[s]['swap_rate_ku'] for s in per_seed]),
        'swap_rate_kk_diffmod': _ms([per_seed[s]['swap_rate_kk_diffmod']
                                     for s in per_seed]),
        'cls_acc_stored_labels': _ms([per_seed[s]['cls_acc_stored_labels']
                                      for s in per_seed]),
        'cls_acc_truth_anchored': _ms([per_seed[s]['cls_acc_truth_anchored']
                                       for s in per_seed]),
    }

    out = {
        'experiment': 'truth-anchored PIT label correction (follow-up to a '
                      'second label-alignment bug found in '
                      'evaluate._collect_predictions during review-2 checks)',
        'bug': 'content aligned to truth but labels swapped with the output '
               'slot -> ku OOD pool contaminated at swapped samples; kk/uu '
               'pools, refpool, and SNR labels unaffected',
        'config': {'seeds': seeds, 'n_per_snr': n_per_snr,
                   'n_per_snr_uu': n_uu, 'test_seed': 99999,
                   'smoke': bool(args.smoke),
                   'scorer_fit': 'refpool npz (kk, seed 88888); odin '
                                 'recomputed (eps=0.005, T=1000)'},
        'per_seed': per_seed,
        'across_seeds': across,
        'diagnostics': diag,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    fname = 'truth_anchor_smoke.json' if args.smoke else 'truth_anchor.json'
    fpath = os.path.join(args.out_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(_round(out), f, indent=1)
    print(f'\nSaved {fpath}')

    print('\n=== ACROSS SEEDS: truth-anchored combined (headline) wavg ===')
    for m in METHODS + ['routed']:
        a = across['combined']['wavg'][m]
        print(f'  {m:>11}: {a["mean"]:.4f} ± {a["std"]:.4f}')


if __name__ == '__main__':
    main()
