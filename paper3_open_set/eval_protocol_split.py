"""
Paper 3 — Review-2 experiment A: protocol-split OOD evaluation
(reviewer Major Concern 3: kk/ku/uu must be reported separately).

The headline pooled comparison mixes pool provenance: the known pool comes
entirely from kk mixtures while the unknown pool mixes ku (OOD slot) and uu
(both slots).  A scorer could therefore appear to work by detecting mixture
composition or separation difficulty rather than modulation novelty.  This
script re-runs inference on the EXACT evaluate.py test protocols
(deterministic test seed 99999, kk n_per_snr=200, ku 200, uu 100, batch 16)
and reports per-SNR-bin AUROC + pair-weighted wavg for three split
comparisons plus the combined sanity comparison:

  C1 kkK_vs_kuU : K(kk both slots)   vs U(ku OOD slot)
  C2 kkK_vs_uuU : K(kk both slots)   vs U(uu both slots)
  C3 kuK_vs_kuU : K(ku known slot)   vs U(ku OOD slot)   <-- same-mixture
  CS combined   : K(kk both slots)   vs U(ku OOD + uu both)  == headline

(The reviewer's hypothetical "K in UU vs U in UU" row is vacuous: uu has no
known sources.)

Six scorers, fitted exactly as revision2_tables.py section s1: energy/msp
from logits; odin recomputed with odin_dump.collect_odin (eps=0.005,
T=1000, same PIT alignment); prototype/vos/mahalanobis fit on the disjoint
reference pool npz (kk, seed 88888).  AUROC = open_set_metrics.auroc
(Mann-Whitney).  wavg = per-bin AUROC weighted by n_known*n_unknown
(uniform within every comparison -> arithmetic mean over the 7 bins).

Sanity (full runs only):
  * recomputed ODIN pools vs stored *_odin_eps0.005_T1000.npz (max |diff|);
  * CS across-seed wavg must reproduce the corrected baselines
    (revision2_tables.json s1) within +-0.002:
    energy 0.4755, msp 0.4306, odin 0.4798, mahalanobis 0.5203,
    prototype 0.5055, vos 0.5056, routed 0.4984.

Label convention: slot arrays are stacked slot1-block then slot2-block, so
per-slot SNR labels are np.tile(snr, 2) — never np.repeat (the fixed bug).

Outputs: results/protocol_split.json (smoke: protocol_split_smoke.json).

Usage:
  python eval_protocol_split.py --smoke
  python eval_protocol_split.py                      # 5 seeds, full
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
# revision2_tables.json s1 (corrected, refpool-fitted) across-seed wavg means
BASELINE_WAVG = {'energy': 0.4755, 'msp': 0.4306, 'odin': 0.4798,
                 'mahalanobis': 0.5203, 'prototype': 0.5055, 'vos': 0.5056,
                 'routed': 0.4984}
SANITY_TOL = 0.002


def build_loaders(n_per_snr, n_per_snr_uu, batch_size):
    """EXACTLY evaluate.py main()'s dataset construction (test seed 99999)."""
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
    return loaders, {p: len(d) for p, d in ds.items()}


def _vos(emb, protos, chunk=1024):
    return np.concatenate([
        vos_score(emb[i:i + chunk], protos, alpha=C.VOS_ALPHA,
                  n_per_class=C.VOS_N_SYNTHETIC, seed=0)
        for i in range(0, len(emb), chunk)])


def score_slots(pred, odin, ref_emb, ref_mods):
    """Six scorers for BOTH PIT-aligned slots of one protocol."""
    protos = compute_prototypes(ref_emb, ref_mods, C.NUM_KNOWN_CLASSES)
    out = {}
    for slot in ('1', '2'):
        emb, logits = pred[f'emb_{slot}'], pred[f'logits_{slot}']
        out[slot] = {
            'energy': energy_score(logits),
            'msp': msp_scores(logits),
            'odin': np.asarray(odin[f'odin_{slot}'], dtype=np.float64),
            'mahalanobis': mahalanobis_scores(ref_emb, ref_mods, emb,
                                              n_classes=C.NUM_KNOWN_CLASSES,
                                              shrink=0.1),
            'prototype': prototype_score(emb, protos),
            'vos': _vos(emb, protos),
        }
    return out


def _stack(sc, m1, m2):
    """Stack per-method score arrays for a slot selection.
    m1/m2: boolean masks (or None = all) applied to slot 1 / slot 2."""
    s1 = sc['1'] if m1 is None else {m: v[m1] for m, v in sc['1'].items()}
    s2 = sc['2'] if m2 is None else {m: v[m2] for m, v in sc['2'].items()}
    return {m: np.concatenate([s1[m], s2[m]]) for m in METHODS}


def eval_comparison(k_scores, k_snr, u_scores, u_snr):
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
    return {'mean': float(v.mean()), 'std': float(v.std())}


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
    p = argparse.ArgumentParser(description='Protocol-split OOD evaluation')
    p.add_argument('--seeds', type=str, default='42,43,44,45,46')
    p.add_argument('--n_per_snr', type=int, default=200)
    p.add_argument('--n_per_snr_uu', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--smoke', action='store_true',
                   help='1 seed, n_per_snr=8; writes protocol_split_smoke.json '
                        'and skips the baseline sanity check.')
    p.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    return p.parse_args()


def main():
    args = get_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    n_per_snr, n_uu = args.n_per_snr, args.n_per_snr_uu
    if args.smoke:
        seeds, n_per_snr, n_uu = seeds[:1], 8, 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[eval_protocol_split] seeds={seeds} n_per_snr={n_per_snr} '
          f'n_per_snr_uu={n_uu} device={device}')

    loaders, sizes = build_loaders(n_per_snr, n_uu, args.batch_size)
    print(f'  dataset sizes: {sizes}')

    per_seed = {}
    odin_max_diff = {}
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

        # ---- ODIN recompute check vs stored dump ----
        opath = os.path.join(C.RESULTS_DIR,
                             BASE.format(seed) + f'_odin_eps{ODIN_EPS:g}_T{ODIN_T:g}.npz')
        if os.path.exists(opath) and not args.smoke:
            o = np.load(opath)
            m1, m2 = odin['ku']['is_ood_1'], odin['ku']['is_ood_2']
            known_rec = np.concatenate([odin['kk']['odin_1'], odin['kk']['odin_2']])
            unk_rec = np.concatenate([odin['ku']['odin_1'][m1], odin['ku']['odin_2'][m2],
                                      odin['uu']['odin_1'], odin['uu']['odin_2']])
            d = max(float(np.abs(known_rec - o['odin_score_known']).max()),
                    float(np.abs(unk_rec - o['odin_score_unknown']).max()))
            odin_max_diff[str(seed)] = d
            print(f'  ODIN recompute vs stored npz: max|diff| = {d:.2e}')

        # ---- pools ----
        snr_kk = np.tile(pred['kk']['snr'], 2)          # slot1 block, slot2 block
        m1, m2 = pred['ku']['is_ood_1'], pred['ku']['is_ood_2']
        K_kk = _stack(sc['kk'], None, None)
        U_uu = _stack(sc['uu'], None, None)
        U_ku = _stack(sc['ku'], m1, m2)
        K_ku = _stack(sc['ku'], ~m1, ~m2)
        snr_uu = np.tile(pred['uu']['snr'], 2)
        snr_uku = np.concatenate([pred['ku']['snr'][m1], pred['ku']['snr'][m2]])
        snr_kku = np.concatenate([pred['ku']['snr'][~m1], pred['ku']['snr'][~m2]])
        U_all = {m: np.concatenate([U_ku[m], U_uu[m]]) for m in METHODS}
        snr_uall = np.concatenate([snr_uku, snr_uu])

        comps = {
            'kkK_vs_kuU': eval_comparison(K_kk, snr_kk, U_ku, snr_uku),
            'kkK_vs_uuU': eval_comparison(K_kk, snr_kk, U_uu, snr_uu),
            'kuK_vs_kuU': eval_comparison(K_ku, snr_kku, U_ku, snr_uku),
            'combined_sanity': eval_comparison(K_kk, snr_kk, U_all, snr_uall),
        }
        per_seed[str(seed)] = comps
        for cname, r in comps.items():
            print(f"  {cname:<16} wavg: " + '  '.join(
                f'{m}={r["wavg"][m]:.4f}' for m in ('mahalanobis', 'prototype', 'routed')))

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---- across-seed aggregation ----
    comp_names = ['kkK_vs_kuU', 'kkK_vs_uuU', 'kuK_vs_kuU', 'combined_sanity']
    across = {}
    for cname in comp_names:
        agg = {'wavg': {}, 'pooled': {}, 'per_bin': {}}
        for m in METHODS + ['routed']:
            agg['wavg'][m] = _ms([per_seed[s][cname]['wavg'][m] for s in per_seed])
        for m in METHODS:
            agg['pooled'][m] = _ms([per_seed[s][cname]['pooled'][m] for s in per_seed])
        for s in BINS:
            key = str(int(s))
            agg['per_bin'][key] = {
                m: _ms([per_seed[sd][cname]['per_bin'][key][m] for sd in per_seed])
                for m in METHODS + ['routed']}
        across[cname] = agg

    # ---- sanity vs corrected baselines (full runs only) ----
    sanity = {'checked': False}
    if not args.smoke:
        devs = {m: abs(across['combined_sanity']['wavg'][m]['mean'] - BASELINE_WAVG[m])
                for m in BASELINE_WAVG}
        passed = all(d <= SANITY_TOL for d in devs.values())
        sanity = {'checked': True, 'tol': SANITY_TOL, 'baseline': BASELINE_WAVG,
                  'combined_wavg_mean': {m: across['combined_sanity']['wavg'][m]['mean']
                                         for m in BASELINE_WAVG},
                  'abs_deviation': devs, 'passed': bool(passed),
                  'odin_recompute_max_abs_diff': odin_max_diff}
        print('\n=== SANITY: combined pool vs corrected baselines ===')
        for m in BASELINE_WAVG:
            print(f"  {m:>11}: {across['combined_sanity']['wavg'][m]['mean']:.4f} "
                  f"vs {BASELINE_WAVG[m]:.4f}  (|dev|={devs[m]:.4f})")
        print(f"  PASSED: {passed}")

    out = {
        'experiment': 'protocol_split_ood (review-2 Major Concern 3)',
        'config': {'seeds': seeds, 'n_per_snr': n_per_snr, 'n_per_snr_uu': n_uu,
                   'test_seed': 99999, 'snr_bins': BINS, 'smoke': bool(args.smoke),
                   'scorer_fit': 'refpool npz (kk, seed 88888); odin recomputed '
                                 f'(eps={ODIN_EPS}, T={ODIN_T:g})',
                   'note': 'uu has no known sources, so no K-in-uu row exists.'},
        'dataset_sizes': sizes,
        'per_seed': per_seed,
        'across_seeds': across,
        'sanity': sanity,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    fname = 'protocol_split_smoke.json' if args.smoke else 'protocol_split.json'
    fpath = os.path.join(args.out_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(_round(out), f, indent=1)
    print(f'\nSaved {fpath}')


if __name__ == '__main__':
    main()
