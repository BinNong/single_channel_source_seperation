"""
Paper 3 — Truth-anchored standard dumps (review-2 follow-up).

Regenerates the standard per-seed dump artifacts with TRUTH-ANCHORED pool
membership (see EXPERIMENT_LOG.md 2026-09-30: evaluate._collect_predictions
swaps per-source labels along with the PIT content swap; the src1-aligned
slot's true label is always the dataset's mod1/ood1).  Contents are aligned
correctly by _collect_predictions; ONLY the pool membership changes.

Protocol: test seed 99999, n_per_snr=192 (kk, ku), n_per_snr_uu=96 — i.e.
exactly 192/192/96 mixtures per bin, the letter's documented protocol
(identical samples to the n_per_snr=200/100 runs: n_per_pair is 12/6/6 in
both cases, same seeded RNG stream).

Per seed (42-46), writes:

  results/{base}_ood_scores_ta.npz          — same keys/schema as
        {base}_ood_scores.npz: energy/prototype/vos _score_known/_unknown,
        known/unknown_emb, known/unknown_logits, prototypes, known_mods,
        unknown_mods, known_snr (np.tile-corrected), unknown_snr, snr_kk.
        Known pool = kk both slots (content PIT-anchored; labels truth).
        Unknown pool = ku TRUE-OOD slot + uu both slots, evaluate.py
        ordering: ku entries first, then uu slot1, then uu slot2.
        Stored prototype/vos scores follow evaluate.py's convention
        (prototypes fit on the dump's own known pool) but with TA labels.
  results/{base}_odin_eps0.005_T1000_ta.npz — ODIN recomputed per slot
        (odin_dump.collect_odin, same eps/T), TA pools, same key schema.
  results/{base}_refpool_ta.npz             — the seed-88888 kk reference
        pool with corrected labels (ref_emb is content-aligned and must be
        bit-close to the stored refpool npz; only ref_mods changes).

Sanity checks (printed + stored in results/ta_dumps_sanity.json):
  (a) known-pool arrays vs the existing corrected dumps: known_emb /
      known_logits / energy_score_known must be bit-close (kk content is
      unaffected by the label bug); prototype/vos known scores may differ
      ONLY through the old dump's swapped-label prototype fit — verified
      by recomputing with swapped-label prototypes (must be bit-close).
  (b) pool sizes 2688/2688; per-bin counts 384/384.
  (c) per-bin known modulation multiset exactly 96/96/96/96.
  (d) combined-pool six-scorer wavg AUROC (revision2Tables-style,
      refpool-fitted) vs results/truth_anchor.json combined wavg within
      ±0.005 (same underlying data; the 200/100 vs 192/96 protocol
      difference is nil — identical n_per_pair).

Usage:  python dump_ta_scores.py [--seeds 42,43,44,45,46] [--smoke]
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

BASE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed{}_best'
ODIN_EPS, ODIN_T = 0.005, 1000.0
BINS = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
METHODS = ['energy', 'msp', 'odin', 'mahalanobis', 'prototype', 'vos']
SANITY_TOL = 0.005


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
        'ref': CommBSSOpenSetTestDataset(           # refpool_dump.py's build
            n_per_snr=200, snr_points=C.SNR_TEST_POINTS,
            signal_length=C.SIGNAL_LENGTH, sample_rate=C.SAMPLE_RATE,
            seed=88888, protocol='kk'),
    }
    loaders = {p: DataLoader(d, batch_size=batch_size, num_workers=2)
               for p, d in ds.items()}
    return loaders, ds


def truth(ds):
    return {'mod1': np.asarray([s['mod1_idx'] for s in ds.samples], dtype=np.int64),
            'mod2': np.asarray([s['mod2_idx'] for s in ds.samples], dtype=np.int64),
            'ood1': np.asarray([s['mod1_is_ood'] for s in ds.samples], dtype=bool),
            'ood2': np.asarray([s['mod2_is_ood'] for s in ds.samples], dtype=bool)}


def _vos(emb, protos, chunk=1024):
    return np.concatenate([
        vos_score(emb[i:i + chunk], protos, alpha=C.VOS_ALPHA,
                  n_per_class=C.VOS_N_SYNTHETIC, seed=0)
        for i in range(0, len(emb), chunk)])


def combined_wavg(k_sc, k_snr, u_sc, u_snr):
    """revision2_tables-style pair-weighted wavg of the six scorers + routed."""
    per_bin = {}
    for s in BINS:
        mk, mu = k_snr == s, u_snr == s
        aucs = {m: auroc(k_sc[m][mk], u_sc[m][mu]) for m in METHODS}
        per_bin[s] = {'w': int(mk.sum()) * int(mu.sum()), **aucs,
                      'routed': aucs['energy' if s <= 0 else 'prototype']}
    out = {}
    for m in METHODS + ['routed']:
        out[m] = sum(v[m] * v['w'] for v in per_bin.values()) / \
                 sum(v['w'] for v in per_bin.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default='42,43,44,45,46')
    ap.add_argument('--n_per_snr', type=int, default=192)
    ap.add_argument('--n_per_snr_uu', type=int, default=96)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--smoke', action='store_true',
                   help='1 seed, n=8/8, no dumps written (sanity dry-run)')
    ap.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    if args.smoke:
        seeds = seeds[:1]
        args.n_per_snr, args.n_per_snr_uu = 8, 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[dump_ta_scores] seeds={seeds} n={args.n_per_snr}/{args.n_per_snr_uu} '
          f'device={device} smoke={args.smoke}')

    loaders, ds = build_loaders(args.n_per_snr, args.n_per_snr_uu, args.batch_size)
    tr = {p: truth(d) for p, d in ds.items()}
    print('sizes:', {p: len(d) for p, d in ds.items()})

    # truth_anchor.json combined wavg = sanity-(d) target (200/100 protocol,
    # identical samples to 192/96)
    ta_path = os.path.join(C.RESULTS_DIR, 'truth_anchor.json')
    ta_target = None
    if os.path.exists(ta_path) and not args.smoke:
        ta = json.load(open(ta_path))
        ta_target = {m: ta['across_seeds']['combined']['wavg'][m]['mean']
                     for m in METHODS + ['routed']}
        ta_per_seed = {s: ta['per_seed'][s]['comparisons']['combined']['wavg']
                       for s in ta['per_seed']}

    sanity = {}
    for seed in seeds:
        base = BASE.format(seed)
        print(f'\n=== seed {seed} ===')
        model = _build_model_from_ckpt(
            os.path.join(C.CHECKPOINT_DIR, base + '.pt'), device)
        pred, odin = {}, {}
        for p in ('kk', 'ku', 'uu', 'ref'):
            pred[p] = _collect_predictions(model, loaders[p], device)
            if p != 'ref':
                odin[p] = collect_odin(model, loaders[p], device, ODIN_T, ODIN_EPS)

        # ---------------- TA refpool ----------------
        ref_emb = np.concatenate([pred['ref']['emb_1'], pred['ref']['emb_2']])
        ref_mods_ta = np.concatenate([tr['ref']['mod1'], tr['ref']['mod2']])
        ref_snr = np.tile(pred['ref']['snr'], 2)
        ref_check = {}
        old_rp_path = os.path.join(C.RESULTS_DIR, base + '_refpool.npz')
        if os.path.exists(old_rp_path):
            old_rp = np.load(old_rp_path)
            ref_check['ref_emb_maxdiff_vs_stored'] = float(
                np.abs(ref_emb - old_rp['ref_emb']).max())
            ref_check['ref_snr_equal'] = bool(np.array_equal(ref_snr,
                                                             old_rp['ref_snr']))
            ref_check['ref_mods_changed_frac'] = float(
                (ref_mods_ta != old_rp['ref_mods']).mean())
            print(f"  refpool: emb max|diff|={ref_check['ref_emb_maxdiff_vs_stored']:.2e}, "
                  f"labels corrected in {ref_check['ref_mods_changed_frac']:.3f} of slots")
        if not args.smoke:
            np.savez(os.path.join(args.out_dir, base + '_refpool_ta.npz'),
                     ref_emb=ref_emb.astype(np.float32),
                     ref_mods=ref_mods_ta.astype(np.int64),
                     ref_snr=ref_snr.astype(np.float32))

        # ---------------- TA pools ----------------
        known_emb = np.concatenate([pred['kk']['emb_1'], pred['kk']['emb_2']])
        known_logits = np.concatenate([pred['kk']['logits_1'], pred['kk']['logits_2']])
        known_mods = np.concatenate([tr['kk']['mod1'], tr['kk']['mod2']])
        known_snr = np.tile(pred['kk']['snr'], 2)

        to1, to2 = tr['ku']['ood1'], tr['ku']['ood2']
        unknown_emb = np.concatenate([
            pred['ku']['emb_1'][to1], pred['ku']['emb_2'][to2],
            pred['uu']['emb_1'], pred['uu']['emb_2']])
        unknown_logits = np.concatenate([
            pred['ku']['logits_1'][to1], pred['ku']['logits_2'][to2],
            pred['uu']['logits_1'], pred['uu']['logits_2']])
        unknown_mods = np.concatenate([
            tr['ku']['mod1'][to1], tr['ku']['mod2'][to2],
            tr['uu']['mod1'], tr['uu']['mod2']])
        unknown_snr = np.concatenate([
            pred['ku']['snr'][to1], pred['ku']['snr'][to2],
            pred['uu']['snr'], pred['uu']['snr']])

        protos_ta = compute_prototypes(known_emb, known_mods, C.NUM_KNOWN_CLASSES)
        scores_known = {
            'energy': energy_score(known_logits),
            'prototype': prototype_score(known_emb, protos_ta),
            'vos': _vos(known_emb, protos_ta),
        }
        scores_unknown = {
            'energy': energy_score(unknown_logits),
            'prototype': prototype_score(unknown_emb, protos_ta),
            'vos': _vos(unknown_emb, protos_ta),
        }

        # ---------------- sanity (a): known-side vs existing dump --------
        a = {}
        old_path = os.path.join(C.RESULTS_DIR, base + '_ood_scores.npz')
        if os.path.exists(old_path) and not args.smoke:
            old = np.load(old_path)
            a['known_emb_maxdiff'] = float(np.abs(known_emb - old['known_emb']).max())
            a['known_logits_maxdiff'] = float(np.abs(known_logits - old['known_logits']).max())
            a['energy_known_maxdiff'] = float(np.abs(
                scores_known['energy'] - old['energy_score_known']).max())
            # old stored prototype/vos used SWAPPED-label prototypes:
            protos_sw = compute_prototypes(old['known_emb'], old['known_mods'],
                                           C.NUM_KNOWN_CLASSES)
            a['prototype_known_maxdiff_if_swapped_fit'] = float(np.abs(
                prototype_score(known_emb, protos_sw)
                - old['prototype_score_known']).max())
            a['vos_known_maxdiff_if_swapped_fit'] = float(np.abs(
                _vos(known_emb, protos_sw) - old['vos_score_known']).max())
            a['prototype_known_maxdiff_ta_fit'] = float(np.abs(
                scores_known['prototype'] - old['prototype_score_known']).max())
            a['vos_known_maxdiff_ta_fit'] = float(np.abs(
                scores_known['vos'] - old['vos_score_known']).max())
            print(f"  (a) emb/logits/energy max|diff|: "
                  f"{a['known_emb_maxdiff']:.2e} / {a['known_logits_maxdiff']:.2e} / "
                  f"{a['energy_known_maxdiff']:.2e};  proto/vos with swapped-fit "
                  f"reproduction: {a['prototype_known_maxdiff_if_swapped_fit']:.2e} / "
                  f"{a['vos_known_maxdiff_if_swapped_fit']:.2e} (TA-fit diff: "
                  f"{a['prototype_known_maxdiff_ta_fit']:.3f} / "
                  f"{a['vos_known_maxdiff_ta_fit']:.3f})")

        # ---------------- sanity (b)/(c): counts --------------------------
        b_ok = len(known_emb) == 2688 and len(unknown_emb) == 2688
        counts = {}
        for s in BINS:
            nk = int((known_snr == s).sum())
            nu = int((unknown_snr == s).sum())
            mods = [int(((known_mods == k) & (known_snr == s)).sum())
                    for k in range(C.NUM_KNOWN_CLASSES)]
            counts[str(int(s))] = {'n_known': nk, 'n_unknown': nu,
                                   'known_mod_counts': mods}
            b_ok &= (nk == 384 and nu == 384)
        c_ok = all(v['known_mod_counts'] == [96, 96, 96, 96]
                   for v in counts.values())
        print(f'  (b) 384/384 per bin: {b_ok};  (c) 96/96/96/96 mods per bin: {c_ok}')

        # ---------------- write dumps ------------------------------------
        if not args.smoke:
            np.savez(os.path.join(args.out_dir, base + '_ood_scores_ta.npz'),
                     energy_score_known=scores_known['energy'],
                     energy_score_unknown=scores_unknown['energy'],
                     prototype_score_known=scores_known['prototype'],
                     prototype_score_unknown=scores_unknown['prototype'],
                     vos_score_known=scores_known['vos'],
                     vos_score_unknown=scores_unknown['vos'],
                     known_emb=known_emb.astype(np.float32),
                     unknown_emb=unknown_emb.astype(np.float32),
                     known_logits=known_logits.astype(np.float32),
                     unknown_logits=unknown_logits.astype(np.float32),
                     prototypes=protos_ta,
                     known_mods=known_mods.astype(np.int64),
                     unknown_mods=unknown_mods.astype(np.int64),
                     known_snr=known_snr.astype(np.float32),
                     unknown_snr=unknown_snr.astype(np.float32),
                     snr_kk=pred['kk']['snr'].astype(np.float32))
            od_known = np.concatenate([odin['kk']['odin_1'], odin['kk']['odin_2']])
            od_unknown = np.concatenate([
                odin['ku']['odin_1'][to1], odin['ku']['odin_2'][to2],
                odin['uu']['odin_1'], odin['uu']['odin_2']])
            np.savez(os.path.join(args.out_dir,
                                  base + f'_odin_eps{ODIN_EPS:g}_T{ODIN_T:g}_ta.npz'),
                     odin_score_known=od_known.astype(np.float32),
                     odin_score_unknown=od_unknown.astype(np.float32),
                     known_snr=known_snr.astype(np.float32),
                     unknown_snr=unknown_snr.astype(np.float32))

        # ---------------- sanity (d): combined wavg vs truth_anchor.json --
        d_res = {}
        if ta_target is not None:
            for fit_name, (re_emb, re_mods) in {
                    'swapped_refpool': (old_rp['ref_emb'], old_rp['ref_mods']),
                    'ta_refpool': (ref_emb, ref_mods_ta)}.items():
                protos_fit = compute_prototypes(re_emb, re_mods, C.NUM_KNOWN_CLASSES)
                k_sc = {
                    'energy': energy_score(known_logits),
                    'msp': msp_scores(known_logits),
                    'odin': od_known.astype(np.float64),
                    'mahalanobis': mahalanobis_scores(re_emb, re_mods, known_emb,
                                                      n_classes=4, shrink=0.1),
                    'prototype': prototype_score(known_emb, protos_fit),
                    'vos': _vos(known_emb, protos_fit),
                }
                u_sc = {
                    'energy': energy_score(unknown_logits),
                    'msp': msp_scores(unknown_logits),
                    'odin': od_unknown.astype(np.float64),
                    'mahalanobis': mahalanobis_scores(re_emb, re_mods, unknown_emb,
                                                      n_classes=4, shrink=0.1),
                    'prototype': prototype_score(unknown_emb, protos_fit),
                    'vos': _vos(unknown_emb, protos_fit),
                }
                d_res[fit_name] = combined_wavg(k_sc, known_snr, u_sc, unknown_snr)
            devs = {m: abs(d_res['swapped_refpool'][m] - ta_per_seed[str(seed)][m])
                    for m in METHODS + ['routed']}
            d_ok = all(v <= SANITY_TOL for v in devs.values())
            print('  (d) combined wavg (swapped-refpool fit) vs truth_anchor.json '
                  f'seed {seed}: max|dev|={max(devs.values()):.5f} -> {"OK" if d_ok else "FAIL"}')
            print('      (d) same with TA-refpool fit: ' + '  '.join(
                f'{m}={d_res["ta_refpool"][m]:.4f}' for m in METHODS + ['routed']))

        sanity[str(seed)] = {'a': a, 'b_counts_ok': bool(b_ok),
                             'c_mods_ok': bool(c_ok), 'counts': counts,
                             'refpool': ref_check, 'd': d_res,
                             'd_target_per_seed': ta_per_seed.get(str(seed))
                             if ta_target else None}
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    spath = os.path.join(args.out_dir,
                         'ta_dumps_sanity_smoke.json' if args.smoke
                         else 'ta_dumps_sanity.json')
    with open(spath, 'w') as f:
        json.dump(sanity, f, indent=1,
                  default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f'\nSaved {spath}')
    if not args.smoke:
        # across-seed TA-refpool combined headline (definitive corrected number)
        if ta_target is not None:
            agg = {m: {'mean': float(np.mean([sanity[s]['d']['ta_refpool'][m]
                                              for s in sanity])),
                       'std': float(np.std([sanity[s]['d']['ta_refpool'][m]
                                            for s in sanity]))}
                   for m in METHODS + ['routed']}
            print('\n=== TA combined wavg, TA-refpool fit (5-seed mean±std) ===')
            for m in METHODS + ['routed']:
                print(f'  {m:>11}: {agg[m]["mean"]:.4f} ± {agg[m]["std"]:.4f}')


if __name__ == '__main__':
    main()
