"""
Paper 3 — Revision experiment: separation quality, known vs unknown sources.

Question: does the frozen OpenSetCSE separator recover KNOWN-modulation
sources better than UNKNOWN-modulation sources?  For every seed checkpoint
we run the exact evaluate.py test pipeline (kk / ku / uu protocols,
deterministic test seed 99999) and tabulate mean SI-SDR (dB) per cell:

    protocol {kk, ku, uu}  x  SNR {-10,-5,0,5,10,15,20}  x  {known, unknown}

"knownness" uses the PIT-aligned is_ood_1/is_ood_2 flags returned by
evaluate._collect_predictions: kk → both sources known, uu → both unknown,
ku → one of each.  We also report pooled-per-protocol means, the pooled
overall known-vs-unknown means, and per-SNR known-vs-unknown means pooled
across protocols.

Sanity: the kk pooled SI-SDR (both slots) must reproduce the closed-set
si_sdr_mean stored in the existing per-seed summary json
(results/openset_..._seedXX_best_summary.json, ≈ −0.9 dB).  Deviations
> 0.2 dB are flagged in the output and on stdout (check is skipped in
--smoke mode, whose tiny cells need not match the 200-sample summaries).

Note: the three test datasets are deterministic (fixed seed 99999), so they
are constructed ONCE and the same loaders are reused for every seed
checkpoint — rebuilding them per seed would produce bit-identical data.

Outputs
-------
  results/sep_quality.json         (full run)
  results/sep_quality_smoke.json   (--smoke)

Usage
-----
  python eval_sep_quality.py --smoke                 # 1 seed, n_per_snr=8
  python eval_sep_quality.py                         # 5 seeds, full cells
  python eval_sep_quality.py --seeds 42,43
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

PROTOCOLS = ('kk', 'ku', 'uu')
SNR_POINTS = [int(s) for s in C.SNR_TEST_POINTS]
CKPT_TEMPLATE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed{seed}_best.pt'
SUMMARY_TEMPLATE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed{seed}_best_summary.json'
SANITY_TOL_DB = 0.2


# ----------------------------------------------------------------------------
# Dataset construction — EXACTLY as evaluate.py main() (default carrier gap)
# ----------------------------------------------------------------------------
def build_loaders(n_per_snr: int, n_per_snr_uu: int, batch_size: int):
    common = dict(
        n_per_snr=n_per_snr,
        snr_points=C.SNR_TEST_POINTS,
        signal_length=C.SIGNAL_LENGTH,
        sample_rate=C.SAMPLE_RATE,
        seed=99999,
    )
    ds_kk = CommBSSOpenSetTestDataset(protocol='kk', **common)
    ds_ku = CommBSSOpenSetTestDataset(protocol='ku', **common)
    ds_uu = CommBSSOpenSetTestDataset(
        n_per_snr=n_per_snr_uu, snr_points=C.SNR_TEST_POINTS,
        signal_length=C.SIGNAL_LENGTH, sample_rate=C.SAMPLE_RATE,
        seed=99999, protocol='uu', carrier_freq_2=2005.0,
    )
    loaders = {
        'kk': DataLoader(ds_kk, batch_size=batch_size, num_workers=2),
        'ku': DataLoader(ds_ku, batch_size=batch_size, num_workers=2),
        'uu': DataLoader(ds_uu, batch_size=batch_size, num_workers=2),
    }
    sizes = {p: len(d) for p, d in (('kk', ds_kk), ('ku', ds_ku), ('uu', ds_uu))}
    return loaders, sizes


# ----------------------------------------------------------------------------
# Cell bookkeeping
# ----------------------------------------------------------------------------
def _flat(pred: dict):
    """Flatten the two PIT-aligned slots into (si_sdr, is_unknown, snr).

    The concatenation order (all slot-1 entries, then all slot-2) matches
    evaluate._collect_predictions, so the per-sample snr labels must be
    np.tile'd — NOT np.repeat'd (the 2026-09 per-source SNR-labeling bug).
    """
    si_sdr = np.concatenate([pred['si_sdr_1'], pred['si_sdr_2']])
    is_unknown = np.concatenate([pred['is_ood_1'], pred['is_ood_2']])
    snr = np.tile(pred['snr'], 2)
    return si_sdr, is_unknown, snr


def _mean(x: np.ndarray):
    return float(np.mean(x)) if x.size else None


def per_seed_metrics(preds: dict) -> dict:
    flats = {p: _flat(preds[p]) for p in PROTOCOLS}

    cells = {}
    for p in PROTOCOLS:
        si, unk, snr = flats[p]
        cells[p] = {}
        for s in SNR_POINTS:
            m = snr == s
            cells[p][str(s)] = {
                'known': _mean(si[m & ~unk]),
                'unknown': _mean(si[m & unk]),
                'n_known': int((m & ~unk).sum()),
                'n_unknown': int((m & unk).sum()),
            }

    pooled_per_protocol = {
        p: {'known': _mean(flats[p][0][~flats[p][1]]),
            'unknown': _mean(flats[p][0][flats[p][1]])}
        for p in PROTOCOLS
    }

    si_all = np.concatenate([flats[p][0] for p in PROTOCOLS])
    unk_all = np.concatenate([flats[p][1] for p in PROTOCOLS])
    snr_all = np.concatenate([flats[p][2] for p in PROTOCOLS])
    pooled_overall = {'known': _mean(si_all[~unk_all]),
                      'unknown': _mean(si_all[unk_all])}

    per_snr_known_unknown = {}
    for s in SNR_POINTS:
        m = snr_all == s
        per_snr_known_unknown[str(s)] = {
            'known': _mean(si_all[m & ~unk_all]),
            'unknown': _mean(si_all[m & unk_all]),
            'n_known': int((m & ~unk_all).sum()),
            'n_unknown': int((m & unk_all).sum()),
        }

    return {
        'cells': cells,
        'pooled_per_protocol': pooled_per_protocol,
        'pooled_overall': pooled_overall,
        'per_snr_known_unknown': per_snr_known_unknown,
        'kk_pooled_si_sdr': _mean(flats['kk'][0]),
    }


# ----------------------------------------------------------------------------
# Across-seed aggregation (mean/std over seeds; None cells stay None)
# ----------------------------------------------------------------------------
def _agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
            'n_seeds': len(vals)}


def aggregate(per_seed: dict) -> dict:
    seeds = list(per_seed.keys())
    cells = {}
    for p in PROTOCOLS:
        cells[p] = {}
        for s in SNR_POINTS:
            s = str(s)
            cells[p][s] = {
                'known': _agg([per_seed[sd]['cells'][p][s]['known'] for sd in seeds]),
                'unknown': _agg([per_seed[sd]['cells'][p][s]['unknown'] for sd in seeds]),
            }
    pooled_per_protocol = {
        p: {k: _agg([per_seed[sd]['pooled_per_protocol'][p][k] for sd in seeds])
            for k in ('known', 'unknown')}
        for p in PROTOCOLS
    }
    pooled_overall = {
        k: _agg([per_seed[sd]['pooled_overall'][k] for sd in seeds])
        for k in ('known', 'unknown')
    }
    per_snr_known_unknown = {}
    for s in SNR_POINTS:
        s = str(s)
        per_snr_known_unknown[s] = {
            k: _agg([per_seed[sd]['per_snr_known_unknown'][s][k] for sd in seeds])
            for k in ('known', 'unknown')
        }
    return {
        'cells': cells,
        'pooled_per_protocol': pooled_per_protocol,
        'pooled_overall': pooled_overall,
        'per_snr_known_unknown': per_snr_known_unknown,
        'kk_pooled_si_sdr': _agg([per_seed[sd]['kk_pooled_si_sdr'] for sd in seeds]),
        'sanity_all_passed': all(
            per_seed[sd].get('sanity', {}).get('passed', True) for sd in seeds),
    }


def _round(obj):
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 3)
    return obj


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description='Separation quality: known vs unknown')
    p.add_argument('--seeds', type=str, default='42,43,44,45,46')
    p.add_argument('--n_per_snr', type=int, default=200)
    p.add_argument('--n_per_snr_uu', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--smoke', action='store_true',
                   help='Tiny cells (n_per_snr=8, n_per_snr_uu=8), first seed only; '
                        'writes sep_quality_smoke.json and skips the summary-json '
                        'sanity check.')
    p.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    return p.parse_args()


def main():
    args = get_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    n_per_snr, n_per_snr_uu = args.n_per_snr, args.n_per_snr_uu
    if args.smoke:
        seeds = seeds[:1]
        n_per_snr, n_per_snr_uu = 8, 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'[eval_sep_quality] seeds={seeds} n_per_snr={n_per_snr} '
          f'n_per_snr_uu={n_per_snr_uu} device={device}')
    print('Building deterministic test sets (seed 99999) ...')
    loaders, sizes = build_loaders(n_per_snr, n_per_snr_uu, args.batch_size)
    print(f'  sizes: {sizes}')

    per_seed = {}
    for seed in seeds:
        ckpt = os.path.join(C.CHECKPOINT_DIR, CKPT_TEMPLATE.format(seed=seed))
        print(f'\n=== seed {seed}: {os.path.basename(ckpt)} ===')
        model = _build_model_from_ckpt(ckpt, device)
        preds = {p: _collect_predictions(model, loaders[p], device)
                 for p in PROTOCOLS}
        m = per_seed_metrics(preds)

        # ---- sanity vs existing closed-set summary json (full runs only) ----
        sanity = {'checked': False}
        if not args.smoke:
            spath = os.path.join(C.RESULTS_DIR, SUMMARY_TEMPLATE.format(seed=seed))
            if os.path.exists(spath):
                with open(spath) as f:
                    ref = json.load(f)['closed']['si_sdr_mean']
                dev = abs(m['kk_pooled_si_sdr'] - ref)
                sanity = {'checked': True, 'kk_pooled': m['kk_pooled_si_sdr'],
                          'summary_json': ref, 'abs_deviation': dev,
                          'passed': bool(dev <= SANITY_TOL_DB)}
                tag = 'OK' if sanity['passed'] else 'FAIL'
                print(f'  sanity kk pooled SI-SDR = {m["kk_pooled_si_sdr"]:.3f} dB '
                      f'vs summary {ref:.3f} dB (|dev|={dev:.3f}) [{tag}]')
            else:
                print(f'  [sanity] summary json not found: {spath} — skipped')
        m['sanity'] = sanity
        per_seed[str(seed)] = m

        print(f'  pooled overall: known={m["pooled_overall"]["known"]:.3f} dB  '
              f'unknown={m["pooled_overall"]["unknown"]:.3f} dB')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    out = {
        'experiment': 'sep_quality_known_vs_unknown',
        'config': {'seeds': seeds, 'n_per_snr': n_per_snr,
                   'n_per_snr_uu': n_per_snr_uu, 'test_seed': 99999,
                   'snr_points': SNR_POINTS, 'smoke': bool(args.smoke)},
        'dataset_sizes': sizes,
        'per_seed': per_seed,
        'across_seeds': aggregate(per_seed),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    fname = 'sep_quality_smoke.json' if args.smoke else 'sep_quality.json'
    fpath = os.path.join(args.out_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(_round(out), f, indent=2)
    print(f'\nSaved {fpath}')

    agg = out['across_seeds']
    print('\n=== Across-seed pooled known vs unknown SI-SDR (dB) ===')
    for k in ('known', 'unknown'):
        a = agg['pooled_overall'][k]
        print(f'  {k:>8s}: {a["mean"]:.3f} ± {a["std"]:.3f}')
    print('\n=== Per-SNR known vs unknown (across-seed mean, dB) ===')
    print(f'  {"SNR":>5s} | {"known":>8s} | {"unknown":>8s}')
    for s in SNR_POINTS:
        r = agg['per_snr_known_unknown'][str(s)]
        ks = f'{r["known"]["mean"]:.3f}' if r['known'] else '  n/a'
        us = f'{r["unknown"]["mean"]:.3f}' if r['unknown'] else '  n/a'
        print(f'  {s:+5d} | {ks:>8s} | {us:>8s}')
    if not args.smoke:
        print(f'\nSanity all passed: {agg["sanity_all_passed"]}')


if __name__ == '__main__':
    main()
