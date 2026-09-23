"""
Paper 3 — Open-Set SC-BSS: R1-6 robustness evaluation.

Added for the post-rejection revision (reviewer demand R1-6): how robust is
OOD detection (and the SNR-routed ensemble) to
  1. unequal source powers  — fixed SIR sweep {-6, -3, 0, +3, +6} dB
     (default training/test mixing uses alpha ~ U(0.4, 0.6), i.e. roughly
     balanced sources; here the SIR is forced to an exact ratio);
  2. timing offsets         — source-2 symbol grid circularly shifted by a
     uniform random integer offset in [0, sps) samples per mixture;
  3. carrier-frequency offsets — per-source carrier jitter widened from the
     default ±5 Hz to ±25 Hz.

For every condition, on the standard test seed 99999 (kk + ku protocols):
  - per-SNR AUROC profiles (Energy / Prototype / VOS),
  - pooled AUROC per scorer,
  - routed weighted-avg AUROC with the paper's fixed 0 dB boundary using the
    ground-truth SNR (no re-tuning — the point is robustness of the EXISTING
    operating point).

Runs against EXISTING baseline checkpoints (no retraining).  All generator
knobs are opt-in; the 'baseline' condition reproduces the standard test set
bit-for-bit and serves as an in-script sanity anchor.

Usage:
    python eval_robustness.py --checkpoint checkpoints/openset_cse_..._seed42_best.pt
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
from evaluate import _build_model_from_ckpt, _collect_predictions, evaluate_ood
from ensemble_analysis import analyze_arrays

TEST_SEED = 99999
ROUTING_BOUNDARY = 0.0    # the paper's fixed operating point (not re-tuned)

# condition name -> generator kwargs (all opt-in; {} = legacy default set)
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


def run_condition(model, device, gen_kwargs, n_per_snr, batch_size):
    common = dict(
        n_per_snr=n_per_snr,
        snr_points=C.SNR_TEST_POINTS,
        signal_length=C.SIGNAL_LENGTH,
        sample_rate=C.SAMPLE_RATE,
        seed=TEST_SEED,
        **gen_kwargs,
    )
    ds_kk = CommBSSOpenSetTestDataset(protocol='kk', **common)
    ds_ku = CommBSSOpenSetTestDataset(protocol='ku', **common)
    kk = _collect_predictions(
        model, DataLoader(ds_kk, batch_size=batch_size, num_workers=2), device)
    ku = _collect_predictions(
        model, DataLoader(ds_ku, batch_size=batch_size, num_workers=2), device)
    ood = evaluate_ood(kk, ku, uu=None)   # 4 known classes (baseline ckpt)
    scores = {m: (ood['methods'][m]['score_known'],
                  ood['methods'][m]['score_unknown'])
              for m in ('energy', 'prototype', 'vos')}
    routed = analyze_arrays(ood['known_snr'], ood['unknown_snr'], scores,
                            threshold=ROUTING_BOUNDARY)
    pooled = {m: float(ood['methods'][m]['AUROC'])
              for m in ('energy', 'prototype', 'vos')}
    return {
        'per_snr_auroc': ood['per_snr_auroc'],
        'pooled_auroc': pooled,
        'routed_per_snr': {s: v for s, v in routed['per_snr'].items()},
        'routed_wavg': routed['wavg'],
        'closed_si_sdr_mean': float(np.concatenate([kk['si_sdr_1'],
                                                     kk['si_sdr_2']]).mean()),
    }


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='R1-6 robustness evaluation')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--n_per_snr', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--conditions', type=str, nargs='*', default=None,
                   help='Subset of conditions to run (default: all). '
                        f'Choices: {list(CONDITIONS)}')
    p.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cond_names = args.conditions or list(CONDITIONS)
    for c in cond_names:
        if c not in CONDITIONS:
            raise ValueError(f"unknown condition {c}; choices {list(CONDITIONS)}")

    print(f"Loading model from {args.checkpoint} ...")
    model = _build_model_from_ckpt(args.checkpoint, device)
    run_base = os.path.splitext(os.path.basename(args.checkpoint))[0]

    results = {}
    for name in cond_names:
        print(f"\n=== Condition: {name}  (gen kwargs: {CONDITIONS[name]}) ===")
        r = run_condition(model, device, CONDITIONS[name],
                          args.n_per_snr, args.batch_size)
        results[name] = r
        w = r['routed_wavg']
        print(f"  SI-SDR(kk)={r['closed_si_sdr_mean']:.2f} dB  "
              f"pooled AUROC: energy={r['pooled_auroc']['energy']:.3f}  "
              f"proto={r['pooled_auroc']['prototype']:.3f}  "
              f"vos={r['pooled_auroc']['vos']:.3f}")
        print(f"  routed wavg AUROC (boundary {ROUTING_BOUNDARY:g} dB) = "
              f"{w['routed']:.3f}   oracle={w['oracle']:.3f}   "
              f"energy={w['energy']:.3f}  proto={w['prototype']:.3f}")
        print(f"  per-SNR routed: " +
              "  ".join(f"{int(s):+d}:{v['routed']:.3f}"
                        for s, v in sorted(r['routed_per_snr'].items())))

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, f"robustness_{run_base}.json")
    payload = {
        'checkpoint': os.path.basename(args.checkpoint),
        'test_seed': TEST_SEED,
        'routing_boundary_db': ROUTING_BOUNDARY,
        'n_per_snr': args.n_per_snr,
        'conditions': {name: {
            'gen_kwargs': CONDITIONS[name],
            'per_snr_auroc': {m: {str(s): v for s, v in d.items()}
                               for m, d in r['per_snr_auroc'].items()},
            'pooled_auroc': r['pooled_auroc'],
            'routed_per_snr': {str(s): v for s, v in r['routed_per_snr'].items()},
            'routed_wavg': r['routed_wavg'],
            'closed_si_sdr_mean': r['closed_si_sdr_mean'],
        } for name, r in results.items()},
    }
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"\nSaved {out_json}")

    # Compact cross-condition table (goes to stdout and the driver log)
    print(f"\n=== Robustness summary ({run_base}) ===")
    hdr = (f"  {'condition':<10} {'SI-SDR':>7} | {'E pooled':>8} {'P pooled':>8} "
           f"{'V pooled':>8} | {'routed':>7} {'oracle':>7}")
    print(hdr)
    for name in cond_names:
        r = results[name]
        w = r['routed_wavg']
        print(f"  {name:<10} {r['closed_si_sdr_mean']:>7.2f} | "
              f"{r['pooled_auroc']['energy']:>8.3f} {r['pooled_auroc']['prototype']:>8.3f} "
              f"{r['pooled_auroc']['vos']:>8.3f} | {w['routed']:>7.3f} {w['oracle']:>7.3f}")


if __name__ == '__main__':
    main()
