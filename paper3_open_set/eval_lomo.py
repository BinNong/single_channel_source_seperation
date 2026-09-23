"""
Paper 3 — Open-Set SC-BSS: Leave-One-Modulation-Out (LOMO) evaluation.

Added for the post-rejection revision (reviewer demands R1-5 / R2-3):
  (1) multiple known/unknown splits — a model is trained on 3 of the 4
      known modulations (train.py --known_mods) and evaluated with the
      held-out KNOWN-class modulation treated as an additional unknown;
  (2) the SNR-routing boundary between the Energy and Prototype scorers is
      selected WITHOUT any test unknown-class data, via a pseudo-OOD
      validation split (seed 77777) in which the "unknown" source is the
      held-out known-class modulation (genuinely OOD for this model, but
      drawn from the same signal family as training — no test-set
      information is touched).

Protocol
--------
(a) PSEUDO-OOD VALIDATION (seed 77777):
      kk pool = 3 trained mods, ku "unknown" pool = {held-out mod} only.
      Per-SNR AUROC profiles of Energy and Prototype -> routing boundary =
      their crossover SNR (fallback 0 dB if no clean crossover).
(b) TEST (seed 99999):
      known pool = 3 trained mods,
      unknown pool = {held-out mod} + {64QAM, PI4_DQPSK, MSK, OFDM_QPSK}
      (5 unknown classes).  Per-SNR profiles, pooled AUROC, and routed
      weighted-avg AUROC using ONLY the boundary from (a).

Usage:
    python eval_lomo.py --checkpoint checkpoints/..._kmQPSK-8PSK-16QAM_seed42_best.pt
    # known set / held-out mod are recovered from the checkpoint's stored
    # args (--known_mods); both can be overridden on the CLI.
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
    CommBSSOpenSetTestDataset,
    MOD_KNOWN, MOD_UNKNOWN, MOD_TO_IDX,
)
from evaluate import _collect_predictions, evaluate_ood
from models import OpenSetCSE
from ensemble_analysis import analyze_arrays

VAL_SEED = 77777    # pseudo-OOD validation split (never the test seed)
TEST_SEED = 99999   # paper's fixed test seed


# ============================================================================
# Helpers
# ============================================================================
def _build_model(ckpt_path: str, device: torch.device,
                 num_known: int) -> tuple[OpenSetCSE, dict]:
    """Build the model with the number of classes it was TRAINED with.

    (evaluate.py's builder hard-codes C.NUM_KNOWN_CLASSES=4, which is wrong
    for LOMO checkpoints; the class count is taken from the CLI/stored known
    set instead.)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get('args', {})
    model = OpenSetCSE(
        hidden_channels=a.get('hidden', C.BACKBONE_HIDDEN_CHANNELS),
        n_layers=a.get('layers', C.BACKBONE_N_LAYERS),
        use_se=not a.get('no_se', False),
        embed_dim=a.get('embed_dim', C.EMBED_DIM),
        num_known_classes=num_known,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, a


def _remap_labels(preds: dict, known_mods: list[str]) -> dict:
    """Remap global MOD_TO_IDX labels to local [0, len(known_mods)) indices.

    The classifier head of a LOMO model has len(known_mods) outputs, so
    prototype computation / OSCR must use local labels.  Unknown-side labels
    (only used for reporting) are set to -1.
    """
    lut = {MOD_TO_IDX[m]: i for i, m in enumerate(known_mods)}
    out = dict(preds)
    for key in ('mod1_idx', 'mod2_idx'):
        out[key] = np.array([lut.get(int(v), -1) for v in preds[key]],
                            dtype=preds[key].dtype)
    return out


def select_boundary(per_snr_energy: dict, per_snr_proto: dict,
                    fallback: float = 0.0) -> tuple[float, str]:
    """Pick the SNR-routing boundary from validation per-SNR AUROC profiles.

    Routing rule (fixed a priori): Energy for snr <= boundary, Prototype
    above.  The boundary is the Energy->Prototype crossover:
      - exactly one sign change of (AUROC_energy - AUROC_prototype) from
        non-negative to negative as SNR increases  ->  boundary = last SNR
        where Energy still wins;
      - one scorer wins at every SNR  ->  boundary routes everything to
        that scorer (snrs[-1] = all-Energy, snrs[0]-1 = all-Prototype);
      - anything else (non-monotone / multiple crossings)  ->  fallback.
    """
    snrs = sorted(set(per_snr_energy) & set(per_snr_proto))
    if not snrs:
        return fallback, 'fallback (no shared SNR bins)'
    e = np.array([per_snr_energy[s] for s in snrs], dtype=float)
    p = np.array([per_snr_proto[s] for s in snrs], dtype=float)
    energy_wins = e >= p
    transitions = int(np.sum(energy_wins[:-1] != energy_wins[1:]))
    if transitions == 0:
        if energy_wins[0]:
            return float(snrs[-1]), 'energy >= prototype at every validation SNR (all-energy route)'
        return float(snrs[0] - 1), 'prototype > energy at every validation SNR (all-prototype route)'
    if transitions == 1 and energy_wins[0] and not energy_wins[-1]:
        i = int(np.nonzero(energy_wins[:-1] != energy_wins[1:])[0][0])
        return float(snrs[i]), (f'clean crossover between {snrs[i]} and '
                                f'{snrs[i + 1]} dB')
    return fallback, (f'no clean crossover ({transitions} sign changes) '
                      f'-> fallback {fallback} dB')


def _run_split(model, device, known_mods, unknown_pool, seed,
               n_per_snr, batch_size, tag):
    """Build kk+ku sets for one split, collect predictions, compute OOD."""
    common = dict(
        n_per_snr=n_per_snr,
        snr_points=C.SNR_TEST_POINTS,
        signal_length=C.SIGNAL_LENGTH,
        sample_rate=C.SAMPLE_RATE,
        seed=seed,
        mod_known_pool=list(known_mods),
        mod_unknown_pool=list(unknown_pool),
    )
    ds_kk = CommBSSOpenSetTestDataset(protocol='kk', **common)
    ds_ku = CommBSSOpenSetTestDataset(protocol='ku', **common)
    print(f"[{tag}] kk={len(ds_kk)}  ku={len(ds_ku)}  "
          f"(known={known_mods}, unknown={unknown_pool}, seed={seed})")
    kk = _collect_predictions(
        model, DataLoader(ds_kk, batch_size=batch_size, num_workers=2), device)
    ku = _collect_predictions(
        model, DataLoader(ds_ku, batch_size=batch_size, num_workers=2), device)
    kk = _remap_labels(kk, known_mods)
    ku = _remap_labels(ku, known_mods)
    return evaluate_ood(kk, ku, uu=None, num_known=len(known_mods))


# ============================================================================
# CLI
# ============================================================================
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='LOMO open-set evaluation')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--known_mods', type=str, default='',
                   help='Comma-separated trained known mods; default = read '
                        'from the checkpoint\'s stored args.')
    p.add_argument('--held_out', type=str, default='',
                   help='Held-out modulation; default = the one MOD_KNOWN '
                        'entry missing from --known_mods.')
    p.add_argument('--n_per_snr', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Recover the known set / held-out modulation ----
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    stored = ckpt.get('args', {})
    del ckpt
    known_str = args.known_mods or stored.get('known_mods', '')
    known_mods = ([m.strip() for m in known_str.split(',') if m.strip()]
                  if known_str else list(MOD_KNOWN))
    bad = [m for m in known_mods if m not in MOD_KNOWN]
    if bad:
        raise ValueError(f"known mods must be a subset of {MOD_KNOWN}; got {bad}")
    if args.held_out:
        held_out = args.held_out
    else:
        missing = [m for m in MOD_KNOWN if m not in known_mods]
        if len(missing) != 1:
            raise ValueError(f"cannot infer held-out mod from known={known_mods}; "
                             f"pass --held_out explicitly")
        held_out = missing[0]
    if held_out in known_mods:
        raise ValueError(f"held-out mod {held_out} is in the known set")
    unknown_pool = [held_out] + list(MOD_UNKNOWN)
    print(f"LOMO: known={known_mods}  held_out={held_out}  "
          f"test unknown pool={unknown_pool}")

    model, _stored_args = _build_model(args.checkpoint, device, len(known_mods))

    # ---- (a) Pseudo-OOD validation: boundary WITHOUT test unknown data ----
    ood_val = _run_split(model, device, known_mods, [held_out], VAL_SEED,
                         args.n_per_snr, args.batch_size, tag='val')
    val_e = ood_val['per_snr_auroc']['energy']
    val_p = ood_val['per_snr_auroc']['prototype']
    boundary, reason = select_boundary(val_e, val_p, fallback=0.0)
    print(f"\n=== Pseudo-OOD validation (seed {VAL_SEED}, unknown={held_out}) ===")
    for s in sorted(val_e):
        print(f"  SNR {s:+5d}: energy={val_e[s]:.3f}  prototype={val_p[s]:.3f}")
    print(f"  -> routing boundary = {boundary:g} dB  ({reason})")

    # ---- (b) Test with 5 unknown classes, boundary fixed from (a) ----
    ood_test = _run_split(model, device, known_mods, unknown_pool, TEST_SEED,
                          args.n_per_snr, args.batch_size, tag='test')
    scores = {m: (ood_test['methods'][m]['score_known'],
                  ood_test['methods'][m]['score_unknown'])
              for m in ('energy', 'prototype', 'vos')}
    routed = analyze_arrays(ood_test['known_snr'], ood_test['unknown_snr'],
                            scores, threshold=boundary)

    print(f"\n=== LOMO test (seed {TEST_SEED}, 5 unknown classes) ===")
    print(f"  {'SNR':>5} | {'energy':>7} {'proto':>7} {'vos':>7} | routed")
    for s, v in sorted(routed['per_snr'].items()):
        print(f"  {s:+5.0f} | {v['energy']:>7.3f} {v['prototype']:>7.3f} "
              f"{v['vos']:>7.3f} | {v['routed']:>7.3f} ({v['rule_method'][:5]})")
    pooled = {m: float(ood_test['methods'][m]['AUROC'])
              for m in ('energy', 'prototype', 'vos')}
    print(f"  pooled AUROC: " + "  ".join(f"{m}={v:.3f}" for m, v in pooled.items()))
    print(f"  routed weighted-avg AUROC (boundary {boundary:g} dB, chosen on "
          f"validation only): {routed['wavg']['routed']:.3f}  "
          f"(oracle {routed['wavg']['oracle']:.3f})")

    # ---- Save ----
    os.makedirs(args.out_dir, exist_ok=True)
    run_base = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_json = os.path.join(args.out_dir, f"lomo_{run_base}.json")
    payload = {
        'checkpoint': os.path.basename(args.checkpoint),
        'known_mods': known_mods,
        'held_out': held_out,
        'unknown_pool': unknown_pool,
        'val_seed': VAL_SEED,
        'test_seed': TEST_SEED,
        'boundary_db': boundary,
        'boundary_reason': reason,
        'val_per_snr_auroc': {m: {str(s): v for s, v in d.items()}
                               for m, d in ood_val['per_snr_auroc'].items()},
        'test_per_snr_auroc': {m: {str(s): v for s, v in d.items()}
                                for m, d in ood_test['per_snr_auroc'].items()},
        'test_pooled_auroc': pooled,
        'test_routed': {
            'per_snr': {str(s): v for s, v in routed['per_snr'].items()},
            'wavg': routed['wavg'],
        },
    }
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    np.savez(os.path.join(args.out_dir, f"lomo_{run_base}_scores.npz"),
             **{f"val_{m}_{k}": v for m in ('energy', 'prototype', 'vos')
                for k, v in ood_val['methods'][m].items()
                if k.startswith('score_')},
             val_known_snr=ood_val['known_snr'],
             val_unknown_snr=ood_val['unknown_snr'],
             **{f"test_{m}_{k}": v for m in ('energy', 'prototype', 'vos')
                for k, v in ood_test['methods'][m].items()
                if k.startswith('score_')},
             test_known_snr=ood_test['known_snr'],
             test_unknown_snr=ood_test['unknown_snr'])
    print(f"\nSaved {out_json}")


if __name__ == '__main__':
    main()
