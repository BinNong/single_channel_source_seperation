"""
Paper 3 — Open-Set SC-BSS: Evaluation script.

Loads a trained OpenSetCSE checkpoint and runs the full evaluation:

  Part A — Closed-set sanity (kk protocol):
    - Per-SNR SDR / SI-SDR / SIR (PIT-invariant)
    - Per-class modulation classification accuracy

  Part B — Open-set detection (ku protocol):
    - For each of 3 OOD scorers (Energy, Prototype, VOS):
      - AUROC, AUPR (in+), FPR @ 95% TPR, OSCR
    - Per-SNR breakdown
    - Per-unknown-modulation breakdown

  Part C — Extreme OOD (uu protocol):
    - Same OOD metrics, both sources unknown

Outputs
-------
  results/eval_<run_name>.csv           : per-row metrics table
  results/eval_<run_name>_summary.txt   : human-readable summary
  results/eval_<run_name>_ood_scores.npz: raw OOD scores (for further plots)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure paper3_open_set/ is first on sys.path so `from models import
# OpenSetCSE` resolves to OUR models.py (not paper1's).
_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import config as C
from data_generator_extended import (
    CommBSSOpenSetTestDataset,
    MOD_KNOWN, MOD_UNKNOWN, MOD_ALL, IDX_TO_MOD, MOD_TO_IDX,
)
from losses import pit_multi_task_loss
from models import OpenSetCSE
from ood_scores import (
    energy_score, prototype_score, vos_score,
    compute_prototypes,
)
from open_set_metrics import (
    auroc, aupr_in, fpr_at_95_tpr, oscr,
)


# ============================================================================
# Helpers
# ============================================================================
def _build_model_from_ckpt(ckpt_path: str, device: torch.device) -> OpenSetCSE:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get('args', {})
    model = OpenSetCSE(
        hidden_channels=a.get('hidden', C.BACKBONE_HIDDEN_CHANNELS),
        n_layers=a.get('layers', C.BACKBONE_N_LAYERS),
        use_se=not a.get('no_se', False),
        embed_dim=a.get('embed_dim', C.EMBED_DIM),
        num_known_classes=C.NUM_KNOWN_CLASSES,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


@torch.no_grad()
def _collect_predictions(model, loader, device, alpha=1.0):
    """Run inference on a test set; collect PIT-aligned per-source outputs.

    Returns a dict of concatenated numpy arrays:
      sdr_1, sdr_2          : per-source SI-SDR (dB)
      correct_1, correct_2   : bool — closed-set top-1 correct
      emb_1, emb_2           : (N, embed_dim)
      logits_1, logits_2     : (N, K)
      mod1_idx, mod2_idx     : (N,)
      is_ood_1, is_ood_2     : (N,) bool
      snr                    : (N,)
    """
    si_sdr_1, si_sdr_2 = [], []
    emb_1, emb_2 = [], []
    logits_1, logits_2 = [], []
    mod1_l, mod2_l = [], []
    ood1_l, ood2_l = [], []
    snr_l = []

    for batch in loader:
        mix, src1, src2, mod1, mod2, ood1, ood2, snr = batch
        mix  = mix.to(device, non_blocking=True)
        src1 = src1.to(device, non_blocking=True)
        src2 = src2.to(device, non_blocking=True)
        mod1 = mod1.to(device, non_blocking=True)
        mod2 = mod2.to(device, non_blocking=True)
        ood1 = ood1.to(device, non_blocking=True)
        ood2 = ood2.to(device, non_blocking=True)

        s1_hat, s2_hat, e1, e2, l1, l2 = model(mix)
        # SI-SDR per source (no permutation needed for collection — we use the
        # better assignment when summarising)
        sdr_11 = _si_sdr_per_sample(s1_hat, src1)
        sdr_22 = _si_sdr_per_sample(s2_hat, src2)
        sdr_12 = _si_sdr_per_sample(s1_hat, src2)
        sdr_21 = _si_sdr_per_sample(s2_hat, src1)
        # Per-sample best assignment by SI-SDR (independent of CE since both
        # sources here are in the test set's pool and have modulation labels).
        swap = (sdr_12 + sdr_21) > (sdr_11 + sdr_22)
        swap_b = swap.view(-1, 1, 1)
        s1_best = torch.where(swap_b, s2_hat, s1_hat)
        s2_best = torch.where(swap_b, s1_hat, s2_hat)
        e1_best = torch.where(swap.view(-1, 1), e2, e1)
        e2_best = torch.where(swap.view(-1, 1), e1, e2)
        l1_best = torch.where(swap.view(-1, 1), l2, l1)
        l2_best = torch.where(swap.view(-1, 1), l1, l2)
        mod1_best = torch.where(swap, mod2, mod1)
        mod2_best = torch.where(swap, mod1, mod2)
        ood1_best = torch.where(swap, ood2, ood1)
        ood2_best = torch.where(swap, ood1, ood2)
        # SI-SDR of the aligned outputs
        si_sdr_1.append(_si_sdr_per_sample(s1_best, src1).cpu().numpy())
        si_sdr_2.append(_si_sdr_per_sample(s2_best, src2).cpu().numpy())

        emb_1.append(e1_best.detach().cpu().numpy())
        emb_2.append(e2_best.detach().cpu().numpy())
        logits_1.append(l1_best.detach().cpu().numpy())
        logits_2.append(l2_best.detach().cpu().numpy())
        mod1_l.append(mod1_best.detach().cpu().numpy())
        mod2_l.append(mod2_best.detach().cpu().numpy())
        ood1_l.append(ood1_best.detach().cpu().numpy().astype(bool))
        ood2_l.append(ood2_best.detach().cpu().numpy().astype(bool))
        snr_l.append(np.asarray(snr, dtype=np.float32))

    return {
        'si_sdr_1': np.concatenate(si_sdr_1),
        'si_sdr_2': np.concatenate(si_sdr_2),
        'emb_1':    np.concatenate(emb_1),
        'emb_2':    np.concatenate(emb_2),
        'logits_1': np.concatenate(logits_1),
        'logits_2': np.concatenate(logits_2),
        'mod1_idx': np.concatenate(mod1_l),
        'mod2_idx': np.concatenate(mod2_l),
        'is_ood_1': np.concatenate(ood1_l),
        'is_ood_2': np.concatenate(ood2_l),
        'snr':      np.concatenate(snr_l),
    }


def _si_sdr_per_sample(estimate, reference, eps=1e-8):
    """Re-uses losses._si_sdr_per_sample."""
    from losses import _si_sdr_per_sample
    return _si_sdr_per_sample(estimate, reference, eps)


# ============================================================================
# Part A — Closed-set
# ============================================================================
def evaluate_closed_set(kk: dict) -> dict:
    """kk: predictions on the (known, known) protocol."""
    # SDR per sample
    si_sdr_all = np.concatenate([kk['si_sdr_1'], kk['si_sdr_2']])
    # Classification accuracy (top-1) per modulation
    pred1 = kk['logits_1'].argmax(axis=-1)
    pred2 = kk['logits_2'].argmax(axis=-1)
    correct1 = (pred1 == kk['mod1_idx'])
    correct2 = (pred2 == kk['mod2_idx'])
    correct_all = np.concatenate([correct1, correct2])

    # Per-class accuracy
    per_class = {}
    for k_idx in range(C.NUM_KNOWN_CLASSES):
        m1 = (kk['mod1_idx'] == k_idx)
        m2 = (kk['mod2_idx'] == k_idx)
        m = np.concatenate([m1, m2])
        if m.any():
            per_class[IDX_TO_MOD[k_idx]] = float(correct_all[m].mean())
        else:
            per_class[IDX_TO_MOD[k_idx]] = float('nan')

    # Per-SNR SI-SDR
    snrs = sorted(set(kk['snr'].tolist()))
    per_snr = {}
    for s in snrs:
        m1 = (kk['snr'] == s)
        m2 = (kk['snr'] == s)
        m = np.concatenate([m1, m2])
        if m.any():
            per_snr[int(s)] = float(si_sdr_all[m].mean())

    return {
        'si_sdr_mean': float(si_sdr_all.mean()),
        'si_sdr_std':  float(si_sdr_all.std()),
        'cls_acc':     float(correct_all.mean()),
        'per_class_acc': per_class,
        'per_snr_si_sdr': per_snr,
    }


# ============================================================================
# Part B — Open-set OOD detection
# ============================================================================
def evaluate_ood(kk: dict, ku: dict, uu: dict | None = None,
                num_known: int = C.NUM_KNOWN_CLASSES,
                vos_alpha: float = C.VOS_ALPHA,
                n_per_class: int = C.VOS_N_SYNTHETIC) -> dict:
    """Compute OOD detection metrics for all three scorers.

    The "known" pool = closed-set samples from the (kk) protocol.
    The "unknown" pool = the (ku) protocol where exactly one source is
    unknown.  We use only that one source's score (after PIT alignment in
    _collect_predictions).
    """
    # ---- Build known pool (both sources from kk) ----
    known_emb = np.concatenate([kk['emb_1'], kk['emb_2']], axis=0)
    known_logits = np.concatenate([kk['logits_1'], kk['logits_2']], axis=0)
    known_mods = np.concatenate([kk['mod1_idx'], kk['mod2_idx']], axis=0)
    # Repeat kk per-sample SNR per source so it aligns with the (2 * N)
    # concatenated arrays above.
    known_snr = np.repeat(kk['snr'], 2)

    # ---- Build unknown pool (ku protocol: pick the OOD source per sample) ----
    # After PIT alignment in _collect_predictions, is_ood_1 / is_ood_2 mark
    # which source is OOD.  We need to extract the *OOD side* only.
    ood_mask_1 = ku['is_ood_1']
    ood_mask_2 = ku['is_ood_2']
    # Sanity: exactly one is OOD per sample for ku protocol; for uu both are.
    if uu is None:
        unknown_emb = np.concatenate([
            ku['emb_1'][ood_mask_1],
            ku['emb_2'][ood_mask_2],
        ], axis=0)
        unknown_logits = np.concatenate([
            ku['logits_1'][ood_mask_1],
            ku['logits_2'][ood_mask_2],
        ], axis=0)
        unknown_snr = np.concatenate([ku['snr'][ood_mask_1],
                                       ku['snr'][ood_mask_2]])
    else:
        unknown_emb = np.concatenate([
            ku['emb_1'][ood_mask_1], ku['emb_2'][ood_mask_2],
            uu['emb_1'], uu['emb_2'],
        ], axis=0)
        unknown_logits = np.concatenate([
            ku['logits_1'][ood_mask_1], ku['logits_2'][ood_mask_2],
            uu['logits_1'], uu['logits_2'],
        ], axis=0)
        unknown_snr = np.concatenate([
            ku['snr'][ood_mask_1], ku['snr'][ood_mask_2],
            uu['snr'], uu['snr'],
        ])

    # ---- Compute prototypes on the known pool ----
    prototypes = compute_prototypes(known_emb, known_mods, num_known)

    # ---- Closed-set correctness (for OSCR) ----
    pred_known = known_logits.argmax(axis=-1)
    correct_known = (pred_known == known_mods)

    # ---- Per-method metrics (aggregate, all SNRs pooled) ----
    out = {}
    for name, score_known, score_unknown in [
        ('energy',    energy_score(known_logits),
                      energy_score(unknown_logits)),
        ('prototype', prototype_score(known_emb, prototypes),
                      prototype_score(unknown_emb, prototypes)),
        ('vos',       vos_score(known_emb, prototypes, vos_alpha, n_per_class, seed=0),
                      vos_score(unknown_emb, prototypes, vos_alpha, n_per_class, seed=0)),
    ]:
        out[name] = {
            'AUROC':         auroc(score_known, score_unknown),
            'AUPR_in':       aupr_in(score_known, score_unknown),
            'FPR@95':        fpr_at_95_tpr(score_known, score_unknown),
            'OSCR':          oscr(correct_known, score_known, score_unknown),
            'score_known':   score_known,
            'score_unknown': score_unknown,
        }

    # ---- Per-SNR OOD metrics (fair comparison: both pools at the same SNR) ----
    per_snr = {'energy': {}, 'prototype': {}, 'vos': {}}
    snrs = sorted(set(known_snr.tolist()))
    for s in snrs:
        m_k = known_snr == s
        m_u = unknown_snr == s
        if m_k.sum() < 5 or m_u.sum() < 5:
            continue
        k_emb_s = known_emb[m_k]
        k_logits_s = known_logits[m_k]
        u_emb_s = unknown_emb[m_u]
        u_logits_s = unknown_logits[m_u]

        # Re-compute prototypes using ONLY the known pool at this SNR — same
        # as the aggregate but restricted, so the per-SNR thresholds reflect
        # the in-distribution statistics at that SNR.
        proto_s = compute_prototypes(k_emb_s, known_mods[m_k], num_known)
        # Energy score
        per_snr['energy'][int(s)] = auroc(
            energy_score(k_logits_s), energy_score(u_logits_s),
        )
        # Prototype score
        per_snr['prototype'][int(s)] = auroc(
            prototype_score(k_emb_s, proto_s),
            prototype_score(u_emb_s, proto_s),
        )
        # VOS score
        per_snr['vos'][int(s)] = auroc(
            vos_score(k_emb_s, proto_s, vos_alpha, n_per_class, seed=0),
            vos_score(u_emb_s, proto_s, vos_alpha, n_per_class, seed=0),
        )

    return {
        'methods': out,
        'known_emb':      known_emb,
        'unknown_emb':    unknown_emb,
        'known_logits':   known_logits,
        'unknown_logits': unknown_logits,
        'prototypes':     prototypes,
        'known_mods':     known_mods,
        'unknown_mods':   np.concatenate([
            ku['mod1_idx'][ood_mask_1], ku['mod2_idx'][ood_mask_2],
        ] if uu is None else [
            ku['mod1_idx'][ood_mask_1], ku['mod2_idx'][ood_mask_2],
            uu['mod1_idx'], uu['mod2_idx'],
        ]),
        'known_snr':      known_snr,
        'unknown_snr':    unknown_snr,
        'per_snr_auroc':  per_snr,
    }


# ============================================================================
# CLI
# ============================================================================
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Evaluate OpenSetCSE')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--n_per_snr', type=int, default=200)
    p.add_argument('--n_per_snr_uu', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--carrier_gap', type=float, default=None,
                   help='If set, override carrier_freq_2 = carrier_freq_1 + gap '
                        '(Hz) for the test sets — frequency-separation '
                        'robustness evaluation.')
    p.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from {args.checkpoint} ...")
    model = _build_model_from_ckpt(args.checkpoint, device)

    # ---- Build three test sets ----
    common = dict(
        n_per_snr=args.n_per_snr,
        snr_points=C.SNR_TEST_POINTS,
        signal_length=C.SIGNAL_LENGTH,
        sample_rate=C.SAMPLE_RATE,
        seed=99999,
    )
    if args.carrier_gap is not None:
        cf2 = 2000.0 + args.carrier_gap
        common['carrier_freq_2'] = cf2
        print(f"[carrier_gap] carrier_freq_2 overridden to {cf2} Hz "
              f"(gap = {args.carrier_gap:g} Hz)")
    ds_kk = CommBSSOpenSetTestDataset(protocol='kk', **common)
    ds_ku = CommBSSOpenSetTestDataset(protocol='ku', **common)
    ds_uu = CommBSSOpenSetTestDataset(
        n_per_snr=args.n_per_snr_uu, snr_points=C.SNR_TEST_POINTS,
        signal_length=C.SIGNAL_LENGTH, sample_rate=C.SAMPLE_RATE,
        seed=99999, protocol='uu',
        carrier_freq_2=common['carrier_freq_2'] if args.carrier_gap is not None
                       else 2005.0,
    )

    loaders = {
        'kk': DataLoader(ds_kk, batch_size=args.batch_size, num_workers=2),
        'ku': DataLoader(ds_ku, batch_size=args.batch_size, num_workers=2),
        'uu': DataLoader(ds_uu, batch_size=args.batch_size, num_workers=2),
    }

    print(f"Collecting predictions: kk={len(ds_kk)}, ku={len(ds_ku)}, uu={len(ds_uu)}")
    kk = _collect_predictions(model, loaders['kk'], device)
    ku = _collect_predictions(model, loaders['ku'], device)
    uu = _collect_predictions(model, loaders['uu'], device)

    # ---- Part A: closed-set ----
    closed = evaluate_closed_set(kk)
    print("\n=== Closed-set (kk protocol) ===")
    print(f"  SI-SDR = {closed['si_sdr_mean']:.3f} ± {closed['si_sdr_std']:.3f} dB")
    print(f"  Closed-set cls_acc = {closed['cls_acc']:.3f}")
    for k, v in closed['per_class_acc'].items():
        print(f"    {k:>8s}: {v:.3f}")

    # ---- Part B + C: OOD ----
    ood = evaluate_ood(kk, ku, uu)
    print("\n=== Open-set OOD detection (aggregate, all SNRs pooled) ===")
    for method in ('energy', 'prototype', 'vos'):
        m = ood['methods'][method]
        print(f"  {method:>9s}  AUROC={m['AUROC']:.3f}  AUPR={m['AUPR_in']:.3f}  "
              f"FPR@95={m['FPR@95']:.3f}  OSCR={m['OSCR']:.3f}")

    # ---- Per-SNR OOD (fair comparison) ----
    per_snr = ood['per_snr_auroc']
    if per_snr['prototype']:
        print("\n=== Per-SNR OOD AUROC (both pools at the same SNR) ===")
        snrs = sorted(per_snr['prototype'].keys())
        print(f"  {'SNR':>5s}  | {'energy':>7s} | {'prototype':>10s} | {'vos':>7s}  | "
              f"{'n_known':>7s} | {'n_unknown':>9s}")
        for s in snrs:
            ks = int((ood['known_snr'] == s).sum())
            us = int((ood['unknown_snr'] == s).sum())
            print(f"  {s:+5d}  | {per_snr['energy'][s]:>7.3f} | "
                  f"{per_snr['prototype'][s]:>10.3f} | {per_snr['vos'][s]:>7.3f}  | "
                  f"{ks:>7d} | {us:>9d}")

    # ---- Save outputs ----
    os.makedirs(args.out_dir, exist_ok=True)
    run_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    if args.carrier_gap is not None:
        run_name += f"_gap{args.carrier_gap:g}"
    np.savez(os.path.join(args.out_dir, f"{run_name}_ood_scores.npz"),
             **{f"{m}_{k}": v
                for m, vals in ood['methods'].items()
                for k, v in vals.items() if k.startswith('score_')},
             known_emb=ood['known_emb'],
             unknown_emb=ood['unknown_emb'],
             known_logits=ood['known_logits'],
             unknown_logits=ood['unknown_logits'],
             prototypes=ood['prototypes'],
             known_mods=ood['known_mods'],
             unknown_mods=ood['unknown_mods'],
             known_snr=ood['known_snr'],
             unknown_snr=ood['unknown_snr'],
             snr_kk=kk['snr'])
    summary = {
        'closed': closed,
        'ood': {m: {k: v for k, v in vals.items() if not k.startswith('score_')}
                 for m, vals in ood['methods'].items()},
        'per_snr_auroc': ood['per_snr_auroc'],
    }
    with open(os.path.join(args.out_dir, f"{run_name}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"\nSaved summary to {args.out_dir}/{run_name}_summary.json")


if __name__ == '__main__':
    main()