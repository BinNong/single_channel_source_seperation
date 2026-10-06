"""Paper 6 — Reviewer-2 (iii): is the baseline separator (SlotSepNet)
genuinely strong?

Three pieces of evidence on the deterministic test grid (seed 99999,
K in {1,2,3} cells; K=4 excluded):

(a) SlotSepNet (5 MSE-anchored checkpoints): per cell, occupancy gating
    at 0.5 (paper5 convention: slots ordered by occupancy probability,
    occupied = sigma(occ) > 0.5), Hungarian match of occupied slots to
    the true sources on SI-SDR; for matched pairs
      SI-SDR  = _si_sdr_np(est, src)          (paper1/utils convention)
      SI-SDRi = SI-SDR(est, src) - SI-SDR(mixture, src)
      NMSE    = ||est - src||^2 / ||src||^2
    reported per K (pooled over SNR) and per K x SNR, means over the 5
    checkpoints.

(b) Occupancy quality per K: TP = occupied slots Hungarian-matched to a
    true source; FP = occupied slots beyond K; FN = true sources with no
    occupied matched slot.  Precision / recall per K.

(c) Cross-family zero-shot reference: paper-1 C-SE CNN
    (ComplexLightweightSepNet, complex_cnn_se_h64_l4_bs16_lr0.005_
    combined_pub_s{42,43,44}_best.pt), evaluated on the K=2 cells only
    (it was trained for 2-source separation).  PIT over its 2 outputs
    vs the 2 true sources (assignment maximising the total SI-SDR);
    same SI-SDR / SI-SDRi / NMSE metrics, compared with SlotSepNet's
    K=2 row.

Usage:
    python eval_separator_quality.py --n_per_cell 3      # smoke
    python eval_separator_quality.py --n_per_cell 100    # full
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# Vendored paper6 modules (self-contained; no sys.path tricks).
import config as C                                          # noqa: E402
from data_generator import (CommBSSVarKTestDataset,         # noqa: E402
                            MOD_TYPES)
from eval_blind_pipeline import (build_slot_from_ckpt,      # noqa: E402
                                 _si_sdr_np)

CKPT_PATTERN = 'slot_h64_l4_k13_bs16_lr0.001_mse_s{seed}_best.pt'
REPO_ROOT = Path(__file__).resolve().parent.parent
P1_DIR = REPO_ROOT / 'paper1_cnn_se'
P1_CKPT_PATTERN = ('complex_cnn_se_h64_l4_bs16_lr0.005_combined_pub_'
                   's{seed}_best.pt')
P1_SEEDS = [42, 43, 44]
OCC_THRESHOLD = 0.5


def _nmse_np(est, ref, eps=1e-12):
    return float(np.sum(np.abs(est - ref) ** 2)
                 / (np.sum(np.abs(ref) ** 2) + eps))


def _pair_metrics(est, src, mix):
    s_est = _si_sdr_np(est, src)
    s_mix = _si_sdr_np(mix, src)
    return {'si_sdr': s_est, 'si_sdri': s_est - s_mix,
            'si_sdr_mix': s_mix, 'nmse': _nmse_np(est, src)}


def _load_p1_models():
    """paper-1 model definitions by explicit file location (avoids
    sys.path pollution; paper6 has its own models.py)."""
    spec = importlib.util.spec_from_file_location(
        'p1_models', P1_DIR / 'models.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_p1_cse(ckpt_path, device):
    p1 = _load_p1_models()
    model = p1.ComplexLightweightSepNet(hidden_channels=64, n_layers=4,
                                        use_se=True)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# (a)+(b) SlotSepNet quality
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_slotsep(ds, cell_idx, seeds, device, progress_every=400):
    records = []            # per matched pair
    occ_counts = {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}  # TP/FP/FN
    for seed in seeds:
        ckpt = os.path.join(C.CHECKPOINT_DIR,
                            CKPT_PATTERN.format(seed=seed))
        model, _a = build_slot_from_ckpt(ckpt, device)
        t0 = time.time()
        B = 16
        for c0 in range(0, len(cell_idx), B):
            chunk = cell_idx[c0:c0 + B]
            mix_b = torch.stack([ds.samples[i]['mixture'][0]
                                 for i in chunk]).to(device)
            with torch.no_grad():
                slots, occ_logits, _ = model(mix_b.unsqueeze(1))
            slots_np = slots.cpu().numpy()
            occ_np = torch.sigmoid(occ_logits).cpu().numpy()
            for bi, i in enumerate(chunk):
                s = ds.samples[i]
                kt = int(s['k'])
                mix_np = s['mixture'].numpy()[0]
                srcs = [s['sources'].numpy()[j, 0] for j in range(kt)]
                order = np.argsort(-occ_np[bi])
                k_occ = int((occ_np[bi] > OCC_THRESHOLD).sum())
                ests = [slots_np[bi, j] for j in order[:k_occ]]
                # occupancy bookkeeping (spec's simple convention)
                occ_counts[kt][0] += min(k_occ, kt)
                occ_counts[kt][1] += max(k_occ - kt, 0)
                occ_counts[kt][2] += max(kt - k_occ, 0)
                if not ests:
                    continue
                cost = np.full((kt, len(ests)), 1e6)
                for a, src in enumerate(srcs):
                    for b, e in enumerate(ests):
                        cost[a, b] = -_si_sdr_np(e, src)
                rows, cols = linear_sum_assignment(cost)
                for a, b in zip(rows, cols):
                    rec = _pair_metrics(ests[b], srcs[a], mix_np)
                    rec.update({'family': 'slotsepnet', 'seed': seed,
                                'k': kt, 'snr': float(s['snr'])})
                    records.append(rec)
            done = min(c0 + B, len(cell_idx))
            if done % progress_every < B or done == len(cell_idx):
                print(f"  slotsep s{seed}: {done}/{len(cell_idx)} "
                      f"({(time.time() - t0) / done:.2f}s/cell)", flush=True)
    return records, occ_counts


# ---------------------------------------------------------------------------
# (c) paper-1 C-SE CNN zero-shot at K=2
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_p1_cse(ds, cell_idx, device, progress_every=200):
    records = []
    for seed in P1_SEEDS:
        ckpt = P1_DIR / 'checkpoints_all' / P1_CKPT_PATTERN.format(seed=seed)
        model = _load_p1_cse(str(ckpt), device)
        t0 = time.time()
        B = 16
        for c0 in range(0, len(cell_idx), B):
            chunk = cell_idx[c0:c0 + B]
            mix_b = torch.stack([ds.samples[i]['mixture'][0]
                                 for i in chunk]).to(device)
            with torch.no_grad():
                s1, s2 = model(mix_b.unsqueeze(1))
            ests_np = np.stack([s1[:, 0, :].cpu().numpy(),
                                s2[:, 0, :].cpu().numpy()], axis=1)  # [B,2,T]
            for bi, i in enumerate(chunk):
                s = ds.samples[i]
                mix_np = s['mixture'].numpy()[0]
                srcs = [s['sources'].numpy()[j, 0] for j in range(2)]
                # PIT over the 2 outputs vs the 2 true sources
                direct = (_si_sdr_np(ests_np[bi, 0], srcs[0])
                          + _si_sdr_np(ests_np[bi, 1], srcs[1]))
                swap = (_si_sdr_np(ests_np[bi, 0], srcs[1])
                        + _si_sdr_np(ests_np[bi, 1], srcs[0]))
                assign = ((0, 1) if direct >= swap else (1, 0))
                for a in range(2):
                    rec = _pair_metrics(ests_np[bi, assign[a]], srcs[a],
                                        mix_np)
                    rec.update({'family': 'p1_cse_cnn', 'seed': seed,
                                'k': 2, 'snr': float(s['snr'])})
                    records.append(rec)
            done = min(c0 + B, len(cell_idx))
            if done % progress_every < B or done == len(cell_idx):
                print(f"  p1-cse s{seed}: {done}/{len(cell_idx)} "
                      f"({(time.time() - t0) / done:.2f}s/cell)", flush=True)
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description='Separator-quality evidence (SI-SDR/SI-SDRi/NMSE, '
                    'occupancy P/R, C-SE zero-shot)')
    p.add_argument('--n_per_cell', type=int, default=100)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--snr_points', type=float, nargs='+',
                   default=list(C.SignalConfig.snr_test_points))
    p.add_argument('--out', type=str,
                   default=os.path.join(C.RESULTS_DIR,
                                        'separator_quality.json'))
    args = p.parse_args()
    device = C.DEVICE

    print("Building deterministic test grid (seed 99999) ...", flush=True)
    ds = CommBSSVarKTestDataset(
        n_per_cell=args.n_per_cell,
        snr_points=list(C.SignalConfig.snr_test_points),
        mod_types=MOD_TYPES, k_lo=1, k_hi=3, k_extrap=4, k_slots=4,
        signal_length=C.SignalConfig.signal_length,
        sample_rate=C.SignalConfig.sample_rate,
        carrier_base=C.SignalConfig.carrier_base,
        freq_gap_range=C.SignalConfig.freq_gap_range,
        seed=C.DataConfig.test_seed, return_carriers=True)
    cells_k = {k: [i for i in range(len(ds))
                   if ds.samples[i]['k'] == k
                   and float(ds.samples[i]['snr']) in set(args.snr_points)]
               for k in (1, 2, 3)}
    all_idx = cells_k[1] + cells_k[2] + cells_k[3]
    print(f"  cells: K1={len(cells_k[1])} K2={len(cells_k[2])} "
          f"K3={len(cells_k[3])}", flush=True)

    slot_records, occ_counts = eval_slotsep(ds, all_idx, args.seeds, device)
    cse_records = eval_p1_cse(ds, cells_k[2], device)

    out = {'n_per_cell': args.n_per_cell, 'seeds': args.seeds,
           'occ_threshold': OCC_THRESHOLD,
           'slotsepnet_records': slot_records,
           'p1_cse_records': cse_records,
           'occupancy_counts': {str(k): {'TP': v[0], 'FP': v[1], 'FN': v[2]}
                                for k, v in occ_counts.items()},
           'summary': _summarize(slot_records, cse_records, occ_counts)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f)
    print(f"saved {args.out}", flush=True)
    _print_summary(out['summary'])


def _agg(recs, key):
    v = [r[key] for r in recs]
    return float(np.mean(v)) if v else float('nan')


def _summarize(slot_records, cse_records, occ_counts):
    summ = {}
    for k in (1, 2, 3):
        rk = [r for r in slot_records if r['k'] == k]
        summ[f'slotsepnet_k{k}'] = {
            'n_pairs': len(rk),
            'si_sdr': _agg(rk, 'si_sdr'),
            'si_sdri': _agg(rk, 'si_sdri'),
            'nmse': _agg(rk, 'nmse'),
            'per_snr': {f'{s:g}': {'si_sdr': _agg(
                [r for r in rk if r['snr'] == s], 'si_sdr'),
                'si_sdri': _agg([r for r in rk if r['snr'] == s], 'si_sdri'),
                'nmse': _agg([r for r in rk if r['snr'] == s], 'nmse')}
                for s in sorted({r['snr'] for r in rk})},
        }
        tp, fp, fn = occ_counts[k]
        summ[f'occupancy_k{k}'] = {
            'TP': tp, 'FP': fp, 'FN': fn,
            'precision': tp / (tp + fp) if tp + fp else float('nan'),
            'recall': tp / (tp + fn) if tp + fn else float('nan'),
        }
    rk2 = [r for r in slot_records if r['k'] == 2]
    summ['p1_cse_k2'] = {
        'n_pairs': len(cse_records),
        'si_sdr': _agg(cse_records, 'si_sdr'),
        'si_sdri': _agg(cse_records, 'si_sdri'),
        'nmse': _agg(cse_records, 'nmse'),
        'per_snr': {f'{s:g}': {'si_sdr': _agg(
            [r for r in cse_records if r['snr'] == s], 'si_sdr'),
            'si_sdri': _agg([r for r in cse_records if r['snr'] == s],
                            'si_sdri'),
            'nmse': _agg([r for r in cse_records if r['snr'] == s], 'nmse')}
            for s in sorted({r['snr'] for r in cse_records})},
        'slotsepnet_k2_si_sdri': _agg(rk2, 'si_sdri'),
    }
    return summ


def _print_summary(summ):
    print("\n=== Separator quality (SlotSepNet, mean over 5 ckpts) ===")
    hdr = (f"{'K':>3s} | {'SI-SDR':>8s} | {'SI-SDRi':>8s} | {'NMSE':>8s}"
           f" | {'occ P':>6s} | {'occ R':>6s}")
    print(hdr)
    print('-' * len(hdr))
    for k in (1, 2, 3):
        r = summ[f'slotsepnet_k{k}']
        o = summ[f'occupancy_k{k}']
        print(f"{k:>3d} | {r['si_sdr']:8.3f} | {r['si_sdri']:8.3f} | "
              f"{r['nmse']:8.4f} | {o['precision']:6.3f} | {o['recall']:6.3f}")
    print("\nK=2 per-SNR SI-SDRi: SlotSepNet vs paper-1 C-SE CNN (zero-shot)")
    s2 = summ['slotsepnet_k2']['per_snr']
    c2 = summ['p1_cse_k2']['per_snr']
    for s in sorted(s2, key=float):
        print(f"  SNR {s:>5s}: SlotSepNet {s2[s]['si_sdri']:7.3f}  |  "
              f"C-SE {c2[s]['si_sdri']:7.3f}")
    print(f"\npooled K=2 SI-SDRi: SlotSepNet "
          f"{summ['slotsepnet_k2']['si_sdri']:.3f}  |  C-SE zero-shot "
          f"{summ['p1_cse_k2']['si_sdri']:.3f}")


if __name__ == '__main__':
    main()
