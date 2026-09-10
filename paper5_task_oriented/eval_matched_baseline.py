"""Paper 5 — reviewer experiment (3): MATCHED-SUBSET mixture baseline for the
recursive (OneAndRestNet) arch.

Motivation: the S1 headline recursive SER (0.4477, matched pairs only) is
computed on the subset of true sources the model's PIT/Hungarian step managed
to match; a source that is missed contributes NOTHING to the SER.  The plain
mixture baseline (0.5049 pooled) is computed on ALL sources.  A reviewer can
therefore ask whether the recursive "gain" is a selection-bias artefact.

This script answers that: for every test sample it records which true
sources the recursive model matched, then scores the MIXTURE-as-estimate
compensated SER/BER (ser_comp.py) on EXACTLY that matched subset.

  - Same test grid as S1: CommBSSVarKTestDataset seed=99999, n_per_cell=100,
    K in {1,2,3} (K=4 cell skipped, as in S1), return_carriers=True.
  - Same checkpoints as S1: paper4 MSE-anchored recursive runs, seeds 42-46,
    loaded through evaluate.build_model_from_ckpt (paper5 models.py).
  - Same matching as evaluate.py: stop-rule occupancy (first step whose stop
    head drops below 0.5), Hungarian with cost = -SI-SDR (evaluate._si_sdr_np
    + scipy linear_sum_assignment, 1e6 fill) — replicated verbatim so the
    matched subsets are IDENTICAL to evaluate.py's.
  - Per matched pair: (recursive SER, BER) and (mixture SER, BER) via
    ser_comp.ser_comp_one — same compensated receiver, same decisions.

Outputs: results/matched_baseline/matched_baseline_s<seed>.json (per seed)
         + matched_baseline_summary.json (mean/std over seeds).

Usage:
    python eval_matched_baseline.py [--n_per_cell 100] [--seeds 42 43 44 45 46]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import config as C
from data_generator_vark import CommBSSVarKTestDataset, MOD_TYPES
from evaluate import build_model_from_ckpt, _si_sdr_np
from ser_comp import ser_comp_one


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_per_cell', type=int, default=100,
                   help='S1 used 100 (run_ser_comp.sh); config default is 200')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--ckpt_dir', type=str,
                   default=os.path.join(_HERE, '..', 'paper4_open_world',
                                        'checkpoints'))
    p.add_argument('--ckpt_glob', type=str,
                   default='recursive_h64_l4_k13_bs16_lr0.001_mse_s{}_best.pt')
    p.add_argument('--out_dir', type=str,
                   default=os.path.join(C.RESULTS_DIR, 'matched_baseline'))
    return p.parse_args()


@torch.no_grad()
def collect_matched(model, loader, device, k_slots: int) -> list[dict]:
    """Per-sample recursive forward + Hungarian match; score BOTH the model
    estimate and the raw mixture on each matched pair (compensated SER/BER).

    The batch unpacking, stop rule and Hungarian cost are line-for-line the
    recursive branch of evaluate.collect_records + evaluate._match_and_measure
    (evaluate.py lines 176-180), so the matched subsets equal the S1
    evaluation's; the only addition is the matched source INDEX per pair and
    the paired mixture-baseline score.
    """

    sample_records = []
    for batch in loader:
        mix, sources, _occ, k, snr, mods, carriers = batch   # 7-tuple
        carriers_np = carriers.numpy()
        mix = mix.to(device)
        B = mix.shape[0]

        ones, stop_logits = model(mix, max_steps=k_slots)
        stop_np = torch.sigmoid(stop_logits).cpu().numpy()   # [B, S]
        ones_np = ones.cpu().numpy()                         # [B, S, T]

        mix_np = mix.cpu().numpy()
        src_np = sources.numpy()                             # [B, S, 1, T]
        k_np = k.numpy()
        snr_np = np.asarray(snr, dtype=np.float32)
        mods_np = mods.numpy()

        for b in range(B):
            kt = int(k_np[b])
            if kt > 3:
                continue                    # K=4 cell: counting-only (as S1)
            srcs = [src_np[b, j, 0] for j in range(kt)]
            mod_ids = [int(m) for m in mods_np[b, :kt]]
            src_carriers = [float(c) for c in carriers_np[b, :kt]]

            # Stop rule (evaluate.py lines 357-366, verbatim semantics).
            k_stop = k_slots
            for i in range(k_slots):
                if stop_np[b, i] < 0.5:
                    k_stop = i + 1
                    break
            occ_slots = [ones_np[b, j] for j in range(k_stop)]

            # Hungarian match, identical to _match_and_measure (cost=-SI-SDR).
            if len(occ_slots) == 0:
                sample_records.append({
                    'snr': float(snr_np[b]), 'k_true': kt, 'n_pairs': 0,
                    'n_miss': kt, 'pairs': []})
                continue
            K_hat = len(occ_slots)
            cost = np.full((kt, K_hat), 1e6)
            for i, s in enumerate(srcs):
                for j, e in enumerate(occ_slots):
                    cost[i, j] = -_si_sdr_np(e, s)
            rows, cols = linear_sum_assignment(cost)

            pairs = []
            for i, j in zip(rows, cols):
                est_t = torch.from_numpy(occ_slots[j])
                src_t = torch.from_numpy(srcs[i])
                ser_m, ber_m = ser_comp_one(est_t, src_t, mod_ids[i],
                                            src_carriers[i])
                # THE paired baseline: SAME matched source, mixture as est.
                ser_b, ber_b = ser_comp_one(torch.from_numpy(mix_np[b, 0]),
                                            src_t, mod_ids[i],
                                            src_carriers[i])
                si = -cost[i, j]
                pairs.append({
                    'src_idx': int(i),
                    'si_sdr': si,
                    'si_sdri': si - _si_sdr_np(mix_np[b, 0], srcs[i]),
                    'mod': mod_ids[i],
                    'ser_rec': ser_m, 'ber_rec': ber_m,
                    'ser_base': ser_b, 'ber_base': ber_b,
                })
            sample_records.append({
                'snr': float(snr_np[b]), 'k_true': kt,
                'n_pairs': len(pairs), 'n_miss': kt - len(rows),
                'pairs': pairs})
    return sample_records


def _mean(recs, field):
    vals = [p[field] for r in recs for p in r['pairs']]
    return float(np.mean(vals)) if vals else float('nan')


def _block(recs):
    n_pairs = sum(r['n_pairs'] for r in recs)
    n_true = sum(r['k_true'] for r in recs)
    return {
        'n_samples': len(recs),
        'n_pairs': n_pairs,
        'miss_rate': 1.0 - n_pairs / max(n_true, 1),
        'ser_recursive': _mean(recs, 'ser_rec'),
        'ber_recursive': _mean(recs, 'ber_rec'),
        'ser_baseline_matched': _mean(recs, 'ser_base'),
        'ber_baseline_matched': _mean(recs, 'ber_base'),
        'si_sdri': _mean(recs, 'si_sdri'),
    }


def summarize(recs):
    snrs = sorted({r['snr'] for r in recs})
    return {
        'overall': _block(recs),
        'per_snr': {str(s): _block([r for r in recs if r['snr'] == s])
                    for s in snrs},
        'per_k': {str(kk): _block([r for r in recs if r['k_true'] == kk])
                  for kk in (1, 2, 3)},
    }


def main():
    args = get_args()
    device = C.DEVICE
    os.makedirs(args.out_dir, exist_ok=True)

    ds = CommBSSVarKTestDataset(
        n_per_cell=args.n_per_cell,
        snr_points=C.SignalConfig.snr_test_points,
        mod_types=MOD_TYPES,
        k_lo=C.VarKConfig.k_min, k_hi=C.VarKConfig.k_max,
        k_extrap=C.VarKConfig.k_extrap,
        k_slots=C.VarKConfig.k_slots,
        signal_length=C.SignalConfig.signal_length,
        sample_rate=C.SignalConfig.sample_rate,
        carrier_base=C.SignalConfig.carrier_base,
        freq_gap_range=C.SignalConfig.freq_gap_range,
        seed=C.DataConfig.test_seed,          # 99999 — same as S1
        return_carriers=True)
    print(f"Test set: {len(ds)} samples (seed=99999, "
          f"n_per_cell={args.n_per_cell}) — identical to S1")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    per_seed = {}
    for s in args.seeds:
        ckpt = os.path.join(args.ckpt_dir, args.ckpt_glob.format(s))
        print(f"\n=== seed {s}: {os.path.basename(ckpt)} ===", flush=True)
        model, arch, _ = build_model_from_ckpt(ckpt, device)
        assert arch == 'recursive', f'expected recursive arch, got {arch}'
        recs = collect_matched(model, loader, device, C.VarKConfig.k_slots)
        res = summarize(recs)
        res['checkpoint'] = os.path.basename(ckpt)
        res['seed'] = s
        res['n_per_cell'] = args.n_per_cell
        per_seed[str(s)] = res
        out = os.path.join(args.out_dir, f'matched_baseline_s{s}.json')
        with open(out, 'w') as f:
            json.dump(res, f, indent=2)
        o = res['overall']
        print(f"  overall: recursive SER={o['ser_recursive']:.4f} vs "
              f"matched-subset baseline SER={o['ser_baseline_matched']:.4f} "
              f"(n_pairs={o['n_pairs']}, miss={o['miss_rate']:.4f})",
              flush=True)
        print(f"  saved {out}", flush=True)

    # Cross-seed summary (mean/std of the per-seed headline numbers).
    def agg(field):
        v = [per_seed[s]['overall'][field] for s in per_seed]
        return {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                'per_seed': v}
    summary = {
        'seeds': args.seeds,
        'n_per_cell': args.n_per_cell,
        'ser_recursive': agg('ser_recursive'),
        'ser_baseline_matched': agg('ser_baseline_matched'),
        'ber_recursive': agg('ber_recursive'),
        'ber_baseline_matched': agg('ber_baseline_matched'),
        'si_sdri': agg('si_sdri'),
        'miss_rate': agg('miss_rate'),
        'reference': {
            's1_recursive_ser_comp': 0.4477,
            's1_baseline_all_sources_ser': 0.5049,
        },
        'per_seed_files': {s: per_seed[s] for s in per_seed},
    }
    out = os.path.join(args.out_dir, 'matched_baseline_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n' + '=' * 72)
    print('MATCHED-BASELINE SUMMARY (mean over seeds)')
    print('=' * 72)
    print(f"  recursive SER (matched pairs)      : "
          f"{summary['ser_recursive']['mean']:.4f} "
          f"+/- {summary['ser_recursive']['std']:.4f}   (S1: 0.4477)")
    print(f"  mixture baseline on SAME subset    : "
          f"{summary['ser_baseline_matched']['mean']:.4f} "
          f"+/- {summary['ser_baseline_matched']['std']:.4f}")
    print(f"  baseline on ALL sources (S1)       : 0.5049")
    print(f"  recursive BER / baseline BER       : "
          f"{summary['ber_recursive']['mean']:.4f} / "
          f"{summary['ber_baseline_matched']['mean']:.4f}")
    print('\n  per-SNR (recursive SER vs matched baseline SER):')
    s0 = per_seed[str(args.seeds[0])]
    for snr in sorted(s0['per_snr'], key=float):
        rs = np.mean([per_seed[s]['per_snr'][snr]['ser_recursive']
                      for s in per_seed])
        bs = np.mean([per_seed[s]['per_snr'][snr]['ser_baseline_matched']
                      for s in per_seed])
        print(f"    SNR {float(snr):>6.1f}: {rs:.4f} vs {bs:.4f}")
    print(f"\nsaved {out}")


if __name__ == '__main__':
    main()
