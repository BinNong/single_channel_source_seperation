"""Paper 6 — E3 addendum: END-TO-END scoring of the separation pipeline.

The main E3 evaluation (eval_blind_pipeline.py, paper5 convention) scores
SER over MATCHED slot-source pairs only — a conditional SER given
successful source-slot matching.  For a receiver whose deliverable is
bits, a missed source or a phantom slot must cost something.  This
script keeps the identical pipeline and matching, but additionally
records per-SAMPLE accounting:

  - count_hat = #{slots with occupancy > 0.5}, vs the true K
    -> count accuracy, per K;
  - n_missed = true sources left unmatched (K - #matched) -> miss
    probability per source;
  - n_false = occupied slots beyond K -> false-alarm slot probability;
  - conditional SER (matched pairs only; identical to E3);
  - END-TO-END SER: per-sample (sum of matched-pair SERs + n_missed * 1)
    / K — a missed source counts as a total detection failure for that
    source (BER penalty 0.5); reported for the oracle-carrier and the
    blind receiver arms alike.

Usage:
    python eval_e2e_e3.py --n_per_cell 100 --configs mse
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import config as C                                          # noqa: E402
from data_generator import (CommBSSVarKTestDataset,         # noqa: E402
                            MOD_TYPES)
from eval_blind_pipeline import (CKPT_PATTERNS, _si_sdr_np,  # noqa: E402
                                 blind_ser_ber, build_slot_from_ckpt,
                                 oracle_ser_ber, ref_labels)


@torch.no_grad()
def eval_checkpoint_e2e(ckpt_path, ds, device, occ_threshold):
    """Per-sample records: counting + matching + matched-pair SERs."""
    model, _a = build_slot_from_ckpt(ckpt_path, device)
    samples = []
    batch = []
    for idx in range(len(ds)):
        batch.append(ds[idx])
        if len(batch) == 16 or idx == len(ds) - 1:
            mix_b = torch.stack([b[0] for b in batch]).to(device)
            slots, occ_logits, _cnt = model(mix_b)
            slots_np = slots.cpu().numpy()
            occ_np = torch.sigmoid(occ_logits).cpu().numpy()
            for bi, b in enumerate(batch):
                mix, sources, _occ, k, snr, mods, carriers = b
                kt = int(k)
                if kt > 3:
                    continue                     # K=4 cell: excluded
                srcs = [sources.numpy()[j, 0] for j in range(kt)]
                mod_ids = [int(m) for m in mods.numpy()[:kt]]
                cars = [float(c) for c in carriers.numpy()[:kt]]
                order = np.argsort(-occ_np[bi])
                k_occ = int((occ_np[bi] > occ_threshold).sum())
                ests = [slots_np[bi, j] for j in order[:k_occ]]
                ser_o, ser_b = [], []
                n_matched = 0
                if ests:
                    cost = np.full((kt, len(ests)), 1e6)
                    for i, s in enumerate(srcs):
                        for j, e in enumerate(ests):
                            cost[i, j] = -_si_sdr_np(e, s)
                    rows, cols = linear_sum_assignment(cost)
                    n_matched = len(rows)
                    for i, j in zip(rows, cols):
                        e, s = ests[j], srcs[i]
                        midx, car = mod_ids[i], cars[i]
                        so, _bo = oracle_ser_ber(e, s, midx, car)
                        rl = ref_labels(s, midx, car)
                        sb, _bb, _df = blind_ser_ber(e, midx, rl)
                        ser_o.append(so)
                        ser_b.append(sb)
                n_missed = kt - n_matched
                n_false = max(0, k_occ - kt)
                samples.append({
                    'snr': float(snr), 'k': kt, 'k_occ': k_occ,
                    'count_correct': int(k_occ == kt),
                    'n_missed': n_missed, 'n_false': n_false,
                    'ser_oracle_cond': ser_o, 'ser_blind_cond': ser_b,
                    # end-to-end: missed source = total failure (SER 1,
                    # BER 0.5) for that source
                    'ser_oracle_e2e': (sum(ser_o) + n_missed * 1.0) / kt,
                    'ser_blind_e2e': (sum(ser_b) + n_missed * 1.0) / kt,
                })
            batch = []
        if (idx + 1) % 500 == 0 or idx == len(ds) - 1:
            print(f"    {idx + 1}/{len(ds)} samples", flush=True)
    return samples


def aggregate(samples, seeds):
    def pool(recs):
        if not recs:
            return None
        cond_o = [s for r in recs for s in r['ser_oracle_cond']]
        cond_b = [s for r in recs for s in r['ser_blind_cond']]
        return {
            'n_samples': len(recs),
            'count_acc': float(np.mean([r['count_correct'] for r in recs])),
            'miss_prob_per_source': float(
                np.sum([r['n_missed'] for r in recs])
                / np.sum([r['k'] for r in recs])),
            'false_slot_prob': float(np.mean(
                [1.0 if r['n_false'] > 0 else 0.0 for r in recs])),
            'ser_oracle_cond': float(np.mean(cond_o)) if cond_o else None,
            'ser_blind_cond': float(np.mean(cond_b)) if cond_b else None,
            'ser_oracle_e2e': float(np.mean(
                [r['ser_oracle_e2e'] for r in recs])),
            'ser_blind_e2e': float(np.mean(
                [r['ser_blind_e2e'] for r in recs])),
        }
    out = {'pooled': pool(samples)}
    for k in (1, 2, 3):
        out[f'k{k}'] = pool([r for r in samples if r['k'] == k])
    per_seed = {}
    for sd in seeds:
        rs = [r for r in samples if r['seed'] == sd]
        per_seed[str(sd)] = {
            'count_acc_k2': float(np.mean(
                [r['count_correct'] for r in rs if r['k'] == 2])),
            'ser_oracle_e2e_k2': float(np.mean(
                [r['ser_oracle_e2e'] for r in rs if r['k'] == 2])),
            'ser_blind_e2e_k2': float(np.mean(
                [r['ser_blind_e2e'] for r in rs if r['k'] == 2])),
        }
    out['per_seed_k2'] = per_seed
    return out


def main():
    p = argparse.ArgumentParser(description='E3 end-to-end scoring')
    p.add_argument('--n_per_cell', type=int, default=100)
    p.add_argument('--seeds', type=int, nargs='+',
                   default=[42, 43, 44, 45, 46])
    p.add_argument('--configs', type=str, nargs='+', default=['mse'],
                   choices=sorted(CKPT_PATTERNS))
    p.add_argument('--ckpt_dir', type=str, default=C.CHECKPOINT_DIR)
    p.add_argument('--occ_threshold', type=float, default=0.5)
    p.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    args = p.parse_args()
    device = C.DEVICE

    print("Building deterministic test grid (seed 99999) ...", flush=True)
    ds = CommBSSVarKTestDataset(
        n_per_cell=args.n_per_cell,
        snr_points=C.SignalConfig.snr_test_points,
        mod_types=MOD_TYPES, k_lo=1, k_hi=3, k_extrap=4, k_slots=4,
        signal_length=C.SignalConfig.signal_length,
        sample_rate=C.SignalConfig.sample_rate,
        carrier_base=C.SignalConfig.carrier_base,
        freq_gap_range=C.SignalConfig.freq_gap_range,
        seed=C.DataConfig.test_seed, return_carriers=True)
    print(f"  {len(ds)} samples", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    for cfg in args.configs:
        all_samples = []
        for seed in args.seeds:
            ckpt = os.path.join(args.ckpt_dir,
                                CKPT_PATTERNS[cfg].format(seed=seed))
            print(f"[{cfg}] seed {seed}", flush=True)
            ss = eval_checkpoint_e2e(ckpt, ds, device, args.occ_threshold)
            for r in ss:
                r['seed'] = seed
            all_samples.extend(ss)
            tmp = {'config': cfg,
                   'seeds_done': sorted({r['seed'] for r in all_samples}),
                   'n_per_cell': args.n_per_cell, 'samples': all_samples}
            with open(os.path.join(args.out_dir,
                                   f'e3_e2e_{cfg}_samples.json'), 'w') as f:
                json.dump(tmp, f)
        agg = aggregate(all_samples, sorted({r['seed'] for r in all_samples}))
        out = {'config': cfg, 'n_per_cell': args.n_per_cell,
               'occ_threshold': args.occ_threshold, 'aggregate': agg}
        apath = os.path.join(args.out_dir, f'e3_e2e_{cfg}.json')
        with open(apath, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\n[{cfg}] saved {apath}")
        # reviewer-requested standalone artifact: the UNCONDITIONAL
        # (end-to-end) accounting table cited in the E3 discussion
        acct = {
            'config': cfg, 'n_per_cell': args.n_per_cell,
            'occ_threshold': args.occ_threshold,
            'seeds': sorted({r['seed'] for r in all_samples}),
            'test_seed': C.DataConfig.test_seed,
            'definition': (
                'unconditional (end-to-end) SER: per burst, '
                '(sum of matched-pair SERs + n_missed * 1) / K — a missed '
                'source counts as a total detection failure (SER 1); '
                'a phantom slot costs nothing beyond the count statistics'),
            'unconditional_ser': {
                arm: {key: (agg[key][f'ser_{arm}_e2e']
                            if agg[key] is not None else None)
                      for key in ('k1', 'k2', 'k3', 'pooled')}
                for arm in ('oracle', 'blind')},
            'conditional_ser_matched_only': {
                arm: {key: (agg[key][f'ser_{arm}_cond']
                            if agg[key] is not None else None)
                      for key in ('k1', 'k2', 'k3', 'pooled')}
                for arm in ('oracle', 'blind')},
            'count_accuracy': {key: (agg[key]['count_acc']
                                     if agg[key] is not None else None)
                               for key in ('k1', 'k2', 'k3', 'pooled')},
            'miss_prob_per_source': {
                key: (agg[key]['miss_prob_per_source']
                      if agg[key] is not None else None)
                for key in ('k1', 'k2', 'k3', 'pooled')},
            'false_slot_prob': {key: (agg[key]['false_slot_prob']
                                      if agg[key] is not None else None)
                                for key in ('k1', 'k2', 'k3', 'pooled')},
            'n_samples': {key: (agg[key]['n_samples']
                                if agg[key] is not None else 0)
                          for key in ('k1', 'k2', 'k3', 'pooled')},
        }
        bpath = os.path.join(args.out_dir, 'e3_unconditional_accounting.json')
        with open(bpath, 'w') as f:
            json.dump(acct, f, indent=2)
        print(f"[{cfg}] saved {bpath}")
        hdr = (f"{'K':>3s} | {'cnt_acc':>7s} | {'miss/src':>8s} | "
               f"{'false':>5s} | {'condSER o/b':>17s} | {'e2eSER o/b':>17s}")
        print(hdr)
        print('-' * len(hdr))
        for key in ('k1', 'k2', 'k3', 'pooled'):
            r = agg[key]
            if r is None:
                continue
            print(f"{key:>3s} | {r['count_acc']:7.4f} | "
                  f"{r['miss_prob_per_source']:8.4f} | "
                  f"{r['false_slot_prob']:5.3f} | "
                  f"{r['ser_oracle_cond'] or float('nan'):8.4f}/"
                  f"{r['ser_blind_cond'] or float('nan'):8.4f} | "
                  f"{r['ser_oracle_e2e']:8.4f}/{r['ser_blind_e2e']:8.4f}")

    print("\nE3 end-to-end evaluation done.")


if __name__ == '__main__':
    main()
