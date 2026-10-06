"""Paper 6 — E-C (review 3): does demodulation-aware fine-tuning remove the
separate -> oracle-sync -> detect floor?

Compares, on the SAME K=2 deterministic test grid (seed 99999) and with the
SAME scoring as the E4 waveform route (eval_psp_baseline.py):

  baseline  : the MSE-anchored SlotSepNet checkpoints (E3/E4 evaluation
              checkpoints, slot_h64_l4_k13_bs16_lr0.001_mse_s{seed}_best.pt)
  ft_lser*  : demodulation-aware fine-tuned checkpoints produced by
              train_finetune_demod.py (same architecture/parameter budget,
              same generator; baseline recipe + lambda_ser soft-SER term)

Pipeline per burst (identical for every arm): occupancy top-2 slots ->
Hungarian match to the 2 true sources on SI-SDR -> oracle front-end
(ser_comp-compensated receiver: true-carrier down-conversion -> RRC matched
filter -> 0::sps grid -> unit power) -> symbol-wise ('sym') and PSP
('psp') detection -> eval_joint_k2.score_decisions (PIT + M-fold rotation
genie) -> compensated SER + Gray BER from the same decisions
(ser_comp.GRAY_BITS).  Per-pair SI-SDR / SI-SDRi of the matched slots is
recorded alongside, so the waveform-metric side of the comparison is in
the same file.

Usage:
    python eval_finetune_demod.py --n_per_cell 2 --snr_points 10 \
        --seeds 42 --lser_values 1.0 --variants sym          # smoke
    python eval_finetune_demod.py                            # full (server)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# Vendored paper6 modules (self-contained; no sys.path tricks).
import config as C                                          # noqa: E402
from data_generator import (CommBSSVarKTestDataset,         # noqa: E402
                            MOD_TYPES)
from joint_detect import constellation_np                   # noqa: E402
from eval_joint_k2 import score_decisions, _ref_labels      # noqa: E402
from eval_psp_baseline import (oracle_frontend, psp_decisions,  # noqa: E402
                               CKPT_PATTERN)
from eval_blind_pipeline import (build_slot_from_ckpt,      # noqa: E402
                                 _si_sdr_np)

FT_CKPT_PATTERN = ('slot_h{hidden}_l{layers}_k13_bs16_lr{lr}'
                   '_mse_ftser{lser}_s{seed}_best.pt')
FT_LR = 1e-4                # train_finetune_demod.py default (checkpoint name)


def ft_ckpt_path(ckpt_dir, lser, seed, hidden=64, layers=4):
    return os.path.join(ckpt_dir, FT_CKPT_PATTERN.format(
        hidden=hidden, layers=layers, lr=FT_LR, lser=lser, seed=seed))


def build_cells(args):
    """K=2 cells of the deterministic test grid (same as eval_psp_baseline),
    with per-cell shared data incl. the mixture SI-SDR for SI-SDRi."""
    print(f"Building test grid (seed {args.test_seed}, "
          f"n_per_cell={args.n_per_cell}) ...", flush=True)
    ds = CommBSSVarKTestDataset(
        n_per_cell=args.n_per_cell,
        snr_points=list(C.SignalConfig.snr_test_points),
        mod_types=MOD_TYPES, k_lo=1, k_hi=3, k_extrap=4, k_slots=4,
        signal_length=C.SignalConfig.signal_length,
        sample_rate=C.SignalConfig.sample_rate,
        carrier_base=C.SignalConfig.carrier_base,
        freq_gap_range=C.SignalConfig.freq_gap_range,
        seed=args.test_seed, return_carriers=True)
    cells = []
    for i in range(len(ds)):
        s = ds.samples[i]
        if s['k'] == 2 and float(s['snr']) in set(args.snr_points):
            cells.append(i)
    print(f"  {len(cells)} K=2 cells", flush=True)

    cell_data = []
    for i in cells:
        s = ds.samples[i]
        srcs = [s['sources'].numpy()[j, 0] for j in range(2)]
        mods_idx = [int(m) for m in s['mods'].numpy()[:2]]
        cars = [float(c) for c in s['carriers'][:2]]
        refs = [_ref_labels(srcs[j], mods_idx[j], cars[j]) for j in range(2)]
        mix_np = s['mixture'].numpy()[0]
        cell_data.append({'snr': float(s['snr']),
                          'mix': mix_np,
                          'srcs': srcs, 'mods_idx': mods_idx,
                          'cars': cars, 'refs': refs,
                          'si_sdr_mix': [_si_sdr_np(mix_np, srcs[j])
                                         for j in range(2)]})
    return cell_data


def eval_arm(arm, ckpt, cell_data, args, device):
    """separate -> oracle-sync -> detect for ONE checkpoint; returns records
    (one per variant per burst)."""
    model, _a = build_slot_from_ckpt(ckpt, device)
    consts = [constellation_np(m) for m in range(len(MOD_TYPES))]
    records = []
    t0 = time.time()
    B = 16
    for c0 in range(0, len(cell_data), B):
        chunk = cell_data[c0:c0 + B]
        mix_b = torch.stack([
            torch.from_numpy(cd['mix']).view(1, -1)
            for cd in chunk]).to(torch.complex64).to(device)
        with torch.no_grad():
            slots, occ_logits, _ = model(mix_b)
        slots_np = slots.cpu().numpy()
        occ_np = torch.sigmoid(occ_logits).cpu().numpy()
        for bi, cd in enumerate(chunk):
            top2 = np.argsort(-occ_np[bi])[:2]       # oracle count (E4 V2)
            waves = [slots_np[bi, j] for j in top2]
            # Hungarian match to the true sources on SI-SDR (V2 path)
            cost = np.full((2, 2), 1e6)
            for i, s in enumerate(cd['srcs']):
                for j, w in enumerate(waves):
                    cost[i, j] = -_si_sdr_np(w, s)
            rows, cols = linear_sum_assignment(cost)
            slot_of_src = {int(i): waves[j] for i, j in zip(rows, cols)}
            # rows come back sorted (source index); SI-SDR per matched pair
            si_sdr = [float(-cost[i, j]) for i, j in zip(rows, cols)]
            decs = {v: [None, None] for v in args.variants}
            for i in range(2):
                z = oracle_frontend(slot_of_src[i], cd['cars'][i])
                const = consts[cd['mods_idx'][i]]
                if 'sym' in args.variants:
                    decs['sym'][i] = np.argmin(
                        np.abs(z[:, None] - const[None, :]), axis=1)
                if 'psp' in args.variants:
                    decs['psp'][i] = psp_decisions(z, const)
            same = cd['mods_idx'][0] == cd['mods_idx'][1]
            mods_names = [MOD_TYPES[m] for m in cd['mods_idx']]
            for var in args.variants:
                ser1, ber1, ser2, ber2 = score_decisions(
                    decs[var][0], decs[var][1],
                    consts[cd['mods_idx'][0]], consts[cd['mods_idx'][1]],
                    cd['refs'][0], cd['refs'][1], mods_names,
                    allow_swap=same)
                records.append({'arm': arm, 'variant': var,
                                'seed': args._seed, 'snr': cd['snr'],
                                'mods': cd['mods_idx'],
                                'ser1': ser1, 'ser2': ser2,
                                'ber1': ber1, 'ber2': ber2,
                                'ser': 0.5 * (ser1 + ser2),
                                'ber': 0.5 * (ber1 + ber2),
                                'si_sdr1': si_sdr[0], 'si_sdr2': si_sdr[1],
                                'si_sdr': 0.5 * (si_sdr[0] + si_sdr[1]),
                                'si_sdri': 0.5 * (si_sdr[0] - cd['si_sdr_mix'][0]
                                                  + si_sdr[1]
                                                  - cd['si_sdr_mix'][1])})
        done = min(c0 + B, len(cell_data))
        if done % 100 < B or done == len(cell_data):
            print(f"  {arm} s{args._seed}: {done}/{len(cell_data)} cells "
                  f"({(time.time() - t0) / done:.2f}s/cell)", flush=True)
    print(f"{arm} seed {args._seed} done in {time.time() - t0:.0f}s",
          flush=True)
    return records


def _agg(records, key='ser'):
    vals = [r[key] for r in records]
    return float(np.mean(vals)) if vals else None


def aggregate(records, arms, seeds, variants):
    """Per arm x variant: per-seed pooled/per-SNR blocks, then mean/std
    over seeds (E4 group-averaging convention)."""
    out = {}
    snrs = sorted({r['snr'] for r in records})
    for arm in arms:
        out[arm] = {}
        for var in variants:
            rs = [r for r in records
                  if r['arm'] == arm and r['variant'] == var]
            blocks = []
            for seed in seeds:
                gr = [r for r in rs if r['seed'] == seed]
                if not gr:
                    continue
                blk = {'pooled_ser': _agg(gr, 'ser'),
                       'pooled_ber': _agg(gr, 'ber'),
                       'pooled_si_sdr': _agg(gr, 'si_sdr'),
                       'pooled_si_sdri': _agg(gr, 'si_sdri'),
                       'n': len(gr),
                       'per_snr': {f'{s:g}': {'ser': _agg(
                           [r for r in gr if r['snr'] == s], 'ser'),
                           'ber': _agg([r for r in gr if r['snr'] == s],
                                       'ber'),
                           'si_sdr': _agg([r for r in gr if r['snr'] == s],
                                          'si_sdr'),
                           'n': len([r for r in gr if r['snr'] == s])}
                           for s in sorted({r['snr'] for r in gr})}}
                blocks.append(blk)

            def gm(key):
                vals = [b[key] for b in blocks if b[key] is not None]
                return ((float(np.mean(vals)), float(np.std(vals)))
                        if vals else (None, None))

            out[arm][var] = {
                'n_seeds': len(blocks),
                'pooled': {'ser': gm('pooled_ser'), 'ber': gm('pooled_ber'),
                           'si_sdr': gm('pooled_si_sdr'),
                           'si_sdri': gm('pooled_si_sdri')},
                'per_snr': {f'{s:g}': {
                    'ser': (float(np.mean([b['per_snr'][f'{s:g}']['ser']
                                           for b in blocks
                                           if f'{s:g}' in b['per_snr']])),
                            float(np.std([b['per_snr'][f'{s:g}']['ser']
                                          for b in blocks
                                          if f'{s:g}' in b['per_snr']])))
                    if any(f'{s:g}' in b['per_snr'] for b in blocks)
                    else (None, None),
                    'ber': (float(np.mean([b['per_snr'][f'{s:g}']['ber']
                                           for b in blocks
                                           if f'{s:g}' in b['per_snr']])),
                            float(np.std([b['per_snr'][f'{s:g}']['ber']
                                          for b in blocks
                                          if f'{s:g}' in b['per_snr']])))
                    if any(f'{s:g}' in b['per_snr'] for b in blocks)
                    else (None, None),
                    'n': sum(b['per_snr'][f'{s:g}']['n'] for b in blocks
                             if f'{s:g}' in b['per_snr'])}
                    for s in snrs},
                'per_seed': blocks,
            }
    return out


def _report(arms, variants, summary):
    for var in variants:
        print(f"\n=== E-C summary, variant '{var}' (mean±std over seeds) "
              f"===")
        print(f"{'arm':>14s} | {'pooled SER':>16s} | {'pooled BER':>16s}"
              f" | {'SI-SDR':>14s} | {'SI-SDRi':>14s}")
        for arm in arms:
            p = summary[arm][var]['pooled']
            fmt = lambda v: (f"{v[0]:.4f}±{v[1]:.4f}"
                             if v[0] is not None else "   --")
            print(f"{arm:>14s} | {fmt(p['ser']):>16s} | {fmt(p['ber']):>16s}"
                  f" | {fmt(p['si_sdr']):>14s} | {fmt(p['si_sdri']):>14s}")
        snrs = sorted(summary[arms[0]][var]['per_snr'], key=float)
        print(f"\n  per-SNR SER ({var}):")
        print(f"  {'arm':>14s} | " + ' | '.join(f"{s:>8s}" for s in snrs))
        for arm in arms:
            row = []
            for s in snrs:
                v = summary[arm][var]['per_snr'][s]['ser']
                row.append(f"{v[0]:8.4f}" if v[0] is not None else "      --")
            print(f"  {arm:>14s} | " + ' | '.join(row))


def main():
    p = argparse.ArgumentParser(
        description='E-C: baseline vs demodulation-aware fine-tuned '
                    'separators on the K=2 separate->oracle-sync->detect '
                    'pipeline (SER + Gray BER + SI-SDRi)')
    p.add_argument('--n_per_cell', type=int, default=100)
    p.add_argument('--seeds', type=int, nargs='+',
                   default=[42, 43, 44, 45, 46])
    p.add_argument('--lser_values', type=float, nargs='+',
                   default=[0.1, 1.0],
                   help='lambda_ser values of the fine-tuned arms')
    p.add_argument('--variants', type=str, nargs='+',
                   default=['sym', 'psp'], choices=['sym', 'psp'])
    p.add_argument('--snr_points', type=float, nargs='+',
                   default=list(C.SignalConfig.snr_test_points))
    p.add_argument('--ckpt_dir', type=str, default=C.CHECKPOINT_DIR)
    p.add_argument('--test_seed', type=int, default=C.DataConfig.test_seed)
    p.add_argument('--out', type=str,
                   default=os.path.join(C.RESULTS_DIR, 'finetune_demod.json'))
    p.add_argument('--no_baseline', action='store_true',
                   help='skip the baseline arm (its results are cached in '
                        'results/psp_baseline.json)')
    args = p.parse_args()
    device = C.DEVICE

    cell_data = build_cells(args)
    arms = ([] if args.no_baseline else ['baseline']) + \
        [f'ft_lser{lser:g}' for lser in args.lser_values]

    records = []
    for seed in args.seeds:
        args._seed = seed
        for arm in arms:
            ckpt = (os.path.join(args.ckpt_dir,
                                 CKPT_PATTERN.format(seed=seed))
                    if arm == 'baseline'
                    else ft_ckpt_path(args.ckpt_dir,
                                      float(arm.replace('ft_lser', '')),
                                      seed))
            assert os.path.exists(ckpt), f"missing checkpoint: {ckpt}"
            records.extend(eval_arm(arm, ckpt, cell_data, args, device))
        _save(args, arms, records)

    _save(args, arms, records)
    _report(arms, args.variants, _load_summary(args.out))


def _save(args, arms, records):
    summary = aggregate(records, arms, args.seeds, args.variants)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'experiment': 'E-C demodulation-aware fine-tuning '
                                 'control (review 3)',
                   'n_per_cell': args.n_per_cell, 'seeds': args.seeds,
                   'lser_values': args.lser_values,
                   'variants': args.variants,
                   'snr_points': args.snr_points,
                   'test_seed': args.test_seed,
                   'pipeline': 'separate (occupancy top-2, Hungarian on '
                               'SI-SDR) -> oracle-sync front-end -> detect '
                               '(sym/psp), scored by score_decisions with '
                               'ser_comp.GRAY_BITS',
                   'summary': summary,
                   'records': records}, f)
    print(f"saved {args.out} ({len(records)} records)", flush=True)


def _load_summary(path):
    return json.load(open(path))['summary']


if __name__ == '__main__':
    main()
