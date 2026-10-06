"""Paper 6 — CNSE strong-baseline evaluation through the E4 waveform route.

Evaluates the vendored CNSE separator (Hou & Gao 2022; trained by
train_cnse_baseline.py on K=2 mixtures) through the EXACT same scoring
pipeline as eval_psp_baseline.py: per matched output waveform -> oracle
front-end (true-carrier down-conversion -> RRC matched filter -> 0::sps
symbol grid -> unit-power normalisation) -> either symbol-wise
nearest-point decisions ('sym') or PSP/Viterbi sequence detection
('psp', L=3 ISI taps, 3 rounds), scored by eval_joint_k2.score_decisions
(PIT over slot<->source assignments + per-source M-fold rotation
resolution) against eval_joint_k2._ref_labels.

Differences from eval_psp_baseline.py:
  (1) CNSE checkpoints are loaded instead of SlotSepNet; the forward pass
      yields 2 complex sources directly (no occupancy / slot selection);
  (2) the 2 outputs are Hungarian-matched to the 2 true sources on SI-SDR
      via eval_blind_pipeline._si_sdr_np (same matching as the template);
  (3) each record additionally carries sisdr1/sisdr2 (matched SI-SDR, dB)
      and sisdr_mix1/sisdr_mix2 (mixture-reference SI-SDR vs each true
      source) so SI-SDRi can be computed;
  (4) output goes to results/cnse_baseline.json.

Checkpoints are located by glob (batch size / lr may differ per seed on
the shared GPU): cnse_h*_s*_bs*_lr*_mse_s{seed}_best.pt

Usage:
    python eval_cnse_baseline.py --n_per_cell 2 --seeds 42   # smoke
    python eval_cnse_baseline.py --n_per_cell 100            # full
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# Vendored paper6 modules (self-contained; no sys.path tricks).
import config as C                                          # noqa: E402
from cnse import CNSE                                       # noqa: E402
from data_generator import (CommBSSVarKTestDataset,         # noqa: E402
                            MOD_TYPES)
from joint_detect import constellation_np                   # noqa: E402
from eval_joint_k2 import score_decisions, _ref_labels      # noqa: E402
from eval_blind_pipeline import _si_sdr_np                  # noqa: E402
from eval_psp_baseline import (L_TAPS, PSP_ROUNDS,          # noqa: E402
                               oracle_frontend, psp_decisions)

CKPT_GLOB = 'cnse_h*_s*_bs*_lr*_mse_s{seed}_best.pt'


def build_cnse_from_ckpt(ckpt_path, device):
    """Load a CNSE checkpoint trained by train_cnse_baseline.py."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get('args', {})
    model = CNSE(hidden=a.get('hidden', 256),
                 n_stacks=a.get('n_stacks', 3))
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model, a


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description='CNSE strong baseline through the E4 waveform route (K=2)')
    p.add_argument('--n_per_cell', type=int, default=100)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--snr_points', type=float, nargs='+',
                   default=list(C.SignalConfig.snr_test_points))
    p.add_argument('--ckpt_dir', type=str, default=C.CHECKPOINT_DIR)
    p.add_argument('--test_seed', type=int, default=C.DataConfig.test_seed)
    p.add_argument('--out', type=str,
                   default=os.path.join(C.RESULTS_DIR, 'cnse_baseline.json'))
    args = p.parse_args()
    device = C.DEVICE

    print(f"Building test grid (seed {args.test_seed}) ...", flush=True)
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

    # per-cell shared data (independent of the checkpoint)
    cell_data = []
    for i in cells:
        s = ds.samples[i]
        srcs = [s['sources'].numpy()[j, 0] for j in range(2)]
        mods_idx = [int(m) for m in s['mods'].numpy()[:2]]
        cars = [float(c) for c in s['carriers'][:2]]
        refs = [_ref_labels(srcs[j], mods_idx[j], cars[j]) for j in range(2)]
        cell_data.append({'snr': float(s['snr']),
                          'mix': s['mixture'].numpy()[0],
                          'srcs': srcs, 'mods_idx': mods_idx,
                          'cars': cars, 'refs': refs})
    consts = [constellation_np(m) for m in range(len(MOD_TYPES))]

    records = []
    for seed in args.seeds:
        hits = sorted(glob.glob(os.path.join(
            args.ckpt_dir, CKPT_GLOB.format(seed=seed))))
        if not hits:
            print(f"WARNING: no CNSE checkpoint for seed {seed} "
                  f"in {args.ckpt_dir} — skipping", flush=True)
            continue
        ckpt = hits[0]
        print(f"seed {seed}: {os.path.basename(ckpt)}", flush=True)
        model, _a = build_cnse_from_ckpt(ckpt, device)
        t0 = time.time()
        # batched forwards (batch 16), then per-cell detection
        B = 16
        for c0 in range(0, len(cell_data), B):
            chunk = cell_data[c0:c0 + B]
            mix_b = torch.stack([
                torch.from_numpy(cd['mix']).view(1, -1)
                for cd in chunk]).to(torch.complex64).to(device)
            with torch.no_grad():
                s1_b, s2_b = model(mix_b)
            out_np = torch.stack([s1_b.squeeze(1), s2_b.squeeze(1)],
                                 dim=1).cpu().numpy()        # [B, 2, T]
            for bi, cd in enumerate(chunk):
                waves = [out_np[bi, 0], out_np[bi, 1]]
                # Hungarian match to the true sources on SI-SDR
                cost = np.full((2, 2), 1e6)
                for i, s in enumerate(cd['srcs']):
                    for j, w in enumerate(waves):
                        cost[i, j] = -_si_sdr_np(w, s)
                rows, cols = linear_sum_assignment(cost)
                # wave j (waves index) serves source i = rows position
                wave_of_src = {int(i): waves[j]
                               for i, j in zip(rows, cols)}
                sisdr = [_si_sdr_np(wave_of_src[i], cd['srcs'][i])
                         for i in range(2)]
                sisdr_mix = [_si_sdr_np(cd['mix'], cd['srcs'][i])
                             for i in range(2)]
                decs = {'sym': [None, None], 'psp': [None, None]}
                for i in range(2):
                    z = oracle_frontend(wave_of_src[i], cd['cars'][i])
                    const = consts[cd['mods_idx'][i]]
                    decs['sym'][i] = np.argmin(
                        np.abs(z[:, None] - const[None, :]), axis=1)
                    decs['psp'][i] = psp_decisions(z, const)
                same = cd['mods_idx'][0] == cd['mods_idx'][1]
                mods_names = [MOD_TYPES[m] for m in cd['mods_idx']]
                for var in ('sym', 'psp'):
                    ser1, ber1, ser2, ber2 = score_decisions(
                        decs[var][0], decs[var][1],
                        consts[cd['mods_idx'][0]], consts[cd['mods_idx'][1]],
                        cd['refs'][0], cd['refs'][1], mods_names,
                        allow_swap=same)
                    # sisdr fields are checkpoint-level, not variant-level;
                    # they are duplicated across variants and aggregated over
                    # one variant only in _report.
                    records.append({'variant': var, 'seed': seed,
                                    'snr': cd['snr'],
                                    'mods': cd['mods_idx'],
                                    'ser1': ser1, 'ser2': ser2,
                                    'ber1': ber1, 'ber2': ber2,
                                    'ser': 0.5 * (ser1 + ser2),
                                    'ber': 0.5 * (ber1 + ber2),
                                    'sisdr1': sisdr[0], 'sisdr2': sisdr[1],
                                    'sisdr_mix1': sisdr_mix[0],
                                    'sisdr_mix2': sisdr_mix[1]})
            done = min(c0 + B, len(cell_data))
            if done % 100 < B or done == len(cell_data):
                el = time.time() - t0
                print(f"  s{seed}: {done}/{len(cell_data)} cells "
                      f"({el / done:.2f}s/cell)", flush=True)
        print(f"seed {seed} done in {time.time() - t0:.0f}s", flush=True)
        _save(args.out, records, args)

    _save(args.out, records, args)
    _report(records)


def _save(out_path, records, args):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'n_per_cell': args.n_per_cell, 'seeds': args.seeds,
                   'snr_points': args.snr_points,
                   'test_seed': args.test_seed,
                   'L_taps': L_TAPS, 'psp_rounds': PSP_ROUNDS,
                   'records': records}, f)
    print(f"saved {out_path} ({len(records)} records)", flush=True)


def _agg(records, key='ser'):
    return float(np.mean([r[key] for r in records])) if records \
        else float('nan')


def _report(records):
    print("\n=== CNSE baseline summary (SER; per-source average) ===")
    snrs = sorted({r['snr'] for r in records})
    print(f"{'variant':>8s} | " + ' | '.join(f"{s:>6g}" for s in snrs)
          + " | pooled")
    for var in ('sym', 'psp'):
        row = [_agg([r for r in records if r['variant'] == var
                     and r['snr'] == s]) for s in snrs]
        pooled = _agg([r for r in records if r['variant'] == var])
        print(f"{var:>8s} | " + ' | '.join(f"{v:6.4f}" for v in row)
              + f" | {pooled:.4f}")
    for var in ('sym', 'psp'):
        print(f"{var} pooled BER: "
              f"{_agg([r for r in records if r['variant'] == var], 'ber'):.4f}")
    psk = [r for r in records if r['variant'] == 'psp'
           and all(MOD_TYPES[m] != '16QAM' for m in r['mods'])]
    if psk:
        print(f"PSK-only @ PSP pooled SER: {_agg(psk):.4f} (n={len(psk)})")

    # SI-SDR / SI-SDRi at K=2 (checkpoint-level: one variant only)
    uni = [r for r in records if r['variant'] == 'sym']
    sisdr = float(np.mean([(r['sisdr1'] + r['sisdr2']) / 2 for r in uni]))
    sisdr_mix = float(np.mean([(r['sisdr_mix1'] + r['sisdr_mix2']) / 2
                               for r in uni]))
    print(f"\nK=2 separation quality: mean SI-SDR = {sisdr:.3f} dB, "
          f"mixture ref = {sisdr_mix:.3f} dB, "
          f"SI-SDRi = {sisdr - sisdr_mix:.3f} dB")


if __name__ == '__main__':
    main()
