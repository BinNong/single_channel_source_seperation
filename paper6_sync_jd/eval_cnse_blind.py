"""Paper 6 — BLIND CNSE waveform route + genie-free differential scoring.

Companion to eval_cnse_baseline.py (which scored the ORACLE-sync CNSE
route, ABS only).  This script fills the missing blind-vs-blind cell:
the same 5 CNSE checkpoints (Hou & Gao 2022, K=2, test seed 99999) run
through FOUR variants per cell:

  oracle_sym / oracle_psp : oracle_frontend (true-carrier down-conversion
      -> RRC MF -> 0::sps grid -> unit power) + nearest-point / PSP
      decisions — identical to eval_cnse_baseline, recomputed here as the
      paired reference (the cached JSON stores only scores);
  blind_sym / blind_psp   : sync.blind_sync_known_mod (known-modulation
      semi-blind convention, exactly as eval_blind_pipeline.py's blind
      arm) on the matched CNSE output waveform -> nearest-point decisions
      on the synced symbol stream (blind_sym) or psp_decisions on that
      stream (blind_psp).  Per-source frequency error
      |df_hat - (f_true - f_nominal)| is recorded (E3's spectral-line
      capture diagnostic).

EVERY variant is scored twice from the same decisions:
  (a) ABS  — eval_joint_k2.score_decisions vs _ref_labels (genie M-fold
      rotation resolution; the published E4 convention);
  (b) DIFF — eval_e4_ber_diff.score_diff_pair (genie-free
      differential-equivalent scoring; PSK pairs only, 16QAM-involving
      pairs get NaN exactly as in E-B).

CNSE forward pass is batched on GPU (16/batch); checkpoints are located
by glob (eval_cnse_baseline.CKPT_GLOB).  Saves incrementally after each
seed to results/cnse_blind.json.

Usage:
    python eval_cnse_blind.py --n_per_cell 2 --seeds 42   # smoke
    python eval_cnse_blind.py --n_per_cell 100            # full
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
from data_generator import (CommBSSVarKTestDataset,         # noqa: E402
                            MOD_TYPES)
from joint_detect import constellation_np                   # noqa: E402
from sync import blind_sync_known_mod                       # noqa: E402
from eval_joint_k2 import score_decisions, _ref_labels      # noqa: E402
from eval_blind_pipeline import _si_sdr_np                  # noqa: E402
from eval_psp_baseline import (L_TAPS, PSP_ROUNDS,          # noqa: E402
                               oracle_frontend, psp_decisions)
from eval_e4_ber_diff import score_diff_pair, PSK_MODS      # noqa: E402
from eval_cnse_baseline import (CKPT_GLOB,                  # noqa: E402
                               build_cnse_from_ckpt)

NAN = float('nan')


# ---------------------------------------------------------------------------
# Per-cell scoring: ABS (genie rotation) + DIFF (genie-free, PSK-only)
# ---------------------------------------------------------------------------
def score_both(variant, seed, cd, dec1, dec2, c1, c2, extra=None):
    """ABS + DIFF from the same decisions.  DIFF is NaN for 16QAM-involving
    pairs (E1-D convention)."""
    mods_names = [MOD_TYPES[m] for m in cd['mods_idx']]
    same = cd['mods_idx'][0] == cd['mods_idx'][1]
    ser1, ber1, ser2, ber2 = score_decisions(
        dec1, dec2, c1, c2, cd['refs'][0], cd['refs'][1], mods_names,
        allow_swap=same)
    rec = {'variant': variant, 'seed': seed, 'snr': cd['snr'],
           'mods': cd['mods_idx'],
           'ser1': ser1, 'ser2': ser2, 'ber1': ber1, 'ber2': ber2,
           'ser': 0.5 * (ser1 + ser2), 'ber': 0.5 * (ber1 + ber2),
           'ser_diff': NAN, 'ber_diff': NAN}
    if all(m in PSK_MODS for m in mods_names):
        ds1, db1, ds2, db2 = score_diff_pair(
            dec1, dec2, c1, c2, cd['refs'][0], cd['refs'][1], mods_names,
            allow_swap=same)
        rec['ser_diff'] = 0.5 * (ds1 + ds2)
        rec['ber_diff'] = 0.5 * (db1 + db2)
    if extra:
        rec.update(extra)
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description='Blind CNSE waveform route + DIFF scoring (K=2)')
    p.add_argument('--n_per_cell', type=int, default=100)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--snr_points', type=float, nargs='+',
                   default=list(C.SignalConfig.snr_test_points))
    p.add_argument('--ckpt_dir', type=str, default=C.CHECKPOINT_DIR)
    p.add_argument('--test_seed', type=int, default=C.DataConfig.test_seed)
    p.add_argument('--out', type=str,
                   default=os.path.join(C.RESULTS_DIR, 'cnse_blind.json'))
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
    n_psk = sum(1 for cd in cell_data
                if all(MOD_TYPES[m] in PSK_MODS for m in cd['mods_idx']))
    print(f"  PSK-only pairs: {n_psk}/{len(cell_data)} bursts "
          f"(16QAM-involving pairs are ABS-only, E1-D convention)",
          flush=True)

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
        # batched forwards (batch 16), then per-cell sync + detection
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
                c1 = consts[cd['mods_idx'][0]]
                c2 = consts[cd['mods_idx'][1]]

                # --- oracle arms (paired reference; == eval_cnse_baseline)
                decs = {'oracle_sym': [None, None],
                        'oracle_psp': [None, None]}
                for i in range(2):
                    z = oracle_frontend(wave_of_src[i], cd['cars'][i])
                    const = consts[cd['mods_idx'][i]]
                    decs['oracle_sym'][i] = np.argmin(
                        np.abs(z[:, None] - const[None, :]), axis=1)
                    decs['oracle_psp'][i] = psp_decisions(z, const)
                for var in ('oracle_sym', 'oracle_psp'):
                    records.append(score_both(
                        var, seed, cd, decs[var][0], decs[var][1], c1, c2))

                # --- blind arms (BlindCarrierSync, known-mod convention)
                decs_b = {'blind_sym': [None, None],
                          'blind_psp': [None, None]}
                freq_err = [NAN, NAN]
                for i in range(2):
                    res = blind_sync_known_mod(
                        np.asarray(wave_of_src[i], dtype=np.complex128),
                        cd['mods_idx'][i])
                    zb = np.asarray(res['z'], dtype=np.complex128)
                    decs_b['blind_sym'][i] = np.asarray(res['decisions'])
                    decs_b['blind_psp'][i] = psp_decisions(
                        zb, consts[cd['mods_idx'][i]])
                    freq_err[i] = abs(float(res['df'])
                                      - (cd['cars'][i]
                                         - C.SyncConfig.nominal_carrier))
                for var in ('blind_sym', 'blind_psp'):
                    records.append(score_both(
                        var, seed, cd, decs_b[var][0], decs_b[var][1],
                        c1, c2,
                        extra={'freq_err1': freq_err[0],
                               'freq_err2': freq_err[1]}))
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
    vals = [r[key] for r in records if not np.isnan(r[key])]
    return float(np.mean(vals)) if vals else NAN


VARIANTS = ('oracle_sym', 'oracle_psp', 'blind_sym', 'blind_psp')


def _report(records):
    snrs = sorted({r['snr'] for r in records})
    for key, tag in (('ser', 'ABS SER'), ('ber', 'ABS BER')):
        print(f"\n=== CNSE blind-route summary ({tag}; per-source avg) ===")
        print(f"{'variant':>11s} | " + ' | '.join(f"{s:>6g}" for s in snrs)
              + " | pooled")
        for var in VARIANTS:
            row = [_agg([r for r in records if r['variant'] == var
                         and r['snr'] == s], key) for s in snrs]
            pooled = _agg([r for r in records if r['variant'] == var], key)
            print(f"{var:>11s} | " + ' | '.join(f"{v:6.4f}" for v in row)
                  + f" | {pooled:.4f}")
    for key, tag in (('ser_diff', 'DIFF SER'), ('ber_diff', 'DIFF BER')):
        print(f"\n=== {tag} (genie-free differential; PSK-only pairs) ===")
        for var in VARIANTS:
            psk = [r for r in records if r['variant'] == var
                   and not np.isnan(r['ser_diff'])]
            row = [_agg([r for r in psk if r['snr'] == s], key)
                   for s in snrs]
            print(f"{var:>11s} | " + ' | '.join(f"{v:6.4f}" for v in row)
                  + f" | {_agg(psk, key):.4f}  (n={len(psk)})")

    # blind-arm frequency-error diagnostic (spectral-line capture)
    print("\n=== blind-arm |freq err| (Hz): median / fraction > 1 Hz ===")
    print(f"{'variant':>11s} | " + ' | '.join(f"{s:>6g}" for s in snrs)
          + " | pooled")
    for var in ('blind_sym', 'blind_psp'):
        meds, fracs = [], []
        for s in list(snrs) + [None]:
            sel = [r for r in records if r['variant'] == var
                   and (s is None or r['snr'] == s)]
            errs = [r[k] for r in sel for k in ('freq_err1', 'freq_err2')
                    if not np.isnan(r[k])]
            meds.append(float(np.median(errs)) if errs else NAN)
            fracs.append(float(np.mean([e > 1.0 for e in errs]))
                         if errs else NAN)
        print(f"{var:>11s} med | "
              + ' | '.join(f"{v:6.3f}" for v in meds))
        print(f"{'':>11s} >1Hz| "
              + ' | '.join(f"{v:6.3f}" for v in fracs))

    # per-seed blind_psp pooled SER spread
    print("\nper-seed blind_psp pooled ABS SER: "
          + '  '.join(
              f"s{s}={_agg([r for r in records if r['variant'] == 'blind_psp'
                           and r['seed'] == s]):.4f}"
              for s in sorted({r['seed'] for r in records})))


if __name__ == '__main__':
    main()
