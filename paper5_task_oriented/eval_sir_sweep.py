"""Paper 5 — reviewer experiment (1): SIR sweep at fixed K=2.

Question: is the "SI-SDRi does not convert to SER/BER gain" conclusion
specific to SIR ~ 0 dB (the training distribution, weights ~ U(0.4, 0.6)),
or does it persist when one source dominates?  If the regime is truly
interference-limited only at SIR ~ 0, a high-SIR cell should approach the
noise-limited single-source behaviour for the strong source and the SER
gain of separation should become visible.

Design
  - K = 2 fixed; SIR in {0, 5, 10, 20} dB with GEOMETRICALLY SYMMETRIC
    weights:  w_strong = 0.5 * 10^(SIR/40),  w_weak = 0.5 * 10^(-SIR/40)
    (SIR = 0 -> 0.5/0.5, the centre of the training weight distribution).
    Per sample a fair coin (deterministic RNG) decides which of source 0/1
    is the strong one.
  - SNR grid = the standard 7 test points (-10 .. 20 dB), 100 samples per
    (SIR, SNR) cell.
  - Deterministic INDEPENDENT seed 88888 (NOT 99999 — that is the main
    test set; disclosed in the paper).
  - Models: the 20 existing S2 checkpoints (sisdr / mse / ser / ser_mse
    x seeds 42-46).  NO retraining.
  - Pipeline reuse: evaluate.build_model_from_ckpt, evaluate._match_and_measure
    (Hungarian + compensated SER/BER via ser_comp.py), evaluate.aggregate.
    The batch loop below mirrors evaluate.collect_records' slot branch; the
    only additions are the 'sir' tag and the matched source INDEX per pair
    (needed for the strong/weak split; collect_records does not expose it).
  - Mixture baseline per cell: mixture-as-estimate scored against EVERY
    true source (same 口径 as ser_comp.py's published baseline), computed
    once (model-independent).

Outputs: results/sir_sweep/sir_sweep_<config>_s<seed>.json (20 files)
         + sir_sweep_baseline.json (model-independent baseline).

Usage:
    python eval_sir_sweep.py [--n_per_cell 100] [--configs sisdr mse ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import config as C
from data_generator_vark import generate_vark_mixture, MOD_TYPES
from evaluate import (build_model_from_ckpt, aggregate,
                      _match_and_measure, _si_sdr_np)
from ser_comp import ser_comp_one

SIRS = [0.0, 5.0, 10.0, 20.0]
SWEEP_SEED = 88888          # independent of the main test seed (99999)

CONFIGS = {
    'sisdr':   'slot_h64_l4_k13_bs16_lr0.001_sisdr_s{}_best.pt',
    'mse':     'slot_h64_l4_k13_bs16_lr0.001_mse_s{}_best.pt',
    'ser':     'slot_h64_l4_k13_bs16_lr0.001_ser_s{}_best.pt',
    'ser_mse': 'slot_h64_l4_k13_bs16_lr0.001_ser_mse_s{}_best.pt',
}


class SirSweepDataset(Dataset):
    """Deterministic K=2 SIR-sweep set, eager/in-memory (test-set style).

    __getitem__ returns the SAME 7-tuple format as CommBSSVarKTestDataset
    with return_carriers=True:  (mixture[1,T], sources[S,1,T], occ[S],
    k, snr, mods[S], carriers[S]).  Per-sample weights are stashed in
    self.samples[i]['weights'] for the strong/weak split.
    """

    def __init__(self, sir_db: float, n_per_cell: int = 100,
                 snr_points=None, seed: int = SWEEP_SEED,
                 k_slots: int = 4, rng_continuation: bool = False):
        if snr_points is None:
            snr_points = C.SignalConfig.snr_test_points
        self.k_slots = k_slots
        self.samples = []
        w_strong = 0.5 * 10 ** (sir_db / 40.0)
        w_weak = 0.5 * 10 ** (-sir_db / 40.0)
        # NOTE: rng_continuation=True assumes the global stream is already
        # positioned by the caller (used to give each SIR block a distinct,
        # deterministic slice of ONE stream seeded with SWEEP_SEED).
        if not rng_continuation:
            np.random.seed(seed)
        for snr in snr_points:
            for _ in range(n_per_cell):
                # fair coin: which source is strong (documented draw order:
                # the flip is drawn BEFORE generate_vark_mixture's internal
                # draws for this sample)
                flip = bool(np.random.rand() < 0.5)
                weights = ([w_weak, w_strong] if flip
                           else [w_strong, w_weak])
                mix, srcs, mod_idx, cars = generate_vark_mixture(
                    C.SignalConfig.signal_length, C.SignalConfig.sample_rate,
                    float(snr), MOD_TYPES, 2,
                    carrier_base=C.SignalConfig.carrier_base,
                    freq_gap_range=C.SignalConfig.freq_gap_range,
                    n_symbols=C.SignalConfig.n_symbols,
                    roll_off=C.SignalConfig.roll_off,
                    num_taps=C.SignalConfig.num_taps,
                    apply_fading=C.SignalConfig.apply_fading,
                    fading_taps=C.SignalConfig.fading_taps,
                    weights=weights,
                    return_carriers=True)
                sources = np.zeros((k_slots, C.SignalConfig.signal_length),
                                   dtype=np.complex64)
                for j, s in enumerate(srcs):
                    sources[j] = s
                occ = np.zeros(k_slots, dtype=np.float32)
                occ[:2] = 1.0
                mods = np.full(k_slots, -1, dtype=np.int64)
                mods[:2] = mod_idx
                cars_pad = np.full(k_slots, np.nan)
                cars_pad[:2] = cars
                self.samples.append({
                    'mixture': torch.from_numpy(mix).unsqueeze(0)
                               .to(torch.complex64),
                    'sources': torch.from_numpy(sources).unsqueeze(1)
                               .to(torch.complex64),
                    'occ_mask': torch.from_numpy(occ),
                    'k': 2,
                    'snr': float(snr),
                    'mods': torch.from_numpy(mods),
                    'carriers': cars_pad,
                    'weights': list(weights),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (s['mixture'], s['sources'], s['occ_mask'],
                torch.tensor(s['k'], dtype=torch.long),
                float(s['snr']), s['mods'],
                torch.from_numpy(s['carriers']))


def build_all_datasets(n_per_cell: int, seed: int = SWEEP_SEED):
    """One stashed stream seeded with `seed`; each SIR block consumes a
    deterministic slice of it (block order = SIRS)."""
    st = np.random.get_state()
    np.random.seed(seed)
    try:
        out = {}
        for sir in SIRS:
            out[sir] = SirSweepDataset(sir, n_per_cell=n_per_cell,
                                       rng_continuation=True)
    finally:
        np.random.set_state(st)
    return out


@torch.no_grad()
def collect_records_sir(model, dataset, loader, device,
                        occ_threshold: float, k_slots: int) -> list[dict]:
    """Slot-arch batch loop, mirroring evaluate.collect_records (lines
    274-386) with two additions: 'sir'-level pairing info is recoverable
    because every pair is annotated with the matched TRUE SOURCE INDEX
    (src_idx), and records carry no counting fields the aggregate pipeline
    does not already handle.

    Matching + pair scoring are done by evaluate._match_and_measure
    (verbatim reuse); the source indices are recovered by recomputing the
    same deterministic Hungarian cost matrix (identical inputs -> identical
    assignment).
    """
    records = []
    for batch in loader:
        mix, sources, _occ, k, snr, mods, carriers = batch
        carriers_np = carriers.numpy()
        mix = mix.to(device)
        B = mix.shape[0]

        out = model(mix)
        if len(out) == 4:
            slots, occ_logits, count_logits, _sym = out
        else:
            slots, occ_logits, count_logits = out
        occ_np = torch.sigmoid(occ_logits).cpu().numpy()     # [B, S]
        slots_np = slots.cpu().numpy()                       # [B, S, T]
        cnt_hat = (count_logits.argmax(dim=1) + 1).cpu().numpy() \
            if count_logits is not None else None

        mix_np = mix.cpu().numpy()
        src_np = sources.numpy()
        k_np = k.numpy()
        snr_np = np.asarray(snr, dtype=np.float32)
        mods_np = mods.numpy()

        for b in range(B):
            kt = int(k_np[b])
            srcs = [src_np[b, j, 0] for j in range(kt)]
            mod_ids = [int(m) for m in mods_np[b, :kt]]
            src_carriers = [float(c) for c in carriers_np[b, :kt]]

            order = np.argsort(-occ_np[b])
            k_occ = int((occ_np[b] > occ_threshold).sum())
            occ_slots = [slots_np[b, j] for j in order[:k_occ]]

            pairs, n_miss, n_halluc, hrel = _match_and_measure(
                occ_slots, srcs, mix_np[b, 0], mod_ids,
                True, True, src_carriers)

            # Recover matched source indices (same deterministic Hungarian).
            src_idx = []
            if pairs:
                cost = np.full((kt, len(occ_slots)), 1e6)
                for i, s in enumerate(srcs):
                    for j, e in enumerate(occ_slots):
                        cost[i, j] = -_si_sdr_np(e, s)
                rows, _cols = linear_sum_assignment(cost)
                src_idx = [int(i) for i in rows]
            for p, i in zip(pairs, src_idx):
                p['src_idx'] = i

            records.append({
                'snr': float(snr_np[b]),
                'k_true': kt,
                'mods': mod_ids,
                'k_hat_occ': max(k_occ, 1),
                'k_hat_cnt': int(cnt_hat[b]) if cnt_hat is not None else None,
                'pairs': pairs,
                'n_miss': n_miss,
                'n_halluc': n_halluc,
                'halluc_rel_power': hrel,
                'n_occupied': len(occ_slots),
                'oracle_pair_si_sdri': [],
            })
    return records


def baseline_scores(dataset) -> dict:
    """Mixture-as-estimate compensated SER/BER against EVERY true source —
    the published baseline 口径 (ser_comp.py main), per SNR + pooled."""
    per_snr_ser, per_snr_ber = {}, {}
    all_ser, all_ber = [], []
    for i in range(len(dataset)):
        s = dataset.samples[i]
        mix_t = s['mixture'][0]
        for j in range(2):
            ser_v, ber_v = ser_comp_one(
                mix_t, s['sources'][j, 0], int(s['mods'][j]),
                float(s['carriers'][j]))
            per_snr_ser.setdefault(s['snr'], []).append(ser_v)
            per_snr_ber.setdefault(s['snr'], []).append(ber_v)
            all_ser.append(ser_v)
            all_ber.append(ber_v)
    return {
        'overall_ser': float(np.mean(all_ser)),
        'overall_ber': float(np.mean(all_ber)),
        'per_snr_ser': {str(k): float(np.mean(v))
                        for k, v in per_snr_ser.items()},
        'per_snr_ber': {str(k): float(np.mean(v))
                        for k, v in per_snr_ber.items()},
        'n_pairs': len(all_ser),
    }


def strong_weak_split(records, dataset) -> dict:
    """Per-pair SER split by whether the matched source was the strong one
    (undefined at SIR=0, where both weights are 0.5)."""
    out = {}
    for rec, sample in zip(records, dataset.samples):
        w = sample['weights']
        if abs(w[0] - w[1]) < 1e-9:
            continue
        strong = int(np.argmax(w))
        for p in rec['pairs']:
            role = 'strong' if p['src_idx'] == strong else 'weak'
            out.setdefault(role, []).append(p['ser_comp'])
    return {role: float(np.mean(v)) for role, v in out.items()}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_per_cell', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--seeds', type=int, nargs='+',
                   default=[42, 43, 44, 45, 46])
    p.add_argument('--configs', type=str, nargs='+',
                   default=list(CONFIGS.keys()), choices=list(CONFIGS.keys()))
    p.add_argument('--ckpt_dir', type=str,
                   default=os.path.join(_HERE, 'checkpoints'))
    p.add_argument('--out_dir', type=str,
                   default=os.path.join(C.RESULTS_DIR, 'sir_sweep'))
    return p.parse_args()


def main():
    args = get_args()
    device = C.DEVICE
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Building SIR-sweep datasets (seed={SWEEP_SEED}, K=2, "
          f"SIRs={SIRS}, {args.n_per_cell} per (SIR,SNR) cell) ...",
          flush=True)
    datasets = build_all_datasets(args.n_per_cell)
    for sir, ds in datasets.items():
        print(f"  SIR={sir:>5.1f} dB: {len(ds)} samples", flush=True)

    # Model-independent mixture baseline (once).
    print("\nComputing mixture baseline (compensated SER/BER) ...", flush=True)
    baseline = {str(sir): baseline_scores(ds) for sir, ds in datasets.items()}
    bpath = os.path.join(args.out_dir, 'sir_sweep_baseline.json')
    with open(bpath, 'w') as f:
        json.dump({'seed': SWEEP_SEED, 'n_per_cell': args.n_per_cell,
                   'baseline': baseline}, f, indent=2)
    for sir in SIRS:
        b = baseline[str(sir)]
        print(f"  SIR={sir:>5.1f} dB: baseline SER={b['overall_ser']:.4f} "
              f"BER={b['overall_ber']:.4f} (n={b['n_pairs']})", flush=True)
    print(f"  saved {bpath}", flush=True)

    loaders = {sir: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=0)
               for sir, ds in datasets.items()}

    for cfg in args.configs:
        for seed in args.seeds:
            name = CONFIGS[cfg].format(seed)
            ckpt = os.path.join(args.ckpt_dir, name)
            out_path = os.path.join(args.out_dir,
                                    f'sir_sweep_{cfg}_s{seed}.json')
            if os.path.exists(out_path):
                print(f"\nskip {cfg} s{seed} (exists)", flush=True)
                continue
            print(f"\n=== {cfg} seed {seed}: {name} ===", flush=True)
            model, arch, _ = build_model_from_ckpt(ckpt, device)
            assert arch == 'slot', f'expected slot arch, got {arch}'
            res = {'checkpoint': name, 'config': cfg, 'seed': seed,
                   'n_per_cell': args.n_per_cell, 'sweep_seed': SWEEP_SEED,
                   'per_sir': {}}
            for sir in SIRS:
                records = collect_records_sir(
                    model, datasets[sir], loaders[sir], device,
                    C.VarKConfig.occ_threshold, C.VarKConfig.k_slots)
                agg = aggregate(records, 'slot')
                sep = agg['separation']
                sw = strong_weak_split(records, datasets[sir])
                res['per_sir'][str(sir)] = {
                    'si_sdri': sep['si_sdri'],
                    'ser_comp': sep['ser_comp'],
                    'ber_comp': sep['ber_comp'],
                    'miss_rate': sep['miss_rate'],
                    'per_snr': {str(s): {
                        'si_sdri': sep['per_snr'][s]['si_sdri'] if sep['per_snr'][s] else None,
                        'ser_comp': sep['per_snr'][s].get('ser_comp') if sep['per_snr'][s] else None,
                        'ber_comp': sep['per_snr'][s].get('ber_comp') if sep['per_snr'][s] else None,
                    } for s in (sep['per_snr'] or {})},
                    'ser_strong': sw.get('strong'),
                    'ser_weak': sw.get('weak'),
                }
                print(f"  SIR={sir:>5.1f}: SI-SDRi={sep['si_sdri']:.3f} dB  "
                      f"SER={sep['ser_comp']:.4f}  BER={sep['ber_comp']:.4f}"
                      + (f"  (strong {sw['strong']:.4f} / weak {sw['weak']:.4f})"
                         if sw else ''), flush=True)
            with open(out_path, 'w') as f:
                json.dump(res, f, indent=2,
                          default=lambda x: None if (
                              isinstance(x, float) and np.isnan(x)) else x)
            print(f"  saved {out_path}", flush=True)

    print("\nAll SIR-sweep evaluations done.")


if __name__ == '__main__':
    main()
