"""Paper 6 — E3: end-to-end blind pipeline evaluation.

Question (plan §4 E3): does the blind pipeline (pretrained SlotSepNet
slots -> BlindCarrierSync -> decisions) match paper 5's ORACLE-scored
numbers?  I.e., remove the oracle carrier assumption from the evaluation
pipeline.

Conventions are copied EXACTLY from paper5_task_oriented/evaluate.py so
the oracle arm reproduces the published config-(b) numbers:
  - deterministic test grid (seed 99999, SNR x K{1,2,3} cells,
    K=4 generated but excluded from scoring);
  - occupancy gating at 0.5, slots ordered by occupancy probability,
    occ_slots = order[:k_occ] (empty slots never scored);
  - Hungarian matching (cost = -SI-SDR) of occupied slots to true
    sources; only MATCHED pairs enter SER (paper5's pooling);
  - oracle arm: ser_comp.compute_ser_ber_compensated (same calls as
    paper5's --ser_comp route).

Blind arm per matched pair (no oracle information in the estimate path):
  blind_sync_known_mod(est_waveform, true mod_idx) -> synced symbols ->
  decisions, scored against ser_comp.ref_labels_only(ref, true carrier),
  minimised over the M-fold symmetry rotations (genie ambiguity
  resolution, same convention as E1); BER via ser_comp.GRAY_BITS from
  the same decisions.  The per-pair frequency error |Δf̂ − Δf_true| is
  recorded (leakage diagnostic for K >= 2 slots).

Diagnostics:
  - K=2 blind MIXTURE baseline: blind_sync on the raw mixture vs each
    source (config-independent, computed once).
  - --sic: crude successive-interference-cancellation variant — sync the
    slot AFTER subtracting the other matched slot estimates from the
    mixture (mix - sum_{j' != j} est_j'), for K >= 2 pairs.

Usage:
    python eval_blind_pipeline.py --n_per_cell 25 --configs mse --sic
    python eval_blind_pipeline.py --n_per_cell 100 --configs mse ser_mse
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# Vendored paper6 modules (self-contained; no sys.path tricks).
import config as C                                          # noqa: E402
from data_generator import (CommBSSVarKTestDataset,         # noqa: E402
                            MOD_TYPES, IDX_TO_MOD)
from models import SlotSepNet                               # noqa: E402
import ser_comp                                             # noqa: E402
from sync import blind_sync_known_mod, constellation_np     # noqa: E402

CKPT_PATTERNS = {
    'mse': 'slot_h64_l4_k13_bs16_lr0.001_mse_s{seed}_best.pt',      # cfg (b)
    'ser_mse': 'slot_h64_l4_k13_bs16_lr0.001_ser_mse_s{seed}_best.pt',  # cfg (d)
}


# ---------------------------------------------------------------------------
# Functions copied from paper5_task_oriented/evaluate.py (verbatim logic;
# copied not imported because that module does `import config` expecting
# paper5's config.py).
# ---------------------------------------------------------------------------
def _si_sdr_np(est, ref, eps=1e-8):
    """Copied from paper5 evaluate.py."""
    e = np.stack([est.real, est.imag], axis=-1).ravel()
    r = np.stack([ref.real, ref.imag], axis=-1).ravel()
    e = e - e.mean()
    r = r - r.mean()
    alpha = np.dot(r, e) / (np.dot(r, r) + eps)
    target = alpha * r
    noise = e - target
    return float(10 * np.log10(np.dot(target, target)
                               / (np.dot(noise, noise) + eps) + eps))


def build_slot_from_ckpt(ckpt_path, device):
    """Slot-branch of paper5 evaluate.build_model_from_ckpt."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get('args', {})
    model = SlotSepNet(
        hidden_channels=a.get('hidden', 64),
        n_layers=a.get('layers', 4),
        k_max=a.get('k_slots', 4),
        use_se=not a.get('no_se', False),
        head_embed_dim=64,
        use_count_head=not a.get('no_count_head', False),
        use_joint_head=a.get('lambda_llr', 0.0) > 0,
    )
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model, a


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def blind_ser_ber(est_np, mod_idx, ref_lab):
    """Blind receiver decisions vs reference labels, M-fold genie rotation
    resolution; BER via GRAY_BITS from the same decisions.
    Returns (ser, ber, df_est)."""
    res = blind_sync_known_mod(np.asarray(est_np, dtype=np.complex128),
                               mod_idx)
    z = res['z']
    const = constellation_np(mod_idx)
    m_order = C.SyncConfig.sym_order[MOD_TYPES[mod_idx]]
    best_ser, best_dec = 1.0, None
    for k in range(m_order):
        zr = z * np.exp(-1j * 2 * np.pi * k / m_order)
        d = np.argmin(np.abs(zr[:, None] - const[None, :]), axis=1)
        s = float(np.mean(d != ref_lab))
        if s < best_ser:
            best_ser, best_dec = s, d
    bits = ser_comp.GRAY_BITS[MOD_TYPES[mod_idx]]
    ber = float(np.mean(bits[best_dec] != bits[ref_lab]))
    return best_ser, ber, float(res['df'])


def ref_labels(src_np, mod_idx, carrier):
    return ser_comp.ref_labels_only(
        torch.from_numpy(np.asarray(src_np, dtype=np.complex64)),
        mod_type=MOD_TYPES[mod_idx],
        sample_rate=C.SignalConfig.sample_rate,
        n_symbols=C.SignalConfig.n_symbols,
        roll_off=C.SignalConfig.roll_off,
        num_taps=C.SignalConfig.num_taps,
        carrier_freq=float(carrier))


def oracle_ser_ber(est_np, src_np, mod_idx, carrier):
    return ser_comp.compute_ser_ber_compensated(
        torch.from_numpy(np.asarray(est_np, dtype=np.complex64)),
        torch.from_numpy(np.asarray(src_np, dtype=np.complex64)),
        mod_type=MOD_TYPES[mod_idx],
        sample_rate=C.SignalConfig.sample_rate,
        n_symbols=C.SignalConfig.n_symbols,
        roll_off=C.SignalConfig.roll_off,
        num_taps=C.SignalConfig.num_taps,
        carrier_freq=float(carrier))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_checkpoint(ckpt_path, ds, device, occ_threshold, use_sic,
                    progress_every=200):
    """One checkpoint over the test grid.  Returns a list of pair records
    plus the K=2 blind-mixture records are handled separately."""
    model, ckpt_args = build_slot_from_ckpt(ckpt_path, device)
    pairs = []
    n = len(ds)
    batch = []
    for idx in range(n):
        batch.append(ds[idx])
        if len(batch) == 16 or idx == n - 1:
            mix_b = torch.stack([b[0] for b in batch]).to(device)  # [B,1,T]
            slots, occ_logits, _cnt = model(mix_b)
            slots_np = slots.cpu().numpy()
            occ_np = torch.sigmoid(occ_logits).cpu().numpy()
            for bi, b in enumerate(batch):
                mix, sources, _occ, k, snr, mods, carriers = b
                kt = int(k)
                if kt > 3:
                    continue                     # K=4 cell: excluded
                mix_np = mix.numpy()[0]
                srcs = [sources.numpy()[j, 0] for j in range(kt)]
                mod_ids = [int(m) for m in mods.numpy()[:kt]]
                cars = [float(c) for c in carriers.numpy()[:kt]]
                # occupancy gating (paper5 convention)
                order = np.argsort(-occ_np[bi])
                k_occ = int((occ_np[bi] > occ_threshold).sum())
                ests = [slots_np[bi, j] for j in order[:k_occ]]
                if not ests:
                    continue                     # all sources missed
                # Hungarian match, cost = -SI-SDR (paper5 convention)
                cost = np.full((kt, len(ests)), 1e6)
                for i, s in enumerate(srcs):
                    for j, e in enumerate(ests):
                        cost[i, j] = -_si_sdr_np(e, s)
                rows, cols = linear_sum_assignment(cost)
                for i, j in zip(rows, cols):
                    e, s = ests[j], srcs[i]
                    midx, car = mod_ids[i], cars[i]
                    ser_o, ber_o = oracle_ser_ber(e, s, midx, car)
                    rl = ref_labels(s, midx, car)
                    ser_b, ber_b, df_b = blind_ser_ber(e, midx, rl)
                    rec = {
                        'snr': float(snr), 'k': kt, 'mod': midx,
                        'ser_oracle': ser_o, 'ber_oracle': ber_o,
                        'ser_blind': ser_b, 'ber_blind': ber_b,
                        'df_err_blind': abs(df_b - (car
                                        - C.SyncConfig.nominal_carrier)),
                    }
                    if use_sic and kt >= 2:
                        e_sic = mix_np - sum(ests[jj] for jj in range(len(ests))
                                             if jj != j)
                        ser_s, ber_s, _ = blind_ser_ber(e_sic, midx, rl)
                        rec['ser_sic'] = ser_s
                        rec['ber_sic'] = ber_s
                    pairs.append(rec)
            batch = []
            done = idx + 1
            if done % progress_every == 0 or done == n:
                print(f"    {done}/{n} samples, {len(pairs)} pairs",
                      flush=True)
    return pairs


def mixture_blind_k2(ds, progress_every=200):
    """K=2 blind mixture baseline: blind_sync on the raw mixture scored
    against each source (config-independent)."""
    recs = []
    n = len(ds)
    for idx in range(n):
        mix, sources, _occ, k, snr, mods, carriers = ds[idx]
        if int(k) != 2:
            continue
        mix_np = mix.numpy()[0]
        for i in range(2):
            midx = int(mods.numpy()[i])
            car = float(carriers.numpy()[i])
            rl = ref_labels(sources.numpy()[i, 0], midx, car)
            ser_b, ber_b, df_b = blind_ser_ber(mix_np, midx, rl)
            recs.append({'snr': float(snr), 'mod': midx,
                         'ser_mixblind': ser_b, 'ber_mixblind': ber_b,
                         'df_err': abs(df_b - (car
                                 - C.SyncConfig.nominal_carrier))})
        if (idx + 1) % progress_every == 0:
            print(f"    mixture-blind: {idx + 1}/{n}", flush=True)
    return recs


def _mean(recs, key):
    v = [r[key] for r in recs if key in r]
    return float(np.mean(v)) if v else None


def aggregate(pairs, seeds):
    """Per-K x per-SNR means + pooled, per seed and seed-averaged."""
    snrs = sorted({r['snr'] for r in pairs})
    out = {}
    for seed in sorted(seeds):
        ps = [r for r in pairs if r['seed'] == seed]
        for k in (1, 2, 3):
            for snr in snrs:
                cell = [r for r in ps if r['k'] == k and r['snr'] == snr]
                if not cell:
                    continue
                out[f's{seed}/k{k}/{snr:g}'] = {
                    'n': len(cell),
                    'ser_oracle': _mean(cell, 'ser_oracle'),
                    'ser_blind': _mean(cell, 'ser_blind'),
                    'ber_oracle': _mean(cell, 'ber_oracle'),
                    'ber_blind': _mean(cell, 'ber_blind'),
                    'ser_sic': _mean(cell, 'ser_sic'),
                    'df_err_blind_med': float(np.median(
                        [r['df_err_blind'] for r in cell])),
                }
    # seed-averaged summary per K and pooled
    def pool(recs):
        keys = ('ser_oracle', 'ser_blind', 'ber_oracle', 'ber_blind',
                'ser_sic')
        return {k: _mean(recs, k) for k in keys}
    summ = {'pooled': pool(pairs)}
    for k in (1, 2, 3):
        summ[f'k{k}'] = pool([r for r in pairs if r['k'] == k])
    per_mod = {}
    for m in range(len(MOD_TYPES)):
        per_mod[MOD_TYPES[m]] = pool([r for r in pairs if r['mod'] == m])
    summ['per_mod'] = per_mod
    out['summary'] = summ
    return out


def main():
    p = argparse.ArgumentParser(description='E3: blind pipeline evaluation')
    p.add_argument('--n_per_cell', type=int, default=100)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--configs', type=str, nargs='+', default=['mse'],
                   choices=sorted(CKPT_PATTERNS))
    p.add_argument('--sic', action='store_true',
                   help='also run the crude-SIC diagnostic at K>=2')
    p.add_argument('--ckpt_dir', type=str, default=C.CHECKPOINT_DIR)
    p.add_argument('--occ_threshold', type=float, default=0.5,
                   help='paper5 occupancy gating threshold')
    p.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    p.add_argument('--skip_mixture_baseline', action='store_true')
    args = p.parse_args()
    device = C.DEVICE

    print("Building deterministic test grid (seed 99999) ...", flush=True)
    ds = CommBSSVarKTestDataset(
        n_per_cell=args.n_per_cell,
        snr_points=C.SignalConfig.snr_test_points,
        mod_types=MOD_TYPES,
        k_lo=1, k_hi=3, k_extrap=4, k_slots=4,
        signal_length=C.SignalConfig.signal_length,
        sample_rate=C.SignalConfig.sample_rate,
        carrier_base=C.SignalConfig.carrier_base,
        freq_gap_range=C.SignalConfig.freq_gap_range,
        seed=C.DataConfig.test_seed,
        return_carriers=True)
    print(f"  {len(ds)} samples", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # K=2 blind mixture baseline (config-independent; run once)
    if not args.skip_mixture_baseline:
        print("K=2 blind mixture baseline ...", flush=True)
        mrecs = mixture_blind_k2(ds)
        snrs = sorted({r['snr'] for r in mrecs})
        mout = {'per_snr': {f'{s:g}': {
            'ser_mixblind': _mean([r for r in mrecs if r['snr'] == s],
                                  'ser_mixblind'),
            'ber_mixblind': _mean([r for r in mrecs if r['snr'] == s],
                                  'ber_mixblind'),
            'n': sum(1 for r in mrecs if r['snr'] == s)} for s in snrs},
            'pooled_ser': _mean(mrecs, 'ser_mixblind'),
            'pooled_ber': _mean(mrecs, 'ber_mixblind')}
        mpath = os.path.join(args.out_dir, 'e3_mixture_blind_k2.json')
        with open(mpath, 'w') as f:
            json.dump(mout, f, indent=2)
        print(f"  pooled blind-mixture SER={mout['pooled_ser']:.4f} "
              f"(saved {mpath})", flush=True)

    for cfg in args.configs:
        all_pairs = []
        for seed in args.seeds:
            ckpt = os.path.join(
                args.ckpt_dir, CKPT_PATTERNS[cfg].format(seed=seed))
            print(f"[{cfg}] seed {seed}: {os.path.basename(ckpt)}",
                  flush=True)
            pairs = eval_checkpoint(ckpt, ds, device,
                                    args.occ_threshold, args.sic)
            for r in pairs:
                r['seed'] = seed
                r['config'] = cfg
            all_pairs.extend(pairs)
            # incremental save (long runs survive interruptions)
            os.makedirs(args.out_dir, exist_ok=True)
            tmp = {'config': cfg, 'seeds_done': sorted(
                {r['seed'] for r in all_pairs}),
                'n_per_cell': args.n_per_cell, 'sic': args.sic,
                'pairs': all_pairs}
            with open(os.path.join(args.out_dir,
                                   f'e3_blind_pipeline_{cfg}_pairs.json'),
                      'w') as f:
                json.dump(tmp, f)
        agg = aggregate(all_pairs, sorted({r['seed'] for r in all_pairs}))
        agg_out = {'config': cfg, 'n_per_cell': args.n_per_cell,
                   'sic': args.sic, 'aggregate': agg}
        apath = os.path.join(args.out_dir, f'e3_blind_pipeline_{cfg}.json')
        with open(apath, 'w') as f:
            json.dump(agg_out, f, indent=2)
        print(f"\n[{cfg}] saved {apath}", flush=True)
        # print the summary table
        s = agg['summary']
        hdr = f"{'K':>3s} | {'SER ora':>8s} | {'SER blind':>9s} | " \
              f"{'SER sic':>8s} | {'BER ora':>8s} | {'BER blind':>9s}"
        print(hdr)
        print('-' * len(hdr))
        for key in ('k1', 'k2', 'k3', 'pooled'):
            r = s[key]
            print(f"{key:>3s} | {r['ser_oracle']:8.4f} | "
                  f"{r['ser_blind']:9.4f} | "
                  f"{(r['ser_sic'] if r['ser_sic'] is not None else float('nan')):8.4f} | "
                  f"{r['ber_oracle']:8.4f} | {r['ber_blind']:9.4f}")
        print("per-mod (ser oracle/blind): "
              + '  '.join(f"{m}: {v['ser_oracle']:.4f}/{v['ser_blind']:.4f}"
                          for m, v in s['per_mod'].items()))

    print("\nE3 evaluation done.")


if __name__ == '__main__':
    main()
