"""
Paper 3 — Revision experiment: oracle clean-source OOD probe.

Question: with NO interferer at all (a single source + AWGN), can post-hoc
OOD scorers tell the unknown modulations from the known ones using the SAME
frozen OpenSetCSE embeddings?  This disentangles "the separator distorts
the representations" from "the embedding has no known/unknown margin".

Protocol
--------
EVAL set: for each mod in MOD_ALL (all 8), each SNR in
{-10,-5,0,5,10,15,20}, n=192 samples:

    x = s + n,   s unit-power via generate_open_set_signal (the project's
    own dispatcher — OFDM through _apply_ofdm_tail, 64QAM / pi/4-DQPSK
    through _apply_paper1_pipeline, known 4 + MSK through paper1's
    generate_single_signal; same defaults as the test set:
    carrier_freq 2000.0 Hz — no interferer here — sample_rate 16000,
    signal_length 4096, n_symbols 256, roll_off 0.35, num_taps 64,
    apply_fading=True, fading_taps=3, default ±5 Hz carrier jitter),

    n complex AWGN with noise_power = 1 / 10^(snr/10) relative to the
    unit-power source (per-dimension scale sqrt(noise_power/2), matching
    generate_open_set_mixture).

REFERENCE set (scorer fitting only): same construction, KNOWN mods only.

Determinism (documented per spec): each sample is self-seeded from a
dedicated master stream, consumed in a FIXED nested-loop order —

    EVAL:      master = np.random.RandomState(55555)
               for mod in MOD_ALL (fixed order BPSK, QPSK, 8PSK, 16QAM,
                                   64QAM, PI4_DQPSK, MSK, OFDM_QPSK)
                 for snr in [-10, -5, 0, 5, 10, 15, 20]
                   for i in range(n)
                     np.random.seed(master.randint(0, 2**31 - 1))
                     -> generate_open_set_signal(...) then the AWGN draw
    REFERENCE: identical, master = np.random.RandomState(66666),
               MOD_KNOWN only.

Because every sample seeds the global RNG itself, the sets are bit-
reproducible independently of any other consumer of np.random.  The sets
depend on no model seed, so they are generated ONCE and reused for all
seed checkpoints.

Inference (batch 64): model(x) -> PIT-align the two output slots to the
single true source s by SI-SDR (evaluate._si_sdr_per_sample) -> keep the
better slot's embedding and logits.

Scorers (larger = more OOD), fitted on the clean REFERENCE set only:
  energy (T=1.0) and msp from logits; prototype (class means of reference
  embeddings); mahalanobis (shared covariance, shrinkage 0.1 —
  ood_baselines.mahalanobis_scores); vos (alpha=2.0, n_per_class=100,
  seed=0 — ood_scores.vos_score).  AUROC = Mann-Whitney rank AUROC from
  open_set_metrics.auroc (no sklearn).

Metrics per scorer: per-SNR AUROC (eval known pool vs eval unknown pool at
the same SNR), pooled AUROC, pair-weighted wavg across SNR bins
(weights n_known * n_unknown), and per-unknown-class AUROC (pooled across
SNR against the full known pool).

Outputs
-------
  results/oracle_clean_ood.json         (full run)
  results/oracle_clean_ood_smoke.json   (--smoke)

Usage
-----
  python eval_oracle_clean.py --smoke       # 1 seed, n=8
  python eval_oracle_clean.py               # 5 seeds, n=192
  python eval_oracle_clean.py --seeds 42,43 --n 192
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import config as C
from data_generator_extended import (
    generate_open_set_signal, MOD_ALL, MOD_KNOWN, MOD_UNKNOWN, MOD_TO_IDX,
)
from evaluate import _build_model_from_ckpt, _si_sdr_per_sample
from ood_baselines import mahalanobis_scores, msp_scores
from ood_scores import (
    compute_prototypes, energy_score, prototype_score, vos_score,
)
from open_set_metrics import auroc

SNR_POINTS = [int(s) for s in C.SNR_TEST_POINTS]
CKPT_TEMPLATE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed{seed}_best.pt'
SCORERS = ('energy', 'msp', 'prototype', 'mahalanobis', 'vos')
EVAL_MASTER_SEED = 55555
REF_MASTER_SEED = 66666
CARRIER_FREQ = 2000.0   # no interferer — same carrier for every source


# ----------------------------------------------------------------------------
# Deterministic single-source set generation
# ----------------------------------------------------------------------------
def gen_single_source_set(mods, n_per_cell: int, master_seed: int):
    """Returns (x, s, mod_idx, snr) numpy arrays; see header for the RNG
    protocol (fixed nested-loop order: mod -> snr -> sample)."""
    master = np.random.RandomState(master_seed)
    xs, ss, mods_l, snrs_l = [], [], [], []
    for mod in mods:
        for snr in SNR_POINTS:
            for _ in range(n_per_cell):
                np.random.seed(int(master.randint(0, 2 ** 31 - 1)))
                s, _symbols = generate_open_set_signal(
                    n_symbols=C.N_SYMBOLS,
                    carrier_freq=CARRIER_FREQ,
                    sample_rate=C.SAMPLE_RATE,
                    signal_length=C.SIGNAL_LENGTH,
                    mod_type=mod,
                    roll_off=C.ROLL_OFF,
                    num_taps=C.NUM_TAPS,
                    apply_fading=True,
                    fading_taps=C.FADING_TAPS,
                )
                noise_power = 1.0 / (10.0 ** (snr / 10.0))
                noise = np.sqrt(noise_power / 2.0) * (
                    np.random.randn(C.SIGNAL_LENGTH)
                    + 1j * np.random.randn(C.SIGNAL_LENGTH)
                )
                xs.append((s + noise).astype(np.complex64))
                ss.append(np.asarray(s, dtype=np.complex64))
                mods_l.append(MOD_TO_IDX[mod])
                snrs_l.append(float(snr))
    return (np.stack(xs), np.stack(ss),
            np.asarray(mods_l, dtype=np.int64),
            np.asarray(snrs_l, dtype=np.float32))


# ----------------------------------------------------------------------------
# Inference: PIT-align both slots to the single true source, keep the better
# ----------------------------------------------------------------------------
@torch.no_grad()
def infer_best_slot(model, xs, ss, device, batch_size: int = 64):
    embs, logits = [], []
    for i in range(0, len(xs), batch_size):
        x = torch.from_numpy(xs[i:i + batch_size]).unsqueeze(1).to(device)
        s = torch.from_numpy(ss[i:i + batch_size]).unsqueeze(1).to(device)
        s1, s2, e1, e2, l1, l2 = model(x)
        d1 = _si_sdr_per_sample(s1, s)
        d2 = _si_sdr_per_sample(s2, s)
        pick2 = (d2 > d1).view(-1, 1)
        embs.append(torch.where(pick2, e2, e1).cpu().numpy())
        logits.append(torch.where(pick2, l2, l1).cpu().numpy())
    return np.concatenate(embs), np.concatenate(logits)


# ----------------------------------------------------------------------------
# Scorers (fitted on the reference set only)
# ----------------------------------------------------------------------------
def _chunked_vos(emb, prototypes, chunk: int = 2048):
    """vos_score materialises an (N, K*n_per_class, D) distance tensor;
    chunk it to bound memory on the 10k-sample eval set."""
    return np.concatenate([
        vos_score(emb[i:i + chunk], prototypes,
                  alpha=C.VOS_ALPHA, n_per_class=C.VOS_N_SYNTHETIC, seed=0)
        for i in range(0, len(emb), chunk)
    ])


def score_eval_set(ref_emb, ref_mods, ev_emb, ev_logits):
    protos = compute_prototypes(ref_emb, ref_mods, C.NUM_KNOWN_CLASSES)
    return {
        'energy': energy_score(ev_logits, temperature=C.ENERGY_TEMPERATURE),
        'msp': msp_scores(ev_logits),
        'prototype': prototype_score(ev_emb, protos),
        'mahalanobis': mahalanobis_scores(ref_emb, ref_mods, ev_emb,
                                          n_classes=C.NUM_KNOWN_CLASSES,
                                          shrink=0.1),
        'vos': _chunked_vos(ev_emb, protos),
    }


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def scorer_metrics(scores, ev_mods, ev_snr):
    is_unk = ev_mods >= C.NUM_KNOWN_CLASSES
    out = {'pooled': auroc(scores[~is_unk], scores[is_unk])}

    per_snr, w_num, w_den = {}, 0.0, 0.0
    for s in SNR_POINTS:
        mk = (ev_snr == s) & ~is_unk
        mu = (ev_snr == s) & is_unk
        a = auroc(scores[mk], scores[mu])
        w = int(mk.sum()) * int(mu.sum())
        per_snr[str(s)] = {'auroc': a, 'n_known': int(mk.sum()),
                           'n_unknown': int(mu.sum())}
        w_num += a * w
        w_den += w
    out['per_snr'] = per_snr
    out['wavg'] = w_num / w_den if w_den else float('nan')

    out['per_unknown_class'] = {
        mod: auroc(scores[~is_unk], scores[ev_mods == MOD_TO_IDX[mod]])
        for mod in MOD_UNKNOWN
    }
    return out


# ----------------------------------------------------------------------------
# Across-seed aggregation
# ----------------------------------------------------------------------------
def _ms(values):
    v = np.asarray(values, dtype=np.float64)
    return {'mean': float(v.mean()), 'std': float(v.std()),
            'n_seeds': int(v.size)}


def aggregate(per_seed: dict) -> dict:
    seeds = list(per_seed.keys())
    agg = {}
    for sc in SCORERS:
        agg[sc] = {
            'pooled': _ms([per_seed[sd][sc]['pooled'] for sd in seeds]),
            'wavg': _ms([per_seed[sd][sc]['wavg'] for sd in seeds]),
            'per_snr': {
                str(s): _ms([per_seed[sd][sc]['per_snr'][str(s)]['auroc']
                             for sd in seeds])
                for s in SNR_POINTS
            },
            'per_unknown_class': {
                mod: _ms([per_seed[sd][sc]['per_unknown_class'][mod]
                          for sd in seeds])
                for mod in MOD_UNKNOWN
            },
        }
    return agg


def _round(obj):
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 3)
    return obj


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description='Oracle clean-source OOD probe')
    p.add_argument('--seeds', type=str, default='42,43,44,45,46')
    p.add_argument('--n', type=int, default=192,
                   help='Samples per (mod, SNR) cell for both sets.')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--smoke', action='store_true',
                   help='Tiny sets (n=8), first seed only; writes '
                        'oracle_clean_ood_smoke.json.')
    p.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    return p.parse_args()


def main():
    args = get_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    n = args.n
    if args.smoke:
        seeds = seeds[:1]
        n = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'[eval_oracle_clean] seeds={seeds} n={n} device={device}')
    print(f'Generating EVAL set (8 mods x {len(SNR_POINTS)} SNRs x {n}, '
          f'master seed {EVAL_MASTER_SEED}) ...')
    ev_x, ev_s, ev_mods, ev_snr = gen_single_source_set(
        MOD_ALL, n, EVAL_MASTER_SEED)
    print(f'  eval: {len(ev_x)} samples')
    print(f'Generating REFERENCE set (4 known mods, master seed '
          f'{REF_MASTER_SEED}) ...')
    rf_x, rf_s, rf_mods, _rf_snr = gen_single_source_set(
        MOD_KNOWN, n, REF_MASTER_SEED)
    print(f'  reference: {len(rf_x)} samples')

    per_seed = {}
    for seed in seeds:
        ckpt = os.path.join(C.CHECKPOINT_DIR, CKPT_TEMPLATE.format(seed=seed))
        print(f'\n=== seed {seed}: {os.path.basename(ckpt)} ===')
        model = _build_model_from_ckpt(ckpt, device)
        ev_emb, ev_logits = infer_best_slot(model, ev_x, ev_s, device,
                                            args.batch_size)
        rf_emb, rf_logits = infer_best_slot(model, rf_x, rf_s, device,
                                            args.batch_size)
        del rf_logits  # reference logits unused: scorers are logit-free fits
        scores = score_eval_set(rf_emb, rf_mods, ev_emb, ev_logits)
        per_seed[str(seed)] = {
            sc: scorer_metrics(scores[sc], ev_mods, ev_snr) for sc in SCORERS
        }
        line = '  '.join(
            f'{sc}: pooled={per_seed[str(seed)][sc]["pooled"]:.3f} '
            f'wavg={per_seed[str(seed)][sc]["wavg"]:.3f}' for sc in SCORERS)
        print('  ' + line)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    out = {
        'experiment': 'oracle_clean_source_ood_probe',
        'config': {
            'seeds': seeds, 'n_per_cell': n, 'snr_points': SNR_POINTS,
            'eval_master_seed': EVAL_MASTER_SEED,
            'ref_master_seed': REF_MASTER_SEED,
            'carrier_freq': CARRIER_FREQ, 'smoke': bool(args.smoke),
            'loop_order': 'for mod in MOD_ALL -> for snr in SNR_POINTS -> '
                          'for i in range(n); np.random.seed(master.randint) '
                          'per sample',
        },
        'per_seed': per_seed,
        'across_seeds': aggregate(per_seed),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    fname = 'oracle_clean_ood_smoke.json' if args.smoke else 'oracle_clean_ood.json'
    fpath = os.path.join(args.out_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(_round(out), f, indent=2)
    print(f'\nSaved {fpath}')

    print('\n=== Across-seed AUROC (mean ± std) ===')
    for sc in SCORERS:
        a = out['across_seeds'][sc]
        print(f'  {sc:>11s}: pooled={a["pooled"]["mean"]:.3f} ± '
              f'{a["pooled"]["std"]:.3f}   wavg={a["wavg"]["mean"]:.3f} ± '
              f'{a["wavg"]["std"]:.3f}')


if __name__ == '__main__':
    main()
