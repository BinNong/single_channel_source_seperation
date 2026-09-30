"""
Paper 3 — Review-2 experiment B: SIR sweep attribution
(reviewer Major Concern 2: "interference destroys the OOD margin" needs a
controlled interference-strength sweep, not just clean-vs-mixture endpoints).

Design
------
Fixed mixture SNR = 10 dB.  SIR grid {-20, -15, -10, -5, 0, 5, 10} dB plus
a CLEAN cell (no interferer, recorded as 'inf').  ku-type mixtures: one
known-modulation source + one unknown-modulation source.

SIR definition: power ratio of the UNKNOWN (target) source to the KNOWN
(interferer) source.  Sources are unit power, so the mixing gains satisfy
alpha_u^2 / alpha_k^2 = 10^(SIR/10).  We do NOT roll our own mixing: we use
the project's own generate_open_set_mixture with its native exact-SIR knob
sir_db (source-1/source-2 power ratio in dB; alpha = 1/(1+10^(-sir_db/20)),
which skips the legacy alpha~U(0.4,0.6) draw).  RRC pulse shaping, 3-tap
multipath, carrier offsets (2000/2005 Hz, ±5 Hz jitter) and the AWGN model
(noise_power = mean|mix_clean|^2 / 10^(SNR/10)) are therefore IDENTICAL to
the main experiments.  The unknown source alternates between the
source-1/source-2 side per repetition (side-balanced).

Per SIR cell: 16 (known mod x unknown mod) pairs x 16 repetitions = 256
mixtures per seed.  CLEAN cell: 64 single sources per modulation (8 mods x
64 = 512) at SNR 10 dB, generated exactly as eval_oracle_clean.py
(generate_open_set_signal + AWGN at noise_power = 1/10^(SNR/10)), pushed
through the network with the better-of-two-slots selection.

Determinism: one master stream np.random.RandomState(77777); loop order
SIR -> pair -> repetition, then CLEAN mod -> repetition; every sample
self-seeds the global RNG via np.random.seed(master.randint(0, 2**31-1))
before generation, so the set is bit-reproducible and identical across
model seeds.

Inference: frozen checkpoint, batch 64; mixture slots PIT-aligned to the
true sources by SI-SDR (same swap rule as evaluate._collect_predictions).
Scorers fitted on the per-seed refpool npz (kk, seed 88888):
mahalanobis (shared cov, shrink 0.1), prototype, vos (alpha 2.0, n 100,
seed 0).  Metric: same-mixture Mann-Whitney AUROC of the known slot vs the
unknown slot (open_set_metrics.auroc).  SI-SDR of each side is recorded as
a separation-quality diagnostic.

Expected sanity: SIR=0 dB -> AUROC ~= 0.5 (consistent with the headline
result at SIR~0); CLEAN -> mahalanobis ~= 0.77 (oracle_clean 10-dB value,
modulo the different reference fit — oracle_clean fit on clean references,
here everything fits on the refpool mixtures).

Outputs: results/sir_sweep.json (smoke: sir_sweep_smoke.json).

Usage:
  python eval_sir_sweep.py --smoke      # 1 seed, 2 reps/pair
  python eval_sir_sweep.py              # 5 seeds, full
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
    generate_open_set_mixture, generate_open_set_signal,
    MOD_KNOWN, MOD_UNKNOWN, MOD_ALL, MOD_TO_IDX,
)
from evaluate import _build_model_from_ckpt, _si_sdr_per_sample
from ood_baselines import mahalanobis_scores
from ood_scores import compute_prototypes, prototype_score, vos_score
from open_set_metrics import auroc

SIR_GRID = [-20, -15, -10, -5, 0, 5, 10]
MIXTURE_SNR = 10.0
N_REPS = 16                  # 16 pairs x 16 reps = 256 mixtures per SIR
N_CLEAN_PER_MOD = 64
MASTER_SEED = 77777
BASE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed{}_best'
SCORERS = ('mahalanobis', 'prototype', 'vos')
PAIRS = [(k, u) for k in MOD_KNOWN for u in MOD_UNKNOWN]


# ----------------------------------------------------------------------------
# Deterministic data generation
# ----------------------------------------------------------------------------
def gen_sir_mixtures(n_reps):
    """Returns per-SIR dict of arrays (mix, s1, s2, mod1, mod2)."""
    master = np.random.RandomState(MASTER_SEED)
    data = {}
    for sir in SIR_GRID:
        xs, s1s, s2s, m1s, m2s = [], [], [], [], []
        for mk, mu in PAIRS:
            for i in range(n_reps):
                np.random.seed(int(master.randint(0, 2 ** 31 - 1)))
                if i % 2 == 0:
                    # source1 = known (interferer), source2 = unknown (target)
                    mix, s1, s2, m1, m2 = generate_open_set_mixture(
                        C.SIGNAL_LENGTH, C.SAMPLE_RATE, MIXTURE_SNR, mk, mu,
                        sir_db=-float(sir))
                else:
                    # source1 = unknown (target), source2 = known (interferer)
                    mix, s1, s2, m1, m2 = generate_open_set_mixture(
                        C.SIGNAL_LENGTH, C.SAMPLE_RATE, MIXTURE_SNR, mu, mk,
                        sir_db=+float(sir))
                xs.append(mix)
                s1s.append(s1)
                s2s.append(s2)
                m1s.append(m1)
                m2s.append(m2)
        data[sir] = {
            'x': np.stack(xs).astype(np.complex64),
            's1': np.stack(s1s).astype(np.complex64),
            's2': np.stack(s2s).astype(np.complex64),
            'mod1': np.asarray(m1s, dtype=np.int64),
            'mod2': np.asarray(m2s, dtype=np.int64),
        }
    return data, master


def gen_clean(master, n_per_mod):
    """Single sources + AWGN at MIXTURE_SNR; continues the master stream."""
    xs, ss, ms = [], [], []
    for mod in MOD_ALL:
        for _ in range(n_per_mod):
            np.random.seed(int(master.randint(0, 2 ** 31 - 1)))
            s, _ = generate_open_set_signal(
                n_symbols=C.N_SYMBOLS, carrier_freq=2000.0,
                sample_rate=C.SAMPLE_RATE, signal_length=C.SIGNAL_LENGTH,
                mod_type=mod, roll_off=C.ROLL_OFF, num_taps=C.NUM_TAPS,
                apply_fading=True, fading_taps=C.FADING_TAPS)
            noise_power = 1.0 / (10.0 ** (MIXTURE_SNR / 10.0))
            noise = np.sqrt(noise_power / 2.0) * (
                np.random.randn(C.SIGNAL_LENGTH)
                + 1j * np.random.randn(C.SIGNAL_LENGTH))
            xs.append((s + noise).astype(np.complex64))
            ss.append(np.asarray(s, dtype=np.complex64))
            ms.append(MOD_TO_IDX[mod])
    return (np.stack(xs), np.stack(ss), np.asarray(ms, dtype=np.int64))


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
@torch.no_grad()
def infer_mixtures(model, cell, device, batch_size):
    """PIT-align slots to (s1, s2); returns aligned emb + si_sdr + mods."""
    x_all = cell['x']
    embs, sis, mods = [], [], []
    for i in range(0, len(x_all), batch_size):
        x = torch.from_numpy(x_all[i:i + batch_size]).unsqueeze(1).to(device)
        s1 = torch.from_numpy(cell['s1'][i:i + batch_size]).unsqueeze(1).to(device)
        s2 = torch.from_numpy(cell['s2'][i:i + batch_size]).unsqueeze(1).to(device)
        s1h, s2h, e1, e2, _l1, _l2 = model(x)
        d11 = _si_sdr_per_sample(s1h, s1)
        d22 = _si_sdr_per_sample(s2h, s2)
        d12 = _si_sdr_per_sample(s1h, s2)
        d21 = _si_sdr_per_sample(s2h, s1)
        swap = ((d12 + d21) > (d11 + d22)).view(-1, 1)
        # per-sample alignment to TRUE source 1 / source 2
        e_src1 = torch.where(swap, e2, e1)
        e_src2 = torch.where(swap, e1, e2)
        si_src1 = _si_sdr_per_sample(torch.where(swap.view(-1, 1, 1), s2h, s1h), s1)
        si_src2 = _si_sdr_per_sample(torch.where(swap.view(-1, 1, 1), s1h, s2h), s2)
        embs.append(torch.stack([e_src1, e_src2], dim=1).cpu().numpy())  # (B,2,D)
        sis.append(torch.stack([si_src1, si_src2], dim=1).cpu().numpy())  # (B,2)
        m1 = cell['mod1'][i:i + batch_size]
        m2 = cell['mod2'][i:i + batch_size]
        mods.append(np.stack([m1, m2], axis=1))                           # (B,2)
    return (np.concatenate(embs), np.concatenate(sis), np.concatenate(mods))


@torch.no_grad()
def infer_clean(model, xs, ss, device, batch_size):
    """Better-of-two-slots embedding for a single source (oracle_clean style)."""
    embs = []
    for i in range(0, len(xs), batch_size):
        x = torch.from_numpy(xs[i:i + batch_size]).unsqueeze(1).to(device)
        s = torch.from_numpy(ss[i:i + batch_size]).unsqueeze(1).to(device)
        s1, s2, e1, e2, _l1, _l2 = model(x)
        d1 = _si_sdr_per_sample(s1, s)
        d2 = _si_sdr_per_sample(s2, s)
        pick2 = (d2 > d1).view(-1, 1)
        embs.append(torch.where(pick2, e2, e1).cpu().numpy())
    return np.concatenate(embs)


# ----------------------------------------------------------------------------
# Scorers (refpool-fitted)
# ----------------------------------------------------------------------------
def _vos(emb, protos, chunk=1024):
    return np.concatenate([
        vos_score(emb[i:i + chunk], protos, alpha=C.VOS_ALPHA,
                  n_per_class=C.VOS_N_SYNTHETIC, seed=0)
        for i in range(0, len(emb), chunk)])


def score_all(ref_emb, ref_mods, emb):
    protos = compute_prototypes(ref_emb, ref_mods, C.NUM_KNOWN_CLASSES)
    return {
        'mahalanobis': mahalanobis_scores(ref_emb, ref_mods, emb,
                                          n_classes=C.NUM_KNOWN_CLASSES,
                                          shrink=0.1),
        'prototype': prototype_score(emb, protos),
        'vos': _vos(emb, protos),
    }


def _ms(vals):
    v = np.asarray(vals, dtype=np.float64)
    return {'mean': float(v.mean()), 'std': float(v.std())}


def _round(obj):
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 4)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def get_args():
    p = argparse.ArgumentParser(description='SIR-sweep OOD attribution')
    p.add_argument('--seeds', type=str, default='42,43,44,45,46')
    p.add_argument('--n_reps', type=int, default=N_REPS)
    p.add_argument('--n_clean_per_mod', type=int, default=N_CLEAN_PER_MOD)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--smoke', action='store_true',
                   help='1 seed, 2 reps/pair (32 mixtures/SIR), 8 clean/mod; '
                        'writes sir_sweep_smoke.json.')
    p.add_argument('--ta', action='store_true',
                   help='Fit prototype/vos/mahalanobis on the truth-anchored '
                        '*_refpool_ta.npz (dump_ta_scores.py) instead of the '
                        'legacy swap-contaminated refpool; writes '
                        'sir_sweep_ta.json.')
    p.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    return p.parse_args()


def main():
    args = get_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    n_reps, n_clean = args.n_reps, args.n_clean_per_mod
    if args.smoke:
        seeds, n_reps, n_clean = seeds[:1], 2, 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[eval_sir_sweep] seeds={seeds} reps/pair={n_reps} '
          f'({len(PAIRS) * n_reps} mixtures/SIR) clean/mod={n_clean} '
          f'device={device} ta_refpool={args.ta}')

    print('Generating SIR-sweep mixtures (master seed', MASTER_SEED, ') ...')
    data, master = gen_sir_mixtures(n_reps)
    print('Generating CLEAN single-source set ...')
    cx, cs, cmods = gen_clean(master, n_clean)
    print(f'  clean: {len(cx)} samples')

    per_seed = {}
    for seed in seeds:
        ckpt = os.path.join(C.CHECKPOINT_DIR, BASE.format(seed) + '.pt')
        print(f'\n=== seed {seed} ===')
        model = _build_model_from_ckpt(ckpt, device)
        rp = np.load(os.path.join(
            C.RESULTS_DIR,
            BASE.format(seed) + ('_refpool_ta.npz' if args.ta
                                 else '_refpool.npz')))
        ref_emb, ref_mods = rp['ref_emb'], rp['ref_mods']

        seed_res = {}
        for sir in SIR_GRID:
            embs, sis, mods = infer_mixtures(model, data[sir], device,
                                             args.batch_size)
            flat_emb = embs.reshape(-1, embs.shape[-1])
            sc = score_all(ref_emb, ref_mods, flat_emb)
            flat_mods = mods.reshape(-1)
            flat_sis = sis.reshape(-1)
            is_unk = flat_mods >= C.NUM_KNOWN_CLASSES
            row = {'n_known': int((~is_unk).sum()), 'n_unknown': int(is_unk.sum()),
                   'si_sdr_known': float(flat_sis[~is_unk].mean()),
                   'si_sdr_unknown': float(flat_sis[is_unk].mean())}
            for m in SCORERS:
                row[m] = auroc(sc[m][~is_unk], sc[m][is_unk])
            seed_res[str(sir)] = row
            print(f"  SIR {sir:+4d} dB: " + '  '.join(
                f'{m}={row[m]:.3f}' for m in SCORERS)
                + f"   (SI-SDR k={row['si_sdr_known']:.2f} / "
                  f"u={row['si_sdr_unknown']:.2f} dB)")

        # ---- clean (inf) cell ----
        cemb = infer_clean(model, cx, cs, device, args.batch_size)
        csc = score_all(ref_emb, ref_mods, cemb)
        c_unk = cmods >= C.NUM_KNOWN_CLASSES
        crow = {'n_known': int((~c_unk).sum()), 'n_unknown': int(c_unk.sum())}
        for m in SCORERS:
            crow[m] = auroc(csc[m][~c_unk], csc[m][c_unk])
        seed_res['inf'] = crow
        print('  CLEAN (inf): ' + '  '.join(f'{m}={crow[m]:.3f}' for m in SCORERS))

        per_seed[str(seed)] = seed_res
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---- across-seed aggregation ----
    cells = [str(s) for s in SIR_GRID] + ['inf']
    across = {}
    for cell in cells:
        agg = {m: _ms([per_seed[s][cell][m] for s in per_seed]) for m in SCORERS}
        for k in ('si_sdr_known', 'si_sdr_unknown'):
            if k in per_seed[seeds[0].__str__()][cell]:
                agg[k] = _ms([per_seed[s][cell][k] for s in per_seed])
        across[cell] = agg

    # ---- sanity ----
    s0 = across['0']
    sc_inf = across['inf']
    sanity = {
        'sir0_in_chance_band': bool(all(0.40 <= s0[m]['mean'] <= 0.60
                                        for m in SCORERS)),
        'sir0_means': {m: s0[m]['mean'] for m in SCORERS},
        'clean_mahalanobis': sc_inf['mahalanobis']['mean'],
        'clean_mahalanobis_target': 0.770,
        'clean_mahalanobis_note': 'target = eval_oracle_clean 10-dB value '
                                  '(which fitted on a clean reference; here '
                                  'the fit is the refpool mixtures), seed '
                                  'noise expected',
        'clean_mahalanobis_in_band': bool(0.65 <= sc_inf['mahalanobis']['mean'] <= 0.88),
    }
    if args.ta:
        # WP1 fully-TA same-mixture reference (protocol_split_ta.json,
        # kuK_vs_kuU): wavg 0.543, 10-dB bin 0.613 — the SIR=0 cell here is
        # at SNR=10, so compare against the 10-dB bin within seed noise.
        sanity['wp1_same_mixture_prototype'] = {'wavg': 0.5433, 'bin10': 0.613}
        sanity['sir0_prototype_vs_wp1_bin10'] = {
            'value': s0['prototype']['mean'],
            'within_seed_noise': bool(abs(s0['prototype']['mean'] - 0.613) <= 0.10),
        }
        print(f"  SIR=0 prototype = {s0['prototype']['mean']:.4f} "
              f"(WP1 same-mixture 10-dB bin 0.613) -> "
              f"{'OK' if sanity['sir0_prototype_vs_wp1_bin10']['within_seed_noise'] else 'CHECK'}")
    print('\n=== SANITY ===')
    print(f"  SIR=0 means: {sanity['sir0_means']} "
          f"-> in [0.40,0.60]: {sanity['sir0_in_chance_band']}")
    print(f"  clean mahalanobis = {sanity['clean_mahalanobis']:.4f} "
          f"(target ~0.77) -> in [0.65,0.88]: {sanity['clean_mahalanobis_in_band']}")

    out = {
        'experiment': 'sir_sweep_attribution (review-2 Major Concern 2)',
        'config': {
            'seeds': seeds, 'mixture_snr_db': MIXTURE_SNR, 'sir_grid': SIR_GRID,
            'n_pairs': len(PAIRS), 'n_reps_per_pair': n_reps,
            'n_mixtures_per_sir': len(PAIRS) * n_reps,
            'n_clean_per_mod': n_clean, 'master_seed': MASTER_SEED,
            'smoke': bool(args.smoke),
            'sir_definition': 'P_unknown / P_known in dB; unit-power sources, '
                              'exact gains via generate_open_set_mixture '
                              'sir_db (alpha = 1/(1+10^(-sir_db/20)) = '
                              'source1/source-2 ratio in dB; unknown side '
                              'alternates per repetition)',
            'noise_model': 'noise_power = mean|mix_clean|^2 / 10^(SNR/10) '
                           '(project-native; identical to main experiments)',
            'scorer_fit': ('per-seed TRUTH-ANCHORED refpool npz '
                           '(*_refpool_ta.npz, kk, seed 88888, TA labels)'
                           if args.ta else
                           'per-seed refpool npz (kk, seed 88888)'),
            'metric': 'same-mixture Mann-Whitney AUROC, known slot vs '
                      'unknown slot (PIT-aligned by SI-SDR)',
            'clean_cell': 'single sources + AWGN at 10 dB through the '
                          'network, better-of-two-slots (eval_oracle_clean '
                          'style), refpool-fitted scorers',
        },
        'per_seed': per_seed,
        'across_seeds': across,
        'sanity': sanity,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    if args.smoke:
        fname = 'sir_sweep_ta_smoke.json' if args.ta else 'sir_sweep_smoke.json'
    else:
        fname = 'sir_sweep_ta.json' if args.ta else 'sir_sweep.json'
    fpath = os.path.join(args.out_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(_round(out), f, indent=1)
    print(f'\nSaved {fpath}')


if __name__ == '__main__':
    main()
