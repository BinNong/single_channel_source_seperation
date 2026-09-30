"""
Paper 3 — Review-2 experiment B follow-up: controls for the SIR=0 cell.

eval_sir_sweep.py found the same-mixture AUROC at exact SIR=0 / SNR=10 dB
elevated above chance for prototype/vos (~0.68), while the main-test-set
same-mixture comparison (eval_protocol_split.py C3, kuK_vs_kuU, 10-dB bin)
sits at ~0.51.  The two cells differ in exactly two design choices:

  D1: exact SIR=0 (alpha=0.5 fixed) vs legacy alpha~U(0.4,0.6) mixing;
  D2: data RNG stream (77777 vs test seed 99999).

This control script disentangles them at SNR=10 dB, 16 (known,unknown)
pairs x 16 reps = 256 mixtures per cell, 5 seeds:

  cell exact_sir0 : sir_db=0 (B's design), fresh stream 77999
  cell legacy_alpha: sir_db=None -> alpha~U(0.4,0.6) (main-test design),
                     same fresh stream 77999

It also re-scores a freshly generated clean single-source cell (SNR=10,
64/mod, stream 77888) with mahalanobis fitted on a CLEAN known-only
reference (stream 66700) — reproducing eval_oracle_clean.py's fitting
choice — next to the refpool fit used in eval_sir_sweep.py, to document
that the clean-cell 0.62 vs 0.77 gap is a reference-fit effect, not a
data/pipeline discrepancy.

Output: results/sir_sweep_controls.json.
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
from eval_sir_sweep import (
    infer_mixtures, infer_clean, score_all, _vos, PAIRS, MIXTURE_SNR,
)

BASE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed{}_best'
SCORERS = ('mahalanobis', 'prototype', 'vos')
N_REPS = 16
CONTROL_STREAM = 77999
CLEAN_EVAL_STREAM = 77888
CLEAN_REF_STREAM = 66700
N_CLEAN_PER_MOD = 64


def gen_cell(sir_db_mode, master_seed, n_reps):
    """One 256-mixture SNR=10 cell.  sir_db_mode: 'exact0' or None (legacy
    alpha~U(0.4,0.6)).  Same loop order / self-seeding as eval_sir_sweep."""
    master = np.random.RandomState(master_seed)
    xs, s1s, s2s, m1s, m2s = [], [], [], [], []
    for mk, mu in PAIRS:
        for i in range(n_reps):
            np.random.seed(int(master.randint(0, 2 ** 31 - 1)))
            if i % 2 == 0:
                args = (mk, mu)
            else:
                args = (mu, mk)
            # at SIR=0 the sir_db sign is irrelevant (alpha=0.5 either way)
            kw = {} if sir_db_mode is None else {'sir_db': 0.0}
            mix, s1, s2, m1, m2 = generate_open_set_mixture(
                C.SIGNAL_LENGTH, C.SAMPLE_RATE, MIXTURE_SNR, *args, **kw)
            xs.append(mix); s1s.append(s1); s2s.append(s2)
            m1s.append(m1); m2s.append(m2)
    return {'x': np.stack(xs).astype(np.complex64),
            's1': np.stack(s1s).astype(np.complex64),
            's2': np.stack(s2s).astype(np.complex64),
            'mod1': np.asarray(m1s, dtype=np.int64),
            'mod2': np.asarray(m2s, dtype=np.int64)}


def gen_singles(master_seed, mods, n_per_mod):
    master = np.random.RandomState(master_seed)
    xs, ss, ms = [], [], []
    for mod in mods:
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
    return np.stack(xs), np.stack(ss), np.asarray(ms, dtype=np.int64)


def mixture_aurocs(model, cell, ref_emb, ref_mods, device, batch_size):
    embs, _sis, mods = infer_mixtures(model, cell, device, batch_size)
    flat_emb = embs.reshape(-1, embs.shape[-1])
    sc = score_all(ref_emb, ref_mods, flat_emb)
    is_unk = mods.reshape(-1) >= C.NUM_KNOWN_CLASSES
    return {m: auroc(sc[m][~is_unk], sc[m][is_unk]) for m in SCORERS}


def _ms(vals):
    v = np.asarray(vals, dtype=np.float64)
    return {'mean': float(v.mean()), 'std': float(v.std()), 'n': int(v.size)}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default='42,43,44,45,46')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[sir_sweep_controls] seeds={seeds} device={device}')

    print('Generating control cells (stream', CONTROL_STREAM, ') ...')
    cell_exact = gen_cell('exact0', CONTROL_STREAM, N_REPS)
    cell_legacy = gen_cell(None, CONTROL_STREAM, N_REPS)
    print('Generating clean eval + clean reference singles ...')
    cx, cs, cmods = gen_singles(CLEAN_EVAL_STREAM, MOD_ALL, N_CLEAN_PER_MOD)
    rx, rs, rmods = gen_singles(CLEAN_REF_STREAM, MOD_KNOWN, N_CLEAN_PER_MOD)

    per_seed = {}
    for seed in seeds:
        ckpt = os.path.join(C.CHECKPOINT_DIR, BASE.format(seed) + '.pt')
        print(f'\n=== seed {seed} ===')
        model = _build_model_from_ckpt(ckpt, device)
        rp = np.load(os.path.join(C.RESULTS_DIR, BASE.format(seed) + '_refpool.npz'))
        ref_emb, ref_mods = rp['ref_emb'], rp['ref_mods']

        res = {
            'exact_sir0': mixture_aurocs(model, cell_exact, ref_emb, ref_mods,
                                         device, args.batch_size),
            'legacy_alpha': mixture_aurocs(model, cell_legacy, ref_emb, ref_mods,
                                           device, args.batch_size),
        }
        print('  exact_sir0  :', {m: round(v, 4) for m, v in res['exact_sir0'].items()})
        print('  legacy_alpha:', {m: round(v, 4) for m, v in res['legacy_alpha'].items()})

        # clean eval cell, two fits: refpool vs clean reference
        cemb = infer_clean(model, cx, cs, device, args.batch_size)
        c_unk = cmods >= C.NUM_KNOWN_CLASSES
        sc_pool = score_all(ref_emb, ref_mods, cemb)
        remb = infer_clean(model, rx, rs, device, args.batch_size)
        sc_clean = score_all(remb, rmods, cemb)
        res['clean_refpool_fit'] = {m: auroc(sc_pool[m][~c_unk], sc_pool[m][c_unk])
                                    for m in SCORERS}
        res['clean_clean_fit'] = {m: auroc(sc_clean[m][~c_unk], sc_clean[m][c_unk])
                                  for m in SCORERS}
        print('  clean (refpool fit):',
              {m: round(v, 4) for m, v in res['clean_refpool_fit'].items()})
        print('  clean (clean fit)  :',
              {m: round(v, 4) for m, v in res['clean_clean_fit'].items()})
        per_seed[str(seed)] = res
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    across = {}
    for cell in ('exact_sir0', 'legacy_alpha', 'clean_refpool_fit', 'clean_clean_fit'):
        across[cell] = {m: _ms([per_seed[s][cell][m] for s in per_seed])
                        for m in SCORERS}
    print('\n=== across seeds ===')
    for cell, r in across.items():
        print(f'  {cell:>18}: ' + '  '.join(
            f'{m}={r[m]["mean"]:.4f}±{r[m]["std"]:.3f}' for m in SCORERS))

    out = {
        'experiment': 'sir_sweep SIR=0 controls (follow-up to eval_sir_sweep)',
        'config': {
            'seeds': seeds, 'snr_db': MIXTURE_SNR, 'n_reps': N_REPS,
            'control_stream': CONTROL_STREAM,
            'clean_eval_stream': CLEAN_EVAL_STREAM,
            'clean_ref_stream': CLEAN_REF_STREAM,
            'purpose': 'disentangle exact-SIR=0 vs legacy alpha~U(0.4,0.6) '
                       '(main-test design) for the same-mixture AUROC, and '
                       'refpool-fit vs clean-fit for the clean cell'},
        'per_seed': per_seed,
        'across_seeds': across,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    fpath = os.path.join(args.out_dir, 'sir_sweep_controls.json')
    with open(fpath, 'w') as f:
        json.dump(_round(out), f, indent=1)
    print(f'\nSaved {fpath}')


if __name__ == '__main__':
    main()
