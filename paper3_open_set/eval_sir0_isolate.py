"""
Paper 3 — Review-2 experiment B follow-up 2: isolate the SIR~0 same-mixture
AUROC discrepancy between the canonical ku test set (eval_protocol_split C3,
10-dB bin ~= 0.51) and the control cells (eval_sir_sweep_controls, ~= 0.64).

Same statistic (same-mixture prototype/mahalanobis/vos AUROC, refpool fit)
on four cells at SNR=10 dB:

  I1: the CANONICAL ku test set (CommBSSOpenSetTestDataset, protocol='ku',
      seed 99999, n_per_snr=200), restricted to the SNR=10 dB bin
      (192 mixtures) — replicates C3's 10-dB cell through the
      eval_sir_sweep inference machinery;
  I2: legacy alpha~U(0.4,0.6) control, stream 77999, 6 reps/pair (192 —
      matches I1's per-bin count);
  I3: legacy alpha control, fresh stream 78000, 16 reps/pair (256);
  I4: exact SIR=0 control, fresh stream 78001, 16 reps/pair (256).

5 model seeds.  Output: results/sir0_isolate.json.
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
    CommBSSOpenSetTestDataset, generate_open_set_mixture,
    MOD_KNOWN, MOD_UNKNOWN, MOD_TO_IDX,
)
from evaluate import _build_model_from_ckpt
from open_set_metrics import auroc
from eval_sir_sweep import infer_mixtures, score_all, PAIRS, MIXTURE_SNR

BASE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed{}_best'
SCORERS = ('mahalanobis', 'prototype', 'vos')


def gen_control_cell(sir_db, master_seed, n_reps):
    master = np.random.RandomState(master_seed)
    xs, s1s, s2s, m1s, m2s = [], [], [], [], []
    for mk, mu in PAIRS:
        for i in range(n_reps):
            np.random.seed(int(master.randint(0, 2 ** 31 - 1)))
            args = (mk, mu) if i % 2 == 0 else (mu, mk)
            kw = {} if sir_db is None else {'sir_db': float(sir_db)}
            mix, s1, s2, m1, m2 = generate_open_set_mixture(
                C.SIGNAL_LENGTH, C.SAMPLE_RATE, MIXTURE_SNR, *args, **kw)
            xs.append(mix); s1s.append(s1); s2s.append(s2)
            m1s.append(m1); m2s.append(m2)
    return {'x': np.stack(xs).astype(np.complex64),
            's1': np.stack(s1s).astype(np.complex64),
            's2': np.stack(s2s).astype(np.complex64),
            'mod1': np.asarray(m1s, dtype=np.int64),
            'mod2': np.asarray(m2s, dtype=np.int64)}


def ku_testset_snr10_cell():
    """The canonical ku test set's SNR=10 bin as an infer_mixtures cell."""
    ds = CommBSSOpenSetTestDataset(
        n_per_snr=200, snr_points=C.SNR_TEST_POINTS,
        signal_length=C.SIGNAL_LENGTH, sample_rate=C.SAMPLE_RATE,
        seed=99999, protocol='ku')
    keep = [i for i, s in enumerate(ds.samples) if s['snr'] == 10.0]
    xs = np.stack([ds.samples[i]['mixture'].numpy()[0] for i in keep])
    s1 = np.stack([ds.samples[i]['source1'].numpy()[0] for i in keep])
    s2 = np.stack([ds.samples[i]['source2'].numpy()[0] for i in keep])
    m1 = np.asarray([ds.samples[i]['mod1_idx'] for i in keep], dtype=np.int64)
    m2 = np.asarray([ds.samples[i]['mod2_idx'] for i in keep], dtype=np.int64)
    return {'x': xs, 's1': s1, 's2': s2, 'mod1': m1, 'mod2': m2}


def cell_aurocs(model, cell, ref_emb, ref_mods, device, batch_size):
    embs, _sis, mods = infer_mixtures(model, cell, device, batch_size)
    sc = score_all(ref_emb, ref_mods, embs.reshape(-1, embs.shape[-1]))
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
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default='42,43,44,45,46')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('[sir0_isolate] building cells ...')
    cells = {
        'I1_ku99999_snr10': ku_testset_snr10_cell(),
        'I2_legacy_77999_n192': gen_control_cell(None, 77999, 6),
        'I3_legacy_78000_n256': gen_control_cell(None, 78000, 16),
        'I4_exact0_78001_n256': gen_control_cell(0.0, 78001, 16),
    }
    print({k: len(v['x']) for k, v in cells.items()})

    per_seed = {}
    for seed in seeds:
        ckpt = os.path.join(C.CHECKPOINT_DIR, BASE.format(seed) + '.pt')
        print(f'\n=== seed {seed} ===')
        model = _build_model_from_ckpt(ckpt, device)
        rp = np.load(os.path.join(C.RESULTS_DIR, BASE.format(seed) + '_refpool.npz'))
        res = {name: cell_aurocs(model, cell, rp['ref_emb'], rp['ref_mods'],
                                 device, args.batch_size)
               for name, cell in cells.items()}
        for name, r in res.items():
            print(f'  {name:>22}: ' + '  '.join(f'{m}={r[m]:.4f}' for m in SCORERS))
        per_seed[str(seed)] = res
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    across = {name: {m: _ms([per_seed[s][name][m] for s in per_seed])
                     for m in SCORERS} for name in cells}
    print('\n=== across seeds ===')
    for name, r in across.items():
        print(f'  {name:>22}: ' + '  '.join(
            f'{m}={r[m]["mean"]:.4f}±{r[m]["std"]:.3f}' for m in SCORERS))

    out = {
        'experiment': 'sir0 same-mixture AUROC isolation (follow-up to '
                      'eval_sir_sweep_controls)',
        'cells': {k: {'n': len(v['x'])} for k, v in cells.items()},
        'per_seed': per_seed,
        'across_seeds': across,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    fpath = os.path.join(args.out_dir, 'sir0_isolate.json')
    with open(fpath, 'w') as f:
        json.dump(_round(out), f, indent=1)
    print(f'\nSaved {fpath}')


if __name__ == '__main__':
    main()
