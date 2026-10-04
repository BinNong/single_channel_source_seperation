"""Aggregate the output-SIR evidence for the interference-limited claim.

Two halves, no new training:

1. Separator output SIR: read the per-SNR / per-K `sir` fields already
   stored in results/s1_mismatch/eval_*.json (evaluate.py computes SIR
   via per-interferer projection power).  Reports the honest pool
   (slot + specialist, 10 runs); recursive is selection-biased and shown
   only for reference.

2. Mixture baseline SIR: generate the deterministic test grid
   (seed 99999) locally and score the UNSEPARATED mixture as the
   estimate with the same _sir_np, so separator output SIR can be
   compared against the input SIR cell-by-cell.

Writes results/sir_evidence.json and prints a summary table.
"""

import glob
import json

import numpy as np

import config as C
from data_generator_vark import CommBSSVarKTestDataset
from evaluate import _sir_np

N_PER_CELL_MIX = 30  # mixture-as-estimate baseline (local, CPU)


def _mean_std(xs):
    xs = [x for x in xs if x == x]
    return (float(np.mean(xs)), float(np.std(xs))) if xs else (float('nan'),) * 2


def separator_sir():
    groups = {}
    for f in sorted(glob.glob('results/s1_mismatch/eval_*.json')):
        arch = f.split('eval_')[1].split('_h')[0]
        groups.setdefault(arch, []).append(json.load(open(f)))
    out = {}
    for arch, ds in groups.items():
        per_snr, per_k = {}, {}
        for s in ds[0]['separation']['per_snr']:
            per_snr[s] = _mean_std([d['separation']['per_snr'][s]['sir'] for d in ds])
        for k in ('1', '2', '3'):
            per_k[k] = _mean_std([d['separation']['per_k'][k]['sir'] for d in ds])
            per_k[k] += _mean_std([d['separation']['per_k'][k]['si_sdri'] for d in ds])
        out[arch] = {'n_seeds': len(ds), 'per_snr_sir': per_snr,
                     'per_k': {k: {'sir': v[:2], 'si_sdri': v[2:]}
                               for k, v in per_k.items()}}
    return out


def mixture_sir():
    ds = CommBSSVarKTestDataset(n_per_cell=N_PER_CELL_MIX,
                                seed=C.DataConfig.test_seed)
    per_k, per_snr = {}, {}
    for mix, sources, occ, k, snr, mods in ds:
        k, snr = int(k), float(snr)
        if k < 2:
            continue
        srcs = [sources[j].numpy() for j in range(k)]
        m = mix.numpy().reshape(-1)
        for i, s in enumerate(srcs):
            interferers = [g for j, g in enumerate(srcs) if j != i]
            v = _sir_np(m, s, interferers)
            per_k.setdefault(k, []).append(v)
            per_snr.setdefault(snr, []).append(v)
    return {
        'n_per_cell': N_PER_CELL_MIX,
        'per_k': {str(k): _mean_std(v) for k, v in sorted(per_k.items())},
        'per_snr': {str(s): _mean_std(v) for s, v in sorted(per_snr.items())},
    }


def main():
    sep = separator_sir()
    mix = mixture_sir()
    res = {'separator_output_sir': sep, 'mixture_input_sir': mix}
    with open('results/sir_evidence.json', 'w') as f:
        json.dump(res, f, indent=2)

    print('=== Mixture input SIR (dB, mean±std) ===')
    for k, (m, s) in mix['per_k'].items():
        print(f'  K={k}: {m:+.2f}±{s:.2f}')
    print('\n=== Separator output SIR per K (SIR | SI-SDRi, dB) ===')
    for arch in ('slot', 'specialist', 'recursive'):
        if arch not in sep:
            continue
        print(f'  {arch}:')
        for k, v in sep[arch]['per_k'].items():
            print(f"    K={k}: SIR {v['sir'][0]:+.2f}±{v['sir'][1]:.2f} | "
                  f"SI-SDRi {v['si_sdri'][0]:+.2f}±{v['si_sdri'][1]:.2f}")
    print('\n=== Per-SNR output SIR (slot, honest pool reference) ===')
    for s, (m, sd) in sorted(sep['slot']['per_snr_sir'].items(), key=lambda x: float(x[0])):
        print(f'  SNR {s:>4}: {m:+.2f}±{sd:.2f}')
    print('\nWrote results/sir_evidence.json')


if __name__ == '__main__':
    main()
