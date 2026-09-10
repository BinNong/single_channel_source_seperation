"""Paper 5 — aggregate the SIR-sweep results (reviewer experiment 1).

Reads results/sir_sweep/sir_sweep_<config>_s<seed>.json (+ the baseline
file), averages over seeds, and prints one table per config:
rows = SIR (dB), columns = SI-SDRi / SER / BER / baseline SER /
strong-source SER / weak-source SER.

Writes results/sir_sweep/sir_sweep_summary.json.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

SIRS = [0.0, 5.0, 10.0, 20.0]
CONFIGS = ['sisdr', 'mse', 'ser', 'ser_mse']


def _ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def main(out_dir='results/sir_sweep'):
    with open(os.path.join(out_dir, 'sir_sweep_baseline.json')) as f:
        baseline = json.load(f)['baseline']

    summary = {'baseline': baseline, 'configs': {}}
    for cfg in CONFIGS:
        files = sorted(glob.glob(os.path.join(
            out_dir, f'sir_sweep_{cfg}_s*.json')))
        if not files:
            continue
        runs = [json.load(open(f)) for f in files]
        rows = {}
        for sir in SIRS:
            key = str(sir)
            si_m, si_s = _ms([r['per_sir'][key]['si_sdri'] for r in runs])
            se_m, se_s = _ms([r['per_sir'][key]['ser_comp'] for r in runs])
            be_m, be_s = _ms([r['per_sir'][key]['ber_comp'] for r in runs])
            st_m, _ = _ms([r['per_sir'][key]['ser_strong'] for r in runs])
            wk_m, _ = _ms([r['per_sir'][key]['ser_weak'] for r in runs])
            rows[key] = {
                'si_sdri_mean': si_m, 'si_sdri_std': si_s,
                'ser_mean': se_m, 'ser_std': se_s,
                'ber_mean': be_m, 'ber_std': be_s,
                'ser_strong': st_m, 'ser_weak': wk_m,
                'baseline_ser': baseline[key]['overall_ser'],
                'baseline_ber': baseline[key]['overall_ber'],
                # per-(SIR,SNR) matrices, mean over seeds
                'per_snr_ser': {snr: float(np.mean(
                    [r['per_sir'][key]['per_snr'][snr]['ser_comp']
                     for r in runs]))
                    for snr in runs[0]['per_sir'][key]['per_snr']},
                'per_snr_si_sdri': {snr: float(np.mean(
                    [r['per_sir'][key]['per_snr'][snr]['si_sdri']
                     for r in runs]))
                    for snr in runs[0]['per_sir'][key]['per_snr']},
            }
        summary['configs'][cfg] = {'n_seeds': len(runs), 'rows': rows}

        print(f"\n=== {cfg}  (n_seeds={len(runs)}, K=2, pooled over SNR) ===")
        print(f"{'SIR(dB)':>8s} | {'SI-SDRi':>14s} | {'SER':>14s} | "
              f"{'BER':>14s} | {'base SER':>8s} | {'SER str':>8s} | "
              f"{'SER weak':>8s}")
        for sir in SIRS:
            r = rows[str(sir)]
            sw = (f"{r['ser_strong']:>8.4f} | {r['ser_weak']:>8.4f}"
                  if r['ser_strong'] is not None else f"{'—':>8s} | {'—':>8s}")
            print(f"{sir:>8.1f} | {r['si_sdri_mean']:>7.3f}±{r['si_sdri_std']:.3f}"
                  f" | {r['ser_mean']:>7.4f}±{r['ser_std']:.4f}"
                  f" | {r['ber_mean']:>7.4f}±{r['ber_std']:.4f}"
                  f" | {r['baseline_ser']:>8.4f} | {sw}")

    out = os.path.join(out_dir, 'sir_sweep_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == '__main__':
    main()
