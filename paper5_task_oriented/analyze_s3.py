"""Paper 5 — S3 three-way comparison on K=2 cells (the SIR~0 regime).

Routes (same K=2 test cells, same compensated receiver truth):
  baseline : mixture, no separation            (results/ser/baseline_ser_comp.json)
  A        : S2 (b) mse  — separate -> hard demod     (variable-K, K=2 cells)
  B        : S2 (c) ser  — separate (soft-SER-trained) -> hard demod
  C-wave   : S3 jointK2 model's WAVEFORM route (ser_comp; sanity: does the
             joint head hurt separation?)
  C-joint  : S3 jointK2 model's JointLLRHead route (ser_joint / ber_joint)

Outputs: results/s3/summary_s3.json + console table + paired bootstrap CIs
(C-joint vs C-wave, C-joint vs B, both seed-paired on K=2 cells).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S2 = os.path.join(HERE, 'results', 's2')
S3 = os.path.join(HERE, 'results', 's3')
BASELINE = os.path.join(HERE, 'results', 'ser', 'baseline_ser_comp.json')


def paired_boot_ci(a, b, n_boot=10000, seed=0):
    """95% CI of mean(a - b) by paired bootstrap."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), \
        float(np.percentile(boots, 97.5))


def main():
    base = json.load(open(BASELINE))
    s2 = {}
    for f in sorted(glob.glob(os.path.join(S2, 'eval_*.json'))):
        n = os.path.basename(f)[5:-5]
        cfg = 'ser_mse' if '_ser_mse_' in n else 'sisdr' if '_sisdr_' in n \
            else 'mse' if '_mse_' in n else 'ser'
        s2.setdefault(cfg, {})[int(n.split('_s')[-1])] = json.load(open(f))
    s3 = {}
    for f in sorted(glob.glob(os.path.join(S3, 'eval_*.json'))):
        n = os.path.basename(f)[5:-5]
        s3[int(n.split('_s')[-1])] = json.load(open(f))

    print('=== K=2 cells (the interference-limited regime, SIR~0) ===')
    print(f"baseline mixture: SER {base['per_k']['2']:.4f}  "
          f"BER {base['per_k_ber']['2']:.4f}")

    rows = {}
    for cfg in ('sisdr', 'mse', 'ser'):
        ser_k2 = [r['separation']['per_k']['2']['ser_comp']
                  for r in s2[cfg].values()]
        ber_k2 = [r['separation']['per_k']['2']['ber_comp']
                  for r in s2[cfg].values()]
        rows[cfg] = (ser_k2, ber_k2)
        print(f"S2 {cfg:>6s} (waveform route): SER {np.mean(ser_k2):.4f}±"
              f"{np.std(ser_k2):.4f}  BER {np.mean(ber_k2):.4f}±"
              f"{np.std(ber_k2):.4f}")

    cw_ser, cw_ber, cj_ser, cj_ber, cj_si, cw_si = [], [], [], [], [], []
    for seed, r in sorted(s3.items()):
        k2 = r['separation']['per_k']['2']
        cw_ser.append(k2['ser_comp'])
        cw_ber.append(k2['ber_comp'])
        cj_ser.append(k2['ser_joint'])
        cj_ber.append(k2['ber_joint'])
        cj_si.append(k2['si_sdri'])
        cw_si.append(k2['si_sdri'])
    print(f"S3 waveform route (sanity):  SER {np.mean(cw_ser):.4f}±"
          f"{np.std(cw_ser):.4f}  BER {np.mean(cw_ber):.4f}±"
          f"{np.std(cw_ber):.4f}  SI-SDRi(K=2) {np.mean(cw_si):+.3f} dB")
    print(f"S3 JOINT head route:         SER {np.mean(cj_ser):.4f}±"
          f"{np.std(cj_ser):.4f}  BER {np.mean(cj_ber):.4f}±"
          f"{np.std(cj_ber):.4f}")

    print('\n--- paired bootstrap (positive = first route better) ---')
    out = {'baseline_k2': {'ser': base['per_k']['2'],
                           'ber': base['per_k_ber']['2']},
           's3_waveform_k2': {'ser': cw_ser, 'ber': cw_ber},
           's3_joint_k2': {'ser': cj_ser, 'ber': cj_ber},
           's2_routes_k2': {c: {'ser': v[0], 'ber': v[1]}
                            for c, v in rows.items()},
           'paired': {}}
    for name, a, b in (
            ('joint_vs_baseline_SER', [base['per_k']['2']] * len(cj_ser),
             cj_ser),
            ('joint_vs_baseline_BER', [base['per_k_ber']['2']] * len(cj_ber),
             cj_ber),
            ('joint_vs_s3waveform_SER', cw_ser, cj_ser),
            ('joint_vs_s3waveform_BER', cw_ber, cj_ber),
            ('joint_vs_s2ser_SER', rows['ser'][0], cj_ser),
            ('joint_vs_s2ser_BER', rows['ser'][1], cj_ber),
            ('joint_vs_s2mse_SER', rows['mse'][0], cj_ser),
            ('joint_vs_s2mse_BER', rows['mse'][1], cj_ber)):
        m, lo, hi = paired_boot_ci(b, a)
        print(f'  {name:>26s}: Δ={m:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]')
        out['paired'][name] = {'delta': m, 'ci95': [lo, hi]}

    # per-SNR joint-vs-waveform gap (the S3 money table)
    print('\n--- per-SNR (K=2 cells): joint vs waveform SER/BER ---')
    print(f'{"SNR":>5s} | {"wf SER":>7s} {"joint SER":>9s} | {"wf BER":>7s} '
          f'{"joint BER":>9s} | {"base SER":>8s}')
    per_snr = {}
    for s in (-10, -5, 0, 5, 10, 15, 20):
        wv_s = [r['separation']['per_snr'][str(s)]['ser_comp']
                for r in s3.values() if 'ser_comp' in
                r['separation']['per_snr'][str(s)]]
        # restrict to K=2 within the SNR cell is not stored; per_snr mixes K.
        # -> per-SNR numbers here mix K in {1,2,3}; the K=2-only readout is
        # the table above.  We still report the trend.
        wv = [r['separation']['per_snr'][str(s)] for r in s3.values()]
        per_snr[int(s)] = {
            'wf_ser': float(np.mean([b['ser_comp'] for b in wv])),
            'joint_ser': float(np.mean([b['ser_joint'] for b in wv])),
            'wf_ber': float(np.mean([b['ber_comp'] for b in wv])),
            'joint_ber': float(np.mean([b['ber_joint'] for b in wv])),
        }
        print(f"{s:>5d} | {per_snr[int(s)]['wf_ser']:7.4f} "
              f"{per_snr[int(s)]['joint_ser']:9.4f} | "
              f"{per_snr[int(s)]['wf_ber']:7.4f} "
              f"{per_snr[int(s)]['joint_ber']:9.4f} | "
              f"{base['per_snr'][str(float(s))]:8.4f}")
    out['per_snr_allk'] = per_snr

    with open(os.path.join(S3, 'summary_s3.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {S3}/summary_s3.json")


if __name__ == '__main__':
    main()
