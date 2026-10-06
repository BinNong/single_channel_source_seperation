"""Paper 6 — paired per-burst statistics for the E4 headline contrasts
(review 2, Major Concern 7: the V4-vs-V1 margin of 0.8 pts must be
tested PAIRED — same test cells — not only by two independent seed-level
CIs, which overlap).

Contrasts (per K=2 test cell, per-burst SER):
  C1  V1 - V4     (ISI-model ablation; 5 independent test grids)
  C2  waveform-route oracle SER - V4   (headline; reference grid
      seed 99999, 5 separator checkpoints averaged per cell)

For each contrast: mean paired difference, paired bootstrap 95% CI
(10^4 resamples), Wilcoxon signed-rank p-value, and the per-SNR split.

Inputs (results/): e4_joint_k2_mse[_tsXXXXX].json (V1V4/V3 records),
e3_blind_pipeline_mse_pairs.json (E3 oracle/blind per-pair records).
Output: results/e4_paired_stats.json + printed summary.  Deterministic.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import wilcoxon

import config as C  # noqa: E402

RESULTS = C.RESULTS_DIR
TEST_SEEDS = [99999, 88888, 77777, 55555, 31337]
E4_FILES = {99999: 'e4_joint_k2_mse.json',
            88888: 'e4_joint_k2_mse_ts88888.json',
            77777: 'e4_joint_k2_mse_ts77777.json',
            55555: 'e4_joint_k2_mse_ts55555.json',
            31337: 'e4_joint_k2_mse_ts31337.json'}
SNRS = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
N_BOOT = 10_000


def _load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def _cells_by_snr(records, variant):
    """{snr: [records in dataset iteration order]} for one variant."""
    out = {}
    for r in records:
        if r['variant'] != variant:
            continue
        out.setdefault(float(r['snr']), []).append(r)
    return out


def paired_stats(delta, n_boot=N_BOOT, seed=12345):
    """delta: paired per-cell differences (a - b)."""
    delta = np.asarray(delta, dtype=float)
    n = len(delta)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(delta, n, replace=True).mean()
                      for _ in range(n_boot)])
    ci = np.percentile(boots, [2.5, 97.5])
    try:
        w = wilcoxon(delta)
        p = float(w.pvalue)
    except ValueError:
        p = float('nan')
    return {'n': int(n), 'mean': float(delta.mean()),
            'ci95': [float(ci[0]), float(ci[1])],
            'wilcoxon_p': p,
            'frac_positive': float((delta > 0).mean())}


def main():
    # ---------------- C1: V1 - V4, per test grid, then pooled ----------
    c1 = {}
    all_delta = []
    for ts in TEST_SEEDS:
        d = _load(E4_FILES[ts])
        v1 = _cells_by_snr(d['records'], 'V1')
        v4 = _cells_by_snr(d['records'], 'V4')
        delta = []
        per_snr = {}
        for s in SNRS:
            r1, r4 = v1[s], v4[s]
            assert len(r1) == len(r4)
            # same cell order (V1V4 appends V1 then V4 per cell)
            dd = [a['ser'] - b['ser'] for a, b in zip(r1, r4)]
            assert all(a['mods'] == b['mods'] for a, b in zip(r1, r4))
            per_snr[f'{s:g}'] = paired_stats(dd)
            delta.extend(dd)
        c1[f'ts{ts}'] = {'pooled': paired_stats(delta), 'per_snr': per_snr}
        all_delta.extend(delta)
    c1['pooled_all_grids'] = paired_stats(all_delta)

    # ---------------- C2: waveform route - V4, reference grid ----------
    # Waveform-route per-cell SER comes from eval_psp_baseline.py's
    # symbol-wise arm (same cells, same dataset order, 5 checkpoints);
    # E3's pairs JSON cannot be position-paired (Hungarian matching drops
    # unmatched pairs, so cell positions are not reconstructible).
    c2 = None
    psp_path = os.path.join(RESULTS, 'psp_baseline.json')
    if os.path.exists(psp_path):
        d = _load(E4_FILES[99999])
        v4 = _cells_by_snr(d['records'], 'V4')
        psp = _load(psp_path)
        prec = psp['records'] if 'records' in psp else psp

        def _contrast(arm):
            ww = [r for r in prec if r.get('variant') == arm]
            by_snr = {}
            for r in ww:
                by_snr.setdefault(float(r['snr']), []).append(r)
            delta, per_snr = [], {}
            for s in SNRS:
                recs = by_snr[s]
                seeds = sorted({r['seed'] for r in recs})
                n_cells = len(v4[s])
                route_cell = np.zeros(n_cells)
                for sd in seeds:
                    sub = [r for r in recs if r['seed'] == sd]
                    assert len(sub) == n_cells, (s, sd, len(sub), n_cells)
                    route_cell += np.array([r['ser'] for r in sub])
                route_cell /= len(seeds)
                v4_ser = np.array([r['ser'] for r in v4[s]])
                dd = route_cell - v4_ser
                per_snr[f'{s:g}'] = paired_stats(dd)
                delta.extend(dd)
            return {'pooled': paired_stats(delta), 'per_snr': per_snr}

        c2 = {'symbolwise_minus_V4': _contrast('sym'),
              'psp_minus_V4': _contrast('psp'),
              'note': 'waveform route = eval_psp_baseline.py arms '
                      '(5-checkpoint mean per cell); positive delta = '
                      'V4 wins'}
    else:
        print(f"[skip] C2: {psp_path} not found yet "
              f"(run eval_psp_baseline.py first)")

    out = {'C1_V1_minus_V4': c1}
    if c2 is not None:
        out['C2_waveform_minus_V4'] = c2
    path = os.path.join(RESULTS, 'e4_paired_stats.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"saved {path}\n")
    print("=== C1: V1 - V4 (positive = V4 better) ===")
    for k in list(c1):
        p = c1[k] if k == 'pooled_all_grids' else c1[k]['pooled']
        print(f"  {k:>16s}: mean {p['mean']:+.4f}  CI "
              f"[{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}]  "
              f"Wilcoxon p={p['wilcoxon_p']:.2e}  n={p['n']}")
    if c2 is not None:
        for arm in ('symbolwise_minus_V4', 'psp_minus_V4'):
            print(f"\n=== C2: {arm} (positive = V4 better) ===")
            p = c2[arm]['pooled']
            print(f"  pooled: mean {p['mean']:+.4f}  CI "
                  f"[{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}]  "
                  f"Wilcoxon p={p['wilcoxon_p']:.2e}  n={p['n']}")
            print("  per-SNR:")
            for s, v in c2[arm]['per_snr'].items():
                print(f"  {s:>4s} dB: mean {v['mean']:+.4f}  CI "
                      f"[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]  "
                      f"p={v['wilcoxon_p']:.1e}")
    print("\nper-SNR (C1, pooled over grids):")
    for s in SNRS:
        dd = []
        for ts in TEST_SEEDS:
            d = _load(E4_FILES[ts])
            v1 = _cells_by_snr(d['records'], 'V1')
            v4 = _cells_by_snr(d['records'], 'V4')
            dd.extend([a['ser'] - b['ser'] for a, b in zip(v1[s], v4[s])])
        v = paired_stats(dd)
        print(f"  {s:>4g} dB: mean {v['mean']:+.4f}  CI "
              f"[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]  "
              f"p={v['wilcoxon_p']:.1e}")


if __name__ == '__main__':
    main()
