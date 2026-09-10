"""
Paper 5 — Task-Oriented SC-BSS: Multi-seed aggregation.

Globs results/eval_*.json, groups by run name with the seed suffix
(_s<seed>) stripped, and prints mean +/- std of the headline metrics
(including compensated SER/BER when present).

Usage:
    python aggregate_results.py [results_dir]
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np


def _flatten_metrics(res: dict) -> dict:
    """Pull the headline scalars out of an eval JSON."""
    out = {}
    cnt = res.get('counting', {})
    for route, blk in cnt.items():
        out[f'{route}_acc'] = blk.get('overall_acc')
        out[f'{route}_tol1'] = blk.get('tol1_acc')
        out[f'{route}_k4_acc'] = blk.get('k4_acc')
    sep = res.get('separation', {})
    for key in ('si_sdr', 'si_sdri', 'sdr', 'sir', 'miss_rate',
                'halluc_rate', 'k1_halluc_rate', 'ser_comp', 'ber_comp'):
        out[key] = sep.get(key)
    cas = res.get('cascade', {})
    out['si_sdri_cnt_correct'] = cas.get('si_sdri_count_correct')
    out['si_sdri_cnt_incorrect'] = cas.get('si_sdri_count_incorrect')
    o = cas.get('oracle_k_si_sdri_per_snr', {})
    e = cas.get('e2e_si_sdri_per_snr', {})
    if o and e:
        gaps = [o[s] - e[s] for s in o
                if o.get(s) is not None and e.get(s) is not None]
        out['oracle_gap_mean'] = float(np.mean(gaps)) if gaps else None
    return out


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results')
    files = sorted(glob.glob(os.path.join(results_dir, 'eval_*.json')))
    if not files:
        print(f"No eval_*.json under {results_dir}")
        return

    groups: dict[str, list[dict]] = {}
    for f in files:
        with open(f) as fh:
            res = json.load(fh)
        name = os.path.splitext(os.path.basename(f))[0][len('eval_'):]
        group = re.sub(r'_s\d+$', '', name)
        groups.setdefault(group, []).append(_flatten_metrics(res))

    for group, runs in groups.items():
        print(f"\n=== {group}  ({len(runs)} seeds) ===")
        keys = sorted({k for r in runs for k in r})
        for k in keys:
            vals = [r[k] for r in runs if r.get(k) is not None]
            if not vals:
                continue
            m, s = float(np.mean(vals)), float(np.std(vals))
            print(f"  {k:>24s}: {m:8.4f} ± {s:.4f}   n={len(vals)}")


if __name__ == '__main__':
    main()
