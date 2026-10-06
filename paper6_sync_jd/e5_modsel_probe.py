"""Paper 6 — E5 part 1 probe: blind modulation-pair selection at K=2.

Question: how much does the joint ECM pipeline (V4) lose when the
per-source modulations are NOT known?  Two selectors per burst:

  - energy: run the full ECM under each of the 10 unordered mod-pair
    hypotheses, keep the lowest converged residual energy.  Raw energy is
    complexity-biased (a 16x16 grid overfits vs a 2x2 one), so we also
    record a penalty-corrected score E + penalty * (M1*M2) (measured,
    see EXPERIMENT_LOG).
  - lock: two-stage — per-source modulation from the single-source
    constellation-lock score (E1 style; known density-biased), then one
    ECM under the selected pair.

Output: per-hypothesis energy margins + selection accuracy vs SNR
(subset: SNR {0, 10, 20}, n bursts CLI).  This probe decides the E5
blind-mod scheme; the full-grid blind-vs-known comparison is in
eval_joint_k2.py --variants V4B.

__main__ runs the probe.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

# Vendored paper6 modules (self-contained; no sys.path tricks).
import config as C                                          # noqa: E402
from data_generator import CommBSSVarKTestDataset, MOD_TYPES  # noqa: E402
from joint_detect import constellation_np                    # noqa: E402
from joint_sync_detect import ecm_joint, mixture_symbols     # noqa: E402
from sync import blind_sync_known_mod                        # noqa: E402

PAIRS = list(itertools.combinations_with_replacement(range(4), 2))


def pair_of(mods_idx):
    return tuple(sorted(mods_idx))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_bursts', type=int, default=12)
    p.add_argument('--snr_points', type=float, nargs='+', default=[0, 10, 20])
    args = p.parse_args()

    ds = CommBSSVarKTestDataset(
        n_per_cell=max(2, args.n_bursts // 2), snr_points=args.snr_points,
        seed=C.DataConfig.test_seed, return_carriers=True)
    rng = np.random.RandomState(C.SEED)

    recs = []
    done = 0
    for i in rng.permutation(len(ds)):
        s = ds.samples[i]
        if s['k'] != 2:
            continue
        mix_np = s['mixture'].numpy()[0]
        mods_idx = tuple(int(m) for m in s['mods'].numpy()[:2])
        true_pair = pair_of(mods_idx)
        r = mixture_symbols(mix_np)

        # selector 1: converged ECM energy per hypothesis
        en = {}
        for pr in PAIRS:
            res = ecm_joint(r, constellation_np(pr[0]),
                            constellation_np(pr[1]))
            en[pr] = res['res_energy']
        sel_energy = min(en, key=en.get)
        # margin: best wrong-hypothesis energy vs true-pair energy
        e_true = en[true_pair]
        e_wrong_best = min(e for pr, e in en.items() if pr != true_pair)

        # selector 2: two-stage lock score (E1-style, per-source on the
        # mixture — biased, measured here)
        ls = []
        for j in range(2):
            locks = []
            for m in range(4):
                res = blind_sync_known_mod(mix_np, m)
                locks.append(float(res['lock']))
            ls.append(int(np.argmin(locks)))
        sel_lock = pair_of(ls)

        recs.append({'snr': float(s['snr']), 'true': true_pair,
                     'sel_energy': sel_energy, 'sel_lock': sel_lock,
                     'margin': float(e_wrong_best - e_true),
                     'rel_margin': float((e_wrong_best - e_true)
                                         / max(e_true, 1e-12))})
        done += 1
        if done % 4 == 0:
            print(f"  {done} bursts", flush=True)
        if done >= args.n_bursts:
            break

    print("\nmod-pair selection accuracy:")
    for snr in args.snr_points:
        rs = [r for r in recs if r['snr'] == snr]
        if not rs:
            continue
        ae = np.mean([r['sel_energy'] == r['true'] for r in rs])
        al = np.mean([r['sel_lock'] == r['true'] for r in rs])
        mg = np.median([r['rel_margin'] for r in rs])
        print(f"  SNR {snr:>5.0f}: energy={ae:.2f} lock={al:.2f} "
              f"(median rel energy margin {mg:+.3f}, n={len(rs)})")
    import json, os
    path = os.path.join(C.RESULTS_DIR, 'e5_modsel_probe.json')
    with open(path, 'w') as f:
        json.dump({'n_bursts': len(recs), 'records': [
            {k: (v if not isinstance(v, tuple) else list(v))
             for k, v in r.items()} for r in recs]}, f, indent=2)
    print(f"saved {path}")


if __name__ == '__main__':
    main()
