"""
Paper 3 — One-click label--score alignment audit (review-3 follow-up).

Regenerates the deterministic open-set test labels (seed 99999;
192 kk / 192 ku / 96 uu mixtures per SNR bin) and checks every label
array of a score dump against them. This is the audit whose failure
exposed both misalignments of the letter:

  * bug 1 (repeat-vs-tile SNR labels)  -> check 4/5 fail, and the
    per-bin modulation multiset of check 3 is impossible (114/96/100/74);
  * bug 2 (PIT slot-anchored labels)   -> check 7 fails (~50% of the
    ku-derived unknown labels mismatched).

Checks
------
1. pool sizes (2688 known / 2688 unknown)
2. per-bin pool counts (384 / 384)
3. per-bin known-pool modulation multiset (96/96/96/96)
4. known-pool SNR label sequence == regenerated (tile order)
5. unknown-pool SNR label sequence == regenerated
6. known-pool modulation label sequence == regenerated truth
7. unknown-pool modulation label sequence == regenerated truth
   (PIT source anchoring)
8. internal consistency: known_snr == np.tile(snr_kk, 2)

Exit code 0 iff every check passes.

Usage:
    python audit_labels.py                       # seed-42 TA dump
    python audit_labels.py --dump <path.npz>     # any dump (e.g. the
    #   archived buggy dumps under results/archive_buggy_snr_labels/,
    #   which SHOULD fail checks 3/4/7)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import config as C
from data_generator_extended import CommBSSOpenSetTestDataset

BINS = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
DEFAULT_DUMP = os.path.join(
    C.RESULTS_DIR,
    'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed42_best_ood_scores_ta.npz')


def regenerate_expected():
    """Truth-anchored label sequences in the dump's documented ordering."""
    common = dict(snr_points=C.SNR_TEST_POINTS,
                  signal_length=C.SIGNAL_LENGTH, sample_rate=C.SAMPLE_RATE,
                  seed=99999)
    ds = {
        'kk': CommBSSOpenSetTestDataset(protocol='kk', n_per_snr=192, **common),
        'ku': CommBSSOpenSetTestDataset(protocol='ku', n_per_snr=192, **common),
        'uu': CommBSSOpenSetTestDataset(protocol='uu', n_per_snr=96,
                                        carrier_freq_2=2005.0, **common),
    }
    tr = {}
    for p, d in ds.items():
        tr[p] = {
            'mod1': np.array([s['mod1_idx'] for s in d.samples]),
            'mod2': np.array([s['mod2_idx'] for s in d.samples]),
            'ood1': np.array([s['mod1_is_ood'] for s in d.samples], dtype=bool),
            'ood2': np.array([s['mod2_is_ood'] for s in d.samples], dtype=bool),
            'snr': np.array([s['snr'] for s in d.samples], dtype=np.float32),
        }
    to1, to2 = tr['ku']['ood1'], tr['ku']['ood2']
    return {
        'known_snr': np.tile(tr['kk']['snr'], 2),
        'known_mods': np.concatenate([tr['kk']['mod1'], tr['kk']['mod2']]),
        'unknown_snr': np.concatenate([tr['ku']['snr'][to1],
                                       tr['ku']['snr'][to2],
                                       tr['uu']['snr'], tr['uu']['snr']]),
        'unknown_mods': np.concatenate([tr['ku']['mod1'][to1],
                                        tr['ku']['mod2'][to2],
                                        tr['uu']['mod1'], tr['uu']['mod2']]),
        'snr_kk': tr['kk']['snr'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', type=str, default=DEFAULT_DUMP)
    args = ap.parse_args()

    print(f'[audit] dump: {args.dump}')
    print('[audit] regenerating deterministic test labels (seed 99999) ...')
    exp = regenerate_expected()
    d = np.load(args.dump)

    results = []

    def check(name, ok, detail=''):
        results.append(bool(ok))
        print(f'  [{"PASS" if ok else "FAIL"}] {name}'
              + (f'  ({detail})' if detail else ''))

    nk, nu = len(d['known_snr']), len(d['unknown_snr'])
    check('1 pool sizes 2688/2688', nk == 2688 and nu == 2688,
          f'got {nk}/{nu}')

    cnt_k = [int((d['known_snr'] == s).sum()) for s in BINS]
    cnt_u = [int((d['unknown_snr'] == s).sum()) for s in BINS]
    check('2 per-bin counts 384/384',
          cnt_k == [384] * 7 and cnt_u == [384] * 7,
          f'known={cnt_k} unknown={cnt_u}')

    multisets = [[int(((d['known_mods'] == m) & (d['known_snr'] == s)).sum())
                  for m in range(C.NUM_KNOWN_CLASSES)] for s in BINS]
    check('3 per-bin known modulation multiset 96/96/96/96',
          all(ms == [96, 96, 96, 96] for ms in multisets),
          f'e.g. bin -10 dB: {multisets[0]}')

    def seq(a, b):
        return a.shape == b.shape and np.array_equal(a, b)

    check('4 known SNR label sequence == regenerated (tile)',
          seq(d['known_snr'], exp['known_snr']),
          f'{int((d["known_snr"] != exp["known_snr"]).sum())} mismatches'
          if d['known_snr'].shape == exp['known_snr'].shape else 'shape')
    check('5 unknown SNR label sequence == regenerated',
          seq(d['unknown_snr'], exp['unknown_snr']))
    check('6 known modulation sequence == regenerated truth',
          seq(d['known_mods'], exp['known_mods']))
    mism_u = int((d['unknown_mods'] != exp['unknown_mods']).sum()) \
        if d['unknown_mods'].shape == exp['unknown_mods'].shape else -1
    check('7 unknown modulation sequence == regenerated truth (PIT anchor)',
          seq(d['unknown_mods'], exp['unknown_mods']),
          f'{mism_u} mismatches')

    if 'snr_kk' in d.files:
        check('8 internal: known_snr == np.tile(snr_kk, 2)',
              seq(d['known_snr'], np.tile(d['snr_kk'], 2)))

    n_pass = sum(results)
    print(f'[audit] {n_pass}/{len(results)} checks passed')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
