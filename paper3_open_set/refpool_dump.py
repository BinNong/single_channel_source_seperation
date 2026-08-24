"""
Paper 3 — Held-out reference-pool dump.

The main protocol fits reference statistics (class prototypes for
Prototype/VOS, class-conditional means + shared covariance for
Mahalanobis) on the *test* kk pool — a transductive choice that is
uniform across scorers but may draw reviewer fire.  This script dumps a
held-out reference pool (known-class kk pairs, seed 88888 — disjoint
from the test seed 99999) so the scorers can be re-fit on data that
never touches the test protocols, mirroring a deployed receiver's
reference library of known transmitters.

Usage:
    python refpool_dump.py --checkpoint checkpoints/<name>_best.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import config as C
from data_generator_extended import CommBSSOpenSetTestDataset
from evaluate import _build_model_from_ckpt, _collect_predictions


def main():
    p = argparse.ArgumentParser(description='Dump held-out reference pool')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--n_per_snr', type=int, default=200)
    p.add_argument('--seed', type=int, default=88888,
                   help='Reference-set seed (test protocols use 99999)')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.checkpoint} ...")
    model = _build_model_from_ckpt(args.checkpoint, device)

    ds = CommBSSOpenSetTestDataset(
        n_per_snr=args.n_per_snr,
        snr_points=C.SNR_TEST_POINTS,
        signal_length=C.SIGNAL_LENGTH,
        sample_rate=C.SAMPLE_RATE,
        seed=args.seed,
        protocol='kk',
    )
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=2)
    print(f"Reference pool: kk protocol, seed {args.seed}, {len(ds)} samples")
    ref = _collect_predictions(model, loader, device)

    emb = np.concatenate([ref['emb_1'], ref['emb_2']], axis=0)
    mods = np.concatenate([ref['mod1_idx'], ref['mod2_idx']], axis=0)
    snr = np.repeat(ref['snr'], 2)

    run_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out = os.path.join(args.out_dir, f"{run_name}_refpool.npz")
    np.savez(out, ref_emb=emb, ref_mods=mods, ref_snr=snr)
    print(f"Saved {out}  (emb {emb.shape})")


if __name__ == '__main__':
    main()
