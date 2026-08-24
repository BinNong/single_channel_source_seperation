"""
Paper 3 — ODIN baseline score dump (requires gradients, runs on GPU).

Computes per-source ODIN scores (Liang et al., ICLR 2018) for a trained
OpenSetCSE checkpoint and saves them aligned with the known/unknown pools
of `evaluate.py` (same datasets, same PIT alignment logic), so the scores
can be merged directly with the `<run>_ood_scores.npz` dumps.

  score(x) = -max_k softmax(f_k(x~) / T),   x~ = x + eps * sign(grad_x logit-obj)

where the perturbation is computed per source head and applied to the
(complex) mixture as  eps * (sign(re g) + j sign(im g))  — the complex
extension of ODIN's real-valued sign perturbation.

Usage:
    python odin_dump.py --checkpoint checkpoints/<name>_best.pt \
        --eps 0.005 --temperature 1000
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
from evaluate import _build_model_from_ckpt, _si_sdr_per_sample


def _odin_scores_batch(model, mix, temperature, eps):
    """Per-source ODIN scores for one batch.

    Returns (score_1, score_2) as (B,) numpy arrays, aligned to heads 1/2
    BEFORE PIT swapping (caller applies the same swap as evaluate.py).
    """
    mix = mix.clone().requires_grad_(True)
    _, _, _, _, l1, l2 = model(mix)
    scores = []
    for l in (l1, l2):
        pred = l.argmax(dim=-1)
        obj = torch.log_softmax(l / temperature, dim=-1)
        obj = obj.gather(1, pred.view(-1, 1)).sum()
        # retain_graph: the two heads share one forward graph (trunk), and we
        # backprop through it once per head.
        g = torch.autograd.grad(obj, mix, retain_graph=True)[0]
        pert = eps * (torch.sign(g.real) + 1j * torch.sign(g.imag))
        with torch.no_grad():
            x_pert = mix.detach() + pert
            *_, lp1, lp2 = model(x_pert)
            lp = lp1 if l is l1 else lp2
            p = torch.softmax(lp / temperature, dim=-1)
            s = -p.max(dim=-1).values        # higher = more OOD
        scores.append(s.cpu().numpy())
    return scores


@torch.no_grad()
def _swap_mask(model, mix, src1, src2):
    """PIT swap decision identical to evaluate._collect_predictions."""
    s1_hat, s2_hat, *_ = model(mix)
    sdr_11 = _si_sdr_per_sample(s1_hat, src1)
    sdr_22 = _si_sdr_per_sample(s2_hat, src2)
    sdr_12 = _si_sdr_per_sample(s1_hat, src2)
    sdr_21 = _si_sdr_per_sample(s2_hat, src1)
    return (sdr_12 + sdr_21) > (sdr_11 + sdr_22)


def collect_odin(model, loader, device, temperature, eps):
    """Returns dict with per-sample aligned ODIN scores + labels/snr."""
    o1_l, o2_l, mod1_l, mod2_l, ood1_l, ood2_l, snr_l = [], [], [], [], [], [], []
    for batch in loader:
        mix, src1, src2, mod1, mod2, ood1, ood2, snr = batch
        mix = mix.to(device)
        src1 = src1.to(device)
        src2 = src2.to(device)
        swap = _swap_mask(model, mix, src1, src2)
        s1, s2 = _odin_scores_batch(model, mix, temperature, eps)
        s1_t = torch.from_numpy(s1).to(device)
        s2_t = torch.from_numpy(s2).to(device)
        o1_l.append(torch.where(swap, s2_t, s1_t).cpu().numpy())
        o2_l.append(torch.where(swap, s1_t, s2_t).cpu().numpy())
        swap_c = swap.cpu().numpy()
        mod1_l.append(np.where(swap_c, mod2.numpy(), mod1.numpy()))
        mod2_l.append(np.where(swap_c, mod1.numpy(), mod2.numpy()))
        ood1_l.append(np.where(swap_c, ood2.numpy(), ood1.numpy()).astype(bool))
        ood2_l.append(np.where(swap_c, ood1.numpy(), ood2.numpy()).astype(bool))
        snr_l.append(np.asarray(snr, dtype=np.float32))
    return {
        'odin_1': np.concatenate(o1_l), 'odin_2': np.concatenate(o2_l),
        'mod1_idx': np.concatenate(mod1_l), 'mod2_idx': np.concatenate(mod2_l),
        'is_ood_1': np.concatenate(ood1_l), 'is_ood_2': np.concatenate(ood2_l),
        'snr': np.concatenate(snr_l),
    }


def build_pools(kk, ku, uu, key1, key2):
    """Known pool = both kk sources; unknown pool = ku OOD side + both uu.
    Mirrors evaluate.evaluate_ood's pool construction exactly."""
    known = np.concatenate([kk[key1], kk[key2]], axis=0)
    m1, m2 = ku['is_ood_1'], ku['is_ood_2']
    unknown = np.concatenate([
        ku[key1][m1], ku[key2][m2],
        uu[key1], uu[key2],
    ], axis=0)
    return known, unknown


def main():
    p = argparse.ArgumentParser(description='Dump ODIN OOD scores')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--eps', type=float, default=0.005)
    p.add_argument('--temperature', type=float, default=1000.0)
    p.add_argument('--n_per_snr', type=int, default=200)
    p.add_argument('--n_per_snr_uu', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {args.checkpoint} ...")
    model = _build_model_from_ckpt(args.checkpoint, device)

    common = dict(
        n_per_snr=args.n_per_snr,
        snr_points=C.SNR_TEST_POINTS,
        signal_length=C.SIGNAL_LENGTH,
        sample_rate=C.SAMPLE_RATE,
        seed=99999,
    )
    ds_kk = CommBSSOpenSetTestDataset(protocol='kk', **common)
    ds_ku = CommBSSOpenSetTestDataset(protocol='ku', **common)
    ds_uu = CommBSSOpenSetTestDataset(
        n_per_snr=args.n_per_snr_uu, snr_points=C.SNR_TEST_POINTS,
        signal_length=C.SIGNAL_LENGTH, sample_rate=C.SAMPLE_RATE,
        seed=99999, protocol='uu',
    )
    loaders = {
        k: DataLoader(ds, batch_size=args.batch_size, num_workers=2)
        for k, ds in [('kk', ds_kk), ('ku', ds_ku), ('uu', ds_uu)]
    }

    pools = {}
    for name, loader in loaders.items():
        print(f"ODIN pass: {name} ({len(loader.dataset)} samples) ...")
        pools[name] = collect_odin(model, loader, device,
                                   args.temperature, args.eps)

    known, unknown = build_pools(pools['kk'], pools['ku'], pools['uu'],
                                 'odin_1', 'odin_2')
    # Sanity: ODIN should beat MSP (its unperturbed special case) if eps is
    # in a sane range — print both pool means for a quick eyeball check.
    print(f"  known mean={known.mean():.4f}  unknown mean={unknown.mean():.4f} "
          f"(want unknown > known)")

    run_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out = os.path.join(args.out_dir,
                       f"{run_name}_odin_eps{args.eps:g}_T{args.temperature:g}.npz")
    np.savez(out,
             odin_score_known=known, odin_score_unknown=unknown,
             known_snr=np.repeat(pools['kk']['snr'], 2),
             unknown_snr=np.concatenate([
                 pools['ku']['snr'][pools['ku']['is_ood_1']],
                 pools['ku']['snr'][pools['ku']['is_ood_2']],
                 pools['uu']['snr'], pools['uu']['snr'],
             ]))
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
