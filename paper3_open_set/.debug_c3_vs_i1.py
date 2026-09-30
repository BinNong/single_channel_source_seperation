"""DEBUG: why does the same ku-99999 SNR=10 cell give prototype AUROC 0.51
via eval_protocol_split (C3) but 0.67 via eval_sir_sweep machinery (I1)?

Compares both inference paths elementwise on the same data (seed 42).
"""
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
from ood_scores import compute_prototypes, prototype_score
from open_set_metrics import auroc
from eval_sir0_isolate import ku_testset_snr10_cell
from eval_sir_sweep import infer_mixtures

BASE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed{}_best'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = _build_model_from_ckpt(os.path.join(C.CHECKPOINT_DIR, BASE.format(42) + '.pt'), device)
rp = np.load(os.path.join(C.RESULTS_DIR, BASE.format(42) + '_refpool.npz'))
protos = compute_prototypes(rp['ref_emb'], rp['ref_mods'], 4)

ds = CommBSSOpenSetTestDataset(n_per_snr=200, snr_points=C.SNR_TEST_POINTS,
                               signal_length=C.SIGNAL_LENGTH,
                               sample_rate=C.SAMPLE_RATE, seed=99999,
                               protocol='ku')
loader = DataLoader(ds, batch_size=16, num_workers=2)
pred = _collect_predictions(model, loader, device)

# --- path A partition (eval_protocol_split C3 style) ---
m1, m2 = pred['is_ood_1'], pred['is_ood_2']
embK_A = np.concatenate([pred['emb_1'][~m1], pred['emb_2'][~m2]])
embU_A = np.concatenate([pred['emb_1'][m1], pred['emb_2'][m2]])
snrK_A = np.concatenate([pred['snr'][~m1], pred['snr'][~m2]])
snrU_A = np.concatenate([pred['snr'][m1], pred['snr'][m2]])
scK_A = prototype_score(embK_A, protos)
scU_A = prototype_score(embU_A, protos)
print('path A: AUROC@10 =', auroc(scK_A[snrK_A == 10], scU_A[snrU_A == 10]),
      ' pooled =', auroc(scK_A, scU_A))

# --- path B (eval_sir_sweep machinery on the same dataset, snr=10 cell) ---
cell = ku_testset_snr10_cell()
embs, sis, mods = infer_mixtures(model, cell, device, 64)
flat = embs.reshape(-1, embs.shape[-1])
is_unk = mods.reshape(-1) >= 4
sc = prototype_score(flat, protos)
print('path B: AUROC@10 =', auroc(sc[~is_unk], sc[is_unk]))

# --- elementwise comparison on the same 192 samples ---
# reconstruct path-A per-sample aligned emb for the snr=10 subset, in dataset order
keep = np.asarray([i for i, s in enumerate(ds.samples) if s['snr'] == 10.0])
# per-sample (2, D) aligned emb from path A
embA2 = np.stack([pred['emb_1'][keep], pred['emb_2'][keep]], axis=1)
modA2 = np.stack([pred['mod1_idx'][keep], pred['mod2_idx'][keep]], axis=1)
print('emb max|A-B| =', np.abs(embA2 - embs).max())
print('mods equal:', np.array_equal(modA2, mods))
print('is_unk A vs B equal:', np.array_equal((modA2 >= 4).reshape(-1), is_unk))
scA = prototype_score(embA2.reshape(-1, embA2.shape[-1]), protos)
print('score max|A-B| =', np.abs(scA - sc).max())

# --- side breakdown (path B scores) ---
u1 = is_unk[0::2]   # src1 unknown?
print('AUROC when unknown is src1 (carrier 2000):',
      auroc(sc[~is_unk & ~np.tile(u1, 2)], sc[is_unk & np.tile(u1, 2)]) if False else 'skip')
# proper side-split: restrict to mixtures where unknown is src1
side1 = u1
maskK1 = np.concatenate([~side1 & False, ~side1])  # placeholder no
