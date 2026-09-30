"""
Paper 3 — Shared truth-anchored (TA) evaluation helpers.

Consolidates the TA machinery used by the review-2 follow-up scripts
(eval_protocol_split_ta.py, eval_nose_ta.py, eval_lomo_ta.py,
eval_robustness_ta.py): per-slot scoring with refpool-fitted OOD scorers,
truth-anchored pool assembly (pool membership from the DATASET's
per-source ood flags, never the label-swapped is_ood_1/2 of
evaluate._collect_predictions — see EXPERIMENT_LOG.md 2026-09-30), and the
pair-weighted wavg AUROC table.

All scorers are fitted on a caller-supplied reference (ref_emb, ref_mods)
— the letter's standard is the TA refpool (`*_refpool_ta.npz`, seed 88888).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import config as C
from evaluate import _build_model_from_ckpt, _collect_predictions  # noqa: F401
from odin_dump import collect_odin  # noqa: F401
from ood_baselines import mahalanobis_scores, msp_scores
from ood_scores import (
    compute_prototypes, energy_score, prototype_score, vos_score,
)
from open_set_metrics import auroc  # noqa: F401

BASE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed{}_best'
BINS = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
METHODS6 = ['energy', 'msp', 'odin', 'mahalanobis', 'prototype', 'vos']
ODIN_EPS, ODIN_T = 0.005, 1000.0


def truth_labels(ds):
    """Per-sample true labels in dataset (== sequential loader) order."""
    return {
        'mod1': np.asarray([s['mod1_idx'] for s in ds.samples], dtype=np.int64),
        'mod2': np.asarray([s['mod2_idx'] for s in ds.samples], dtype=np.int64),
        'ood1': np.asarray([s['mod1_is_ood'] for s in ds.samples], dtype=bool),
        'ood2': np.asarray([s['mod2_is_ood'] for s in ds.samples], dtype=bool),
    }


def _vos(emb, protos, chunk=1024):
    return np.concatenate([
        vos_score(emb[i:i + chunk], protos, alpha=C.VOS_ALPHA,
                  n_per_class=C.VOS_N_SYNTHETIC, seed=0)
        for i in range(0, len(emb), chunk)])


def slot_scores(pred, odin, ref_emb, ref_mods, methods, num_known=4):
    """Score BOTH PIT-aligned slots of one protocol's predictions.

    methods: subset of METHODS6 ('odin' requires odin != None).
    Returns {slot: {method: (N,) scores}}.
    """
    protos = compute_prototypes(ref_emb, ref_mods, num_known)
    out = {}
    for slot in ('1', '2'):
        emb, logits = pred[f'emb_{slot}'], pred[f'logits_{slot}']
        sc = {}
        if 'energy' in methods:
            sc['energy'] = energy_score(logits)
        if 'msp' in methods:
            sc['msp'] = msp_scores(logits)
        if 'odin' in methods:
            sc['odin'] = np.asarray(odin[f'odin_{slot}'], dtype=np.float64)
        if 'mahalanobis' in methods:
            sc['mahalanobis'] = mahalanobis_scores(ref_emb, ref_mods, emb,
                                                   n_classes=num_known,
                                                   shrink=0.1)
        if 'prototype' in methods:
            sc['prototype'] = prototype_score(emb, protos)
        if 'vos' in methods:
            sc['vos'] = _vos(emb, protos)
        out[slot] = sc
    return out


def _sel(sc, m1, m2, methods):
    """Concatenate slot-1[m1] and slot-2[m2] per method (m=None -> all)."""
    s1 = sc['1'] if m1 is None else {m: v[m1] for m, v in sc['1'].items()}
    s2 = sc['2'] if m2 is None else {m: v[m2] for m, v in sc['2'].items()}
    return {m: np.concatenate([s1[m], s2[m]]) for m in methods}


def pools_ta(sc, pred, tr, methods):
    """Truth-anchored pools for one protocol set.

    Returns dict with K (kk both slots), U_ku (ku true-OOD side),
    K_ku (ku true-known side), U_uu (uu both slots or None), each
    {'scores': {m: arr}, 'snr': arr}.  Membership uses the dataset truth
    flags; snr is per-mixture (swap-invariant).
    """
    out = {}
    if 'kk' in sc:
        out['K_kk'] = {'scores': _sel(sc['kk'], None, None, methods),
                       'snr': np.tile(pred['kk']['snr'], 2)}
    if 'ku' in sc:
        to1, to2 = tr['ku']['ood1'], tr['ku']['ood2']
        out['U_ku'] = {'scores': _sel(sc['ku'], to1, to2, methods),
                       'snr': np.concatenate([pred['ku']['snr'][to1],
                                              pred['ku']['snr'][to2]])}
        out['K_ku'] = {'scores': _sel(sc['ku'], ~to1, ~to2, methods),
                       'snr': np.concatenate([pred['ku']['snr'][~to1],
                                              pred['ku']['snr'][~to2]])}
    if 'uu' in sc:
        out['U_uu'] = {'scores': _sel(sc['uu'], None, None, methods),
                       'snr': np.tile(pred['uu']['snr'], 2)}
    return out


def combine_pools(*pools, methods):
    """Concatenate pool dicts (same methods)."""
    return {'scores': {m: np.concatenate([p['scores'][m] for p in pools])
                       for m in methods},
            'snr': np.concatenate([p['snr'] for p in pools])}


def wavg_table(K, U, methods, boundary=0.0, bins=BINS):
    """Per-bin AUROC per method + routed (energy<=boundary else prototype) +
    pair-weighted wavg + pooled, on score pools K/U ({'scores', 'snr'})."""
    k_sc, k_snr = K['scores'], K['snr']
    u_sc, u_snr = U['scores'], U['snr']
    per_bin = {}
    for s in bins:
        mk, mu = k_snr == s, u_snr == s
        nk, nu = int(mk.sum()), int(mu.sum())
        if nk < 2 or nu < 2:
            continue
        aucs = {m: auroc(k_sc[m][mk], u_sc[m][mu]) for m in methods}
        per_bin[str(int(s))] = {'n_known': nk, 'n_unknown': nu, **aucs}
        if 'energy' in methods and 'prototype' in methods:
            per_bin[str(int(s))]['routed'] = aucs[
                'energy' if s <= boundary else 'prototype']

    keys = list(methods) + (['routed'] if 'routed' in next(iter(per_bin.values()),
                                                         {}) else [])

    def wavg(m):
        num = sum(v[m] * v['n_known'] * v['n_unknown'] for v in per_bin.values())
        den = sum(v['n_known'] * v['n_unknown'] for v in per_bin.values())
        return num / den if den else float('nan')

    return {'per_bin': per_bin,
            'wavg': {m: wavg(m) for m in keys},
            'pooled': {m: auroc(k_sc[m], u_sc[m]) for m in methods}}


def cls_acc(pred_kk, tr_kk):
    """(stored-label-swapped acc, truth-anchored acc) on the kk protocol."""
    pr1 = pred_kk['logits_1'].argmax(-1)
    pr2 = pred_kk['logits_2'].argmax(-1)
    stored = float(np.mean(np.concatenate([
        pr1 == pred_kk['mod1_idx'], pr2 == pred_kk['mod2_idx']])))
    truth = float(np.mean(np.concatenate([
        pr1 == tr_kk['mod1'], pr2 == tr_kk['mod2']])))
    return stored, truth


def ms(vals):
    v = np.asarray(vals, dtype=np.float64)
    return {'mean': float(v.mean()), 'std': float(v.std()), 'n': int(v.size)}


def round4(obj):
    if isinstance(obj, dict):
        return {k: round4(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round4(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 4)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj
