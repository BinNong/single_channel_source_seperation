"""
Paper 5 — Task-Oriented SC-BSS: Evaluation script.

Deterministic test grid: SNR{-10,-5,0,5,10,15,20} x K{1,2,3} plus a K=4
zero-shot extrapolation cell (seed=99999).  Reports:

  1. Counting — accuracy (overall / per-SNR / per-K), confusion matrix,
     +/-1-tolerance accuracy, for BOTH counting routes of the model:
     occupancy-count (threshold 0.5) and count-head (argmax) for the slot
     arch; stop-rule count for the recursive arch.
  2. Separation (K in {1,2,3} cells) — Hungarian matching (scipy
     linear_sum_assignment, cost = -SI-SDR) between true sources and the
     occupied slots / extracted sources: per-matched-pair SI-SDRi (over
     the mixture-as-estimate baseline), SDR, SIR; miss rate; hallucination
     rate (+ relative power; K=1 subset listed separately).
  3. Cascade analysis — separation quality conditioned on count
     correct/incorrect; oracle-K (top-K slots by occupancy / first-K
     steps) vs end-to-end (predicted K), per SNR.
  4. K=4 extrapolation cell — occupancy/stop counting only.
  5. (Optional, --ser) per-matched-pair SER via
     paper1_cnn_se/utils.py:compute_ser_from_signal (per-modulation).
  6. (Optional, --ser_comp) per-matched-pair compensated SER **and**
     Gray-mapped BER via ser_comp.compute_ser_ber_compensated (paper 5;
     both come from the SAME hard decisions).

Outputs: results/eval_<run_name>.json + readable tables on stdout.

Usage:
    python evaluate.py --checkpoint checkpoints/<name>_best.pt
    python evaluate.py --checkpoint <ckpt> --n_per_cell 4      # smoke
    python evaluate.py --checkpoint <ckpt> --ser_comp          # SER + BER
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

# Ensure paper4_open_world/ is first on sys.path so `from models import ...`
# resolves to OUR models.py (not paper1's).
_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import config as C
from data_generator_vark import (
    CommBSSVarKTestDataset,
    MOD_TYPES,
    IDX_TO_MOD,
)
from models import SlotSepNet, OneAndRestNet, SpecialistBankNet


# ============================================================================
# CLI
# ============================================================================
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Evaluate variable-K SC-BSS models')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--n_per_cell', type=int, default=C.DataConfig.test_n_per_cell,
                   help='Samples per (SNR, K) cell of the test grid')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--occ_threshold', type=float, default=C.VarKConfig.occ_threshold)
    p.add_argument('--ser', action='store_true',
                   help='Also compute per-matched-pair SER (slow)')
    p.add_argument('--ser_comp', action='store_true',
                   help='Compute offset-COMPENSATED SER instead '
                        '(ser_comp.py; oracle phase-ramp removal — '
                        'diagnoses the proxy-receiver floor)')
    # num_workers=0: the test set is eager/in-memory; worker pickling of
    # the sample tensors is pure overhead.
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    return p.parse_args()


# ============================================================================
# Model construction from checkpoint
# ============================================================================
def build_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get('args', {})
    arch = a.get('arch', 'slot')
    common = dict(
        hidden_channels=a.get('hidden', C.ModelConfig.hidden_channels),
        n_layers=a.get('layers', C.ModelConfig.n_layers),
        k_max=a.get('k_slots', C.VarKConfig.k_slots),
        use_se=not a.get('no_se', False),
        head_embed_dim=C.ModelConfig.head_embed_dim,
    )
    if arch == 'slot':
        model = SlotSepNet(use_count_head=not a.get('no_count_head', False),
                           use_joint_head=a.get('lambda_llr', 0.0) > 0,
                           **common)
    elif arch == 'specialist':
        model = SpecialistBankNet(**common)
    else:
        model = OneAndRestNet(**common)
    model.load_state_dict(ckpt['model'])
    # NOTE: load_state_dict only copies tensor data — it does NOT move the
    # model.  Without this .to(device), weights stay on CPU while inputs are
    # on GPU (local CPU smoke tests cannot catch this).
    model.to(device)
    model.eval()
    return model, arch, a


# ============================================================================
# Per-pair metrics (numpy, complex-safe)
# ============================================================================
def _si_sdr_np(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    e = np.stack([est.real, est.imag], axis=-1).ravel()
    r = np.stack([ref.real, ref.imag], axis=-1).ravel()
    e = e - e.mean()
    r = r - r.mean()
    alpha = np.dot(r, e) / (np.dot(r, r) + eps)
    target = alpha * r
    noise = e - target
    return float(10 * np.log10(np.dot(target, target)
                               / (np.dot(noise, noise) + eps) + eps))


def _sdr_np(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    noise = est - ref
    return float(10 * np.log10(
        np.sum(np.abs(ref) ** 2) / (np.sum(np.abs(noise) ** 2) + eps) + eps))


def _sir_np(est: np.ndarray, target: np.ndarray,
            interferers: list[np.ndarray], eps: float = 1e-8) -> float:
    """SIR via per-interferer projection power (NaN when K == 1)."""
    if len(interferers) == 0:
        return float('nan')
    p_t = abs(np.sum(est * np.conj(target))) ** 2 / (
        np.sum(np.abs(target) ** 2) + eps)
    p_i = sum(abs(np.sum(est * np.conj(g))) ** 2 / (
        np.sum(np.abs(g) ** 2) + eps) for g in interferers)
    return float(10 * np.log10(p_t / (p_i + eps) + eps))


# ============================================================================
# Per-sample analysis
# ============================================================================
def _match_and_measure(est_list: list[np.ndarray],
                       src_list: list[np.ndarray],
                       mix: np.ndarray,
                       mods: list[int],
                       compute_ser: bool,
                       ser_comp: bool = False,
                       src_carriers: list[float] | None = None,
                       est_syms: list[np.ndarray] | None = None):
    """Hungarian-match estimates to true sources; measure matched pairs.

    Returns (pair_records, n_miss, n_halluc, halluc_rel_power).
    pair_records: list of dicts with si_sdr / si_sdri / sdr / sir / mod
    and optionally ser (raw proxy) or ser_comp (offset-compensated).
    ser_comp requires src_carriers (true per-source carriers).
    est_syms (paper 5 S3): per-slot JointLLRHead logit arrays [N_sym, 16]
    aligned with est_list; when given (with src_carriers), each matched
    pair additionally gets ser_joint / ber_joint — the head's hard
    decisions (argmax over logits masked to the source's oracle
    constellation) vs the reference receiver's hard labels.
    """
    K = len(src_list)
    K_hat = len(est_list)
    if K_hat == 0:
        return [], K, 0, 0.0

    cost = np.full((K, K_hat), 1e6)
    for i, s in enumerate(src_list):
        for j, e in enumerate(est_list):
            cost[i, j] = -_si_sdr_np(e, s)
    rows, cols = linear_sum_assignment(cost)

    pairs = []
    ser_fn = None
    if compute_ser:
        if ser_comp:
            # Offset-compensated proxy SER + Gray-mapped BER (paper 5):
            # down-conversion at the TRUE per-source carrier (oracle carrier
            # sync; see ser_comp.py).  SER and BER share the same decisions.
            assert src_carriers is not None, \
                'ser_comp needs true carriers (dataset return_carriers=True)'
            from ser_comp import compute_ser_ber_compensated as ser_fn
        else:
            # Lazy import: utils.py (symlink to paper1) pulls in matplotlib;
            # only pay for it when --ser is requested.  compute_ser_from_signal
            # does `from data_generator import rrc_filter` internally, which
            # resolves via the paper1 path appended by data_generator_vark.
            from utils import compute_ser_from_signal as ser_fn
    for i, j in zip(rows, cols):
        s, e = src_list[i], est_list[j]
        si = -cost[i, j]
        si_base = _si_sdr_np(mix, s)
        interferers = [g for m, g in enumerate(src_list) if m != i]
        rec = {
            'si_sdr': si,
            'si_sdri': si - si_base,
            'sdr': _sdr_np(e, s),
            'sir': _sir_np(e, s, interferers),
            'mod': mods[i],
        }
        if ser_fn is not None:
            if ser_comp:
                ser_v, ber_v = ser_fn(
                    torch.from_numpy(e), torch.from_numpy(s),
                    mod_type=IDX_TO_MOD[mods[i]],
                    sample_rate=C.SignalConfig.sample_rate,
                    n_symbols=C.SignalConfig.n_symbols,
                    roll_off=C.SignalConfig.roll_off,
                    num_taps=C.SignalConfig.num_taps,
                    carrier_freq=float(src_carriers[i]),
                )
                rec['ser_comp'] = ser_v
                rec['ber_comp'] = ber_v
            else:
                rec['ser'] = ser_fn(
                    torch.from_numpy(e), torch.from_numpy(s),
                    mod_type=IDX_TO_MOD[mods[i]],
                    sample_rate=C.SignalConfig.sample_rate,
                    n_symbols=C.SignalConfig.n_symbols,
                    roll_off=C.SignalConfig.roll_off,
                    num_taps=C.SignalConfig.num_taps,
                    carrier_freq=C.SignalConfig.carrier_base,
                )
        if est_syms is not None:
            # JointLLRHead route (S3): head decisions on this matched pair,
            # scored against the reference receiver's hard labels.
            from ser_comp import ref_labels_only, GRAY_BITS
            mod_name = IDX_TO_MOD[mods[i]]
            n_pts = len(GRAY_BITS[mod_name])
            est_lab = est_syms[j][:, :n_pts].argmax(axis=1)
            ref_lab = ref_labels_only(
                torch.from_numpy(s), mod_type=mod_name,
                sample_rate=C.SignalConfig.sample_rate,
                n_symbols=C.SignalConfig.n_symbols,
                roll_off=C.SignalConfig.roll_off,
                num_taps=C.SignalConfig.num_taps,
                carrier_freq=float(src_carriers[i]))
            n = min(len(est_lab), len(ref_lab))
            rec['ser_joint'] = float(np.mean(est_lab[:n] != ref_lab[:n]))
            bits = GRAY_BITS[mod_name]
            rec['ber_joint'] = float(np.mean(bits[est_lab[:n]]
                                             != bits[ref_lab[:n]]))
        pairs.append(rec)

    n_miss = K - len(rows)
    n_halluc = K_hat - len(rows)
    halluc_rel = 0.0
    if n_halluc > 0:
        matched_cols = set(cols.tolist())
        p_halluc = sum(np.sum(np.abs(est_list[j]) ** 2)
                       for j in range(K_hat) if j not in matched_cols)
        p_all = sum(np.sum(np.abs(e) ** 2) for e in est_list) + 1e-12
        halluc_rel = float(p_halluc / p_all)
    return pairs, n_miss, n_halluc, halluc_rel


# ============================================================================
# Inference: collect per-sample records over the test grid
# ============================================================================
@torch.no_grad()
def collect_records(model, arch: str, loader, device,
                    occ_threshold: float, k_slots: int,
                    compute_ser: bool, ser_comp: bool = False) -> list[dict]:
    records = []
    for batch in loader:
        # The test dataset yields a 7th element (true per-source
        # carriers) only when built with return_carriers=True
        # (--ser_comp); otherwise a 6-tuple.
        if len(batch) == 7:
            mix, sources, _occ, k, snr, mods, carriers = batch
            carriers_np = carriers.numpy()
        else:
            mix, sources, _occ, k, snr, mods = batch
            carriers_np = None
        mix = mix.to(device)
        B = mix.shape[0]

        if arch == 'slot':
            out = model(mix)
            if len(out) == 4:
                # paper 5 S3: model carries a JointLLRHead
                slots, occ_logits, count_logits, sym_logits = out
                sym_np = sym_logits.cpu().numpy()               # [B, S, N, 16]
            else:
                slots, occ_logits, count_logits = out
                sym_np = None
            occ_prob = torch.sigmoid(occ_logits)                # [B, S]
            slots_np = slots.cpu().numpy()                      # [B, S, T]
            occ_np = occ_prob.cpu().numpy()
            cnt_hat = (count_logits.argmax(dim=1) + 1).cpu().numpy() \
                if count_logits is not None else None
        elif arch == 'specialist':
            # Self-routed: the count head drives the specialist routing.
            slots, count_logits = model(mix)
            slots_np = slots.cpu().numpy()                      # [B, S, T]
            cnt_hat = (count_logits.argmax(dim=1) + 1).cpu().numpy()
            # Teacher-routed pass for the oracle-K cascade reference.
            # K=4 samples are clamped through the K=3 specialist; their
            # oracle slots are never used (separation skips the K=4 cell).
            oracle_t, _ = model(mix,
                                k_route=torch.clamp(k.to(device), max=3))
            oracle_np = oracle_t.cpu().numpy()
        else:
            ones, stop_logits = model(mix, max_steps=k_slots)
            stop_prob = torch.sigmoid(stop_logits)              # [B, S]
            ones_np = ones.cpu().numpy()                        # [B, S, T]
            stop_np = stop_prob.cpu().numpy()
            cnt_hat = None

        mix_np = mix.cpu().numpy()
        src_np = sources.numpy()                                # [B, S, 1, T]
        k_np = k.numpy()
        snr_np = np.asarray(snr, dtype=np.float32)
        mods_np = mods.numpy()

        for b in range(B):
            kt = int(k_np[b])
            srcs = [src_np[b, j, 0] for j in range(kt)]
            mod_ids = [int(m) for m in mods_np[b, :kt]]

            rec = {
                'snr': float(snr_np[b]),
                'k_true': kt,
                'mods': mod_ids,
            }

            # S3: joint-head logits (slot arch only); None otherwise.
            sym_slots = None
            if arch == 'slot':
                order = np.argsort(-occ_np[b])                  # by occupancy
                k_occ = int((occ_np[b] > occ_threshold).sum())
                # Counting clips K_hat to >= 1 (a separator must emit >= 1).
                rec['k_hat_occ'] = max(k_occ, 1)
                rec['k_hat_cnt'] = int(cnt_hat[b]) if cnt_hat is not None else None
                occ_slots = [slots_np[b, j] for j in order[:k_occ]]
                oracle_slots = [slots_np[b, j] for j in order[:kt]]
                # S3: joint-head logits for the SAME slot subset/order.
                sym_slots = ([sym_np[b, j] for j in order[:k_occ]]
                             if sym_np is not None else None)
            elif arch == 'specialist':
                k_hat = int(cnt_hat[b])         # count head is the only route
                rec['k_hat_occ'] = k_hat
                rec['k_hat_cnt'] = None
                # Occupied slots = the first k_hat (routed specialist's).
                occ_slots = [slots_np[b, j] for j in range(k_hat)]
                oracle_slots = [oracle_np[b, j] for j in range(min(kt, 3))]
            else:
                # Stop rule: first step whose stop head says "no more".
                k_stop = k_slots
                for i in range(k_slots):
                    if stop_np[b, i] < 0.5:
                        k_stop = i + 1
                        break
                rec['k_hat_occ'] = max(k_stop, 1)   # stop route == count route
                rec['k_hat_cnt'] = None
                occ_slots = [ones_np[b, j] for j in range(k_stop)]
                oracle_slots = [ones_np[b, j] for j in range(kt)]

            # Separation is evaluated on the K in {1,2,3} cells only
            # (K=4 cell: counting-only probe per spec §8.5).
            if kt <= 3:
                src_carriers = ([float(c) for c in carriers_np[b, :kt]]
                                if carriers_np is not None else None)
                pairs, n_miss, n_halluc, hrel = _match_and_measure(
                    occ_slots, srcs, mix_np[b, 0], mod_ids, compute_ser,
                    ser_comp, src_carriers,
                    est_syms=(sym_slots if src_carriers is not None
                              else None))
                opairs, _, _, _ = _match_and_measure(
                    oracle_slots, srcs, mix_np[b, 0], mod_ids, False)
                rec['pairs'] = pairs
                rec['n_miss'] = n_miss
                rec['n_halluc'] = n_halluc
                rec['halluc_rel_power'] = hrel
                rec['n_occupied'] = len(occ_slots)
                rec['oracle_pair_si_sdri'] = [p['si_sdri'] for p in opairs]
            records.append(rec)
    return records


# ============================================================================
# Aggregation
# ============================================================================
SNRS = C.SignalConfig.snr_test_points
KS = [1, 2, 3, 4]


def _acc(recs, key):
    if not recs:
        return float('nan')
    return float(np.mean([r[key] == r['k_true'] for r in recs]))


def _tol1(recs, key):
    if not recs:
        return float('nan')
    return float(np.mean([abs(r[key] - r['k_true']) <= 1 for r in recs]))


def _confusion(recs, key):
    m = np.zeros((4, 4), dtype=int)
    for r in recs:
        m[r['k_true'] - 1, min(max(r[key], 1), 4) - 1] += 1
    return m


def _counting_block(recs, key, label):
    in_grid = [r for r in recs if r['k_true'] <= 3]
    k4 = [r for r in recs if r['k_true'] == 4]
    return {
        'route': label,
        'overall_acc': _acc(in_grid, key),
        'tol1_acc': _tol1(in_grid, key),
        'per_snr_acc': {int(s): _acc([r for r in in_grid if r['snr'] == s], key)
                        for s in SNRS},
        'per_k_acc': {kk: _acc([r for r in in_grid if r['k_true'] == kk], key)
                      for kk in (1, 2, 3)},
        'confusion': _confusion(in_grid, key).tolist(),
        'k4_acc': _acc(k4, key),
        'k4_per_snr_acc': {int(s): _acc([r for r in k4 if r['snr'] == s], key)
                           for s in SNRS},
        'k4_confusion': _confusion(k4, key).tolist(),
    }


def _pair_field(recs, field):
    vals = [p[field] for r in recs for p in r.get('pairs', [])
            if not (isinstance(p[field], float) and np.isnan(p[field]))]
    return float(np.mean(vals)) if vals else float('nan')


def _sep_block(recs):
    recs = [r for r in recs if 'pairs' in r]
    if not recs:
        return {}
    total_true = sum(r['k_true'] for r in recs)
    total_occ = sum(r['n_occupied'] for r in recs)
    total_miss = sum(r['n_miss'] for r in recs)
    total_halluc = sum(r['n_halluc'] for r in recs)
    k1 = [r for r in recs if r['k_true'] == 1]
    out = {
        'si_sdr': _pair_field(recs, 'si_sdr'),
        'si_sdri': _pair_field(recs, 'si_sdri'),
        'sdr': _pair_field(recs, 'sdr'),
        'sir': _pair_field(recs, 'sir'),
        'miss_rate': total_miss / max(total_true, 1),
        'halluc_rate': total_halluc / max(total_occ, 1),
        'halluc_rel_power': float(np.mean(
            [r['halluc_rel_power'] for r in recs if r['n_halluc'] > 0]))
        if any(r['n_halluc'] > 0 for r in recs) else 0.0,
        # K=1 subset: slot-level halluc rate + sample-level false-alarm rate
        'k1_halluc_rate': (sum(r['n_halluc'] for r in k1)
                           / max(sum(r['n_occupied'] for r in k1), 1)) if k1
        else float('nan'),
        'k1_false_alarm_rate': float(np.mean(
            [r['n_halluc'] > 0 for r in k1])) if k1 else float('nan'),
    }
    if any('ser' in p for r in recs for p in r['pairs']):
        out['ser'] = _pair_field(recs, 'ser')
        out['ser_per_mod'] = {
            IDX_TO_MOD[m]: float(np.mean([
                p['ser'] for r in recs for p in r['pairs']
                if p['mod'] == m and 'ser' in p]))
            for m in range(len(MOD_TYPES))
            if any(p['mod'] == m and 'ser' in p
                   for r in recs for p in r['pairs'])
        }
    if any('ser_comp' in p for r in recs for p in r['pairs']):
        out['ser_comp'] = _pair_field(recs, 'ser_comp')
        out['ser_comp_per_mod'] = {
            IDX_TO_MOD[m]: float(np.mean([
                p['ser_comp'] for r in recs for p in r['pairs']
                if p['mod'] == m and 'ser_comp' in p]))
            for m in range(len(MOD_TYPES))
            if any(p['mod'] == m and 'ser_comp' in p
                   for r in recs for p in r['pairs'])
        }
    if any('ber_comp' in p for r in recs for p in r['pairs']):
        out['ber_comp'] = _pair_field(recs, 'ber_comp')
        out['ber_comp_per_mod'] = {
            IDX_TO_MOD[m]: float(np.mean([
                p['ber_comp'] for r in recs for p in r['pairs']
                if p['mod'] == m and 'ber_comp' in p]))
            for m in range(len(MOD_TYPES))
            if any(p['mod'] == m and 'ber_comp' in p
                   for r in recs for p in r['pairs'])
        }
    if any('ser_joint' in p for r in recs for p in r['pairs']):
        out['ser_joint'] = _pair_field(recs, 'ser_joint')
        out['ser_joint_per_mod'] = {
            IDX_TO_MOD[m]: float(np.mean([
                p['ser_joint'] for r in recs for p in r['pairs']
                if p['mod'] == m and 'ser_joint' in p]))
            for m in range(len(MOD_TYPES))
            if any(p['mod'] == m and 'ser_joint' in p
                   for r in recs for p in r['pairs'])
        }
        out['ber_joint'] = _pair_field(recs, 'ber_joint')
        out['ber_joint_per_mod'] = {
            IDX_TO_MOD[m]: float(np.mean([
                p['ber_joint'] for r in recs for p in r['pairs']
                if p['mod'] == m and 'ber_joint' in p]))
            for m in range(len(MOD_TYPES))
            if any(p['mod'] == m and 'ber_joint' in p
                   for r in recs for p in r['pairs'])
        }
    return out


def aggregate(records: list[dict], arch: str) -> dict:
    grid = [r for r in records if r['k_true'] <= 3]

    route_name = {'slot': 'occupancy', 'recursive': 'stop',
                  'specialist': 'count_head'}[arch]
    counting = {route_name: _counting_block(records, 'k_hat_occ', route_name)}
    if arch == 'slot' and any(r['k_hat_cnt'] is not None for r in records):
        counting['count_head'] = _counting_block(records, 'k_hat_cnt',
                                                 'count_head')

    separation = _sep_block(grid)
    separation['per_snr'] = {int(s): _sep_block([r for r in grid
                                                 if r['snr'] == s])
                             for s in SNRS}
    separation['per_k'] = {kk: _sep_block([r for r in grid
                                           if r['k_true'] == kk])
                           for kk in (1, 2, 3)}

    # Cascade: separation conditioned on counting correctness (occ route)
    correct = [r for r in grid if 'pairs' in r and r['k_hat_occ'] == r['k_true']]
    incorrect = [r for r in grid if 'pairs' in r and r['k_hat_occ'] != r['k_true']]
    oracle_per_snr = {}
    e2e_per_snr = {}
    for s in SNRS:
        rs = [r for r in grid if r['snr'] == s and 'pairs' in r]
        ov = [v for r in rs for v in r['oracle_pair_si_sdri']]
        ev = [p['si_sdri'] for r in rs for p in r['pairs']]
        oracle_per_snr[int(s)] = float(np.mean(ov)) if ov else float('nan')
        e2e_per_snr[int(s)] = float(np.mean(ev)) if ev else float('nan')
    cascade = {
        'n_count_correct': len(correct),
        'n_count_incorrect': len(incorrect),
        'si_sdri_count_correct': _pair_field(correct, 'si_sdri'),
        'si_sdri_count_incorrect': _pair_field(incorrect, 'si_sdri'),
        'oracle_k_si_sdri_per_snr': oracle_per_snr,
        'e2e_si_sdri_per_snr': e2e_per_snr,
    }

    return {
        'counting': counting,
        'separation': separation,
        'cascade': cascade,
    }


# ============================================================================
# Pretty printing
# ============================================================================
def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '  —  '
    return f'{v:.{nd}f}'


def print_results(res: dict, arch: str):
    print('\n' + '=' * 72)
    print('COUNTING')
    print('=' * 72)
    if arch == 'specialist':
        print("  NOTE: SpecialistBankNet (A6) has a 3-way count head and no\n"
              "  K=4 specialist — it CANNOT represent K=4, so 0% accuracy on\n"
              "  the K=4 cell is EXPECTED BY DESIGN (the architectural-failure\n"
              "  demonstration that motivates the unified variable-K archs).")
    for route, blk in res['counting'].items():
        print(f"\n-- route: {route} --")
        print(f"  overall acc = {_fmt(blk['overall_acc'])}   "
              f"+/-1 tol acc = {_fmt(blk['tol1_acc'])}")
        print('  per-K acc:   ' + '  '.join(
            f"K={kk}: {_fmt(blk['per_k_acc'][kk])}" for kk in (1, 2, 3)))
        print('  per-SNR acc: ' + '  '.join(
            f"{int(s)}: {_fmt(blk['per_snr_acc'][int(s)], 2)}" for s in SNRS))
        print('  confusion (rows=true K, cols=pred K):')
        print('        ' + '  '.join(f'K={p}' for p in (1, 2, 3, 4)))
        for i, row in enumerate(blk['confusion']):
            print(f'    K={i+1}  ' + '  '.join(f'{v:>3d}' for v in row))
        print(f"  K=4 extrapolation acc = {_fmt(blk['k4_acc'])}"
              + ('  (route cannot exceed K=3)' if route == 'count_head' else ''))
        print('  K=4 per-SNR acc: ' + '  '.join(
            f"{int(s)}: {_fmt(blk['k4_per_snr_acc'][int(s)], 2)}" for s in SNRS))

    sep = res['separation']
    if sep:
        print('\n' + '=' * 72)
        print('SEPARATION (Hungarian-matched, K in {1,2,3} cells)')
        print('=' * 72)
        print(f"  SI-SDR = {_fmt(sep['si_sdr'])} dB   "
              f"SI-SDRi = {_fmt(sep['si_sdri'])} dB   "
              f"SDR = {_fmt(sep['sdr'])} dB   SIR = {_fmt(sep['sir'])} dB")
        print(f"  miss rate = {_fmt(sep['miss_rate'])}   "
              f"halluc rate = {_fmt(sep['halluc_rate'])} "
              f"(rel. power {_fmt(sep['halluc_rel_power'])})")
        print(f"  K=1 halluc rate = {_fmt(sep['k1_halluc_rate'])}   "
              f"K=1 false-alarm rate = {_fmt(sep['k1_false_alarm_rate'])}")
        if 'ser' in sep:
            print(f"  SER = {_fmt(sep['ser'], 4)}  per-mod: "
                  + '  '.join(f'{m}: {_fmt(v, 4)}'
                              for m, v in sep['ser_per_mod'].items()))
        if 'ser_comp' in sep:
            print(f"  SER (offset-compensated) = {_fmt(sep['ser_comp'], 4)}"
                  "  per-mod: "
                  + '  '.join(f'{m}: {_fmt(v, 4)}'
                              for m, v in sep['ser_comp_per_mod'].items()))
        if 'ber_comp' in sep:
            print(f"  BER (offset-compensated, Gray) = "
                  f"{_fmt(sep['ber_comp'], 4)}  per-mod: "
                  + '  '.join(f'{m}: {_fmt(v, 4)}'
                              for m, v in sep['ber_comp_per_mod'].items()))
        if 'ser_joint' in sep:
            print(f"  SER (joint LLR head) = {_fmt(sep['ser_joint'], 4)}"
                  "  per-mod: "
                  + '  '.join(f'{m}: {_fmt(v, 4)}'
                              for m, v in sep['ser_joint_per_mod'].items()))
            print(f"  BER (joint LLR head, Gray) = "
                  f"{_fmt(sep['ber_joint'], 4)}  per-mod: "
                  + '  '.join(f'{m}: {_fmt(v, 4)}'
                              for m, v in sep['ber_joint_per_mod'].items()))
        print('\n  per-SNR:')
        print(f"    {'SNR':>5s} | {'SI-SDRi':>8s} | {'SDR':>7s} | {'SIR':>7s} | "
              f"{'miss':>6s} | {'halluc':>6s}")
        for s in SNRS:
            b = sep['per_snr'][int(s)]
            if not b:
                continue
            print(f"    {int(s):>5d} | {_fmt(b['si_sdri']):>8s} | "
                  f"{_fmt(b['sdr']):>7s} | {_fmt(b['sir']):>7s} | "
                  f"{_fmt(b['miss_rate'], 2):>6s} | {_fmt(b['halluc_rate'], 2):>6s}")
        print('\n  per-K:')
        print(f"    {'K':>3s} | {'SI-SDRi':>8s} | {'SDR':>7s} | {'SIR':>7s} | "
              f"{'miss':>6s} | {'halluc':>6s}")
        for kk in (1, 2, 3):
            b = sep['per_k'][kk]
            if not b:
                continue
            print(f"    {kk:>3d} | {_fmt(b['si_sdri']):>8s} | "
                  f"{_fmt(b['sdr']):>7s} | {_fmt(b['sir']):>7s} | "
                  f"{_fmt(b['miss_rate'], 2):>6s} | {_fmt(b['halluc_rate'], 2):>6s}")

    cas = res['cascade']
    print('\n' + '=' * 72)
    print('CASCADE ANALYSIS')
    print('=' * 72)
    print(f"  count correct: n={cas['n_count_correct']}  "
          f"SI-SDRi = {_fmt(cas['si_sdri_count_correct'])} dB")
    print(f"  count wrong:   n={cas['n_count_incorrect']}  "
          f"SI-SDRi = {_fmt(cas['si_sdri_count_incorrect'])} dB")
    print('\n  oracle-K vs end-to-end SI-SDRi (dB) per SNR:')
    print(f"    {'SNR':>5s} | {'oracle-K':>9s} | {'end-to-end':>10s} | {'gap':>6s}")
    for s in SNRS:
        o = cas['oracle_k_si_sdri_per_snr'][int(s)]
        e = cas['e2e_si_sdri_per_snr'][int(s)]
        gap = o - e if not (np.isnan(o) or np.isnan(e)) else float('nan')
        print(f"    {int(s):>5d} | {_fmt(o):>9s} | {_fmt(e):>10s} | {_fmt(gap):>6s}")


# ============================================================================
# Main
# ============================================================================
def main():
    args = get_args()
    device = C.DEVICE

    print(f"Loading model from {args.checkpoint} ...")
    model, arch, ckpt_args = build_model_from_ckpt(args.checkpoint, device)
    k_slots = ckpt_args.get('k_slots', C.VarKConfig.k_slots)
    print(f"arch={arch}  hidden={ckpt_args.get('hidden')}  "
          f"layers={ckpt_args.get('layers')}  k_slots={k_slots}")

    test_ds = CommBSSVarKTestDataset(
        n_per_cell=args.n_per_cell,
        snr_points=C.SignalConfig.snr_test_points,
        mod_types=MOD_TYPES,
        k_lo=C.VarKConfig.k_min, k_hi=C.VarKConfig.k_max,
        k_extrap=C.VarKConfig.k_extrap,
        k_slots=k_slots,
        signal_length=C.SignalConfig.signal_length,
        sample_rate=C.SignalConfig.sample_rate,
        carrier_base=C.SignalConfig.carrier_base,
        freq_gap_range=C.SignalConfig.freq_gap_range,
        seed=C.DataConfig.test_seed,
        # True carriers are needed by --ser_comp AND by the JointLLRHead
        # eval route (reference labels), which activates whenever the
        # checkpoint was trained with --lambda_llr > 0.
        return_carriers=(args.ser_comp
                         or ckpt_args.get('lambda_llr', 0.0) > 0),
    )
    print(f"Test grid: {len(test_ds)} samples "
          f"({len(C.SignalConfig.snr_test_points)} SNR x K{{1,2,3}}+K=4 x "
          f"{args.n_per_cell} per cell, seed={C.DataConfig.test_seed})")
    loader = DataLoader(test_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    records = collect_records(model, arch, loader, device,
                              args.occ_threshold, k_slots,
                              args.ser or args.ser_comp, args.ser_comp)
    res = aggregate(records, arch)
    res['checkpoint'] = os.path.basename(args.checkpoint)
    res['arch'] = arch
    res['n_per_cell'] = args.n_per_cell
    res['occ_threshold'] = args.occ_threshold

    print_results(res, arch)

    os.makedirs(args.out_dir, exist_ok=True)
    run = os.path.splitext(os.path.basename(args.checkpoint))[0]
    if run.endswith('_best'):
        run = run[:-5]
    out_path = os.path.join(args.out_dir, f"eval_{run}.json")
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2,
                  default=lambda x: None if (isinstance(x, float)
                                             and np.isnan(x)) else x)
    print(f"\nSaved results to {out_path}")


if __name__ == '__main__':
    main()
