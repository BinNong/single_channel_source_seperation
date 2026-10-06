"""Paper 6 — E4 driver: joint sync+detection on K=2 test cells.

Variants (plan §4-E4; all scored identically against ser_comp reference
labels, PIT over slot<->source assignments, per-source M-fold rotation
resolution as in E1/E3; BER via GRAY_BITS from the same decisions):

  V0  E3 chaining baseline: occupancy top-2 slot waveforms ->
      independent blind_sync_known_mod per slot -> joint_detect
      estimate_A_em.  (Expected weak: sync bias kills naive chaining.)
  V1  headline: ECM joint sync+detection on the RAW K=2 mixture
      (training-free; plan R4 promoted to main line).
  V2  slot-aided: same ECM on the two occupancy-top-2 slot symbol
      streams (R [2,N]), initialised at the per-slot blind-sync
      frequency estimates with a +-4 Hz warm search.
  V3  oracle-carrier bound: ECM with TRUE (df1, df2) (no frequency
      search) — isolates sync error from detection error.

References to beat (paper 5): mixture baseline 0.5611, waveform route
0.5475; E2's union bound / exact floor as the theory context.

Test cells: K=2 of the deterministic grid (seed 99999, same cells as
paper 5 — the dataset is built with the full K{1,2,3}+K4 layout and
filtered, so cells are bit-identical to paper 5's runs).

Usage:
    python eval_joint_k2.py --n_per_cell 25 --snr_points 0 10 20   # pilot
    python eval_joint_k2.py --n_per_cell 100                       # full
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Vendored paper6 modules (self-contained; no sys.path tricks).
import config as C                                          # noqa: E402
from data_generator import (CommBSSVarKTestDataset,         # noqa: E402
                            MOD_TYPES)
import ser_comp                                             # noqa: E402
from sync import blind_sync_known_mod                       # noqa: E402
from joint_detect import constellation_np, estimate_A_em    # noqa: E402
from joint_sync_detect import (ecm_joint, mixture_symbols,  # noqa: E402
                               _steer, marginal_decisions, EM_STATS)
from joint_isi import ecm_joint_isi                          # noqa: E402
from eval_blind_pipeline import (build_slot_from_ckpt,      # noqa: E402
                                 _si_sdr_np)

CKPT_PATTERN = 'slot_h64_l4_k13_bs16_lr0.001_{cfg}_s{seed}_best.pt'


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _rot_ser_ber(dec_syms, const, ref_lab, m_order):
    """Min over M-fold rotations; returns (ser, best_dec)."""
    best = None
    for k in range(m_order):
        dv = dec_syms * np.exp(-1j * 2 * np.pi * k / m_order)
        d = np.argmin(np.abs(dv[:, None] - const[None, :]), axis=1)
        s = float(np.mean(d != ref_lab))
        if best is None or s < best[0]:
            best = (s, d)
    return best


def score_decisions(dec1, dec2, const1, const2, ref1, ref2, mods,
                    allow_swap):
    """PIT over assignments (swap only if the modulations match) +
    per-source rotation resolution.  Returns (ser1, ber1, ser2, ber2)."""
    m1o = C.SyncConfig.sym_order[mods[0]]
    m2o = C.SyncConfig.sym_order[mods[1]]

    def one(dec, const, ref, mo, mod_name):
        bits = ser_comp.GRAY_BITS[mod_name]
        s, d = _rot_ser_ber(const[dec], const, ref, mo)
        return s, float(np.mean(bits[d] != bits[ref]))

    direct = one(dec1, const1, ref1, m1o, mods[0]) + \
        one(dec2, const2, ref2, m2o, mods[1])
    if allow_swap:
        sw = one(dec2, const1, ref1, m1o, mods[0]) + \
            one(dec1, const2, ref2, m2o, mods[1])
        if 0.5 * (sw[0] + sw[2]) < 0.5 * (direct[0] + direct[2]):
            return sw
    return direct


def _ref_labels(src_np, mod_idx, carrier):
    return ser_comp.ref_labels_only(
        torch.from_numpy(np.asarray(src_np, dtype=np.complex64)),
        mod_type=MOD_TYPES[mod_idx],
        sample_rate=C.SignalConfig.sample_rate,
        n_symbols=C.SignalConfig.n_symbols,
        roll_off=C.SignalConfig.roll_off,
        num_taps=C.SignalConfig.num_taps,
        carrier_freq=float(carrier))


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
def run_v1(mix_np, const1, const2):
    """ECM on the raw mixture."""
    r = mixture_symbols(mix_np)
    return ecm_joint(r, const1, const2)


def run_v3(mix_np, const1, const2, df_true):
    """Oracle-frequency ECM (no frequency search)."""
    r = mixture_symbols(mix_np)
    res = ecm_joint(r, const1, const2, df_init=df_true,
                    nls_hw=0.0, nls_step=1.0)
    return res


def run_v4(mix_np, const1, const2, L=5):
    """ISI-aware ECM on the raw mixture (warm-started from its own V1
    solution — run inside, so V4 is self-contained per burst)."""
    r = mixture_symbols(mix_np)
    v1 = ecm_joint(r, const1, const2)
    return ecm_joint_isi(r, const1, const2, L=L, v1=v1)


# ---------------------------------------------------------------------------
# E5 variants
# ---------------------------------------------------------------------------
def run_v4b(mix_np, n_ph=4):
    """Blind-modulation V4: ECM under all 10 unordered mod-pair hypotheses,
    selected by NORMALISED soft evidence (the only selector that worked —
    see e5_modsel_probe.py / EXPERIMENT_LOG: raw energy, held-out energy
    and unnormalised evidence always pick the densest hypothesis (16QAM,
    16QAM); per-hypothesis sigma^2 collapses to the sparsest)."""
    from scipy.special import logsumexp
    import itertools
    r = mixture_symbols(mix_np)
    fits = {}
    for pr in itertools.combinations_with_replacement(range(4), 2):
        c1, c2 = constellation_np(pr[0]), constellation_np(pr[1])
        fits[pr] = run_v4(mix_np, c1, c2)     # run_v4 applies the front-end
    sigma2 = min(f['res_energy'] for f in fits.values()) / len(r)
    best = None
    for pr, res in fits.items():
        c1, c2 = constellation_np(pr[0]), constellation_np(pr[1])
        N = len(r)
        U = np.stack([_steer(res['df1'], np.arange(N)),
                      _steer(res['df2'], np.arange(N))])
        pred = (res['A'][0, 0] * (U[0][:, None, None] * c1[None, :, None])
                + res['A'][0, 1] * (U[1][:, None, None] * c2[None, None, :]))
        d2 = np.abs(r[:, None, None] - pred) ** 2
        ev = float(np.sum(logsumexp(-d2 / (2 * sigma2), axis=(1, 2))
                          - np.log(len(c1) * len(c2))))
        if best is None or ev > best[0]:
            best = (ev, pr, res)
    out = dict(best[2])
    out['mod_pair_hat'] = best[1]
    return out


def run_v4_no_nls(mix_np, const1, const2):
    """Ablation: V4 without the fine NLS frequency polish."""
    r = mixture_symbols(mix_np)
    v1 = ecm_joint(r, const1, const2, polish_rounds=0)
    return ecm_joint_isi(r, const1, const2, L=5, v1=v1, polish_rounds=0)


def run_v4_single_init(mix_np, const1, const2):
    """Ablation: single coarse candidate (no top-4 restart set)."""
    r = mixture_symbols(mix_np)
    v1 = ecm_joint(r, const1, const2, n_coarse_keep=1)
    return ecm_joint_isi(r, const1, const2, L=5, v1=v1)


def run_v4_l3(mix_np, const1, const2):
    """Ablation: L=3 centred taps (dose-response between V1=L1 and L=5)."""
    r = mixture_symbols(mix_np)
    v1 = ecm_joint(r, const1, const2)
    return ecm_joint_isi(r, const1, const2, L=3, v1=v1)


def run_v2(slot_waves, df_inits, const1, const2):
    """ECM on the two slot symbol streams, initialised at the per-slot
    blind-sync estimates; warm local frequency search (stride-4, cheap)
    then the standard full-rate stage C/D via ecm_joint."""
    from joint_sync_detect import em_converged
    R = np.stack([mixture_symbols(w) for w in slot_waves], axis=0)
    res0 = em_converged(R, const1, const2, df_inits[0], df_inits[1],
                        stride=4)
    warm = (res0['A'], res0['dec1'], res0['dec2'])
    best = res0
    for g1 in np.arange(df_inits[0] - 4.0, df_inits[0] + 4.01, 0.5):
        for g2 in np.arange(df_inits[1] - 4.0, df_inits[1] + 4.01, 0.5):
            res = em_converged(R, const1, const2, g1, g2, stride=4,
                               warm=warm)
            if res['E'] < best['E']:
                best = res
    return ecm_joint(R, const1, const2, df_init=(best['df1'], best['df2']),
                     nls_hw=0.15, nls_step=0.05, polish_rounds=3)


def run_v0(slot_waves, mods_idx, consts, ref_labels):
    """E3 chaining: independent blind sync per slot + joint_detect EM-A.
    slot_waves: two waveforms; mods_idx: per-SOURCE mods; assignment by
    PIT at the sync level (each slot synced under each source's mod)."""
    # sync each slot under each candidate modulation
    z = {}
    df = {}
    for j, w in enumerate(slot_waves):
        for m in set(mods_idx):
            res = blind_sync_known_mod(np.asarray(w, dtype=np.complex128), m)
            z[(j, m)] = res['z']
            df[(j, m)] = float(res['df'])
    best = None
    for assign in ((0, 1), (1, 0)):
        # slot j assigned source assign[j]
        if len(set(mods_idx)) < 2 and assign == (1, 0):
            pass                                    # same-mod: both valid
        m1, m2 = mods_idx[assign[0]], mods_idx[assign[1]]
        z1, z2 = z[(0, m1)], z[(1, m2)]
        A_hat, i1, i2 = estimate_A_em(z1, z2, consts[m1], consts[m2])
        ser1 = _rot_ser_ber(consts[m1][i1], consts[m1],
                            ref_labels[assign[0]],
                            C.SyncConfig.sym_order[MOD_TYPES[m1]])[0]
        ser2 = _rot_ser_ber(consts[m2][i2], consts[m2],
                            ref_labels[assign[1]],
                            C.SyncConfig.sym_order[MOD_TYPES[m2]])[0]
        # assignment metric: total SER
        if best is None or (ser1 + ser2) < best[0]:
            best = (ser1 + ser2, (z1, z2), (m1, m2), (i1, i2), assign)
    _, (z1, z2), (m1, m2), (i1, i2), assign = best
    return {'dec1': i1, 'dec2': i2, 'assign': assign, 'mods': (m1, m2)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='E4: joint sync+detection at K=2')
    p.add_argument('--n_per_cell', type=int, default=100)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--config', type=str, default='mse',
                   choices=['mse', 'ser_mse'])
    p.add_argument('--snr_points', type=float, nargs='+',
                   default=list(C.SignalConfig.snr_test_points))
    p.add_argument('--variants', type=str, nargs='+',
                   default=['V0', 'V1', 'V2', 'V3'],
                   choices=['V0', 'V1', 'V2', 'V3', 'V4', 'V1V4',
                            'V4B', 'V4NOPOLISH', 'V4SINGLE', 'V4L3'])
    p.add_argument('--ckpt_dir', type=str,
                   default=C.CHECKPOINT_DIR)
    p.add_argument('--test_seed', type=int, default=C.DataConfig.test_seed,
                   help='test-grid seed (99999 = the paper grid; other '
                        'values give independent grids for multi-seed CIs)')
    p.add_argument('--out', type=str, default=None)
    args = p.parse_args()
    device = C.DEVICE
    out_path = args.out or os.path.join(
        C.RESULTS_DIR, f'e4_joint_k2_{args.config}.json'
        if args.test_seed == C.DataConfig.test_seed else
        f'e4_joint_k2_{args.config}_ts{args.test_seed}.json')

    print(f"Building test grid (seed {args.test_seed}) ...", flush=True)
    ds = CommBSSVarKTestDataset(
        n_per_cell=args.n_per_cell,
        snr_points=list(C.SignalConfig.snr_test_points),
        mod_types=MOD_TYPES, k_lo=1, k_hi=3, k_extrap=4, k_slots=4,
        signal_length=C.SignalConfig.signal_length,
        sample_rate=C.SignalConfig.sample_rate,
        carrier_base=C.SignalConfig.carrier_base,
        freq_gap_range=C.SignalConfig.freq_gap_range,
        seed=args.test_seed, return_carriers=True)
    # K=2 cells on the requested SNR subset
    cells = []
    for i in range(len(ds)):
        s = ds.samples[i]
        if s['k'] == 2 and float(s['snr']) in set(args.snr_points):
            cells.append(i)
    print(f"  {len(cells)} K=2 cells", flush=True)

    # precompute per-cell shared data
    cell_data = []
    for i in cells:
        s = ds.samples[i]
        mix_np = s['mixture'].numpy()[0]
        srcs = [s['sources'].numpy()[j, 0] for j in range(2)]
        mods_idx = [int(m) for m in s['mods'].numpy()[:2]]
        cars = [float(c) for c in s['carriers'][:2]]
        refs = [_ref_labels(srcs[j], mods_idx[j], cars[j]) for j in range(2)]
        cell_data.append({'snr': float(s['snr']), 'mix': mix_np,
                          'srcs': srcs, 'mods': mods_idx, 'cars': cars,
                          'refs': refs, 'mods_idx': mods_idx})

    need_model = ('V0' in args.variants) or ('V2' in args.variants)
    records = []

    # ---- combined V1+V4 (shares V1's frequency search; ~45% cheaper) ----
    if 'V1V4' in args.variants:
        t0 = time.time()
        for ci, cd in enumerate(cell_data):
            c1 = constellation_np(cd['mods_idx'][0])
            c2 = constellation_np(cd['mods_idx'][1])
            same = cd['mods_idx'][0] == cd['mods_idx'][1]
            mods_names = [MOD_TYPES[m] for m in cd['mods_idx']]
            r_m = mixture_symbols(cd['mix'])
            em0 = dict(EM_STATS)
            v1 = ecm_joint(r_m, c1, c2)
            v4 = ecm_joint_isi(r_m, c1, c2, L=5, v1=v1)
            emd = {k: EM_STATS[k] - em0[k] for k in EM_STATS}
            for tag, res in (('V1', v1), ('V4', v4)):
                ser1, ber1, ser2, ber2 = score_decisions(
                    res['dec1'], res['dec2'], c1, c2,
                    cd['refs'][0], cd['refs'][1], mods_names, allow_swap=same)
                extra = {'em_calls': emd['em_calls'],
                         'em_rounds': emd['em_rounds'],
                         'nls_evals': emd['nls_evals']}
                if tag == 'V1':
                    s2_hat = v1['res_energy'] / len(r_m)
                    mm1, mm2 = marginal_decisions(
                        r_m, v1['A'], v1['df1'], v1['df2'], c1, c2,
                        max(s2_hat, 1e-9))
                    ms1, mb1, ms2, mb2 = score_decisions(
                        mm1, mm2, c1, c2, cd['refs'][0], cd['refs'][1],
                        mods_names, allow_swap=same)
                    extra['ser_mmap'] = 0.5 * (ms1 + ms2)
                    extra['ber_mmap'] = 0.5 * (mb1 + mb2)
                records.append({'variant': tag, 'seed': -1, 'snr': cd['snr'],
                                'mods': cd['mods_idx'], 'ser1': ser1,
                                'ser2': ser2, 'ber1': ber1, 'ber2': ber2,
                                'ser': 0.5 * (ser1 + ser2),
                                'ber': 0.5 * (ber1 + ber2),
                                'rounds': res['rounds'],
                                'res_energy': res['res_energy'],
                                'df1_err': abs(res['df1'] - (cd['cars'][0]
                                               - C.SyncConfig.nominal_carrier)),
                                'df2_err': abs(res['df2'] - (cd['cars'][1]
                                               - C.SyncConfig.nominal_carrier)),
                                **extra})
            if (ci + 1) % 25 == 0:
                print(f"  V1V4: {ci + 1}/{len(cell_data)} "
                      f"({(time.time() - t0) / (ci + 1):.2f}s/cell)",
                      flush=True)
        print(f"V1V4 done in {time.time() - t0:.0f}s", flush=True)
        _save(out_path, records, args)

    # ---- V1 / V3 / V4 / E5 variants (no model; once) ----
    for var in ('V1', 'V3', 'V4', 'V4B', 'V4NOPOLISH', 'V4SINGLE', 'V4L3'):
        if var not in args.variants:
            continue
        t0 = time.time()
        for ci, cd in enumerate(cell_data):
            c1 = constellation_np(cd['mods_idx'][0])
            c2 = constellation_np(cd['mods_idx'][1])
            em0 = dict(EM_STATS)
            if var == 'V1':
                res = run_v1(cd['mix'], c1, c2)
            elif var == 'V3':
                res = run_v3(cd['mix'], c1, c2,
                             (cd['cars'][0] - C.SyncConfig.nominal_carrier,
                              cd['cars'][1] - C.SyncConfig.nominal_carrier))
            elif var == 'V4':
                res = run_v4(cd['mix'], c1, c2)
            elif var == 'V4NOPOLISH':
                res = run_v4_no_nls(cd['mix'], c1, c2)
            elif var == 'V4SINGLE':
                res = run_v4_single_init(cd['mix'], c1, c2)
            elif var == 'V4L3':
                res = run_v4_l3(cd['mix'], c1, c2)
            else:  # V4B: blind modulation-pair selection
                res = run_v4b(cd['mix'])
                c1 = constellation_np(res['mod_pair_hat'][0])
                c2 = constellation_np(res['mod_pair_hat'][1])
            same = cd['mods_idx'][0] == cd['mods_idx'][1]
            if var == 'V4B':
                # joint classification+detection convention: a wrong
                # modulation-pair decision loses the burst (SER = 1),
                # since decision indices live in different constellations.
                true_pair = tuple(sorted(cd['mods_idx']))
                rec_extra = {'mod_pair_hat': list(res['mod_pair_hat']),
                             'mod_pair_true': list(true_pair)}
                if res['mod_pair_hat'] != true_pair:
                    ser1 = ber1 = ser2 = ber2 = 1.0
                else:
                    # score in the SORTED-pair frame: decisions under
                    # hat[0]/hat[1] vs the correspondingly-ordered refs
                    order = np.argsort(cd['mods_idx'])
                    refs_s = [cd['refs'][int(j)] for j in order]
                    mods_s = [MOD_TYPES[int(cd['mods_idx'][j])]
                              for j in order]
                    ser1, ber1, ser2, ber2 = score_decisions(
                        res['dec1'], res['dec2'], c1, c2,
                        refs_s[0], refs_s[1], mods_s, allow_swap=same)
            else:
                ser1, ber1, ser2, ber2 = score_decisions(
                    res['dec1'], res['dec2'], c1, c2, cd['refs'][0], cd['refs'][1],
                    [MOD_TYPES[m] for m in cd['mods_idx']], allow_swap=same)
                rec_extra = {}
                if var in ('V1', 'V3'):
                    # marginal-MAP baseline: same ECM fit, per-source
                    # marginal decisions (sigma2 from the residual energy)
                    r_m = mixture_symbols(cd['mix'])
                    s2_hat = res['res_energy'] / len(r_m)
                    mm1, mm2 = marginal_decisions(
                        r_m, res['A'], res['df1'], res['df2'], c1, c2,
                        max(s2_hat, 1e-9))
                    ms1, mb1, ms2, mb2 = score_decisions(
                        mm1, mm2, c1, c2, cd['refs'][0], cd['refs'][1],
                        [MOD_TYPES[m] for m in cd['mods_idx']],
                        allow_swap=same)
                    rec_extra = {
                        'ser_mmap': 0.5 * (ms1 + ms2),
                        'ber_mmap': 0.5 * (mb1 + mb2),
                        'em_calls': EM_STATS['em_calls'] - em0['em_calls'],
                        'em_rounds': EM_STATS['em_rounds'] - em0['em_rounds'],
                        'nls_evals': EM_STATS['nls_evals'] - em0['nls_evals'],
                    }
            records.append({'variant': var, 'seed': -1, 'snr': cd['snr'],
                            'mods': cd['mods_idx'], 'ser1': ser1,
                            'ser2': ser2, 'ber1': ber1, 'ber2': ber2,
                            'ser': 0.5 * (ser1 + ser2),
                            'ber': 0.5 * (ber1 + ber2),
                            'rounds': res['rounds'],
                            'res_energy': res['res_energy'],
                            'df1_err': abs(res['df1'] - (cd['cars'][0]
                                           - C.SyncConfig.nominal_carrier)),
                            'df2_err': abs(res['df2'] - (cd['cars'][1]
                                           - C.SyncConfig.nominal_carrier)),
                            **rec_extra,
                            })
            if (ci + 1) % 25 == 0:
                print(f"  {var}: {ci + 1}/{len(cell_data)} "
                      f"({(time.time() - t0) / (ci + 1):.2f}s/cell)",
                      flush=True)
        print(f"{var} done in {time.time() - t0:.0f}s", flush=True)

    # ---- V0 / V2 (per seed) ----
    if need_model:
        for seed in args.seeds:
            ckpt = os.path.join(
                args.ckpt_dir,
                CKPT_PATTERN.format(cfg=args.config, seed=seed))
            model, _a = build_slot_from_ckpt(ckpt, device)
            t0 = time.time()
            for ci, cd in enumerate(cell_data):
                mix_t = torch.from_numpy(cd['mix']).view(1, 1, -1) \
                    .to(torch.complex64).to(device)
                with torch.no_grad():
                    slots, occ_logits, _ = model(mix_t)
                slots_np = slots[0].cpu().numpy()
                occ_np = torch.sigmoid(occ_logits[0]).cpu().numpy()
                top2 = np.argsort(-occ_np)[:2]      # oracle count (E3
                                                    # measures counting)
                waves = [slots_np[j] for j in top2]
                c1 = constellation_np(cd['mods_idx'][0])
                c2 = constellation_np(cd['mods_idx'][1])
                same = cd['mods_idx'][0] == cd['mods_idx'][1]
                for var in ('V0', 'V2'):
                    if var not in args.variants:
                        continue
                    if var == 'V0':
                        res = run_v0(waves, cd['mods_idx'],
                                     {m: constellation_np(m)
                                      for m in set(cd['mods_idx'])},
                                     cd['refs'])
                        # V0 stores dec on possibly-swapped assignment
                        a1, a2 = res['assign']
                        d1 = res['dec1']  # on mods[a1]... careful below
                        ser1, ber1, ser2, ber2 = score_decisions(
                            res['dec1'], res['dec2'],
                            constellation_np(cd['mods_idx'][a1]),
                            constellation_np(cd['mods_idx'][a2]),
                            cd['refs'][a1], cd['refs'][a2],
                            [MOD_TYPES[cd['mods_idx'][a1]],
                             MOD_TYPES[cd['mods_idx'][a2]]],
                            allow_swap=False)
                        res_energy = None
                        rounds = None
                        df_info = {}
                    else:
                        # per-slot blind sync for the init (true mod per
                        # slot via the source it matches — the E3 pairing
                        # is available here through the occupancy order;
                        # use SI-SDR matching for the init assignment)
                        cost = np.full((2, 2), 1e6)
                        for i, s in enumerate(cd['srcs']):
                            for j, w in enumerate(waves):
                                cost[i, j] = -_si_sdr_np(w, s)
                        from scipy.optimize import linear_sum_assignment
                        rows, cols = linear_sum_assignment(cost)
                        # slot j syncs under its matched source's mod
                        df_inits = [0.0, 0.0]
                        for i, j in zip(rows, cols):
                            res_b = blind_sync_known_mod(
                                np.asarray(waves[j], dtype=np.complex128),
                                cd['mods_idx'][i])
                            df_inits[j] = float(res_b['df'])
                        res = run_v2(waves, df_inits, c1, c2)
                        ser1, ber1, ser2, ber2 = score_decisions(
                            res['dec1'], res['dec2'], c1, c2,
                            cd['refs'][0], cd['refs'][1],
                            [MOD_TYPES[m] for m in cd['mods_idx']],
                            allow_swap=same)
                        res_energy = res['res_energy']
                        rounds = res['rounds']
                    records.append({'variant': var, 'seed': seed,
                                    'snr': cd['snr'], 'mods': cd['mods_idx'],
                                    'ser1': ser1, 'ser2': ser2,
                                    'ber1': ber1, 'ber2': ber2,
                                    'ser': 0.5 * (ser1 + ser2),
                                    'ber': 0.5 * (ber1 + ber2),
                                    'rounds': rounds,
                                    'res_energy': res_energy})
                if (ci + 1) % 25 == 0:
                    print(f"  V0/V2 s{seed}: {ci + 1}/{len(cell_data)} "
                          f"({(time.time() - t0) / (ci + 1):.2f}s/cell)",
                          flush=True)
            print(f"V0/V2 seed {seed} done in {time.time() - t0:.0f}s",
                  flush=True)
            # incremental save
            _save(out_path, records, args)

    _save(out_path, records, args)
    _report(records)


def _save(out_path, records, args):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'config': args.config, 'n_per_cell': args.n_per_cell,
                   'snr_points': args.snr_points, 'variants': args.variants,
                   'test_seed': args.test_seed,
                   'records': records}, f)
    print(f"saved {out_path} ({len(records)} records)", flush=True)


def _agg(records, key='ser'):
    return float(np.mean([r[key] for r in records])) if records else float('nan')


def _report(records):
    print("\n=== E4 summary (SER; per-source average) ===")
    variants = sorted({r['variant'] for r in records})
    snrs = sorted({r['snr'] for r in records})
    seeds = sorted({r['seed'] for r in records})
    print(f"{'variant':>8s} | " + ' | '.join(f"{s:>6g}" for s in snrs)
          + " | pooled")
    for var in variants:
        vseeds = sorted({r['seed'] for r in records if r['variant'] == var})
        row = []
        for s in snrs:
            per_seed = []
            for sd in vseeds:
                rs = [r for r in records if r['variant'] == var
                      and r['snr'] == s and r['seed'] == sd]
                if rs:
                    per_seed.append(_agg(rs))
            row.append(np.mean(per_seed) if per_seed else float('nan'))
        pooled = np.mean([_agg([r for r in records if r['variant'] == var
                                and r['seed'] == sd])
                          for sd in vseeds])
        print(f"{var:>8s} | " + ' | '.join(f"{v:6.4f}" for v in row)
              + f" | {pooled:.4f}")
    # per modulation-pair class
    print("\nper modulation-pair class (pooled SER):")
    for var in variants:
        psk = [r for r in records if r['variant'] == var
               and all(MOD_TYPES[m] != '16QAM' for m in r['mods'])]
        qam = [r for r in records if r['variant'] == var
               and any(MOD_TYPES[m] == '16QAM' for m in r['mods'])]
        print(f"  {var}: PSK-only={_agg(psk):.4f} (n={len(psk)})  "
              f"16QAM-involving={_agg(qam):.4f} (n={len(qam)})")
    # V4B: modulation-pair classification accuracy
    v4b = [r for r in records if r['variant'] == 'V4B']
    if v4b:
        print("\nV4B modulation-pair classification accuracy:")
        for s in sorted({r['snr'] for r in v4b}):
            rs = [r for r in v4b if r['snr'] == s]
            acc = np.mean([r['mod_pair_hat'] == r['mod_pair_true']
                           for r in rs])
            print(f"  SNR {s:>5g}: {acc:.3f} (n={len(rs)})")
    # marginal-MAP comparison (V1/V3 records carry ser_mmap)
    print("\nmarginal-MAP vs joint-ML (pooled SER):")
    for var in ('V1', 'V3'):
        rs = [r for r in records if r['variant'] == var
              and 'ser_mmap' in r]
        if rs:
            print(f"  {var}: joint-ML={_agg(rs):.4f}  "
                  f"marginal-MAP={_agg(rs, 'ser_mmap'):.4f}")
        emc = [r['em_calls'] for r in rs if 'em_calls' in r]
        emr = [r['em_rounds'] for r in rs if 'em_rounds' in r]
        if emc:
            print(f"      EM stats/burst: calls={np.mean(emc):.0f} "
                  f"rounds={np.mean(emr):.0f} "
                  f"(converged-EM invocations and total decide/gain rounds)")
    print("\nreferences: mixture baseline 0.5611, waveform route 0.5475")


if __name__ == '__main__':
    main()
