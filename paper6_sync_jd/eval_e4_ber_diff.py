"""Paper 6 — Review-3 additions for E4: (a) Gray-mapped BER for every E4
route and (b) genie-free differential-decoding scores.

(a) BER FROM CACHE (no re-compute)
----------------------------------
Every E4 scoring path already records a Gray-mapped BER computed from the
SAME hard decisions as the SER (eval_joint_k2.score_decisions and
eval_psp_baseline both call ser_comp.GRAY_BITS — binary-reflected Gray for
BPSK/QPSK/8PSK/16QAM).  This part simply re-aggregates the cached JSON
records:

  V1/V3/V4 : results/e4_joint_k2_mse.json           (test grid 99999)
             results/e4_joint_k2_mse_ts{31337,55555,77777,88888}.json
             (4 extra grids; multi-grid CIs per EXPERIMENT_LOG 2026-09-21)
  V0/V2    : results/e4_joint_k2_slots.json          (grid 99999, 5 ckpt seeds)
  sym/psp  : results/psp_baseline.json               (grid 99999, 5 ckpt seeds;
             psp = separate -> oracle-sync -> PSP/Viterbi, the strongest
             oracle-assisted waveform route, pooled SER 0.5337)

Pooling convention matches the E4 log entries: pooled within a group
(test grid for V1/V3/V4, checkpoint seed for V0/V2/sym/psp), then averaged
over groups; per-modulation-pair-class split (PSK-only vs 16QAM-involving)
as requested by the reviewer, plus a full unordered-pair table.

(b) GENIE-FREE DIFFERENTIAL DECODING
------------------------------------
The published E4 SER/BER resolve the per-source M-fold rotation ambiguity
with a genie (min over the M rotations against the reference labels — the
standard blind-receiver convention).  The deployment answer is differential
coding (E1-D, eval_diff_coding.py, measured the K=1 cost: +0.03..+0.08
high-SNR penalty, oracle pays it too).  This part scores E4 bursts
GENIE-FREE, following the E1-D convention adapted to hard decision streams:

  The data source is NOT differentially encoded, so scoring is
  "differential-equivalent": the receiver's hard symbol stream c[dec[n]] is
  mapped to the differential stream d_n = c[dec[n]] * conj(c[dec[n-1]]),
  n = 1..N-1, and compared against the reference differential stream
  r_n = c[ref[n]] * conj(c[ref[n-1]]).  Products of grid points are again
  grid points (the M-point product grid, possibly rotated), so decisions
  are labeled canonically by the nearest M-th root of unity — a per-burst
  constant rotation e^{j k 2pi/M} of the hard decisions cancels EXACTLY in
  the product, hence no genie enters.  Differential BER maps the
  differential label to bits with the SAME ser_comp.GRAY_BITS table (the
  product grid of an M-PSK has M points, so the modulation's Gray map
  applies to the differential index — the standard differential-equivalent
  bit mapping).  16QAM pairs are EXCLUDED from differential scoring (no
  standard differential coding for square QAM — E1-D convention); they
  still get ABS scores.

PIT over the two source assignments is kept (min average differential SER;
permutation resolution is not the phase genie at issue).  ABS scores are
recomputed from the same decisions on the same bursts, so ABS-vs-DIFF is
paired.  The per-burst decisions are NOT cached by the E4 scripts, so this
part re-runs the detectors; --n_subset takes the FIRST n bursts per SNR
block of the SAME deterministic grid (n_per_cell=100, seed 99999), making
the subset a strict subset of the published runs (cross-checked against the
cached ABS records at runtime).

Usage:
    python eval_e4_ber_diff.py                      # smoke (self-test + tiny run)
    python eval_e4_ber_diff.py --mode ber           # cache aggregation only
    python eval_e4_ber_diff.py --mode diff --n_subset 10
    python eval_e4_ber_diff.py --mode all --n_subset 100   # FULL (server)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# Vendored paper6 modules (self-contained; no sys.path tricks).
import config as C                                          # noqa: E402
from data_generator import (CommBSSVarKTestDataset,         # noqa: E402
                            MOD_TYPES)
import ser_comp                                             # noqa: E402
from joint_detect import constellation_np                   # noqa: E402
from joint_sync_detect import ecm_joint, mixture_symbols    # noqa: E402
from joint_isi import ecm_joint_isi                         # noqa: E402
from eval_joint_k2 import score_decisions, _ref_labels      # noqa: E402
from eval_psp_baseline import (oracle_frontend, psp_decisions,  # noqa: E402
                               CKPT_PATTERN)
from eval_diff_coding import diff_grid, diff_ser            # noqa: E402
from eval_blind_pipeline import (build_slot_from_ckpt,      # noqa: E402
                                 _si_sdr_np)

GRID_FILES = ['e4_joint_k2_mse.json',
              'e4_joint_k2_mse_ts31337.json',
              'e4_joint_k2_mse_ts55555.json',
              'e4_joint_k2_mse_ts77777.json',
              'e4_joint_k2_mse_ts88888.json']
SLOTS_FILE = 'e4_joint_k2_slots.json'
PSP_FILE = 'psp_baseline.json'
PSK_MODS = ('BPSK', 'QPSK', '8PSK')     # 16QAM excluded from DIFF (E1-D)


# ---------------------------------------------------------------------------
# Part (a): cache aggregation
# ---------------------------------------------------------------------------
def _pair_class(mods):
    return ('PSK-only' if all(MOD_TYPES[m] != '16QAM' for m in mods)
            else '16QAM-involving')


def _pair_key(mods):
    return '-'.join(sorted(MOD_TYPES[m] for m in mods))


def _mean(rs, key):
    return float(np.mean([r[key] for r in rs])) if rs else None


def _agg_block(records):
    """Aggregate one variant's records: per-SNR, pooled, pair classes,
    per unordered pair.  `records` must already be group-averaged if a
    multi-grid/multi-seed convention is wanted (see aggregate_cached)."""
    out = {'n_records': len(records)}
    snrs = sorted({r['snr'] for r in records})
    out['per_snr'] = {f'{s:g}': {'ser': _mean([r for r in records
                                               if r['snr'] == s], 'ser'),
                                 'ber': _mean([r for r in records
                                               if r['snr'] == s], 'ber'),
                                 'n': len([r for r in records
                                           if r['snr'] == s])}
                      for s in snrs}
    out['pooled'] = {'ser': _mean(records, 'ser'),
                     'ber': _mean(records, 'ber')}
    for cls in ('PSK-only', '16QAM-involving'):
        rs = [r for r in records if _pair_class(r['mods']) == cls]
        out[cls] = {'ser': _mean(rs, 'ser'), 'ber': _mean(rs, 'ber'),
                    'n': len(rs)}
    pairs = sorted({_pair_key(r['mods']) for r in records})
    out['per_pair'] = {p: {'ser': _mean([r for r in records
                                         if _pair_key(r['mods']) == p],
                                        'ser'),
                           'ber': _mean([r for r in records
                                         if _pair_key(r['mods']) == p],
                                        'ber'),
                           'n': len([r for r in records
                                     if _pair_key(r['mods']) == p])}
                       for p in pairs}
    return out


def _group_average(records, group_of):
    """Replace records by per-group averages (one pseudo-record per
    (group, snr, mods) is wrong for mods — instead average pooled and
    per-snr within each group, then over groups)."""
    groups = {}
    for r in records:
        groups.setdefault(group_of(r), []).append(r)
    return groups


def aggregate_cached(results_dir):
    """Gray BER (+SER for context) of every E4 route from cached JSONs.

    V1/V3/V4: mean over the 5 test grids of per-grid aggregates.
    V0/V2/sym/psp: mean over the 5 checkpoint seeds on grid 99999.
    """
    out = {}
    # --- V1/V3/V4: one file per test grid ---
    per_variant_groups = {}
    for f in GRID_FILES:
        path = os.path.join(results_dir, f)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        tag = f"grid{d.get('test_seed', 99999)}"
        for r in d['records']:
            if r['variant'] in ('V1', 'V3', 'V4'):
                per_variant_groups.setdefault(r['variant'], {}) \
                    .setdefault(tag, []).append(r)
    for var, groups in per_variant_groups.items():
        blocks = [_agg_block(rs) for rs in groups.values()]
        out[var] = _merge_group_blocks(blocks)
        out[var]['groups'] = {g: {'pooled_ser': b['pooled']['ser'],
                                  'pooled_ber': b['pooled']['ber']}
                              for g, b in zip(groups, blocks)}
    # --- V0/V2 (slots) and sym/psp (waveform route): per checkpoint seed ---
    for f, keep in ((SLOTS_FILE, ('V0', 'V2')), (PSP_FILE, ('sym', 'psp'))):
        path = os.path.join(results_dir, f)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        per_variant_groups = {}
        for r in d['records']:
            if r['variant'] in keep:
                per_variant_groups.setdefault(r['variant'], {}) \
                    .setdefault(f"seed{r['seed']}", []).append(r)
        for var, groups in per_variant_groups.items():
            blocks = [_agg_block(rs) for rs in groups.values()]
            out[var] = _merge_group_blocks(blocks)
            out[var]['groups'] = {g: {'pooled_ser': b['pooled']['ser'],
                                      'pooled_ber': b['pooled']['ber']}
                                  for g, b in zip(groups, blocks)}
    return out


def _merge_group_blocks(blocks):
    """Average _agg_block outputs over groups (grids or seeds)."""
    def m(key_path):
        vals = []
        for b in blocks:
            v = b
            for k in key_path:
                v = v.get(k) if isinstance(v, dict) else None
            if isinstance(v, (int, float)) and v is not None:
                vals.append(v)
        return (float(np.mean(vals)), float(np.std(vals))) if vals \
            else (None, None)

    merged = {'n_groups': len(blocks)}
    snrs = sorted({s for b in blocks for s in b['per_snr']})
    merged['per_snr'] = {
        s: {'ser': m(['per_snr', s, 'ser'])[0],
            'ber': m(['per_snr', s, 'ber'])[0],
            'ser_std': m(['per_snr', s, 'ser'])[1],
            'ber_std': m(['per_snr', s, 'ber'])[1],
            'n': sum(b['per_snr'][s]['n'] for b in blocks
                     if s in b['per_snr'])}
        for s in snrs}
    merged['pooled'] = {'ser': m(['pooled', 'ser'])[0],
                        'ber': m(['pooled', 'ber'])[0],
                        'ser_std': m(['pooled', 'ser'])[1],
                        'ber_std': m(['pooled', 'ber'])[1]}
    for cls in ('PSK-only', '16QAM-involving'):
        merged[cls] = {'ser': m([cls, 'ser'])[0], 'ber': m([cls, 'ber'])[0],
                       'n': sum(b[cls]['n'] for b in blocks)}
    pairs = sorted({p for b in blocks for p in b['per_pair']})
    merged['per_pair'] = {p: {'ser': m(['per_pair', p, 'ser'])[0],
                              'ber': m(['per_pair', p, 'ber'])[0],
                              'n': sum(b['per_pair'][p]['n'] for b in blocks)}
                          for p in pairs}
    return merged


# ---------------------------------------------------------------------------
# Part (b): genie-free differential scoring
# ---------------------------------------------------------------------------
def diff_labels(idx_stream, const):
    """Canonical differential labels of a hard-decision index stream.

    d_n = const[idx[n]] * conj(const[idx[n-1]]), labeled by the nearest
    M-th root of unity.  Invariant to any per-burst M-fold rotation of the
    decision stream (the rotation cancels in the product) -> genie-free.
    """
    M = len(const)
    uroots = np.exp(2j * np.pi * np.arange(M) / M)
    prods = const[idx_stream[1:]] * np.conj(const[idx_stream[:-1]])
    return np.argmin(np.abs(prods[:, None] - uroots[None, :]), axis=1)


def diff_ser_ber(dec_idx, ref_lab, const, mod_name):
    """(SER, BER) of differentially decoded hard decisions vs the reference
    differential labels; BER via the modulation's Gray map on the
    differential index (ser_comp.GRAY_BITS — same table as the ABS BER)."""
    dl = diff_labels(dec_idx, const)
    rl = diff_labels(ref_lab, const)
    bits = ser_comp.GRAY_BITS[mod_name]
    return float(np.mean(dl != rl)), float(np.mean(bits[dl] != bits[rl]))


def score_diff_pair(dec1, dec2, c1, c2, ref1, ref2, mods_names, allow_swap):
    """PIT over assignments (min average differential SER); returns
    (ser1, ber1, ser2, ber2) mirroring eval_joint_k2.score_decisions."""
    direct = (diff_ser_ber(dec1, ref1, c1, mods_names[0])
              + diff_ser_ber(dec2, ref2, c2, mods_names[1]))
    if allow_swap:
        sw = (diff_ser_ber(dec2, ref1, c1, mods_names[0])
              + diff_ser_ber(dec1, ref2, c2, mods_names[1]))
        if 0.5 * (sw[0] + sw[2]) < 0.5 * (direct[0] + direct[2]):
            return sw
    return direct


def selftest():
    """Differential scorer self-tests (no data, no model)."""
    print("DIFF scorer self-test ...", flush=True)
    rng = np.random.RandomState(0)
    for mod in PSK_MODS:
        mi = MOD_TYPES.index(mod)
        const = constellation_np(mi)
        M = len(const)
        ref = rng.randint(0, M, size=300)
        # (1) rotation invariance: decisions rotated by k*2pi/M (a physical
        #     rotation of the decided symbols, re-demodulated onto the
        #     grid) must score EXACTLY 0 for every k — this is the genie
        #     the ABS scoring needs and the DIFF scoring must not.
        for k in range(M):
            rot = const[ref] * np.exp(2j * np.pi * k / M)
            dec = np.argmin(np.abs(rot[:, None] - const[None, :]), axis=1)
            assert not np.array_equal(dec, ref) or k == 0 or M == 2
            s, b = diff_ser_ber(dec, ref, const, mod)
            assert s == 0.0 and b == 0.0, \
                f"{mod}: rotation k={k} not absorbed (ser={s}, ber={b})"
        # (2) consistency with E1-D's continuous-stream scorer: feeding
        #     diff_ser the constellation points of the decisions must give
        #     the same SER as the hard-decision path here.
        dec = (ref + rng.randint(0, M, size=300)) % M   # random symbol errs
        s_cont = diff_ser(const[dec], const, ref, diff_grid(const))
        s_hard, _ = diff_ser_ber(dec, ref, const, mod)
        assert abs(s_cont - s_hard) < 1e-12, \
            f"{mod}: hard vs continuous diff scorers disagree"
        # (3) chance level: independent random decisions ~ 1 - 1/M.
        dec_r = rng.randint(0, M, size=20000)
        s_r, _ = diff_ser_ber(dec_r, ref[:1].repeat(20000), const, mod)
        assert abs(s_r - (1 - 1 / M)) < 0.02, \
            f"{mod}: chance level off ({s_r})"
    print("  self-test passed (rotation invariance exact for k=0..M-1, "
          "E1-D scorer parity, chance level)", flush=True)


# ---------------------------------------------------------------------------
# Part (b): detection re-run on a (sub)set of the E4 grid
# ---------------------------------------------------------------------------
def select_cells(args):
    """K=2 cells of the deterministic n_per_cell=100 grid, FIRST
    --n_subset bursts per SNR block (strict subset of the published runs)."""
    print(f"Building test grid (seed {args.test_seed}, n_per_cell=100) ...",
          flush=True)
    ds = CommBSSVarKTestDataset(
        n_per_cell=100,
        snr_points=list(C.SignalConfig.snr_test_points),
        mod_types=MOD_TYPES, k_lo=1, k_hi=3, k_extrap=4, k_slots=4,
        signal_length=C.SignalConfig.signal_length,
        sample_rate=C.SignalConfig.sample_rate,
        carrier_base=C.SignalConfig.carrier_base,
        freq_gap_range=C.SignalConfig.freq_gap_range,
        seed=args.test_seed, return_carriers=True)
    per_snr = {}
    for i in range(len(ds)):
        s = ds.samples[i]
        if s['k'] != 2 or float(s['snr']) not in set(args.snr_points):
            continue
        per_snr.setdefault(float(s['snr']), []).append(i)
    cells = []
    for snr in sorted(per_snr):
        cells.extend(per_snr[snr][:args.n_subset])
    print(f"  {len(cells)} K=2 bursts "
          f"({args.n_subset}/SNR x {len(per_snr)} SNR points)", flush=True)

    cell_data = []
    for i in cells:
        s = ds.samples[i]
        srcs = [s['sources'].numpy()[j, 0] for j in range(2)]
        mods_idx = [int(m) for m in s['mods'].numpy()[:2]]
        cars = [float(c) for c in s['carriers'][:2]]
        refs = [_ref_labels(srcs[j], mods_idx[j], cars[j]) for j in range(2)]
        cell_data.append({'snr': float(s['snr']),
                          'mix': s['mixture'].numpy()[0],
                          'srcs': srcs, 'mods_idx': mods_idx,
                          'cars': cars, 'refs': refs})
    return cell_data


def _score_one(variant, seed, cd, dec1, dec2, c1, c2):
    """ABS (genie rotation, the published convention) + DIFF (genie-free)
    from the same decisions.  DIFF is None for 16QAM-involving pairs."""
    mods_names = [MOD_TYPES[m] for m in cd['mods_idx']]
    same = cd['mods_idx'][0] == cd['mods_idx'][1]
    ser1, ber1, ser2, ber2 = score_decisions(
        dec1, dec2, c1, c2, cd['refs'][0], cd['refs'][1], mods_names,
        allow_swap=same)
    rec = {'variant': variant, 'seed': seed, 'snr': cd['snr'],
           'mods': cd['mods_idx'],
           'abs_ser': 0.5 * (ser1 + ser2), 'abs_ber': 0.5 * (ber1 + ber2),
           'diff_ser': None, 'diff_ber': None}
    if all(m in PSK_MODS for m in mods_names):
        ds1, db1, ds2, db2 = score_diff_pair(
            dec1, dec2, c1, c2, cd['refs'][0], cd['refs'][1], mods_names,
            allow_swap=same)
        rec['diff_ser'] = 0.5 * (ds1 + ds2)
        rec['diff_ber'] = 0.5 * (db1 + db2)
    return rec


def run_diff(args):
    records = []
    cell_data = select_cells(args)
    psk_cells = [cd for cd in cell_data
                 if all(MOD_TYPES[m] in PSK_MODS for m in cd['mods_idx'])]
    print(f"  PSK-only pairs: {len(psk_cells)}/{len(cell_data)} bursts "
          f"(16QAM-involving pairs are ABS-only, E1-D convention)",
          flush=True)

    # ---- ECM routes (V1, V4; V1 is V4's warm start, scored for free) ----
    t0 = time.time()
    for ci, cd in enumerate(cell_data):
        c1 = constellation_np(cd['mods_idx'][0])
        c2 = constellation_np(cd['mods_idx'][1])
        r_m = mixture_symbols(cd['mix'])
        v1 = ecm_joint(r_m, c1, c2)
        v4 = ecm_joint_isi(r_m, c1, c2, L=5, v1=v1)
        for tag, res in (('V1', v1), ('V4', v4)):
            records.append(_score_one(tag, -1, cd,
                                      res['dec1'], res['dec2'], c1, c2))
        if (ci + 1) % 10 == 0 or ci + 1 == len(cell_data):
            print(f"  V1/V4: {ci + 1}/{len(cell_data)} "
                  f"({(time.time() - t0) / (ci + 1):.2f}s/burst)",
                  flush=True)
    print(f"V1/V4 done in {time.time() - t0:.0f}s", flush=True)

    # ---- waveform route (separate -> oracle sync -> sym/psp) ----
    device = C.DEVICE
    for seed in args.seeds:
        ckpt = os.path.join(args.ckpt_dir, CKPT_PATTERN.format(seed=seed))
        model, _a = build_slot_from_ckpt(ckpt, device)
        t0 = time.time()
        B = 16
        for c0 in range(0, len(cell_data), B):
            chunk = cell_data[c0:c0 + B]
            mix_b = torch.stack([
                torch.from_numpy(cd['mix']).view(1, -1)
                for cd in chunk]).to(torch.complex64).to(device)
            with torch.no_grad():
                slots, occ_logits, _ = model(mix_b)
            slots_np = slots.cpu().numpy()
            occ_np = torch.sigmoid(occ_logits).cpu().numpy()
            for bi, cd in enumerate(chunk):
                top2 = np.argsort(-occ_np[bi])[:2]   # oracle count (E4 V2)
                waves = [slots_np[bi, j] for j in top2]
                cost = np.full((2, 2), 1e6)
                for i, s in enumerate(cd['srcs']):
                    for j, w in enumerate(waves):
                        cost[i, j] = -_si_sdr_np(w, s)
                rows, cols = linear_sum_assignment(cost)
                slot_of_src = {int(i): waves[j] for i, j in zip(rows, cols)}
                c1 = constellation_np(cd['mods_idx'][0])
                c2 = constellation_np(cd['mods_idx'][1])
                consts = [c1, c2]
                decs = {'sym': [None, None], 'psp': [None, None]}
                for i in range(2):
                    z = oracle_frontend(slot_of_src[i], cd['cars'][i])
                    decs['sym'][i] = np.argmin(
                        np.abs(z[:, None] - consts[i][None, :]), axis=1)
                    decs['psp'][i] = psp_decisions(z, consts[i])
                for var in ('sym', 'psp'):
                    records.append(_score_one(
                        var, seed, cd, decs[var][0], decs[var][1], c1, c2))
        print(f"  waveform seed {seed}: {len(cell_data)} bursts in "
              f"{time.time() - t0:.0f}s", flush=True)

    # ---- cross-check ABS against the cached full-run records ----
    _crosscheck(records, cell_data, args)
    return records


def _crosscheck(records, cell_data, args):
    """The subset is the FIRST n bursts per SNR block of the published
    grid; the cached ABS records for the same variants/seeds must match
    positionally (small float drift from BLAS threading is expected,
    EXPERIMENT_LOG 2026-09-21: <=0.0006).  Prints max |delta|; never
    raises."""
    def max_delta(mine, cached):
        worst = 0.0
        n = 0
        for s in sorted({r['snr'] for r in mine}):
            m_s = [r for r in mine if r['snr'] == s]
            c_s = [c for c in cached if c['snr'] == s]
            for k, r in enumerate(m_s):
                if k < len(c_s):
                    worst = max(worst, abs(c_s[k]['ser'] - r['abs_ser']))
                    n += 1
        return worst, n

    try:
        cache = json.load(open(os.path.join(
            C.RESULTS_DIR, GRID_FILES[0])))['records']
        for var in ('V1', 'V4'):
            mine = [r for r in records if r['variant'] == var]
            cs = [c for c in cache if c['variant'] == var]
            worst, n = max_delta(mine, cs)
            print(f"  cross-check {var}: max |ABS_SER - cached| = "
                  f"{worst:.5f} over {n} bursts", flush=True)
    except Exception as e:  # pragma: no cover - informational only
        print(f"  cross-check (ECM) skipped ({e})", flush=True)
    try:
        psp_cache = json.load(open(os.path.join(
            C.RESULTS_DIR, PSP_FILE)))['records']
        for var in ('sym', 'psp'):
            for seed in args.seeds:
                mine = [r for r in records
                        if r['variant'] == var and r['seed'] == seed]
                cs = [c for c in psp_cache
                      if c['variant'] == var and c['seed'] == seed]
                worst, n = max_delta(mine, cs)
                print(f"  cross-check {var} s{seed}: max |ABS_SER - cached|"
                      f" = {worst:.5f} over {n} bursts", flush=True)
    except Exception as e:  # pragma: no cover - informational only
        print(f"  cross-check (waveform) skipped ({e})", flush=True)


def aggregate_diff(records):
    out = {}
    for var in sorted({r['variant'] for r in records}):
        rs = [r for r in records if r['variant'] == var]
        # seed/group averaging: ECM variants have seed=-1 (single group)
        groups = sorted({r['seed'] for r in rs})
        gblocks = []
        for g in groups:
            gr = [r for r in rs if r['seed'] == g]
            dr = [r for r in gr if r['diff_ser'] is not None]
            blk = {'n': len(gr), 'n_diff': len(dr),
                   'abs_ser': _mean(gr, 'abs_ser'),
                   'abs_ber': _mean(gr, 'abs_ber'),
                   # ABS restricted to the SAME PSK-only bursts as DIFF
                   # (paired comparison — pooled ABS above includes the
                   # 16QAM pairs and is NOT comparable to DIFF)
                   'abs_ser_psk': _mean(dr, 'abs_ser'),
                   'abs_ber_psk': _mean(dr, 'abs_ber'),
                   'diff_ser': _mean(dr, 'diff_ser'),
                   'diff_ber': _mean(dr, 'diff_ber'),
                   'per_snr': {}}
            for s in sorted({r['snr'] for r in gr}):
                sr = [r for r in gr if r['snr'] == s]
                sd = [r for r in sr if r['diff_ser'] is not None]
                blk['per_snr'][f'{s:g}'] = {
                    'abs_ser': _mean(sr, 'abs_ser'),
                    'abs_ber': _mean(sr, 'abs_ber'),
                    'abs_ser_psk': _mean(sd, 'abs_ser'),
                    'abs_ber_psk': _mean(sd, 'abs_ber'),
                    'diff_ser': _mean(sd, 'diff_ser'),
                    'diff_ber': _mean(sd, 'diff_ber'),
                    'n': len(sr), 'n_diff': len(sd)}
            gblocks.append(blk)
        def gm(key):
            vals = [b[key] for b in gblocks if b[key] is not None]
            return (float(np.mean(vals)), float(np.std(vals))) if vals \
                else (None, None)
        snrs = sorted({s for b in gblocks for s in b['per_snr']})

        def gm_snr(s, key):
            vals = [b['per_snr'][s][key] for b in gblocks
                    if b['per_snr'][s][key] is not None]
            return float(np.mean(vals)) if vals else None

        out[var] = {
            'n_groups': len(gblocks),
            'n_bursts_total': sum(b['n'] for b in gblocks),
            'n_diff_bursts_total': sum(b['n_diff'] for b in gblocks),
            'pooled': {'abs_ser': gm('abs_ser'), 'abs_ber': gm('abs_ber'),
                       'abs_ser_psk': gm('abs_ser_psk'),
                       'abs_ber_psk': gm('abs_ber_psk'),
                       'diff_ser': gm('diff_ser'),
                       'diff_ber': gm('diff_ber')},
            'per_snr': {s: {
                'abs_ser': gm_snr(s, 'abs_ser'),
                'abs_ber': gm_snr(s, 'abs_ber'),
                'abs_ser_psk': gm_snr(s, 'abs_ser_psk'),
                'abs_ber_psk': gm_snr(s, 'abs_ber_psk'),
                'diff_ser': gm_snr(s, 'diff_ser'),
                'diff_ber': gm_snr(s, 'diff_ber'),
                'n': sum(b['per_snr'][s]['n'] for b in gblocks),
                'n_diff': sum(b['per_snr'][s]['n_diff'] for b in gblocks)}
                for s in snrs},
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _report(out):
    if 'ber_cached' in out:
        print("\n=== E4 Gray-mapped BER (cached decisions, group-averaged) "
              "===")
        print(f"{'variant':>8s} | {'pooled SER':>10s} | {'pooled BER':>10s}"
              f" | {'PSK BER':>8s} | {'16QAM BER':>9s}")
        for var, blk in out['ber_cached'].items():
            p = blk['pooled']
            print(f"{var:>8s} | {p['ser']:.4f}±{p['ser_std']:.4f} | "
                  f"{p['ber']:.4f}±{p['ber_std']:.4f} | "
                  f"{blk['PSK-only']['ber']:.4f} | "
                  f"{blk['16QAM-involving']['ber']:.4f}")
    if 'diff' in out:
        d = out['diff']
        print(f"\n=== E4 genie-free differential decoding "
              f"(n_subset={d['config']['n_subset']}/SNR"
              f"{' SUBSET' if d['config']['subset'] else ' FULL'}) ===")
        print(f"{'variant':>8s} | {'ABS* SER':>9s} | {'DIFF SER':>9s}"
              f" | {'ABS* BER':>9s} | {'DIFF BER':>9s} | {'n_diff':>6s}")
        for var, blk in d['variants'].items():
            p = blk['pooled']
            fmt = lambda v: f"{v[0]:.4f}" if v[0] is not None else "   --"
            print(f"{var:>8s} | {fmt(p['abs_ser_psk']):>9s} | "
                  f"{fmt(p['diff_ser']):>9s} | {fmt(p['abs_ber_psk']):>9s} | "
                  f"{fmt(p['diff_ber']):>9s} | {blk['n_diff_bursts_total']:>6d}")
        print("(ABS* and DIFF are paired on the SAME PSK-only bursts; "
              "16QAM-involving pairs are ABS-only — E1-D convention)")


def main(argv=None):
    p = argparse.ArgumentParser(
        description='E4 review-3: cached Gray BER + genie-free differential')
    p.add_argument('--mode', choices=['ber', 'diff', 'all'], default='all')
    p.add_argument('--n_subset', type=int, default=100,
                   help='bursts per SNR block (100 = the full published '
                        'grid; smaller = strict subset, first n per SNR)')
    p.add_argument('--seeds', type=int, nargs='+',
                   default=[42, 43, 44, 45, 46],
                   help='checkpoint seeds for the waveform route')
    p.add_argument('--snr_points', type=float, nargs='+',
                   default=list(C.SignalConfig.snr_test_points))
    p.add_argument('--test_seed', type=int, default=C.DataConfig.test_seed)
    p.add_argument('--ckpt_dir', type=str, default=C.CHECKPOINT_DIR)
    p.add_argument('--results_dir', type=str, default=C.RESULTS_DIR)
    p.add_argument('--out', type=str,
                   default=os.path.join(C.RESULTS_DIR, 'e4_ber_diff.json'))
    args = p.parse_args(argv)

    out = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
           'note': 'ber_cached = Gray BER of the SAME cached decisions as '
                   'the published E4 SER (ser_comp.GRAY_BITS); diff = '
                   'genie-free differential-equivalent scoring of re-run '
                   'detectors (PSK pairs only; E1-D convention).'}
    if args.mode in ('ber', 'all'):
        out['ber_cached'] = aggregate_cached(args.results_dir)
    if args.mode in ('diff', 'all'):
        selftest()
        t0 = time.time()
        records = run_diff(args)
        out['diff'] = {
            'config': {'n_subset': args.n_subset,
                       'subset': args.n_subset < 100,
                       'seeds': args.seeds,
                       'snr_points': args.snr_points,
                       'test_seed': args.test_seed,
                       'runtime_s': round(time.time() - t0, 1)},
            'variants': aggregate_diff(records),
            'records': records}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"saved {args.out}", flush=True)
    _report(out)
    return out


def smoke():
    """__main__ smoke: scorer self-test + tiny end-to-end run (1 burst at
    SNR 10 for the ECM routes + one checkpoint seed for the waveform
    route), plus the full cache BER aggregation.  Writes a SEPARATE smoke
    file; the canonical results/e4_ber_diff.json is only written by an
    explicit --mode run."""
    out = main(['--mode', 'all', '--n_subset', '1', '--snr_points', '10',
                '--seeds', '42',
                '--out', os.path.join(C.RESULTS_DIR,
                                      'e4_ber_diff_smoke.json')])
    d = out['diff']['variants']
    assert 'V4' in d and 'psp' in d
    for var in ('V1', 'V4', 'sym', 'psp'):
        assert d[var]['pooled']['abs_ser'][0] is not None
    print("smoke passed", flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        smoke()
    else:
        main()
