"""Paper 6 — Reviewer-2 (ii): differential decoding vs the genie M-fold
phase-ambiguity resolution (K=1).

E1's blind-sync evaluation resolves the M-fold phase ambiguity of the
M-th-power synchroniser POST-HOC against the reference labels (genie
ambiguity resolution, the standard convention for evaluating blind
synchronisers).  A real deployment instead resolves it with differential
coding.  This script quantifies what that deployment answer costs:

  per K=1 burst, two receivers
    (a) oracle compensated receiver (true-carrier down-conversion +
        phase alignment — mirrors ser_comp._demod_pair_labels);
    (b) BlindCarrierSync (sync.blind_sync_known_mod on the raw burst,
        true modulation);
  each scored two ways
    ABS  : genie M-fold rotation-resolved SER (the E1 convention,
           eval_joint_k2._rot_ser_ber);
    DIFF : differential decisions d_n = z_n conj(z_{n-1}), n = 1..N-1,
           nearest point on the differential constellation.  For the
           offset PSK grids used here (constellation =
           exp(j(2*pi*m/M + phi0))) the product of two constellation
           points is again an M-point grid with offset 2*phi0, so the
           decision lives on that M-point grid and the M-fold ambiguity
           cancels exactly (z_n e^{j th} conj(z_{n-1} e^{j th}) =
           z_n conj(z_{n-1})).

16QAM is skipped (no standard differential coding for square QAM).
Grid: 200 bursts per (modulation, SNR) cell, SNR in {-10..20} dB, two
independent grids (np.random.seed 99999 / 31337), bursts built with
generate_vark_mixture(..., k=1, mods=[mod], return_carriers=True).

Usage:
    python eval_diff_coding.py --n_per_cell 4      # smoke (+ self-test)
    python eval_diff_coding.py --n_per_cell 200    # full
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

# Vendored paper6 modules (self-contained; no sys.path tricks).
import config as C                                          # noqa: E402
from data_generator import (generate_vark_mixture,          # noqa: E402
                            MOD_TYPES)
from signal_utils import rrc_filter                         # noqa: E402
from sync import blind_sync_known_mod                       # noqa: E402
from joint_detect import constellation_np                   # noqa: E402
from eval_joint_k2 import _rot_ser_ber, _ref_labels         # noqa: E402

DIFF_MODS = ['BPSK', 'QPSK', '8PSK']      # 16QAM: no standard diff coding


# ---------------------------------------------------------------------------
# Receivers
# ---------------------------------------------------------------------------
def oracle_z(est_np, ref_np, carrier):
    """Oracle compensated receiver's unit-power symbol stream: true-carrier
    down-conversion -> RRC MF -> 0::sps grid -> unit power -> phase
    alignment to the reference (mirrors ser_comp._demod_pair_labels)."""
    T = len(est_np)
    fs = C.SignalConfig.sample_rate
    n_sym = C.SignalConfig.n_symbols
    sps = T // n_sym
    t = np.arange(T) / fs
    rrc = rrc_filter(C.SignalConfig.num_taps, C.SignalConfig.roll_off, sps)
    out = []
    for w in (est_np, ref_np):
        bb = np.asarray(w, dtype=np.complex128) \
            * np.exp(-1j * 2 * np.pi * carrier * t)
        syms = np.convolve(bb, rrc, mode='same')[0::sps][:n_sym]
        out.append(syms / (np.sqrt(np.mean(np.abs(syms) ** 2)) + 1e-10))
    est_u, ref_u = out
    g = np.sum(ref_u * np.conj(est_u)) / (np.sum(np.abs(est_u) ** 2) + 1e-10)
    return est_u * np.exp(1j * np.angle(g))


# ---------------------------------------------------------------------------
# Differential scoring
# ---------------------------------------------------------------------------
def diff_grid(const):
    """Distinct points of the product set {c_i conj(c_j)}.  For the PSK
    grids this is exactly M points (the constellation rotated by 2*phi0)."""
    prods = (const[:, None] * np.conj(const[None, :])).ravel()
    D = [prods[0]]
    for p in prods:
        if not any(abs(p - d) < 1e-9 for d in D):
            D.append(p)
    return np.asarray(D)


def diff_ser(z, const, ref_lab, D):
    """SER of differential decisions vs the reference differential symbols.

    d_n = z_n conj(z_{n-1}); reference r_n = c_true[n] conj(c_true[n-1])
    with c_true = const[ref_lab] (the E1/E3/E4 scoring truth).  Both are
    demodulated to the same product grid D, so the M-fold ambiguity
    cancels and NO genie information enters this score."""
    d = z[1:] * np.conj(z[:-1])
    r = const[ref_lab[1:]] * np.conj(const[ref_lab[:-1]])
    dec = np.argmin(np.abs(d[:, None] - D[None, :]), axis=1)
    ref = np.argmin(np.abs(r[:, None] - D[None, :]), axis=1)
    return float(np.mean(dec != ref))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def selftest(n_bursts=10):
    """(1) Fading-free 40 dB bursts: DIFF SER must be EXACTLY ~0 — this
    validates the differential grid / index math (any rotation or
    product-grid bug shows up here).  (2) Fading 40 dB bursts
    (informational + loose sanity bounds): the 3-tap channel's ISI phase
    noise is DOUBLED by the differential product while nearest-point ABS
    absorbs it, so with fading the DIFF floor is bursty and heavy-tailed
    (measured 2026-09-22: ABS ~ 0.003 with DIFF up to 0.5 on smeared
    8PSK bursts) — a genuine property of differential detection under
    fading-ISI, not a scoring bug."""
    print("DIFF self-test ...", flush=True)
    np.random.seed(20260922)
    for mod in DIFF_MODS:
        mi = MOD_TYPES.index(mod)
        const = constellation_np(mi)
        D = diff_grid(const)
        assert len(D) == len(const), \
            f"{mod}: product grid has {len(D)} points (expected {len(const)})"
        m_order = C.SyncConfig.sym_order[mod]
        # (1) fading-free: strict
        ser_o, ser_b = [], []
        for _ in range(n_bursts):
            mix, srcs, midx, cars = generate_vark_mixture(
                C.SignalConfig.signal_length, C.SignalConfig.sample_rate,
                40.0, MOD_TYPES, 1, mods=[mod], return_carriers=True,
                apply_fading=False)
            ref = _ref_labels(srcs[0], midx[0], cars[0])
            ser_o.append(diff_ser(oracle_z(mix, srcs[0], cars[0]),
                                  const, ref, D))
            zb = blind_sync_known_mod(np.asarray(mix, dtype=np.complex128),
                                      mi)['z']
            ser_b.append(diff_ser(zb, const, ref, D))
        print(f"  {mod:5s} fading-free 40 dB: oracle DIFF="
              f"{np.mean(ser_o):.5f}  blind DIFF={np.mean(ser_b):.5f}",
              flush=True)
        assert np.mean(ser_o) < 0.01 and np.mean(ser_b) < 0.02, \
            f"{mod}: fading-free DIFF not ~0 (machinery bug)"
        # (2) with fading: informational + loose sanity
        abs_o, dif_o = [], []
        for _ in range(n_bursts):
            mix, srcs, midx, cars = generate_vark_mixture(
                C.SignalConfig.signal_length, C.SignalConfig.sample_rate,
                40.0, MOD_TYPES, 1, mods=[mod], return_carriers=True)
            ref = _ref_labels(srcs[0], midx[0], cars[0])
            zo = oracle_z(mix, srcs[0], cars[0])
            abs_o.append(_rot_ser_ber(zo, const, ref, m_order)[0])
            dif_o.append(diff_ser(zo, const, ref, D))
        print(f"  {mod:5s} fading 40 dB:      oracle ABS="
              f"{np.mean(abs_o):.4f}  DIFF mean={np.mean(dif_o):.4f} "
              f"med={np.median(dif_o):.4f}", flush=True)
        assert np.mean(abs_o) < 0.05, \
            f"{mod}: fading ABS floor {np.mean(abs_o):.4f} unexpectedly high"
    print("  self-test passed", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description='Differential decoding vs genie M-fold resolution (K=1)')
    p.add_argument('--n_per_cell', type=int, default=200)
    p.add_argument('--seeds', type=int, nargs='+', default=[99999, 31337],
                   help='independent grid seeds (burst RNG)')
    p.add_argument('--snr_points', type=float, nargs='+',
                   default=list(C.SignalConfig.snr_test_points))
    p.add_argument('--mods', type=str, nargs='+', default=DIFF_MODS)
    p.add_argument('--out', type=str,
                   default=os.path.join(C.RESULTS_DIR, 'diff_coding.json'))
    p.add_argument('--skip_selftest', action='store_true')
    args = p.parse_args()

    if not args.skip_selftest:
        selftest()

    records = []
    for gseed in args.seeds:
        np.random.seed(gseed)
        for mod in args.mods:
            mi = MOD_TYPES.index(mod)
            const = constellation_np(mi)
            D = diff_grid(const)
            m_order = C.SyncConfig.sym_order[mod]
            for snr in args.snr_points:
                t0 = time.time()
                for _ in range(args.n_per_cell):
                    mix, srcs, midx, cars = generate_vark_mixture(
                        C.SignalConfig.signal_length,
                        C.SignalConfig.sample_rate, float(snr),
                        MOD_TYPES, 1, mods=[mod], return_carriers=True)
                    ref = _ref_labels(srcs[0], midx[0], cars[0])
                    # (a) oracle compensated receiver
                    zo = oracle_z(mix, srcs[0], cars[0])
                    abs_o = _rot_ser_ber(zo, const, ref, m_order)[0]
                    dif_o = diff_ser(zo, const, ref, D)
                    # (b) BlindCarrierSync
                    zb = blind_sync_known_mod(
                        np.asarray(mix, dtype=np.complex128), mi)['z']
                    abs_b = _rot_ser_ber(zb, const, ref, m_order)[0]
                    dif_b = diff_ser(zb, const, ref, D)
                    records.append({'grid_seed': gseed, 'mod': mod,
                                    'snr': float(snr),
                                    'abs_oracle': abs_o, 'diff_oracle': dif_o,
                                    'abs_blind': abs_b, 'diff_blind': dif_b})
                print(f"  grid {gseed} {mod:5s} SNR {snr:>5g}: "
                      f"{args.n_per_cell} bursts "
                      f"({(time.time() - t0) / args.n_per_cell * 1000:.0f}"
                      f" ms/burst)", flush=True)
        _save(args.out, records, args)

    _save(args.out, records, args)
    _report(records, args)


def _save(out_path, records, args):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'n_per_cell': args.n_per_cell, 'grid_seeds': args.seeds,
                   'snr_points': args.snr_points, 'mods': args.mods,
                   'records': records}, f)
    print(f"saved {out_path} ({len(records)} records)", flush=True)


def _mean(recs, key):
    v = [r[key] for r in recs]
    return float(np.mean(v)) if v else float('nan')


def _report(records, args):
    print("\n=== Differential decoding summary (SER; mean over grids) ===")
    hdr = (f"{'mod':>5s} | {'SNR':>5s} | {'ora ABS':>8s} | {'ora DIFF':>8s}"
           f" | {'bli ABS':>8s} | {'bli DIFF':>8s} | {'bD-bA':>7s}")
    print(hdr)
    print('-' * len(hdr))
    for mod in args.mods:
        for snr in args.snr_points:
            rs = [r for r in records if r['mod'] == mod
                  and r['snr'] == float(snr)]
            oa, od = _mean(rs, 'abs_oracle'), _mean(rs, 'diff_oracle')
            ba, bd = _mean(rs, 'abs_blind'), _mean(rs, 'diff_blind')
            print(f"{mod:>5s} | {snr:>5g} | {oa:8.4f} | {od:8.4f} | "
                  f"{ba:8.4f} | {bd:8.4f} | {bd - ba:+7.4f}")
    ba = _mean(records, 'abs_blind')
    bd = _mean(records, 'diff_blind')
    oa = _mean(records, 'abs_oracle')
    od = _mean(records, 'diff_oracle')
    print(f"\npooled: oracle ABS={oa:.4f} DIFF={od:.4f} "
          f"(gap {od - oa:+.4f}) | blind ABS={ba:.4f} DIFF={bd:.4f} "
          f"(gap {bd - ba:+.4f})")
    print("Scientific question: is the blind DIFF-DIFF gap (deployment "
          "answer to the M-fold ambiguity) small vs the genie-resolved "
          "ABS scoring?")


if __name__ == '__main__':
    main()
