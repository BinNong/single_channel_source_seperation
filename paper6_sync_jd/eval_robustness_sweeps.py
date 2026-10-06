"""Paper 6 — reviewer-2 robustness sweeps at K=2 (QPSK+QPSK fixed).

Ties the two-tone CRB theory (Prop 3: frequency-estimation information
collapses when the two carrier offsets differ by less than
~1/T_burst = 3.906 Hz) to receiver performance, plus timing-offset and
amplitude-ratio sensitivity.  Three sweeps, all K=2 with both sources
QPSK so the swept variable is isolated:

  (a) freq: true carrier offsets delta1 = 1.5 Hz,
      delta2 = 1.5 + df with df in {0, 0.5, 1, 2, 4, 8} Hz — all inside
      the receiver's native [-5, +10] Hz search window (NOT widened).
      In-generator jitter is pinned to 0 via the freq_jitter hook of
      signal_utils.generate_single_signal; the mixture is built manually
      (w1, w2 ~ U(0.4, 0.6), AWGN scaled to the TOTAL mixture power —
      the generate_vark_mixture SNR convention).
      Variants: V1, V3 (oracle-frequency reference), V4.
  (b) tau: receiver sampling-grid offset tau in {0, 1.6, 4.0, 8.0}
      samples = {0, 0.1, 0.25, 0.5} T_s, applied inside
      joint_sync_detect.mixture_symbols.  Bursts come from the
      benchmark's default random offsets (generate_vark_mixture with
      return_carriers=True).  Because run_v1/run_v4 call
      mixture_symbols(mix) without tau, this sweep drives the front end
      explicitly: r = mixture_symbols(mix, tau_samples=tau), then
      ecm_joint (V1) / ecm_joint_isi (V4).  Reference labels are
      unaffected by tau.  Variants: V1, V4.
  (c) amp: amplitude ratio rho = 20log10(w2/w1) in {-10,-5,0,5,10} dB
      with w1 = 1.0 fixed, w2 = 10^(rho/20); default jitter (true
      carriers returned).  Manual mixture as in (a).  Variants: V1, V4
      (V3 skipped).  ser1/ser2 recorded separately — the interesting
      quantity is the weak source's SER.

Deterministic seeding per burst: np.random.seed(BASE + 1000*value_idx +
10*snr_idx + burst_idx) with BASE in {91000 (freq), 92000 (tau),
93000 (amp)}.

Efficiency note: for every burst the V1 frequency search is run ONCE
and shared between the V1 and V4 arms (r = mixture_symbols(mix);
v1 = ecm_joint(r, ...); v4 = ecm_joint_isi(r, ..., v1=v1)) — this is
bit-identical to calling eval_joint_k2.run_v1 / run_v4 separately
(mixture_symbols is deterministic and run_v4 warm-starts from the same
internal V1 solution) and mirrors the 'V1V4' combined arm of
eval_joint_k2.py.  V3 uses eval_joint_k2.run_v3 directly.

Usage:
    python eval_robustness_sweeps.py --sweep all --n_per_point 2 \
        --out results/robustness_smoke.json          # smoke
    python eval_robustness_sweeps.py --sweep freq    # full
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

# Vendored paper6 modules (self-contained; no sys.path tricks).
import config as C                                          # noqa: E402
from data_generator import generate_vark_mixture, MOD_TYPES  # noqa: E402
from joint_detect import constellation_np                    # noqa: E402
from signal_utils import generate_single_signal              # noqa: E402
from joint_sync_detect import ecm_joint, mixture_symbols     # noqa: E402
from joint_isi import ecm_joint_isi                          # noqa: E402
from eval_joint_k2 import score_decisions, run_v3, _ref_labels  # noqa: E402

QPSK = 'QPSK'
QPSK_IDX = MOD_TYPES.index(QPSK)

FREQ_GRID = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]      # Delta f [Hz]
TAU_GRID = [0.0, 1.6, 4.0, 8.0]                 # samples (0, 0.1, 0.25, 0.5 Ts)
AMP_GRID = [-10.0, -5.0, 0.0, 5.0, 10.0]        # rho = 20log10(w2/w1) [dB]
SNR_GRID = [0.0, 10.0, 20.0]
SEED_BASE = {'freq': 91000, 'tau': 92000, 'amp': 93000}
DF1_FIXED = 1.5                                  # sweep (a): delta1 [Hz]


# ---------------------------------------------------------------------------
# Burst construction
# ---------------------------------------------------------------------------
def _manual_mixture(carriers_hz, snr, weights, jitter):
    """Build a K=2 QPSK mixture by hand (mirrors generate_vark_mixture's
    SNR convention: noise power = mean(|mix_clean|^2)/10^(snr/10)).

    carriers_hz: (fc1, fc2) base carriers; jitter: None -> default
    U(-5, 5) in-generator jitter, float -> pinned jitter (freq_jitter
    hook).  Returns (mix, sources, true_carriers)."""
    sigs, fcs = [], []
    for cf in carriers_hz:
        kw = {'return_freq': True}
        if jitter is not None:
            kw['freq_jitter'] = float(jitter)
        s, _, fc = generate_single_signal(
            C.SignalConfig.n_symbols, float(cf), C.SignalConfig.sample_rate,
            C.SignalConfig.signal_length, QPSK, C.SignalConfig.roll_off,
            C.SignalConfig.num_taps, C.SignalConfig.apply_fading,
            C.SignalConfig.fading_taps, **kw)
        sigs.append(s)
        fcs.append(float(fc))
    mix_clean = weights[0] * sigs[0] + weights[1] * sigs[1]
    noise_power = np.mean(np.abs(mix_clean) ** 2) / (10 ** (snr / 10))
    T = C.SignalConfig.signal_length
    noise = np.sqrt(noise_power / 2) * (np.random.randn(T)
                                        + 1j * np.random.randn(T))
    return (mix_clean + noise).astype(np.complex64), sigs, fcs


# ---------------------------------------------------------------------------
# Per-burst evaluation (shared V1 search; see module docstring)
# ---------------------------------------------------------------------------
def _eval_burst(mix, srcs, fcs, variants, tau=0.0, df_true=None):
    """Returns {variant: record-dict} for one burst."""
    c1 = c2 = constellation_np(QPSK_IDX)
    ref1 = _ref_labels(srcs[0], QPSK_IDX, fcs[0])
    ref2 = _ref_labels(srcs[1], QPSK_IDX, fcs[1])
    df_off = [fcs[0] - C.SyncConfig.nominal_carrier,
              fcs[1] - C.SyncConfig.nominal_carrier]

    res = {}
    v1 = None
    if 'V1' in variants or 'V4' in variants:
        r = mixture_symbols(mix, tau_samples=tau)
        v1 = ecm_joint(r, c1, c2)
    if 'V1' in variants:
        res['V1'] = v1
    if 'V4' in variants:
        res['V4'] = ecm_joint_isi(r, c1, c2, L=5, v1=v1)
    if 'V3' in variants:
        # run_v3 calls mixture_symbols(mix) without tau — only used in
        # the freq sweep where tau = 0 (identical front end).
        res['V3'] = run_v3(mix, c1, c2, df_true)

    out = {}
    for tag, r_ in res.items():
        ser1, _ber1, ser2, _ber2 = score_decisions(
            r_['dec1'], r_['dec2'], c1, c2, ref1, ref2,
            [QPSK, QPSK], allow_swap=True)
        if tag == 'V3':
            d1e = d2e = float('nan')
        else:
            d1e = abs(r_['df1'] - df_off[0])
            d2e = abs(r_['df2'] - df_off[1])
        out[tag] = {'ser': 0.5 * (ser1 + ser2), 'ser1': ser1, 'ser2': ser2,
                    'df1_err': d1e, 'df2_err': d2e,
                    'res_energy': float(r_['res_energy']),
                    'rounds': int(r_['rounds'])}
    return out


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def sweep_freq(n_per_point):
    """(a) carrier-separation sweep; delta1 = 1.5 Hz, delta2 = 1.5+df."""
    records = []
    for di, df in enumerate(FREQ_GRID):
        for si, snr in enumerate(SNR_GRID):
            t0 = time.time()
            for b in range(n_per_point):
                np.random.seed(SEED_BASE['freq'] + 1000 * di + 10 * si + b)
                w = np.random.uniform(0.4, 0.6, size=2)
                mix, srcs, fcs = _manual_mixture(
                    (C.SyncConfig.nominal_carrier + DF1_FIXED,
                     C.SyncConfig.nominal_carrier + DF1_FIXED + df),
                    snr, w, jitter=0.0)
                df_true = (DF1_FIXED, DF1_FIXED + df)
                recs = _eval_burst(mix, srcs, fcs, ['V1', 'V3', 'V4'],
                                   df_true=df_true)
                for tag, rec in recs.items():
                    records.append({'sweep': 'freq', 'value': float(df),
                                    'snr': float(snr), 'burst': b,
                                    'variant': tag, **rec})
            print(f"  freq df={df:>4.1f} snr={snr:>5.1f}: "
                  f"{n_per_point} bursts in {time.time() - t0:.0f}s",
                  flush=True)
    return records


def sweep_tau(n_per_point):
    """(b) receiver sampling-grid offset sweep (benchmark bursts)."""
    records = []
    for ti, tau in enumerate(TAU_GRID):
        for si, snr in enumerate(SNR_GRID):
            t0 = time.time()
            for b in range(n_per_point):
                np.random.seed(SEED_BASE['tau'] + 1000 * ti + 10 * si + b)
                mix, srcs, _midx, cars = generate_vark_mixture(
                    C.SignalConfig.signal_length, C.SignalConfig.sample_rate,
                    float(snr), [QPSK], k=2,
                    carrier_base=C.SignalConfig.carrier_base,
                    freq_gap_range=C.SignalConfig.freq_gap_range,
                    n_symbols=C.SignalConfig.n_symbols,
                    roll_off=C.SignalConfig.roll_off,
                    num_taps=C.SignalConfig.num_taps,
                    apply_fading=C.SignalConfig.apply_fading,
                    fading_taps=C.SignalConfig.fading_taps,
                    return_carriers=True)
                recs = _eval_burst(mix, srcs, cars, ['V1', 'V4'], tau=tau)
                for tag, rec in recs.items():
                    records.append({'sweep': 'tau', 'value': float(tau),
                                    'snr': float(snr), 'burst': b,
                                    'variant': tag, **rec})
            print(f"  tau={tau:>4.1f} snr={snr:>5.1f}: "
                  f"{n_per_point} bursts in {time.time() - t0:.0f}s",
                  flush=True)
    return records


def sweep_amp(n_per_point):
    """(c) amplitude-ratio sweep; w1 = 1.0, w2 = 10^(rho/20)."""
    records = []
    for ri, rho in enumerate(AMP_GRID):
        for si, snr in enumerate(SNR_GRID):
            t0 = time.time()
            for b in range(n_per_point):
                np.random.seed(SEED_BASE['amp'] + 1000 * ri + 10 * si + b)
                # benchmark carrier draw: base + U(freq_gap_range), then
                # the default in-generator jitter (jitter=None)
                cars0 = C.SignalConfig.carrier_base + np.random.uniform(
                    C.SignalConfig.freq_gap_range[0],
                    C.SignalConfig.freq_gap_range[1], size=2)
                w2 = 10 ** (rho / 20)
                mix, srcs, fcs = _manual_mixture(
                    cars0, snr, (1.0, w2), jitter=None)
                recs = _eval_burst(mix, srcs, fcs, ['V1', 'V4'])
                for tag, rec in recs.items():
                    records.append({'sweep': 'amp', 'value': float(rho),
                                    'snr': float(snr), 'burst': b,
                                    'variant': tag, **rec})
            print(f"  amp rho={rho:>5.1f} snr={snr:>5.1f}: "
                  f"{n_per_point} bursts in {time.time() - t0:.0f}s",
                  flush=True)
    return records


# ---------------------------------------------------------------------------
# Reporting / saving
# ---------------------------------------------------------------------------
def _save(out_path, records, args):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'sweep': args.sweep, 'n_per_point': args.n_per_point,
                   'snr_points': SNR_GRID,
                   'freq_grid': FREQ_GRID, 'tau_grid': TAU_GRID,
                   'amp_grid': AMP_GRID,
                   'records': records}, f)
    print(f"saved {out_path} ({len(records)} records)", flush=True)


def _cell_stats(rs):
    """mean SER / ser1 / ser2, frequency RMSE, acquisition prob."""
    ser = float(np.mean([r['ser'] for r in rs]))
    ser1 = float(np.mean([r['ser1'] for r in rs]))
    ser2 = float(np.mean([r['ser2'] for r in rs]))
    d1 = np.array([r['df1_err'] for r in rs])
    d2 = np.array([r['df2_err'] for r in rs])
    if np.all(np.isnan(d1)):
        frmse, acq = float('nan'), float('nan')
    else:
        frmse = float(np.sqrt(np.nanmean(np.concatenate([d1, d2]) ** 2)))
        acq = float(np.mean((d1 < 1.0) & (d2 < 1.0)))
    return ser, ser1, ser2, frmse, acq


def _report(records):
    """Per (sweep value, SNR, variant): SER / ser1 / ser2 / f-RMSE /
    P(acquisition)."""
    sweeps = sorted({r['sweep'] for r in records})
    for sw in sweeps:
        rs_sw = [r for r in records if r['sweep'] == sw]
        vals = sorted({r['value'] for r in rs_sw})
        snrs = sorted({r['snr'] for r in rs_sw})
        variants = sorted({r['variant'] for r in rs_sw})
        unit = {'freq': 'df[Hz]', 'tau': 'tau[smp]', 'amp': 'rho[dB]'}[sw]
        print(f"\n=== {sw} sweep: mean SER (per-source avg), frequency "
              f"RMSE [Hz], P(both df_err < 1 Hz) ===")
        print(f"{'value':>9s} {'snr':>5s} {'var':>4s} | {'SER':>7s} "
              f"{'ser1':>7s} {'ser2':>7s} {'f-RMSE':>7s} {'P(acq)':>7s}")
        for v in vals:
            for s in snrs:
                for var in variants:
                    rs = [r for r in rs_sw if r['value'] == v
                          and r['snr'] == s and r['variant'] == var]
                    if not rs:
                        continue
                    ser, ser1, ser2, frmse, acq = _cell_stats(rs)
                    print(f"{v:>9.1f} {s:>5.1f} {var:>4s} | {ser:7.4f} "
                          f"{ser1:7.4f} {ser2:7.4f} {frmse:7.3f} "
                          f"{acq:7.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description='Paper 6 reviewer-2 robustness sweeps at K=2 '
                    '(freq / tau / amp, QPSK+QPSK)')
    p.add_argument('--sweep', type=str, default='all',
                   choices=['freq', 'tau', 'amp', 'all'])
    p.add_argument('--n_per_point', type=int, default=100)
    p.add_argument('--out', type=str, default=None,
                   help='default: results/robustness_<sweep>.json '
                        '(per-sweep files; a single file when --out is '
                        'given)')
    args = p.parse_args()

    sweep_fns = {'freq': sweep_freq, 'tau': sweep_tau, 'amp': sweep_amp}
    names = ['freq', 'tau', 'amp'] if args.sweep == 'all' else [args.sweep]

    all_records = []
    for name in names:
        print(f"=== {name} sweep: {args.n_per_point} bursts/cell ===",
              flush=True)
        t0 = time.time()
        recs = sweep_fns[name](args.n_per_point)
        all_records.extend(recs)
        print(f"{name} sweep done in {time.time() - t0:.0f}s "
              f"({len(recs)} records)", flush=True)
        if args.out is None:
            _save(os.path.join(C.RESULTS_DIR, f'robustness_{name}.json'),
                  recs, args)
    if args.out is not None:
        _save(args.out, all_records, args)
    _report(all_records)


if __name__ == '__main__':
    main()
