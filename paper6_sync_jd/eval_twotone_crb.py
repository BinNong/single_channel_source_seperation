"""Paper 6 — review-3 experiment E-A: K=2 two-tone frequency estimation
vs the Proposition-3 CRB.

Closes the loop on Prop 3 (theory_validation.py Part 2): under the
carrier-only (pure-tone) model the Fisher information for the two tone
frequencies couples and inflates as |Df| -> 0.  Prop 3 is a BOUND; this
script measures an actual estimator on the same model and overlays the
two curves:

  Signal model (complex baseband, matches Prop 3 exactly):
      r_n = a1 e^{j(2 pi f1 n Ts + phi1)} + a2 e^{j(2 pi f2 n Ts + phi2)}
            + w_n,     n = 0..N-1
  with N = SignalConfig.signal_length = 4096, fs = SignalConfig.sample_rate
  = 16000 Hz (T_burst = 0.256 s, 1/T_burst = 3.906 Hz), f1 = 0 and
  f2 = Df WLOG (the Fisher matrix depends only on the separation),
  phi_k ~ U[0, 2pi), w ~ CN(0, sigma2).  SNR is PER-STRONG-TONE:
  sigma2 = a1^2 / 10^(snr/10), so tone k sees SNR_k = a_k^2/sigma2
  (identical to the convention of theory_validation.crb_single_tone_hz2,
  whose docstring equates the generator's mixture SNR with the tone SNR).
  The amplitude ratio is rho = 20 log10(a1/a2) >= 0 dB (tone 1 strong).

  Estimator (minimal two-tone spectral-line estimator, implemented HERE
  because sync.py's public API is strictly single-burst/single-carrier —
  BlindCarrierSync has no K=2 mode; its per-source estimate is one line
  per separated slot, while Prop 3 concerns TWO lines in ONE burst).
  Relationship to BlindCarrierSync: it reuses the same core primitive,
  sync._fft_line (zero-padded FFT, SyncConfig.fft_zeropad = x8, peak +
  parabolic interpolation) for the first spectral line, with M = 1 — the
  pure tones are ALREADY spectral lines, so the M-th-power nonlinearity
  and the RRC matched filter of BlindCarrierSync step 2 are identity
  operations on this model.  The two-tone additions are (i) projection
  subtraction of the first line and a masked second-peak pick (exclusion
  +-0.6/T_burst around the first line), and (ii) a Gauss-Seidel
  zoom refinement: each tone in turn is re-estimated on the residual
  after subtracting the other by golden-section maximisation of the
  matched-filter (periodogram) magnitude |<x, e^{j2 pi f t}>| — the ML
  criterion for a single tone — over +-1.2/T_burst around the current
  estimate.  This is the standard CLEAN/zoom-FFT spectral-line estimator;
  no learning, no oracle information.

  Theory overlay: exact numeric Fisher via theory_validation._fisher_tones
  (phases unknown, as in Prop 3's headline curve).  Unequal amplitudes
  enter by congruence scaling J -> S J S / sigma2 with
  S = diag(a1, a2, a1, a2) (every f_k/phi_k derivative scales with a_k);
  at rho = 0 dB this reduces to two_tone_inflation x
  crb_single_tone_closed (asserted in the smoke test).

  Sweeps: |Df| in DF_GRID (10 log-spaced points, 0.25-16 Hz — the paper's
  Fig. 3(b) / robustness-sweep range, bracketing 1/T_burst = 3.906 Hz),
  SNR in {0, 10, 20} dB, rho in {0, 6} dB, n_mc Monte-Carlo bursts per
  cell (default 200; --smoke 20 on a 3-point Df grid).

  Outputs:
    results/eval_twotone_crb.json  — per-cell bias/var/RMSE per tone,
      pooled RMSE, acquisition-failure prob, and the CRB stds.
    figures/fig_twotone_crb.pdf/.png — 2x2 figure: columns = amplitude
      ratio; row (a) measured pooled RMSE (solid) vs sqrt(CRB) (dashed);
      row (b) per-tone bias (symlog).  Vertical line at 1/T_burst.
    figures/fig_twotone_crb_wide.pdf/.png (--wide) — compact 1x3
      textwidth variant for the page budget: (a) rho=0 dB RMSE vs
      sqrt(CRB), (b) rho=6 dB RMSE vs sqrt(CRB), (c) rho=0 dB bias.

  Usage:
      python eval_twotone_crb.py --smoke           # fast self-check
      python eval_twotone_crb.py                   # full grid (~3 min CPU)
      python eval_twotone_crb.py --from_json results/eval_twotone_crb.json \
          --wide                                   # regenerate figures only
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Vendored paper6 modules (self-contained; no sys.path tricks).
import config as C                                              # noqa: E402
from sync import _fft_line                                      # noqa: E402
from theory_validation import (_fisher_tones, two_tone_inflation,
                               crb_single_tone_closed)          # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

DF_GRID = np.geomspace(0.25, 16.0, 10)          # |Df| [Hz], paper range
SNR_GRID = [0.0, 10.0, 20.0]                    # per-strong-tone SNR [dB]
RHO_GRID = [0.0, 6.0]                           # 20 log10(a1/a2) [dB]
SEED_BASE = 94000
SEARCH_WIN = (-4.0, 20.0)                       # estimator search window [Hz]


# ---------------------------------------------------------------------------
# Minimal two-tone spectral-line estimator (see module docstring)
# ---------------------------------------------------------------------------
def _proj(x, t, f):
    """Complex amplitude of the line at f: <x, e^{j2 pi f t}> / N."""
    return complex(np.dot(x, np.exp(-2j * np.pi * f * t))) / len(x)


def _zoom_refine(x, t, f0, half_win, iters=24):
    """Golden-section maximisation of |<x, e^{j2 pi f t}>| on
    [f0 - half_win, f0 + half_win] (single-tone ML criterion)."""
    a, b = f0 - half_win, f0 + half_win
    gr = (np.sqrt(5.0) - 1.0) / 2.0

    def g(f):
        return abs(np.dot(x, np.exp(-2j * np.pi * f * t)))

    c, d = b - gr * (b - a), a + gr * (b - a)
    gc, gd = g(c), g(d)
    for _ in range(iters):
        if gc < gd:
            a, c, gc = c, d, gd
            d = a + gr * (b - a)
            gd = g(d)
        else:
            b, d, gd = d, c, gc
            c = b - gr * (b - a)
            gc = g(c)
    return 0.5 * (a + b)


def two_tone_estimate(x, fs, search_win=SEARCH_WIN):
    """Estimate the two tone frequencies of a noisy two-tone burst.

    Returns (f_lo, f_hi) sorted ascending.  Stage 1 reuses sync._fft_line
    (the BlindCarrierSync step-2 peak picker); stages 2-3 are the
    subtraction + zoom refinement described in the module docstring."""
    T = len(x)
    t = np.arange(T) / fs
    zeropad = C.SyncConfig.fft_zeropad
    t_burst = T / fs

    # 1. first (strongest) spectral line.
    f1 = _fft_line(x, fs, search_win[0], search_win[1], zeropad)
    a1 = _proj(x, t, f1)
    r = x - a1 * np.exp(2j * np.pi * f1 * t)

    # 2. second line on the residual: masked peak (exclusion +-0.6/T
    #    around f1) + parabolic interpolation, mirroring sync._fft_line.
    n_fft = T * zeropad
    spec = np.abs(np.fft.fft(r, n=n_fft))
    freqs = np.fft.fftfreq(n_fft, d=1.0 / fs)
    excl = 0.6 / t_burst
    in_win = ((freqs >= search_win[0]) & (freqs <= search_win[1])
              & (np.abs(freqs - f1) > excl))
    k = int(np.argmax(np.where(in_win, spec, -np.inf)))
    ym, y0, yp = spec[max(k - 1, 0)], spec[k], spec[min(k + 1, n_fft - 1)]
    denom = ym - 2.0 * y0 + yp
    delta = 0.5 * (ym - yp) / denom if abs(denom) > 1e-20 else 0.0
    f2 = (k + float(np.clip(delta, -1.0, 1.0))) * (fs / n_fft)
    if f2 > fs / 2:
        f2 -= fs

    # 3. Gauss-Seidel zoom refinement (2 sweeps are enough to converge).
    est = [[f1, a1], [f2, _proj(r, t, f2)]]
    half = 1.2 / t_burst
    for _ in range(2):
        for kk in (0, 1):
            fo, ao = est[1 - kk]
            res = x - ao * np.exp(2j * np.pi * fo * t)
            fk = _zoom_refine(res, t, est[kk][0], half)
            est[kk] = [fk, _proj(res, t, fk)]
    f_lo, f_hi = sorted((est[0][0], est[1][0]))
    return float(f_lo), float(f_hi)


# ---------------------------------------------------------------------------
# Two-tone CRB with per-tone amplitudes (exact numeric Fisher, phases
# unknown; reduces to two_tone_inflation x crb_single_tone_closed at
# rho = 0 dB — asserted in the smoke test)
# ---------------------------------------------------------------------------
def crb_two_tone_std_hz(df, snr_db, rho_db, n_samples, fs):
    """sqrt(CRB) [Hz] for both tone frequencies.  a1 = 1,
    a2 = 10^(-rho/20), sigma2 = a1^2 / 10^(snr/10)."""
    a2 = 10.0 ** (-rho_db / 20.0)
    sigma2 = 10.0 ** (-snr_db / 10.0)
    J = _fisher_tones([0.0, float(df)], n_samples, fs)
    S = np.diag([1.0, a2, 1.0, a2])     # (f1, f2, phi1, phi2) scale by a_k
    Ji = np.linalg.inv(S @ J @ S / sigma2)
    return float(np.sqrt(Ji[0, 0])), float(np.sqrt(Ji[1, 1]))


# ---------------------------------------------------------------------------
# Monte-Carlo driver
# ---------------------------------------------------------------------------
def run_mc(df_grid, n_mc):
    """Returns {(df, snr, rho): cell-stats dict}."""
    N = C.SignalConfig.signal_length
    fs = C.SignalConfig.sample_rate
    t = np.arange(N) / fs
    cells = {}
    for ri, rho in enumerate(RHO_GRID):
        a2 = 10.0 ** (-rho / 20.0)
        for si, snr in enumerate(SNR_GRID):
            sigma2 = 10.0 ** (-snr / 10.0)
            for di, df in enumerate(df_grid):
                t0 = time.time()
                e1 = np.empty(n_mc)
                e2 = np.empty(n_mc)
                for b in range(n_mc):
                    np.random.seed(SEED_BASE + 1_000_000 * ri
                                   + 100_000 * si + 1000 * di + b)
                    ph = np.random.uniform(0, 2 * np.pi, 2)
                    x = (np.exp(1j * (2 * np.pi * 0.0 * t + ph[0]))
                         + a2 * np.exp(1j * (2 * np.pi * df * t + ph[1])))
                    x = x + np.sqrt(sigma2 / 2) * (
                        np.random.randn(N) + 1j * np.random.randn(N))
                    f_lo, f_hi = two_tone_estimate(x, fs)
                    e1[b] = f_lo - 0.0          # tone 1 (strong, f1 = 0)
                    e2[b] = f_hi - df           # tone 2 (f2 = Df)
                c1, c2 = crb_two_tone_std_hz(df, snr, rho, N, fs)
                cells[(float(df), float(snr), float(rho))] = {
                    'n_mc': n_mc,
                    'bias1_hz': float(np.mean(e1)),
                    'var1_hz2': float(np.var(e1)),
                    'rmse1_hz': float(np.sqrt(np.mean(e1 ** 2))),
                    'bias2_hz': float(np.mean(e2)),
                    'var2_hz2': float(np.var(e2)),
                    'rmse2_hz': float(np.sqrt(np.mean(e2 ** 2))),
                    'rmse_pooled_hz': float(np.sqrt(
                        np.mean(np.concatenate([e1, e2]) ** 2))),
                    'std_pooled_hz': float(np.sqrt(
                        0.5 * (np.var(e1) + np.var(e2)))),
                    'fail_prob_gt1hz': float(np.mean(
                        (np.abs(e1) > 1.0) | (np.abs(e2) > 1.0))),
                    'crb1_std_hz': c1,
                    'crb2_std_hz': c2,
                    'crb_rms_hz': float(np.sqrt(0.5 * (c1 ** 2 + c2 ** 2))),
                }
                print(f"  rho={rho:>3.0f} snr={snr:>5.1f} df={df:>6.2f}: "
                      f"rmse={cells[(float(df), float(snr), float(rho))]['rmse_pooled_hz']:.4f} Hz "
                      f"vs crb={cells[(float(df), float(snr), float(rho))]['crb_rms_hz']:.4f} Hz "
                      f"({time.time() - t0:.1f}s)", flush=True)
    return cells


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(cells, df_grid, fig_path):
    N = C.SignalConfig.signal_length
    fs = C.SignalConfig.sample_rate
    df_dense = np.geomspace(df_grid[0], df_grid[-1], 60)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex=True)
    for ci, rho in enumerate(RHO_GRID):
        ax_a, ax_b = axes[0, ci], axes[1, ci]
        for si, snr in enumerate(SNR_GRID):
            col = f'C{si}'
            rmse = [cells[(float(d), float(snr), float(rho))]['rmse_pooled_hz']
                    for d in df_grid]
            b1 = [cells[(float(d), float(snr), float(rho))]['bias1_hz']
                  for d in df_grid]
            b2 = [cells[(float(d), float(snr), float(rho))]['bias2_hz']
                  for d in df_grid]
            crb = [crb_two_tone_std_hz(d, snr, rho, N, fs) for d in df_dense]
            crb_rms = np.sqrt(0.5 * (np.array([c[0] for c in crb]) ** 2
                                     + np.array([c[1] for c in crb]) ** 2))
            ax_a.loglog(df_grid, rmse, 'o-', ms=3, lw=0.9, color=col,
                        label=f'measured RMSE, SNR {snr:g} dB')
            ax_a.loglog(df_dense, crb_rms, '--', lw=0.9, color=col,
                        alpha=0.65,
                        label=f'$\\sqrt{{\\mathrm{{CRB}}}}$, SNR {snr:g} dB')
            ax_b.plot(df_grid, b1, 'o-', ms=3, lw=0.9, color=col,
                      label=f'tone 1 (strong), SNR {snr:g} dB')
            ax_b.plot(df_grid, b2, 's--', ms=2.5, lw=0.8, color=col,
                      alpha=0.65, label=f'tone 2, SNR {snr:g} dB')
        for ax in (ax_a, ax_b):
            ax.axvline(fs / N, color='k', ls=':', lw=0.8)
            ax.grid(True, which='both', alpha=0.3)
            ax.tick_params(labelsize=6)
        ax_a.set_ylim(1e-3, 30)
        ax_b.axhline(0.0, color='k', lw=0.5)
        ax_b.set_yscale('symlog', linthresh=1e-2)
        ax_b.set_xlabel(r'$|\Delta f|$ (Hz)', fontsize=7)
        ax_b.set_ylim(-20, 20)
    axes[0, 0].set_ylabel('frequency error (Hz)', fontsize=7)
    axes[1, 0].set_ylabel('bias (Hz, symlog)', fontsize=7)
    axes[0, 0].annotate(r'$1/T$', xy=(fs / N, 1e-3),
                        xytext=(fs / N * 1.15, 2e-3), fontsize=6)
    axes[1, 0].annotate(r'$1/T$', xy=(fs / N, 0),
                        xytext=(fs / N * 1.15, 0.03), fontsize=6)
    axes[0, 0].set_title('(a) RMSE vs $\\sqrt{\\mathrm{CRB}}$, '
                         f'amplitude ratio {RHO_GRID[0]:g} dB', fontsize=7)
    axes[0, 1].set_title(f'(a) amplitude ratio {RHO_GRID[1]:g} dB',
                         fontsize=7)
    axes[1, 0].set_title(f'(b) per-tone bias, '
                         f'amplitude ratio {RHO_GRID[0]:g} dB', fontsize=7)
    axes[1, 1].set_title(f'(b) amplitude ratio {RHO_GRID[1]:g} dB',
                         fontsize=7)
    axes[0, 0].legend(fontsize=4.8, loc='lower left')
    axes[1, 0].legend(fontsize=4.8, loc='lower left')
    fig.tight_layout(pad=0.4)
    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    fig.savefig(os.path.splitext(fig_path)[0] + '.png', dpi=200)
    plt.close(fig)
    print(f"saved {fig_path} (+ .png)", flush=True)


def make_figure_wide(cells, df_grid, fig_path):
    """Compact 1x3 textwidth variant (~6.9 x 1.9 in) for the page budget:
    (a) pooled RMSE vs sqrt(CRB) at rho = 0 dB, (b) same at rho = 6 dB,
    (c) per-tone bias at rho = 0 dB (symlog).  Same data as make_figure."""
    N = C.SignalConfig.signal_length
    fs = C.SignalConfig.sample_rate
    df_dense = np.geomspace(df_grid[0], df_grid[-1], 60)
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.9))
    for ci, rho in enumerate(RHO_GRID):
        ax = axes[ci]
        for si, snr in enumerate(SNR_GRID):
            col = f'C{si}'
            rmse = [cells[(float(d), float(snr), float(rho))]['rmse_pooled_hz']
                    for d in df_grid]
            crb = [crb_two_tone_std_hz(d, snr, rho, N, fs) for d in df_dense]
            crb_rms = np.sqrt(0.5 * (np.array([c[0] for c in crb]) ** 2
                                     + np.array([c[1] for c in crb]) ** 2))
            ax.loglog(df_grid, rmse, 'o-', ms=2.5, lw=0.8, color=col,
                      label=f'measured, SNR {snr:g} dB')
            ax.loglog(df_dense, crb_rms, '--', lw=0.7, color=col, alpha=0.65,
                      label=f'$\\sqrt{{\\mathrm{{CRB}}}}$, {snr:g} dB')
        ax.set_ylim(1e-3, 30)
        ax.set_title(f'({"ab"[ci]}) RMSE vs $\\sqrt{{\\mathrm{{CRB}}}}$, '
                     f'$\\rho$ = {rho:g} dB', fontsize=6.5)
        ax.set_xlabel(r'$|\Delta f|$ (Hz)', fontsize=6)
    ax = axes[2]
    rho = RHO_GRID[0]
    for si, snr in enumerate(SNR_GRID):
        col = f'C{si}'
        b1 = [cells[(float(d), float(snr), float(rho))]['bias1_hz']
              for d in df_grid]
        b2 = [cells[(float(d), float(snr), float(rho))]['bias2_hz']
              for d in df_grid]
        ax.plot(df_grid, b1, 'o-', ms=2.5, lw=0.8, color=col,
                label=f'tone 1, SNR {snr:g} dB')
        ax.plot(df_grid, b2, 's--', ms=2, lw=0.7, color=col, alpha=0.65,
                label=f'tone 2, {snr:g} dB')
    ax.axhline(0.0, color='k', lw=0.5)
    ax.set_yscale('symlog', linthresh=1e-2)
    ax.set_ylim(-20, 20)
    ax.set_title(f'(c) per-tone bias, $\\rho$ = {rho:g} dB', fontsize=6.5)
    ax.set_xlabel(r'$|\Delta f|$ (Hz)', fontsize=6)
    for ax in axes:
        ax.axvline(fs / N, color='k', ls=':', lw=0.7)
        ax.grid(True, which='both', alpha=0.3)
        ax.tick_params(labelsize=5.5)
        ax.set_xscale('log')
    axes[0].set_ylabel('frequency error (Hz)', fontsize=6)
    axes[2].set_ylabel('bias (Hz, symlog)', fontsize=6)
    axes[0].annotate(r'$1/T$', xy=(fs / N, 1e-3),
                     xytext=(fs / N * 1.2, 2e-3), fontsize=5.5)
    axes[0].legend(fontsize=4.2, loc='lower left', ncol=2, columnspacing=0.6,
                   handlelength=1.4, labelspacing=0.25)
    axes[2].legend(fontsize=4.2, loc='lower left', ncol=2, columnspacing=0.6,
                   handlelength=1.4, labelspacing=0.25)
    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    fig.savefig(os.path.splitext(fig_path)[0] + '.png', dpi=200)
    plt.close(fig)
    print(f"saved {fig_path} (+ .png)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description='Paper 6 review-3 E-A: two-tone frequency estimation '
                    'vs the Prop-3 CRB')
    p.add_argument('--smoke', action='store_true',
                   help='3-point Df grid x 20 MC bursts (fast self-check)')
    p.add_argument('--n_mc', type=int, default=200,
                   help='Monte-Carlo bursts per cell (full mode)')
    p.add_argument('--out', type=str, default=os.path.join(
        C.RESULTS_DIR, 'eval_twotone_crb.json'))
    p.add_argument('--fig', type=str, default=os.path.join(
        FIG_DIR, 'fig_twotone_crb.pdf'))
    p.add_argument('--wide', action='store_true',
                   help='also write the compact 1x3 textwidth figure '
                        '(--fig_wide, default figures/fig_twotone_crb_wide.pdf)')
    p.add_argument('--fig_wide', type=str, default=os.path.join(
        FIG_DIR, 'fig_twotone_crb_wide.pdf'))
    p.add_argument('--from_json', type=str, default=None,
                   help='load cells from an existing results JSON instead of '
                        're-running the Monte Carlo (figure regeneration)')
    args = p.parse_args()

    N = C.SignalConfig.signal_length
    fs = C.SignalConfig.sample_rate

    if args.from_json:
        data = json.load(open(args.from_json))
        df_grid = np.array(data['config']['df_grid_hz'])
        cells = {(c['df_hz'], c['snr_db'], c['rho_db']): c
                 for c in data['cells']}
        print(f"loaded {len(cells)} cells from {args.from_json}", flush=True)
    else:
        n_mc = 20 if args.smoke else args.n_mc
        df_grid = (np.array([0.5, 4.0, 16.0]) if args.smoke else DF_GRID)
        print(f"E-A two-tone CRB validation: {len(df_grid)} Df x "
              f"{len(SNR_GRID)} SNR x {len(RHO_GRID)} rho x {n_mc} MC "
              f"(N={N}, fs={fs}, 1/T={fs / N:.3f} Hz)", flush=True)

        # --- sanity: the amplitude-scaled Fisher reduces to Prop 3's curves
        for df in (1.0, fs / N, 16.0):
            for snr in (0.0, 20.0):
                c1, c2 = crb_two_tone_std_hz(df, snr, 0.0, N, fs)
                ref = float(np.sqrt(
                    two_tone_inflation([df], N, fs)[0]
                    * crb_single_tone_closed(snr, N, fs)))
                assert abs(c1 - ref) / ref < 1e-9 and abs(c2 - ref) / ref < 1e-9, \
                    (df, snr, c1, c2, ref)
        print("sanity: scaled-Fisher CRB == two_tone_inflation x closed form "
              "at rho=0 dB: OK", flush=True)

        t0 = time.time()
        cells = run_mc(df_grid, n_mc)
        print(f"MC done in {time.time() - t0:.0f}s", flush=True)

        # --- smoke-mode estimator sanity ---------------------------------
        if args.smoke:
            c_lo = cells[(float(df_grid[-1]), 20.0, 0.0)]
            ratio = c_lo['rmse_pooled_hz'] / c_lo['crb_rms_hz']
            assert ratio < 3.0, f"estimator far from CRB at Df=16 Hz: {ratio:.2f}x"
            c_hi = cells[(float(df_grid[0]), 20.0, 0.0)]
            assert c_hi['rmse_pooled_hz'] > 3.0 * c_hi['crb_rms_hz'], \
                "no threshold behaviour at Df=0.5 Hz?"
            print(f"smoke sanity: Df=16 Hz RMSE/CRB = {ratio:.2f}x (<3 OK); "
                  f"Df=0.5 Hz RMSE = {c_hi['rmse_pooled_hz']:.3f} Hz >> CRB "
                  f"{c_hi['crb_rms_hz']:.4f} Hz: OK", flush=True)

        out = {
            'config': {
                'n_samples': N, 'fs': fs, 't_burst_s': N / fs,
                'one_over_T_hz': fs / N,
                'df_grid_hz': [float(d) for d in df_grid],
                'snr_grid_db': SNR_GRID, 'rho_grid_db': RHO_GRID,
                'n_mc': n_mc, 'smoke': bool(args.smoke),
                'seed_base': SEED_BASE, 'search_win_hz': SEARCH_WIN,
                'snr_convention': 'per-strong-tone: sigma2 = a1^2/10^(snr/10)',
                'estimator': 'two-tone CLEAN/zoom spectral-line estimator '
                             '(sync._fft_line first peak + projection '
                             'subtraction + Gauss-Seidel golden-section '
                             'periodogram refinement); see module docstring',
            },
            'cells': [{'df_hz': k[0], 'snr_db': k[1], 'rho_db': k[2], **v}
                      for k, v in sorted(cells.items())],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"saved {args.out}", flush=True)

    make_figure(cells, df_grid, args.fig)
    if args.wide:
        make_figure_wide(cells, df_grid, args.fig_wide)

    # --- headline numbers (for the experiment log) ------------------------
    print("\nheadline (rho=0 dB):")
    for snr in SNR_GRID:
        row = []
        for d in (1.0, fs / N, 16.0):
            j = int(np.argmin(np.abs(df_grid - d)))
            c = cells[(float(df_grid[j]), float(snr), 0.0)]
            row.append(f"Df={df_grid[j]:.2f}: {c['rmse_pooled_hz']:.4f}/"
                       f"{c['crb_rms_hz']:.4f} Hz "
                       f"({c['rmse_pooled_hz'] / c['crb_rms_hz']:.1f}x)")
        print(f"  SNR {snr:>5.1f} dB: " + " | ".join(row), flush=True)
    print("\neval_twotone_crb done.", flush=True)


if __name__ == '__main__':
    main()
