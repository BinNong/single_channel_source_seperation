"""Paper 6 — robustness sweeps figure (review 2, §14).

Four panels from results/robustness_{freq,tau,amp}.json (K=2,
QPSK+QPSK, 100 bursts per cell, deterministic):
  (a) SER vs carrier separation |Df| (V4 at 0/10/20 dB; V3 oracle-
      frequency reference at 20 dB);
  (b) V4 frequency RMSE (lines) and acquisition probability
      P(both |df_err| < 1 Hz) (markers) vs |Df|;
  (c) SER vs timing-grid offset tau (0/0.1/0.25/0.5 T_s);
  (d) SER vs amplitude ratio rho = 20log10(w2/w1).
Output: figures/fig_e6_robustness.{pdf,png}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config as C  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / 'figures'
SNRS = [0.0, 10.0, 20.0]
COLORS = {0.0: 'tab:blue', 10.0: 'tab:orange', 20.0: 'tab:green'}
MARKERS = {0.0: 'o', 10.0: 's', 20.0: '^'}


def _load(name):
    with open(Path(C.RESULTS_DIR) / name) as f:
        return json.load(f)['records']


def _agg(recs, value, snr, variant, key='ser'):
    rs = [r for r in recs if r['value'] == value and r['snr'] == snr
          and r['variant'] == variant]
    return float(np.mean([r[key] for r in rs])) if rs else float('nan')


def main():
    freq = _load('robustness_freq.json')
    tau = _load('robustness_tau.json')
    amp = _load('robustness_amp.json')
    dfs = sorted({r['value'] for r in freq})
    taus = sorted({r['value'] for r in tau})
    rhos = sorted({r['value'] for r in amp})

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.05))

    ax = axes[0]
    for s in SNRS:
        ax.plot(dfs, [_agg(freq, d, s, 'V4') for d in dfs],
                f'-{MARKERS[s]}', ms=3, lw=0.9, color=COLORS[s],
                label=f'V4, {s:g} dB')
    ax.plot(dfs, [_agg(freq, d, 20.0, 'V3') for d in dfs], 'k--', lw=0.9,
            label='V3 (oracle freq), 20 dB')
    ax.axvline(3.906, color='0.5', ls=':', lw=0.8)
    ax.text(3.906, 0.02, '1/T ', fontsize=5.5, color='0.4',
            va='bottom', ha='right')
    ax.set_xlabel(r'$|\Delta f|$ (Hz)', fontsize=7)
    ax.set_ylabel('SER', fontsize=7)
    ax.set_title('(a) carrier separation', fontsize=7)
    ax.legend(fontsize=5.2, loc='upper right')
    ax.grid(alpha=0.3)

    ax = axes[1]
    for s in SNRS:
        rmses = []
        acqs = []
        for d in dfs:
            rs = [r for r in freq if r['value'] == d and r['snr'] == s
                  and r['variant'] == 'V4']
            fe = np.concatenate([[r['df1_err'], r['df2_err']] for r in rs])
            rmses.append(float(np.sqrt(np.mean(fe ** 2))))
            acqs.append(float(np.mean([(r['df1_err'] < 1)
                                       and (r['df2_err'] < 1) for r in rs])))
        ax.plot(dfs, rmses, f'-{MARKERS[s]}', ms=3, lw=0.9,
                color=COLORS[s], label=f'RMSE, {s:g} dB')
        ax.plot(dfs, acqs, f':{MARKERS[s]}', ms=3, lw=0.8, color=COLORS[s],
                alpha=0.55)
    ax.set_xlabel(r'$|\Delta f|$ (Hz)', fontsize=7)
    ax.set_ylabel(r'freq RMSE (Hz) solid / acq. prob. dotted', fontsize=6)
    ax.set_title('(b) V4 frequency acquisition', fontsize=7)
    ax.legend(fontsize=5.2, loc='upper right')
    ax.grid(alpha=0.3)

    ax = axes[2]
    for s in SNRS:
        ax.plot(np.array(taus) / 16.0, [_agg(tau, t, s, 'V4') for t in taus],
                f'-{MARKERS[s]}', ms=3, lw=0.9, color=COLORS[s],
                label=f'{s:g} dB')
    ax.set_xlabel(r'timing offset $\tau$ ($T_s$)', fontsize=7)
    ax.set_ylabel('SER (V4)', fontsize=7)
    ax.set_title('(c) timing-grid sensitivity', fontsize=7)
    ax.legend(fontsize=5.2)
    ax.grid(alpha=0.3)

    ax = axes[3]
    for s in SNRS:
        ax.plot(rhos, [_agg(amp, r, s, 'V4') for r in rhos],
                f'-{MARKERS[s]}', ms=3, lw=0.9, color=COLORS[s],
                label=f'{s:g} dB')
    ax.set_xlabel(r'amplitude ratio $\rho$ (dB)', fontsize=7)
    ax.set_ylabel('SER (V4)', fontsize=7)
    ax.set_title('(d) amplitude-ratio sensitivity', fontsize=7)
    ax.legend(fontsize=5.2)
    ax.grid(alpha=0.3)

    for ax in axes:
        ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'fig_e6_robustness.{ext}', dpi=200)
    print(f"saved {FIG_DIR / 'fig_e6_robustness.pdf'}")


if __name__ == '__main__':
    main()
