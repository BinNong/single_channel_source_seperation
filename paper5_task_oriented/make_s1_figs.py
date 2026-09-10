"""Paper 5 — S1 mismatch quantification figures + summary (motivation section).

Inputs (all local, synced from the server):
  results/s1_mismatch/eval_*_mse_s4[2-6].json   — 15 paper4 checkpoints
      re-scored with the compensated receiver (SER + Gray BER), n=100/cell
  results/ser/baseline_ser_comp.json            — mixture baseline (n=100)

Outputs:
  figures/s1_sisdr_vs_ser_ber.{png,pdf}  — scatter: SI-SDRi vs SER / BER,
      one point per (run, SNR) cell; per-cell Pearson r across runs.
  figures/s1_flat_region.{png,pdf}       — per-SNR spread bands: within a
      cell, SI-SDRi spreads by dB while SER stays flat (the "flat region").
  results/s1_mismatch/summary.json       — all numbers behind the figures.

NOTE (honesty, inherited from paper4): recursive's SER/BER is computed on
Hungarian-matched pairs only and it misses 37% of sources — selection bias;
its numbers are reported but not headline.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results', 's1_mismatch')
BASELINE = os.path.join(HERE, 'results', 'ser', 'baseline_ser_comp.json')
FIGS = os.path.join(HERE, 'figures')
SNRS = [-10, -5, 0, 5, 10, 15, 20]
KS = [1, 2, 3]
FLOOR = {'BPSK': 0.000, 'QPSK': 0.000, '8PSK': 0.014, '16QAM': 0.032}
ARCH_MARK = {'slot': 'o', 'specialist': 's', 'recursive': '^'}


def load_runs():
    runs = []
    for f in sorted(glob.glob(os.path.join(RES, 'eval_*.json'))):
        name = os.path.basename(f)[len('eval_'):-len('.json')]
        arch = name.split('_')[0]
        seed = int(re.search(r'_s(\d+)$', name).group(1))
        with open(f) as fh:
            sep = json.load(fh)['separation']
        runs.append({'arch': arch, 'seed': seed, 'sep': sep})
    assert len(runs) == 15, f"expect 15 runs, got {len(runs)}"
    return runs


def cell_vals(run, metric):
    """Per-(SNR, K) cell values of `metric`, with the cell's SI-SDRi."""
    out = []
    sep = run['sep']
    for s in SNRS:
        b = sep['per_snr'].get(str(s))
        if b and metric in b:
            out.append((('snr', s), b['si_sdri'], b[metric]))
    for k in KS:
        b = sep['per_k'].get(str(k))
        if b and metric in b:
            out.append((('k', k), b['si_sdri'], b[metric]))
    return out


def main():
    os.makedirs(FIGS, exist_ok=True)
    runs = load_runs()
    with open(BASELINE) as fh:
        base = json.load(fh)

    # ---------------- summary table ----------------
    summary = {'baseline': {
        'ser': base['overall'], 'ber': base['overall_ber'],
        'per_mod_ser': base['per_mod'], 'per_mod_ber': base['per_mod_ber'],
        'per_snr_ser': base['per_snr'], 'per_snr_ber': base['per_snr_ber'],
        'n_per_cell': base['n_per_cell']}}
    for arch in ('slot', 'specialist', 'recursive'):
        sub = [r for r in runs if r['arch'] == arch]
        blk = {}
        for m in ('si_sdri', 'ser_comp', 'ber_comp'):
            v = [r['sep'][m] for r in sub]
            blk[m] = {'mean': float(np.mean(v)), 'std': float(np.std(v))}
        # per-modulation SER vs clean-source floor (slot is the headline)
        blk['ser_per_mod'] = {
            mod: float(np.mean([r['sep']['ser_comp_per_mod'][mod]
                                for r in sub if mod in r['sep']['ser_comp_per_mod']]))
            for mod in FLOOR}
        summary[arch] = blk
    summary['clean_source_floor_ser'] = FLOOR

    # per-cell Pearson r between SI-SDRi and SER across runs, reported for
    # THREE subsets: slot only (headline arch), slot+specialist (honest
    # core), and all 15 runs (includes recursive, whose 37% miss rate makes
    # its matched-pair SER a SELECTION-biased figure — it single-handedly
    # creates the strong negative r at high SNR in the all-15 analysis).
    cell_r = {}
    for subset_name, subset in (('slot', ('slot',)),
                                ('slot+specialist', ('slot', 'specialist')),
                                ('all15', ('slot', 'specialist', 'recursive'))):
        sub_runs = [r for r in runs if r['arch'] in subset]
        for axis, key in [(('snr', s), s) for s in SNRS] + [(('k', k), k) for k in KS]:
            bucket = 'per_snr' if axis[0] == 'snr' else 'per_k'
            pts = [(r['sep'][bucket][str(key)]['si_sdri'],
                    r['sep'][bucket][str(key)]['ser_comp'])
                   for r in sub_runs
                   if r['sep'][bucket][str(key)].get('ser_comp') is not None]
            x = np.array([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            cell_r[f"{subset_name}:{axis[0]}={key}"] = {
                'r_si_ser': float(np.corrcoef(x, y)[0, 1]),
                'n_runs': len(pts),
                'si_sdri_spread_db': float(x.max() - x.min()),
                'ser_spread': float(y.max() - y.min())}
    summary['per_cell_correlation'] = cell_r

    with open(os.path.join(RES, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ---------------- console table ----------------
    print(f"baseline (mixture, no separation): SER={base['overall']:.4f}  "
          f"BER={base['overall_ber']:.4f}")
    for arch in ('slot', 'specialist', 'recursive'):
        b = summary[arch]
        print(f"{arch:>11s}: SI-SDRi {b['si_sdri']['mean']:+.3f}±{b['si_sdri']['std']:.3f} dB  "
              f"SER {b['ser_comp']['mean']:.4f}±{b['ser_comp']['std']:.4f}  "
              f"BER {b['ber_comp']['mean']:.4f}±{b['ber_comp']['std']:.4f}")
    print("\nper-cell correlation r(SI-SDRi, SER) across runs:")
    for c, v in cell_r.items():
        if not c.startswith(('slot:', 'slot+specialist:snr')):
            continue        # console shows the honest core; all in summary.json
        print(f"  {c:>26s}: r={v['r_si_ser']:+.3f}  n={v['n_runs']}  "
              f"SI-SDRi spread {v['si_sdri_spread_db']:.2f} dB   "
              f"SER spread {v['ser_spread']:.4f}")

    # ---------------- figure 1: scatter ----------------
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(min(SNRS), max(SNRS))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for ax, metric, ylab in ((axes[0], 'ser_comp', 'SER (compensated)'),
                             (axes[1], 'ber_comp', 'BER (compensated, Gray)')):
        for run in runs:
            for (axis, val), si, y in cell_vals(run, metric):
                if axis != 'snr':
                    continue          # scatter over per-SNR cells only
                ax.scatter(si, y, color=cmap(norm(val)),
                           marker=ARCH_MARK[run['arch']], s=18, alpha=0.75,
                           edgecolors='none')
        ax.axhline(base['overall'] if metric == 'ser_comp'
                   else base['overall_ber'],
                   color='crimson', ls='--', lw=1.2,
                   label='mixture baseline (no separation)')
        ax.set_xlabel('SI-SDRi (dB)')
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=axes, label='SNR (dB)')
    fig.suptitle('SI-SDR vs demodulation metrics are decoupled '
                 '(15 runs x 7 SNR cells; markers = arch)')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGS, f's1_sisdr_vs_ser_ber.{ext}'), dpi=200)
    plt.close(fig)

    # ---------------- figure 2: flat region bands ----------------
    # Bands over the HONEST core (slot+specialist, 10 runs); recursive shown
    # as its own dashed mean because its matched-pair SER is selection-biased
    # (37% miss rate).
    honest = [r for r in runs if r['arch'] in ('slot', 'specialist')]
    rec = [r for r in runs if r['arch'] == 'recursive']
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True)
    def band(ax, ys, **kw):
        lo = [min(y) for y in ys]
        hi = [max(y) for y in ys]
        mu = [float(np.mean(y)) for y in ys]
        ax.fill_between(SNRS, lo, hi, alpha=0.35, **kw)
        ax.plot(SNRS, mu, marker='o', ms=3.5, **kw)
    si = [[r['sep']['per_snr'][str(s)]['si_sdri'] for r in honest] for s in SNRS]
    se = [[r['sep']['per_snr'][str(s)]['ser_comp'] for r in honest] for s in SNRS]
    be = [[r['sep']['per_snr'][str(s)]['ber_comp'] for r in honest] for s in SNRS]
    band(axes[0], si, color='tab:blue')
    axes[0].set_ylabel('SI-SDRi (dB)')
    axes[0].set_title('within-cell spread over 10 slot/specialist runs '
                      '(band = min..max)')
    axes[0].grid(alpha=0.3)
    band(axes[1], se, color='tab:orange')
    band(axes[1], be, color='tab:green')
    axes[1].plot(SNRS, [float(np.mean([r['sep']['per_snr'][str(s)]['ser_comp']
                                       for r in rec])) for s in SNRS],
                 ls='-.', color='tab:orange', lw=1.1,
                 label='recursive SER (selection-biased)')
    axes[1].plot(SNRS, [base['per_snr'][str(float(s))] for s in SNRS],
                 ls='--', color='crimson', lw=1.2, label='mixture SER')
    axes[1].plot(SNRS, [base['per_snr_ber'][str(float(s))] for s in SNRS],
                 ls=':', color='crimson', lw=1.2, label='mixture BER')
    axes[1].set_ylabel('SER / BER')
    axes[1].set_xlabel('SNR (dB)')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGS, f's1_flat_region.{ext}'), dpi=200)
    plt.close(fig)

    print(f"\nSaved figures to {FIGS} and summary to {RES}/summary.json")


if __name__ == '__main__':
    main()
