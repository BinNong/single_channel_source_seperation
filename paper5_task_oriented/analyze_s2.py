"""Paper 5 — S2 main-matrix analysis: paired stats + Pareto figure.

Inputs: results/s2/eval_*.json (20 main runs: 4 configs x 5 seeds + 1
pureser probe), results/ser/baseline_ser_comp.json.

Outputs:
  results/s2/summary_s2.json          — all numbers behind the tables
  figures/s2_pareto.{png,pdf}         — SI-SDRi vs BER/SER per config
                                        (5-seed mean ± std, baseline ref)

Significance: paired-by-seed differences AND cell-level pairing
(7 SNR cells x 5 seeds = 35 paired observations), bootstrap 95% CI
(10k resamples, seed 0).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results', 's2')
BASELINE = os.path.join(HERE, 'results', 'ser', 'baseline_ser_comp.json')
FIGS = os.path.join(HERE, 'figures')
SNRS = [-10, -5, 0, 5, 10, 15, 20]
CONFIGS = ['sisdr', 'mse', 'ser', 'ser_mse', 'pureser']


def cfg_of(name: str) -> str:
    for c in ('ser_mse', 'pureser', 'sisdr', 'mse', 'ser'):
        if f'_{c}_' in name:
            return c
    raise ValueError(name)


def load():
    runs = {c: {} for c in CONFIGS}
    for f in sorted(glob.glob(os.path.join(RES, 'eval_*.json'))):
        n = os.path.basename(f)[5:-5]
        seed = int(n.split('_s')[-1])
        runs[cfg_of(n)][seed] = json.load(open(f))
    return runs


def paired_boot_ci(a, b, n_boot=10000, seed=0):
    """95% CI of mean(a - b) by paired bootstrap.  a, b: equal-length."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), \
        float(np.percentile(boots, 97.5))


def paired_permutation_p(a, b):
    """EXACT two-sided paired permutation (sign-flip) test of mean(a-b)=0.

    Enumerates all 2^n sign assignments (n=5 seeds -> 32, exact).  More
    defensible than a bootstrap CI at this sample size: the percentile
    bootstrap CI on 5 points is coarse, and one S2 comparison's lower
    bound sits at +0.00011 (reviewer bait).
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = abs(d.mean())
    n = len(d)
    count = 0
    for signs in range(1 << n):
        s = np.array([(signs >> i) & 1 for i in range(n)]) * 2 - 1
        if abs((d * s).mean()) >= obs - 1e-15:
            count += 1
    return count / (1 << n)


def main():
    runs = load()
    base = json.load(open(BASELINE))
    out = {'baseline': {'ser': base['overall'], 'ber': base['overall_ber']}}

    # ---------------- headline table ----------------
    print(f"{'cfg':>8s} {'n':>2s} {'SI-SDRi':>14s} {'SER':>14s} {'BER':>14s} "
          f"{'count_acc':>14s} {'halluc':>12s}")
    for c in CONFIGS:
        subs = runs[c]
        if not subs:
            continue
        g = lambda m: [s['separation'][m] for s in subs.values()]
        cnt = [s['counting']['occupancy']['overall_acc']
               for s in subs.values()]
        line = (f"{c:>8s} {len(subs):>2d} "
                f"{np.mean(g('si_sdri')):+7.3f}±{np.std(g('si_sdri')):.3f} "
                f"{np.mean(g('ser_comp')):7.4f}±{np.std(g('ser_comp')):.4f} "
                f"{np.mean(g('ber_comp')):7.4f}±{np.std(g('ber_comp')):.4f} "
                f"{np.mean(cnt):7.4f}±{np.std(cnt):.4f} "
                f"{np.mean(g('halluc_rate')):6.4f}±{np.std(g('halluc_rate')):.4f}")
        print(line)
        out[c] = {
            'n': len(subs),
            'si_sdri': {'mean': float(np.mean(g('si_sdri'))),
                        'std': float(np.std(g('si_sdri')))},
            'ser': {'mean': float(np.mean(g('ser_comp'))),
                    'std': float(np.std(g('ser_comp')))},
            'ber': {'mean': float(np.mean(g('ber_comp'))),
                    'std': float(np.std(g('ber_comp')))},
            'count_acc': {'mean': float(np.mean(cnt)),
                          'std': float(np.std(cnt))},
            'halluc_rate': {'mean': float(np.mean(g('halluc_rate'))),
                            'std': float(np.std(g('halluc_rate')))},
        }

    # ---------------- per-K / per-mod cuts ----------------
    print('\n--- per-K SER ---')
    for c in CONFIGS:
        subs = runs[c]
        if not subs:
            continue
        line = f'{c:>8s}: '
        for k in ('1', '2', '3'):
            v = [s['separation']['per_k'][k]['ser_comp']
                 for s in subs.values()]
            line += f'K={k} {np.mean(v):.4f}  '
        print(line)
        out[c]['per_k_ser'] = {k: float(np.mean(
            [s['separation']['per_k'][k]['ser_comp']
             for s in subs.values()])) for k in ('1', '2', '3')}

    # ---------------- paired significance ----------------
    print('\n--- paired bootstrap (ser-family vs baselines) ---')
    pairs = [('ser', 'sisdr'), ('ser', 'mse'), ('ser_mse', 'mse'),
             ('ser_mse', 'sisdr')]
    out['paired'] = {}
    for a, b in pairs:
        seeds = sorted(set(runs[a]) & set(runs[b]))
        if len(seeds) < 2:
            continue
        for metric, key in (('SER', 'ser_comp'), ('BER', 'ber_comp')):
            va = [runs[a][s]['separation'][key] for s in seeds]
            vb = [runs[b][s]['separation'][key] for s in seeds]
            m, lo, hi = paired_boot_ci(vb, va)   # baseline minus treatment
            pval = paired_permutation_p(vb, va)
            print(f'  {a:>8s} vs {b:<6s} {metric}: Δ={m:+.5f} '
                  f'95%CI=[{lo:+.5f},{hi:+.5f}]  perm_p={pval:.4f}  '
                  f'(seed-paired, n={len(seeds)}; positive = {a} better)')
            out['paired'][f'{a}_vs_{b}_{key}'] = {
                'delta': m, 'ci95': [lo, hi], 'perm_p': pval}
        # cell-level pairing (7 SNR cells x shared seeds)
        for metric, key in (('SER', 'ser_comp'), ('BER', 'ber_comp')):
            ca, cb = [], []
            for s in seeds:
                for snr in SNRS:
                    ca.append(runs[a][s]['separation']['per_snr'][str(snr)][key])
                    cb.append(runs[b][s]['separation']['per_snr'][str(snr)][key])
            m, lo, hi = paired_boot_ci(cb, ca)
            print(f'  {a:>8s} vs {b:<6s} {metric}: Δ={m:+.5f} '
                  f'95%CI=[{lo:+.5f},{hi:+.5f}]  (cell-paired, n={len(ca)})')
            out['paired'][f'{a}_vs_{b}_{key}_cells'] = {'delta': m,
                                                        'ci95': [lo, hi]}

    with open(os.path.join(RES, 'summary_s2.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # ---------------- Pareto figure ----------------
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    colors = {'sisdr': 'tab:gray', 'mse': 'tab:blue', 'ser': 'tab:red',
              'ser_mse': 'tab:purple', 'pureser': 'tab:green'}
    for ax, key, ylab in ((axes[0], 'ser_comp', 'SER (compensated)'),
                          (axes[1], 'ber_comp', 'BER (compensated, Gray)')):
        ybase = base['overall'] if key == 'ser_comp' else base['overall_ber']
        zoom_pts = []          # (mean_x, mean_y, std_x, std_y), trained only
        for c in CONFIGS:
            subs = runs[c]
            if not subs:
                continue
            x = [s['separation']['si_sdri'] for s in subs.values()]
            y = [s['separation'][key] for s in subs.values()]
            ax.errorbar(np.mean(x), np.mean(y), xerr=np.std(x),
                        yerr=np.std(y), fmt='o', color=colors[c],
                        capsize=3, label=f'{c} (n={len(subs)})')
            if c != 'pureser':
                zoom_pts.append((np.mean(x), np.mean(y),
                                 np.std(x), np.std(y)))
        ax.axhline(ybase, color='crimson', ls='--', lw=1.2,
                   label='mixture baseline')
        ax.set_xlabel('SI-SDRi (dB)')
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        # zoom inset on the four trained configs (pureser outlier excluded)
        zp = np.array(zoom_pts)
        x_lo = (zp[:, 0] - zp[:, 2]).min() - 0.35
        x_hi = (zp[:, 0] + zp[:, 2]).max() + 0.35
        y_lo = min((zp[:, 1] - zp[:, 3]).min(), ybase)
        y_hi = max((zp[:, 1] + zp[:, 3]).max(), ybase)
        y_pad = 0.3 * (y_hi - y_lo)
        axins = inset_axes(ax, width='44%', height='44%',
                           loc='center left', borderpad=2.4)
        for c in CONFIGS:
            subs = runs[c]
            if not subs or c == 'pureser':
                continue
            x = [s['separation']['si_sdri'] for s in subs.values()]
            y = [s['separation'][key] for s in subs.values()]
            axins.errorbar(np.mean(x), np.mean(y), xerr=np.std(x),
                           yerr=np.std(y), fmt='o', color=colors[c],
                           capsize=2, ms=4, lw=1)
        axins.axhline(ybase, color='crimson', ls='--', lw=1)
        axins.set_xlim(x_lo, x_hi)
        axins.set_ylim(y_lo - y_pad, y_hi + y_pad)
        axins.grid(alpha=0.3)
        axins.tick_params(labelsize=7)
        mark_inset(ax, axins, loc1=1, loc2=3, fc='none', ec='0.4', ls=':')
    axes[0].set_title('waveform quality vs task metric (5-seed mean ± std)')
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGS, f's2_pareto.{ext}'), dpi=200)
    plt.close(fig)
    print(f"\nSaved figures/s2_pareto.* and {RES}/summary_s2.json")


if __name__ == '__main__':
    main()
