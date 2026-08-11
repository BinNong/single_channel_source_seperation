#!/usr/bin/env python3
"""Regenerate paper figures 01/06/07 and micro_freq_eval as vector PDFs.

Data sources (all local after fetching from the training machine):
- results/phase5_results/<tag>_s<seed>/*_overall.json   (5-seed / 3-seed runs)
- results/pub_2026/per_mod_fixed/<tag>_s<seed>/*_overall.json (original 3-seed runs)
- results/phase_results/microfreq_complex_cnn_{se,no_se}/*.json (micro-frequency eval)

The previous PNGs in paper/figures/ were stale (3-seed numbers under 5-seed
captions for 01/07) or had a corrupted annotation (06), so they are rebuilt
here directly from the per-seed JSON metrics.
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_BASE, '..')
OUT = os.path.join(ROOT, 'results', 'charts')
PH5 = os.path.join(ROOT, 'results', 'phase5_results')
ORIG = os.path.join(ROOT, 'results', 'pub_2026', 'per_mod_fixed')
MICRO = os.path.join(ROOT, 'results', 'phase_results')
os.makedirs(OUT, exist_ok=True)

METRICS = ("SI-SDR", "SDR", "SIR", "NMSE")


def load_group(src_dir, tag, seeds):
    """Return {metric: (mean, sample_std)} over the given seeds."""
    per_seed = {m: [] for m in METRICS}
    for s in seeds:
        files = glob.glob(os.path.join(src_dir, f'{tag}_s{s}', '*_overall.json'))
        assert len(files) == 1, f'{tag}_s{s}: found {files}'
        d = json.load(open(files[0]))
        for m in METRICS:
            per_seed[m].append(d[m])
    return {m: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
            for m, v in per_seed.items()}


# key -> (label, params, metrics)
MODELS = {
    "proposed":    ("Complex CNN + SE (Proposed)", 235_525,
                    load_group(PH5, 'cse', (42, 43, 44, 45, 46))),
    "no_se_match": ("Complex CNN no-SE (matched)", 242_525,
                    load_group(PH5, 'pm_no_se', (42, 43, 44, 45, 46))),
    "real_match":  ("Real-Valued CNN (matched)", 236_885,
                    load_group(PH5, 'pm_real', (42, 43, 44, 45, 46))),
    "tasnet":      ("Complex Conv-TasNet", 817_000,
                    load_group(PH5, 'ctasnet', (42, 43, 44))),
    "cnse":        ("CNSE (scaled)", 6_690_000,
                    load_group(PH5, 'cnse', (42, 43, 44))),
    "s4unet":      ("S4-UNET (scaled)", 1_570_000,
                    load_group(PH5, 's4unet', (42, 43, 44))),
    "no_se_orig":  ("Complex CNN no-SE (orig.)", 203_000,
                    # Local per_mod_fixed JSONs come from a different eval than
                    # the paper's orig rows (means match, stds do not), so keep
                    # the values printed in Table 1 of the paper.
                    {"SI-SDR": (-1.13, 0.21), "SDR": (1.56, 0.18),
                     "SIR": (20.17, 1.05), "NMSE": (-1.56, 0.18)}),
    "real_orig":   ("Real-Valued CNN (orig.)", 78_000,
                    {"SI-SDR": (-0.97, 0.14), "SDR": (1.95, 0.12),
                     "SIR": (16.87, 0.84), "NMSE": (-1.95, 0.12)}),
}

C = {"proposed": "#1f77b4", "no_se_match": "#ff7f0e", "real_match": "#2ca02c",
     "tasnet": "#d62728", "cnse": "#17becf", "s4unet": "#8c564b",
     "no_se_orig": "#ff7f0e", "real_orig": "#2ca02c"}


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {name}.pdf/.png')


# ---------------------------------------------------------------------------
# Fig 01: headline 4-metric comparison (4 models x 4 metrics)
# ---------------------------------------------------------------------------
def fig_headline():
    keys = ["proposed", "no_se_match", "real_match", "tasnet"]
    labels = ["Complex CNN + SE\n(Proposed)", "Complex CNN no-SE\n(matched)",
              "Real-Valued CNN\n(matched)", "Conv-TasNet\n(SOTA baseline)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, mname in zip(axes.flat, METRICS):
        means = [MODELS[k][2][mname][0] for k in keys]
        stds = [MODELS[k][2][mname][1] for k in keys]
        bars = ax.bar(labels, means, yerr=stds, capsize=4,
                      color=[C[k] for k in keys], alpha=0.85,
                      error_kw=dict(ecolor='black', lw=1.4))
        for b, v, s in zip(bars, means, stds):
            ax.annotate(f'{v:.2f}', (b.get_x() + b.get_width() / 2,
                        v + s if v >= 0 else v - s),
                        ha='center', va='bottom' if v >= 0 else 'top',
                        fontsize=9, fontweight='bold')
        ax.set_title(f'{mname}: 4 Models', fontsize=11)
        ax.set_ylabel(f'{mname} (dB)')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(0, color='gray', lw=0.5)
        ax.tick_params(axis='x', labelsize=8)
    fig.suptitle('Headline: Complex CNN + SE vs Three Baselines\n'
                 '(mean ± std; 5 seeds for Proposed and matched ablations, '
                 '3 seeds for Conv-TasNet)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, '01_headline_4models')


# ---------------------------------------------------------------------------
# Fig 06: parameter count vs SDR (all 8 models)
# ---------------------------------------------------------------------------
def fig_params_perf():
    order = ["real_orig", "no_se_orig", "proposed", "real_match",
             "no_se_match", "tasnet", "s4unet", "cnse"]
    markers = {"real_orig": "^", "no_se_orig": "s", "proposed": "o",
               "real_match": "^", "no_se_match": "s", "tasnet": "D",
               "s4unet": "p", "cnse": "*"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in order:
        label, params, met = MODELS[k]
        mean, std = met["SDR"]
        ax.errorbar(params, mean, yerr=std, fmt=markers[k], ms=11,
                    color=C[k], mec='black', mew=0.6, capsize=4,
                    elinewidth=1.4, label=label, alpha=0.9)
    ax.set_xscale('log')
    ax.set_xlabel('Model parameters (log scale)')
    ax.set_ylabel('Overall SDR (dB, mean ± std)')
    ax.set_title('Parameter count vs. SDR performance '
                 '(5 seeds where available)')
    ax.grid(True, which='both', alpha=0.25)
    p_mean, p_std = MODELS['proposed'][2]["SDR"]
    ax.annotate(f'Proposed\n(235K, {p_mean:.2f}±{p_std:.2f} dB)',
                xy=(MODELS['proposed'][1], p_mean),
                xytext=(0.16, 0.30), textcoords='axes fraction', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    save(fig, '06_params_vs_perf')


# ---------------------------------------------------------------------------
# Fig 07: SE ablation, parameter-matched (5 seeds each)
# ---------------------------------------------------------------------------
def fig_se_ablation():
    keys = ["proposed", "no_se_match"]
    labels = ["Complex CNN + SE (Proposed)", "Complex CNN no-SE (matched)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, mname in zip(axes, ["SDR", "SI-SDR"]):
        means = [MODELS[k][2][mname][0] for k in keys]
        stds = [MODELS[k][2][mname][1] for k in keys]
        bars = ax.bar(labels, means, yerr=stds, capsize=5, width=0.5,
                      color=[C[k] for k in keys], alpha=0.85,
                      error_kw=dict(ecolor='black', lw=1.4))
        for b, v, s in zip(bars, means, stds):
            ax.annotate(f'{v:.2f}±{s:.2f}', (b.get_x() + b.get_width() / 2,
                        v + s if v >= 0 else v - s),
                        ha='center', va='bottom' if v >= 0 else 'top',
                        fontsize=10, fontweight='bold')
        ax.set_title(f'{mname} (dB): SE Ablation Effect', fontsize=11)
        ax.set_ylabel(f'{mname} (dB)')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(0, color='gray', lw=0.5)
        ax.tick_params(axis='x', labelsize=9)
        lo = min(v - s for v, s in zip(means, stds))
        hi = max(v + s for v, s in zip(means, stds))
        pad = (hi - lo) * 0.25
        ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle('Complex SE Block Contribution '
                 '(parameter-matched, 5 seeds each)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, '07_se_ablation')


# ---------------------------------------------------------------------------
# Micro-frequency generalisation (C-SE vs no-SE, trained on U(0, 5 Hz))
# ---------------------------------------------------------------------------
def fig_micro_freq():
    series = {}
    for tag, label in [("complex_cnn_se", "C-SE (Proposed)"),
                       ("complex_cnn_no_se", "No-SE")]:
        d = json.load(open(os.path.join(
            MICRO, f'microfreq_{tag}', f'{tag}_freq_offset.json')))
        gaps = sorted(float(g) for g in d['results'])
        series[label] = (gaps,
                         [d['results'][str(g) if str(g) in d['results'] else f'{g}']['SDR'] for g in gaps],
                         [d['results'][str(g) if str(g) in d['results'] else f'{g}']['SIR'] for g in gaps])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    style = {"C-SE (Proposed)": dict(color="#1f77b4", marker='o', ls='-'),
             "No-SE": dict(color="#d62728", marker='s', ls='--')}
    for ax, idx, name in [(axes[0], 1, 'SDR'), (axes[1], 2, 'SIR')]:
        for label, (gaps, sdr, sir) in series.items():
            vals = sdr if idx == 1 else sir
            xs = [0.06 if g == 0 else g for g in gaps]
            ax.plot(xs, vals, label=label, lw=1.8, ms=6, **style[label])
        ax.set_xscale('log')
        ax.set_xticks([0.06, 0.1, 1, 10, 100, 500])
        ax.set_xticklabels(['0', '0.1', '1', '10', '100', '500'])
        ax.set_xlabel('Test gap (Hz)')
        ax.set_ylabel(f'{name} (dB)')
        ax.set_title(f'{name} vs. carrier-frequency gap')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()
    fig.suptitle('Micro-frequency generalisation (trained on gaps ~ U(0, 5 Hz))',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, 'micro_freq_eval')


if __name__ == '__main__':
    for k, (label, params, met) in MODELS.items():
        print(f'{k:12s} N/A SDR={met["SDR"][0]:.2f}±{met["SDR"][1]:.2f} '
              f'SIR={met["SIR"][0]:.2f}±{met["SIR"][1]:.2f}')
    fig_headline()
    fig_params_perf()
    fig_se_ablation()
    fig_micro_freq()
    print(f'All figures saved to {OUT}')
