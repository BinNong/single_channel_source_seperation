"""Regenerate paper-3 REVISION figures (cautionary / negative-result pivot).

Produces, into ../paper3/figures/:

  fig_pitfall_mechanism.{pdf,png}  Schematic of the repeat-vs-tile label
                                   misalignment that manufactured the
                                   per-SNR structure.
  fig_artifact_vs_corrected.*      TWO-PANEL money figure: per-SNR AUROC
                                   profiles of the SAME score dumps under
                                   the buggy (repeat) vs corrected (tile)
                                   known-pool SNR labels.
  fig_per_snr_all6.*               Corrected per-SNR AUROC for all six
                                   scorers (flat at chance).
  fig_snr_estimator.*              True vs estimated SNR for the blind
                                   subspace estimator (M2M4 for contrast),
                                   from results/snr_est_routing.json.
  fig_operating_point.*            Single-threshold operating point
                                   (variant B, refpool-calibrated): per-SNR
                                   Det@tau and FRR@tau of the routed
                                   detector, from
                                   results/routed_threshold_metrics.json.

Every number comes from the npz/json files under results/ (buggy dumps
from results/archive_buggy_snr_labels/); nothing is hard-coded.

Usage:
    python make_figs_revision.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np

# Match the manuscript fonts (Latin Modern text / CM math), as make_figs.py.
for _f in glob.glob('/usr/local/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf'):
    font_manager.fontManager.addfont(_f)
for _f in glob.glob('/Library/TeX/Root/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf'):
    font_manager.fontManager.addfont(_f)
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Latin Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
})

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from open_set_metrics import auroc
from ood_baselines import mahalanobis_scores, msp_scores

SNRS = [-10, -5, 0, 5, 10, 15, 20]
THRESHOLD = 0.0  # a-priori routing boundary
RESULTS = 'results'
ARCHIVE = os.path.join(RESULTS, 'archive_buggy_snr_labels')
OUT = os.path.join('..', 'paper3', 'figures')

BASE_GLOB = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed4[2-6]_best_ood_scores.npz'


def load_scores(f):
    """Six scorers (known, unknown) from one corrected score dump."""
    d = np.load(f)
    scores = {
        'energy':    (d['energy_score_known'], d['energy_score_unknown']),
        'prototype': (d['prototype_score_known'], d['prototype_score_unknown']),
        'vos':       (d['vos_score_known'], d['vos_score_unknown']),
        'mahalanobis': (
            mahalanobis_scores(d['known_emb'], d['known_mods'], d['known_emb']),
            mahalanobis_scores(d['known_emb'], d['known_mods'], d['unknown_emb'])),
        'msp': (msp_scores(d['known_logits']),
                msp_scores(d['unknown_logits'])),
    }
    odin_files = sorted(glob.glob(f.replace('_ood_scores.npz',
                                            '_odin_eps*_T*.npz')))
    if odin_files:
        o = np.load(odin_files[-1])
        scores['odin'] = (o['odin_score_known'], o['odin_score_unknown'])
    return d['known_snr'], d['unknown_snr'], scores


def per_snr_profiles(files, methods):
    """Per-SNR AUROC (5-seed mean, std) for each method + routed."""
    per_seed = {m: [] for m in list(methods) + ['routed']}
    for f in files:
        k_snr, u_snr, scores = load_scores(f)
        row = {m: [] for m in per_seed}
        for s in SNRS:
            mk, mu = k_snr == s, u_snr == s
            aucs = {m: auroc(scores[m][0][mk], scores[m][1][mu])
                    for m in methods}
            for m in methods:
                row[m].append(aucs[m])
            row['routed'].append(aucs['energy'] if s <= THRESHOLD
                                 else aucs['prototype'])
        for m in per_seed:
            per_seed[m].append(row[m])
    return {m: (np.mean(v, axis=0), np.std(v, axis=0))
            for m, v in per_seed.items()}


# ----------------------------------------------------------------------
def fig_pitfall_mechanism(out):
    """Schematic: repeat-labeled SNR array vs tile-stacked score array."""
    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    ax.axis('off')
    repeat = [-5, -5, -5, -5, 10, 10, 10, 10]      # as STORED (repeat)
    tile_true = [-5, -5, 10, 10, -5, -5, 10, 10]  # true SNR of tile-stacked scores
    colors = {-5: 'tab:orange', 10: 'tab:blue'}

    def draw_row(y, values, title):
        ax.text(-0.15, y + 0.25, title, ha='right', va='center',
                fontsize=8.5)
        for i, v in enumerate(values):
            ax.add_patch(Rectangle((i, y), 0.92, 0.5,
                                   facecolor=colors[v], alpha=0.75,
                                   edgecolor='black', linewidth=0.5))
            ax.text(i + 0.46, y + 0.25, f'{v}', ha='center', va='center',
                    fontsize=8, color='white')

    draw_row(1.55, repeat, 'stored SNR label\n(np.repeat, interleaved)')
    draw_row(0.75, tile_true, 'true SNR of score\n(tile-stacked)')
    for i in (2, 3, 4, 5):
        ax.plot(i + 0.46, 0.42, marker='x', color='red', markersize=9,
                markeredgewidth=2.2)
    ax.text(4.0, 0.05, 'misaligned: scores paired with the WRONG SNR label',
            ha='center', fontsize=8.5, color='red')
    ax.text(2.0, 2.25, '$-5$ dB mixtures', ha='center', fontsize=8,
            color='tab:orange')
    ax.text(6.0, 2.25, '$+10$ dB mixtures', ha='center', fontsize=8,
            color='tab:blue')
    ax.set_xlim(-3.6, 8.1)
    ax.set_ylim(-0.1, 2.5)
    fig.savefig(out, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def fig_artifact_vs_corrected(stats_bug, stats_fix, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    style = {'energy':   ('o-', 'tab:orange', 'Energy'),
             'prototype': ('s-', 'tab:blue', 'Prototype'),
             'routed':   ('^-', 'tab:red', 'SNR-routed')}
    for ax, stats, title in zip(
            axes, (stats_bug, stats_fix),
            ['(a) Buggy labels (repeat): apparent complementarity',
             '(b) Corrected labels (tile): chance everywhere']):
        for k, (mk, color, label) in style.items():
            m, s = stats[k]
            ax.plot(SNRS, m, mk, color=color, label=label, linewidth=1.6,
                    markersize=5, zorder=3)
            ax.fill_between(SNRS, m - s, m + s, color=color, alpha=0.15,
                            zorder=2)
        ax.axhline(0.5, color='gray', ls=':', lw=1, zorder=1)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylim(0.1, 0.95)
        ax.set_xticks(SNRS)
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('OOD AUROC')
    axes[0].legend(loc='lower left', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_per_snr_all6(stats, out):
    style = {'energy':      ('o-', 'tab:orange', 'Energy'),
             'msp':         ('v-', 'tab:red', 'MSP'),
             'odin':        ('<-', 'tab:brown', 'ODIN'),
             'prototype':   ('s-', 'tab:blue', 'Prototype'),
             'vos':         ('D-', 'tab:cyan', 'VOS-inspired'),
             'mahalanobis': ('>-', 'tab:purple', 'Mahalanobis')}
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for k, (mk, color, label) in style.items():
        m, s = stats[k]
        ax.plot(SNRS, m, mk, color=color, label=label, linewidth=1.4,
                markersize=4.5, zorder=3)
        ax.fill_between(SNRS, m - s, m + s, color=color, alpha=0.12, zorder=2)
    ax.axhline(0.5, color='gray', ls=':', lw=1, zorder=1)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('OOD AUROC')
    ax.set_ylim(0.3, 0.75)
    ax.set_xticks(SNRS)
    ax.legend(loc='upper right', fontsize=7.5, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_snr_estimator(json_path, out):
    d = json.load(open(json_path))['estimator_quality']
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for name, mk, color, label in [
            ('subspace', 'o-', 'tab:blue', 'Subspace (this work)'),
            ('m2m4', 's-', 'tab:gray', 'M2M4 (inadequate)')]:
        means = [d[name][str(s)]['mean_est'] for s in SNRS]
        stds = [d[name][str(s)]['std'] for s in SNRS]
        ax.errorbar(SNRS, means, yerr=stds, fmt=mk, color=color, capsize=3,
                    linewidth=1.5, markersize=4.5, label=label)
    ax.plot(SNRS, SNRS, 'k:', lw=1.2, label='Ideal')
    ax.axhline(0.0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('True SNR (dB)')
    ax.set_ylabel('Estimated SNR (dB)')
    ax.set_xticks(SNRS)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_operating_point(json_path, out):
    d = json.load(open(json_path))['across_seeds']['B']['routed']['per_snr']
    det = [d[str(s)]['det']['mean'] for s in SNRS]
    det_s = [d[str(s)]['det']['std'] for s in SNRS]
    frr = [d[str(s)]['frr']['mean'] for s in SNRS]
    frr_s = [d[str(s)]['frr']['std'] for s in SNRS]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.errorbar(SNRS, det, yerr=det_s, fmt='o-', color='tab:red', capsize=3,
                linewidth=1.5, markersize=5,
                label='Det@$\\tau$ (unknown detected)')
    ax.errorbar(SNRS, frr, yerr=frr_s, fmt='s--', color='tab:blue', capsize=3,
                linewidth=1.5, markersize=4.5,
                label='FRR@$\\tau$ (known rejected)')
    ax.axhline(0.05, color='gray', ls=':', lw=1,
               label='FRR calibration target 5%')
    ax.axhline(1.0, color='gray', ls='-.', lw=0.8)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Rate')
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(SNRS)
    ax.legend(fontsize=8, loc='center right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    buggy = sorted(glob.glob(os.path.join(ARCHIVE, BASE_GLOB)))
    fixed = sorted(glob.glob(os.path.join(RESULTS, BASE_GLOB)))
    assert len(buggy) == 5 and len(fixed) == 5, (len(buggy), len(fixed))
    print(f'{len(buggy)} archived (buggy) + {len(fixed)} corrected dumps')

    for ext in ('pdf', 'png'):
        fig_pitfall_mechanism(os.path.join(OUT, f'fig_pitfall_mechanism.{ext}'))
    print('fig_pitfall_mechanism done')

    stats_bug = per_snr_profiles(buggy, ['energy', 'prototype'])
    stats_fix = per_snr_profiles(fixed, ['energy', 'prototype'])
    for ext in ('pdf', 'png'):
        fig_artifact_vs_corrected(
            stats_bug, stats_fix,
            os.path.join(OUT, f'fig_artifact_vs_corrected.{ext}'))
    print('fig_artifact_vs_corrected done')

    all6 = ['energy', 'msp', 'odin', 'prototype', 'vos', 'mahalanobis']
    stats6 = per_snr_profiles(fixed, all6)
    for m in all6:
        print(f'  corrected wavg-ish check {m:<11}: '
              f'mean over bins {np.mean(stats6[m][0]):.3f}')
    for ext in ('pdf', 'png'):
        fig_per_snr_all6(stats6, os.path.join(OUT, f'fig_per_snr_all6.{ext}'))
    print('fig_per_snr_all6 done')

    for ext in ('pdf', 'png'):
        fig_snr_estimator(os.path.join(RESULTS, 'snr_est_routing.json'),
                          os.path.join(OUT, f'fig_snr_estimator.{ext}'))
    print('fig_snr_estimator done')

    for ext in ('pdf', 'png'):
        fig_operating_point(os.path.join(RESULTS, 'routed_threshold_metrics.json'),
                            os.path.join(OUT, f'fig_operating_point.{ext}'))
    print('fig_operating_point done')


if __name__ == '__main__':
    main()
