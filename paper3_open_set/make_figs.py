"""Regenerate paper-3 figures from archived OOD score dumps.

Reads results/*_ood_scores.npz (baseline model, seeds 42-46) and writes
publication figures to figures/:

  fig_per_snr_auroc.{pdf,png}   Per-SNR AUROC: energy vs prototype vs
                                SNR-routed ensemble (5-seed mean ± std).
  fig_overall_auroc.{pdf,png}   Weighted-average AUROC: single scorers vs
                                routed ensemble vs oracle bound.
  fig_snr_noise_robustness.*    Routed AUROC vs SNR-estimation noise σ.
  fig_embedding_pca.{pdf,png}   PCA of per-source embeddings (seed 42),
                                known vs unknown modulations, low/high SNR.

Usage:
    python make_figs.py                       # uses results/seed4* npz
    python make_figs.py --npz "results/x*.npz" --out figures
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# Match the manuscript fonts (Latin Modern text / CM math), same as
# make_arch_fig.py, so figure labels use the same typeface as the paper.
for _f in glob.glob('/usr/local/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf'):
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
THRESHOLD = 0.0  # energy at or below, prototype above (a-priori rule)


def per_snr_stats(files):
    """Per-SNR AUROC for energy / prototype / routed, per seed."""
    out = {k: [] for k in ('energy', 'prototype', 'routed')}
    for f in files:
        d = np.load(f)
        k_snr, u_snr = d['known_snr'], d['unknown_snr']
        seed_res = {k: {} for k in out}
        for s in SNRS:
            mk, mu = k_snr == s, u_snr == s
            if mk.sum() < 5 or mu.sum() < 5:
                continue
            e = auroc(d['energy_score_known'][mk], d['energy_score_unknown'][mu])
            p = auroc(d['prototype_score_known'][mk], d['prototype_score_unknown'][mu])
            seed_res['energy'][s] = e
            seed_res['prototype'][s] = p
            seed_res['routed'][s] = e if s <= THRESHOLD else p
        for k in out:
            out[k].append([seed_res[k][s] for s in SNRS])
    return {k: (np.mean(v, axis=0), np.std(v, axis=0)) for k, v in out.items()}


def weighted_avg_stats(files):
    """Weighted-average AUROC per scorer + oracle, per seed.

    Six single scorers: the three stored in the npz (energy / prototype /
    vos) plus mahalanobis and msp (recomputed from embeddings / logits)
    and odin (loaded from the sibling ``*_odin_eps*_T*.npz`` if present).
    """
    keys = ['energy', 'prototype', 'vos', 'mahalanobis', 'msp', 'odin',
            'routed', 'oracle']
    per_seed = {k: [] for k in keys}
    for f in files:
        d = np.load(f)
        k_snr, u_snr = d['known_snr'], d['unknown_snr']
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
        singles = list(scores)
        sums, den = {k: 0.0 for k in keys}, 0.0
        for s in SNRS:
            mk, mu = k_snr == s, u_snr == s
            if mk.sum() < 5 or mu.sum() < 5:
                continue
            w = int(mk.sum()) * int(mu.sum())
            aucs = {m: auroc(sk[mk], su[mu]) for m, (sk, su) in scores.items()}
            for m in singles:
                sums[m] += aucs[m] * w
            sums['routed'] += (aucs['energy'] if s <= THRESHOLD
                               else aucs['prototype']) * w
            sums['oracle'] += max(aucs.values()) * w
            den += w
        for k in keys:
            per_seed[k].append(sums[k] / den)
    return {k: (np.mean(v), np.std(v)) for k, v in per_seed.items()}


def noise_sensitivity(files, sigmas=(0, 1, 3, 6), trials=50, seed=0):
    """Routed weighted-avg AUROC under noisy SNR estimates (MC mean per σ)."""
    rng = np.random.default_rng(seed)
    means, stds = [], []
    for sigma in sigmas:
        file_means = []
        for f in files:
            d = np.load(f)
            k_snr, u_snr = d['known_snr'], d['unknown_snr']
            e_k, e_u = d['energy_score_known'], d['energy_score_unknown']
            p_k, p_u = d['prototype_score_known'], d['prototype_score_unknown']
            t_res = []
            for _ in range(trials if sigma > 0 else 1):
                if sigma > 0:
                    k_est = k_snr + rng.normal(0, sigma, k_snr.shape)
                    u_est = u_snr + rng.normal(0, sigma, u_snr.shape)
                else:
                    k_est, u_est = k_snr, u_snr
                k_r = np.where(k_est <= THRESHOLD, e_k, p_k)
                u_r = np.where(u_est <= THRESHOLD, e_u, p_u)
                num = den = 0.0
                for s in SNRS:
                    mk, mu = k_snr == s, u_snr == s
                    if mk.sum() < 5 or mu.sum() < 5:
                        continue
                    w = int(mk.sum()) * int(mu.sum())
                    num += auroc(k_r[mk], u_r[mu]) * w
                    den += w
                t_res.append(num / den)
            file_means.append(np.mean(t_res))
        means.append(np.mean(file_means))
        stds.append(np.std(file_means))
    return np.array(sigmas), np.array(means), np.array(stds)


def fig_per_snr(stats, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    style = {'energy':   ('o-', 'tab:orange', 'Energy'),
             'prototype': ('s-', 'tab:blue', 'Prototype'),
             'routed':   ('^-', 'tab:red', 'SNR-routed (ours)')}
    for k, (mk, color, label) in style.items():
        m, s = stats[k]
        ax.plot(SNRS, m, mk, color=color, label=label, linewidth=1.6,
                markersize=5, zorder=3)
        ax.fill_between(SNRS, m - s, m + s, color=color, alpha=0.15, zorder=2)
    ax.axhline(0.5, color='gray', ls=':', lw=1, zorder=1)
    ax.axvline(THRESHOLD, color='gray', ls='--', lw=1, zorder=1)
    ax.text(THRESHOLD + 0.3, 0.16, 'routing\nboundary', fontsize=7, color='gray')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('OOD AUROC')
    ax.set_ylim(0.15, 0.95)
    ax.set_xticks(SNRS)
    ax.legend(loc='center right', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_overall(stats, out):
    keys = ['energy', 'msp', 'odin', 'prototype', 'vos', 'mahalanobis',
            'routed']
    labels = ['Energy', 'MSP', 'ODIN', 'Prototype', 'VOS', 'Mahalanobis',
              'SNR-routed\n(ours)']
    means = [stats[k][0] for k in keys]
    stds = [stats[k][1] for k in keys]
    # orange = logit family, blue = geometry family, red = ours
    colors = ['tab:orange'] * 3 + ['tab:blue'] * 3 + ['tab:red']
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    bars = ax.bar(labels, means, yerr=stds, capsize=3, color=colors,
                  alpha=0.85, edgecolor='black', linewidth=0.6)
    om, osd = stats['oracle']
    ax.axhline(om, color='tab:green', ls='--', lw=1.4,
               label=f'Oracle bound = {om:.3f}')
    ax.axhline(0.5, color='gray', ls=':', lw=1)
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.012, f'{m:.3f}',
                ha='center', fontsize=7.5)
    ax.set_ylabel('Weighted-avg OOD AUROC')
    ax.set_ylim(0.0, 0.75)
    ax.tick_params(axis='x', labelsize=8)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_noise(sigmas, means, stds, single_best, oracle, out):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.errorbar(sigmas, means, yerr=stds, fmt='o-', color='tab:red',
                capsize=4, linewidth=1.6, markersize=5,
                label='SNR-routed (ours)')
    ax.axhline(single_best, color='tab:blue', ls='--', lw=1.4,
               label=f'Best single scorer = {single_best:.3f}')
    ax.axhline(oracle, color='tab:green', ls=':', lw=1.4,
               label=f'Oracle (ideal SNR) = {oracle:.3f}')
    ax.set_xlabel('SNR estimation error $\\sigma$ (dB)')
    ax.set_ylabel('Weighted-avg OOD AUROC')
    ax.set_xticks(sigmas)
    ax.set_ylim(0.45, 0.72)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_pca(npz_path, out):
    d = np.load(npz_path)
    k_snr, u_snr = d['known_snr'], d['unknown_snr']
    emb = np.vstack([d['known_emb'], d['unknown_emb']])
    snr = np.concatenate([k_snr, u_snr])
    is_unknown = np.concatenate([np.zeros(len(k_snr), bool),
                                 np.ones(len(u_snr), bool)])
    mods = np.concatenate([d['known_mods'], d['unknown_mods']])

    # PCA via SVD (no sklearn dependency)
    X = emb - emb.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ vt[:2].T

    KNOWN = ['BPSK', 'QPSK', '8PSK', '16QAM']
    UNKNOWN = ['64QAM', '$\\pi$/4-DQPSK', 'MSK', 'OFDM-QPSK']
    cmap = plt.get_cmap('tab10')

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True, sharey=True)
    for ax, (lo, hi), title in zip(axes, [(-99, 0), (10, 99)],
                                   ['Low SNR ($\\leq$ 0 dB)', 'High SNR ($\\geq$ 10 dB)']):
        m = (snr >= lo) & (snr <= hi)
        for c in range(4):
            mk = m & ~is_unknown & (mods == c)
            ax.scatter(Z[mk, 0], Z[mk, 1], s=4, alpha=0.35, color=cmap(c),
                       label=KNOWN[c] if ax is axes[0] else None)
            mu_ = m & is_unknown & (mods == c)
            ax.scatter(Z[mu_, 0], Z[mu_, 1], s=6, alpha=0.5, color=cmap(c),
                       marker='x', label=UNKNOWN[c] + ' (unk)' if ax is axes[0] else None)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('PC 1')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('PC 2')
    axes[0].legend(fontsize=6.5, ncol=2, loc='best', markerscale=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', default='results/openset_cse_h32_l4_bs16_lr0.001'
                                    '_alpha1.0_seed4[2-6]_best_ood_scores.npz')
    p.add_argument('--out', default='figures')
    args = p.parse_args()

    files = sorted(glob.glob(args.npz))
    assert files, f'no npz matched: {args.npz}'
    os.makedirs(args.out, exist_ok=True)
    print(f'{len(files)} score dumps found')

    stats = per_snr_stats(files)
    for ext in ('pdf', 'png'):
        fig_per_snr(stats, os.path.join(args.out, f'fig_per_snr_auroc.{ext}'))
    print('fig_per_snr_auroc done')

    ov = weighted_avg_stats(files)
    for k, (m, s) in ov.items():
        print(f'  {k:<10}: {m:.3f} ± {s:.3f}')
    for ext in ('pdf', 'png'):
        fig_overall(ov, os.path.join(args.out, f'fig_overall_auroc.{ext}'))
    print('fig_overall_auroc done')

    sig, m, s = noise_sensitivity(files)
    singles = ('energy', 'prototype', 'vos', 'mahalanobis', 'msp', 'odin')
    single_best = max(ov[k][0] for k in singles)
    best_name = singles[np.argmax([ov[k][0] for k in singles])]
    for ext in ('pdf', 'png'):
        fig_noise(sig, m, s, single_best, ov['oracle'][0],
                  os.path.join(args.out, f'fig_snr_noise_robustness.{ext}'))
    print(f'fig_snr_noise_robustness done (best single: {best_name} '
          f'= {single_best:.3f}):', dict(zip(sig.tolist(), m.round(3))))

    seed42 = [f for f in files if 'seed42' in f][0]
    for ext in ('pdf', 'png'):
        fig_pca(seed42, os.path.join(args.out, f'fig_embedding_pca.{ext}'))
    print('fig_embedding_pca done')


if __name__ == '__main__':
    main()
