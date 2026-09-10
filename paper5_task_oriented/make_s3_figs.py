"""Paper 5 — S3 joint-LLR-head failure figures (negative result, Sec. S3).

Inputs (all local, synced from the server):
  results/s3/summary_s3.json               — v2 (learned strided conv) summary,
      K=2 cell: mixture baseline, S3 waveform route, S3 joint head (5 seeds),
      S2 waveform routes, paired bootstrap deltas
  results/s3_v1_meanpool/summary_s3.json   — v1 (window-mean downsample) archive
  results/probe_sync_head.log              — controlled sync probe, variants
      P (blind burst sync) and S (oracle sync), val SER at ep 1,5,...,30

Outputs:
  figures/s3_joint_vs_waveform.{png,pdf} — K=2 grouped bars (SER / BER):
      joint head ~= chance, waveform route intact.
  figures/s3_probe_sync.{png,pdf}       — probe learning curves (P vs S) +
      final-epoch per-modulation SER: same arch/data, oracle sync learns,
      blind sync fails => blind burst-level sync is the blocker.
Both PDFs are also copied to ../paper5/figures/ for the manuscript.

All results shown are negative/mismatch results by design: the joint head is
significantly WORSE than the waveform route and the mixture baseline (paired
bootstrap CIs far from 0); the figure must not suggest otherwise.
"""
from __future__ import annotations

import json
import os
import re
import shutil

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIGS = os.path.join(HERE, 'figures')
PAPER_FIGS = os.path.normpath(os.path.join(HERE, '..', 'paper5', 'figures'))
PROBE_LOG = os.path.join(RES, 'probe_sync_head.log')

# Chance SER under uniform decisions, averaged over the modulation pool
# {BPSK, QPSK, 8PSK, 16QAM}: mean(1 - 1/M).
CHANCE_SER = float(np.mean([1 - 1 / m for m in (2, 4, 8, 16)]))
CHANCE_BER = 0.5
MODS = ('BPSK', 'QPSK', '8PSK', '16QAM')


def mean_std(vals):
    return float(np.mean(vals)), float(np.std(vals))


def fig_joint_vs_waveform(s3_v2, s3_v1):
    """Grouped bars for the K=2 cell: baseline / waveform routes / joint heads."""
    def run(ser_list, ber_list):
        sm, ss = mean_std(ser_list)
        bm, bs = mean_std(ber_list)
        return {'ser': (sm, ss), 'ber': (bm, bs)}

    b = s3_v2['baseline_k2']
    bars = [
        {'label': 'Mixture\n(no sep.)', 'color': 'tab:gray',
         'ser': (b['ser'], None), 'ber': (b['ber'], None)},
        {'label': 'S2-ser\nwaveform', 'color': 'tab:blue',
         **run(s3_v2['s2_routes_k2']['ser']['ser'],
               s3_v2['s2_routes_k2']['ser']['ber'])},
        {'label': 'S3 waveform\n(v2 model)', 'color': 'tab:green',
         **run(s3_v2['s3_waveform_k2']['ser'], s3_v2['s3_waveform_k2']['ber'])},
        {'label': 'S3 joint head\nv1 (mean-pool)', 'color': 'tab:orange',
         **run(s3_v1['s3_joint_k2']['ser'], s3_v1['s3_joint_k2']['ber'])},
        {'label': 'S3 joint head\nv2 (strided conv)', 'color': 'tab:red',
         **run(s3_v2['s3_joint_k2']['ser'], s3_v2['s3_joint_k2']['ber'])},
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for ax, key, ylab in ((axes[0], 'ser', 'SER (K=2 cell)'),
                          (axes[1], 'ber', 'BER (K=2 cell, Gray)')):
        xs = np.arange(len(bars))
        means = [b[key][0] for b in bars]
        stds = [b[key][1] for b in bars]
        yerr = [0.0 if s is None else s for s in stds]
        ax.bar(xs, means, yerr=yerr, color=[b['color'] for b in bars],
               width=0.65, capsize=3.5,
               error_kw={'lw': 1.1, 'ecolor': 'black'})
        for x, m, s in zip(xs, means, stds):
            ax.text(x, m + (s or 0) + 0.006, f'{m:.3f}', ha='center',
                    va='bottom', fontsize=7.5)
        chance = CHANCE_SER if key == 'ser' else CHANCE_BER
        ax.axhline(chance, color='crimson', ls='--', lw=1.2,
                   label=f'chance level ({chance:.3f})')
        ax.set_xticks(xs)
        ax.set_xticklabels([b['label'] for b in bars], fontsize=7.5,
                           rotation=18, ha='right')
        ax.set_ylabel(ylab)
        ax.set_ylim(0, chance + 0.08)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')
    axes[0].set_title('SER: joint heads sit at chance', fontsize=10)
    axes[1].set_title('BER: joint heads ~= random bits', fontsize=10)
    fig.suptitle('S3 joint LLR head fails at K=2; waveform route is intact '
                 '(5 seeds, mean $\\pm$ std)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ('png', 'pdf'):
        out = os.path.join(FIGS, f's3_joint_vs_waveform.{ext}')
        fig.savefig(out, dpi=200)
        if ext == 'pdf':
            shutil.copy(out, os.path.join(PAPER_FIGS, os.path.basename(out)))
    plt.close(fig)


def parse_probe_log():
    """Parse probe_sync_head.log -> {variant: {ep, loss, ser, per_mod lists}}."""
    pat = re.compile(
        r'\[(P|S)\]\s+ep\s*(\d+)\s+loss=([\d.]+)\s+val SER=([\d.]+)\s+'
        r'per-mod:\s+16QAM:([\d.]+)\s+8PSK:([\d.]+)\s+BPSK:([\d.]+)\s+QPSK:([\d.]+)')
    out = {}
    with open(PROBE_LOG) as fh:
        for line in fh:
            m = pat.search(line)
            if not m:
                continue
            v = out.setdefault(m.group(1),
                               {'ep': [], 'loss': [], 'ser': [],
                                'per_mod': {k: [] for k in
                                            ('16QAM', '8PSK', 'BPSK', 'QPSK')}})
            v['ep'].append(int(m.group(2)))
            v['loss'].append(float(m.group(3)))
            v['ser'].append(float(m.group(4)))
            for k, g in zip(('16QAM', '8PSK', 'BPSK', 'QPSK'), m.groups()[4:]):
                v['per_mod'][k].append(float(g))
    assert set(out) == {'P', 'S'} and len(out['P']['ep']) == 7, \
        f'unexpected probe log contents: {list(out)}'
    return out


def fig_probe_sync(probe):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0),
                             gridspec_kw={'width_ratios': [1.15, 1]})
    ax = axes[0]
    for v, color, lab in (('P', 'tab:red', 'P: blind sync (as deployed)'),
                          ('S', 'tab:green', 'S: oracle sync (ceiling)')):
        d = probe[v]
        ax.plot(d['ep'], d['ser'], marker='o', ms=4, color=color, label=lab)
    ax.axhline(CHANCE_SER, color='crimson', ls='--', lw=1.2,
               label=f'chance level ({CHANCE_SER:.3f})')
    ax.set_xlabel('epoch')
    ax.set_ylabel('val SER (K=1 probe)')
    ax.set_ylim(0, CHANCE_SER + 0.06)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title('Same arch, same data: oracle sync learns,\n'
                 'blind sync stays at chance', fontsize=10)

    ax = axes[1]
    xs = np.arange(len(MODS))
    w = 0.36
    p_fin = [probe['P']['per_mod'][m][-1] for m in MODS]
    s_fin = [probe['S']['per_mod'][m][-1] for m in MODS]
    ax.bar(xs - w / 2, p_fin, w, color='tab:red', label='P: blind sync')
    ax.bar(xs + w / 2, s_fin, w, color='tab:green', label='S: oracle sync')
    for x, y in zip(xs - w / 2, p_fin):
        ax.text(x, y + 0.008, f'{y:.2f}', ha='center', va='bottom', fontsize=7)
    for x, y in zip(xs + w / 2, s_fin):
        ax.text(x, y + 0.008, f'{y:.2f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(MODS, fontsize=8)
    ax.set_ylabel('val SER @ epoch 30')
    ax.set_ylim(0, max(p_fin) + 0.1)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title('Per-modulation SER at final epoch', fontsize=10)

    fig.suptitle('Controlled probe: burst-level blind carrier/phase sync is '
                 'the bottleneck', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ('png', 'pdf'):
        out = os.path.join(FIGS, f's3_probe_sync.{ext}')
        fig.savefig(out, dpi=200)
        if ext == 'pdf':
            shutil.copy(out, os.path.join(PAPER_FIGS, os.path.basename(out)))
    plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(PAPER_FIGS, exist_ok=True)
    with open(os.path.join(RES, 's3', 'summary_s3.json')) as fh:
        s3_v2 = json.load(fh)
    with open(os.path.join(RES, 's3_v1_meanpool', 'summary_s3.json')) as fh:
        s3_v1 = json.load(fh)
    probe = parse_probe_log()

    # console summary
    b = s3_v2['baseline_k2']
    print(f"mixture baseline K=2:      SER={b['ser']:.4f}  BER={b['ber']:.4f}")
    for name, blk in (('S3 waveform (v2)', s3_v2['s3_waveform_k2']),
                      ('S3 joint v1', s3_v1['s3_joint_k2']),
                      ('S3 joint v2', s3_v2['s3_joint_k2'])):
        sm, ss = mean_std(blk['ser'])
        bm, bs = mean_std(blk['ber'])
        print(f"{name:<24s} SER={sm:.4f}+-{ss:.4f}  BER={bm:.4f}+-{bs:.4f}")
    print(f"chance levels: SER(mod-avg)={CHANCE_SER:.4f}  BER={CHANCE_BER}")
    for v in ('P', 'S'):
        d = probe[v]
        print(f"probe [{v}]: SER {d['ser'][0]:.4f} (ep1) -> {d['ser'][-1]:.4f} "
              f"(ep{d['ep'][-1]})")

    fig_joint_vs_waveform(s3_v2, s3_v1)
    fig_probe_sync(probe)
    print(f"\nSaved figures to {FIGS}; PDFs copied to {PAPER_FIGS}")


if __name__ == '__main__':
    main()
