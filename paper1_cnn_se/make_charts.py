"""
Generate publication-quality charts from evaluation results.

Outputs PNGs to results/charts/. Run on remote after evaluations are done.
"""
import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Global style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 7,
})

MODEL_INFO = {
    'se':   ('Complex CNN + SE (Proposed)', '#1f77b4', 'o'),
    'nose': ('Complex CNN no SE (Ablation)', '#ff7f0e', 's'),
    'real': ('Real-Valued CNN (Baseline)',   '#2ca02c', '^'),
}
MODELS = ['se', 'nose', 'real']
RESULTS = '/data/work/comm_bss_project/results'
LOGS = '/data/work/logs'
CHARTS = f'{RESULTS}/charts'
os.makedirs(CHARTS, exist_ok=True)


def load_snr_csvs():
    """Returns dict: {model_key: DataFrame with columns SI-SDR, SDR, SIR, NMSE, SNR}."""
    out = {}
    for m in MODELS:
        if m == 'se':
            f = f'{RESULTS}/se/complex_cnn_se_results.csv'
        elif m == 'nose':
            f = f'{RESULTS}/nose/complex_cnn_no_se_results.csv'
        else:
            f = f'{RESULTS}/real/real_baseline_results.csv'
        df = pd.read_csv(f)
        out[m] = df
    return out


def load_per_mod():
    """Returns dict: {model_key: {same_mod, diff_mod, per_pair}}."""
    out = {}
    for m in MODELS:
        if m == 'se':
            f = f'{RESULTS}/se/complex_cnn_se_per_mod.json'
        elif m == 'nose':
            f = f'{RESULTS}/nose/complex_cnn_no_se_per_mod.json'
        else:
            f = f'{RESULTS}/real/real_baseline_per_mod.json'
        out[m] = json.load(open(f))
    return out


def load_freq_offset():
    """Returns dict: {model_key: {sep: {SI-SDR, SDR, SIR, NMSE}}}."""
    out = {}
    for m in MODELS:
        f = f'{RESULTS}/freq_offset/{m}_freq_offset.json'
        if not os.path.exists(f):
            # try alt names
            if m == 'se':
                f = f'{RESULTS}/freq_offset/complex_cnn_se_freq_offset.json'
            elif m == 'nose':
                f = f'{RESULTS}/freq_offset/complex_cnn_no_se_freq_offset.json'
            elif m == 'real':
                f = f'{RESULTS}/freq_offset/real_baseline_freq_offset.json'
        d = json.load(open(f))
        out[m] = {int(k): v for k, v in d['results'].items()}
    return out


def parse_training_log(path):
    """Parse training log: returns list of dicts with epoch, train_loss, val_loss, si_sdr, sdr, sir."""
    rows = []
    pat = re.compile(r'\s*(\d+)\s*\|\s*([-\d.eE+]+)\s*\|\s*([-\d.eE+]+)\s*\|\s*([-\d.eE+]+)\s*\|\s*([-\d.eE+]+)\s*\|\s*([-\d.eE+]+)')
    with open(path) as f:
        for line in f:
            m = pat.match(line)
            if m:
                rows.append({
                    'epoch': int(m.group(1)),
                    'train_loss': float(m.group(2)),
                    'val_loss': float(m.group(3)),
                    'si_sdr': float(m.group(4)),
                    'sdr': float(m.group(5)),
                    'sir': float(m.group(6)),
                })
    return rows


# =============================================================================
# Chart 1: SNR Sweep — 4 metrics × 3 models (one big figure with 2x2 subplots)
# =============================================================================
def chart_snr_sweep(snr_data):
    metrics = ['SI-SDR', 'SDR', 'SIR', 'NMSE']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, metric in zip(axes.flat, metrics):
        for m in MODELS:
            df = snr_data[m]
            df_plot = df[df['SNR'] != 'Overall'].copy()
            df_plot['SNR'] = df_plot['SNR'].astype(int)
            label, color, marker = MODEL_INFO[m]
            ax.plot(df_plot['SNR'], df_plot[metric], marker=marker,
                    color=color, label=label)
        ax.set_xlabel('Input SNR (dB)')
        ax.set_ylabel(f'{metric} (dB)')
        ax.set_title(f'{metric} vs SNR')
        ax.legend(loc='best', fontsize=10)
        # x ticks at every SNR
        ax.set_xticks([-10, -5, 0, 5, 10, 15, 20])
    fig.suptitle('SNR-Dependent Performance (5 Hz Carrier Separation)', fontsize=15, y=1.00)
    plt.tight_layout()
    plt.savefig(f'{CHARTS}/01_snr_sweep.png', dpi=140, bbox_inches='tight')
    plt.close()


# =============================================================================
# Chart 2: Per-modulation heatmap (3 models × per-pair SI-SDR)
# =============================================================================
def chart_per_mod_heatmap(per_mod):
    # Order: BPSK, QPSK, 8PSK, 16QAM
    mods = ['BPSK', 'QPSK', '8PSK', '16QAM']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, m in zip(axes, MODELS):
        label = MODEL_INFO[m][0]
        per_pair = per_mod[m]['per_pair']
        mat = np.full((4, 4), np.nan)
        for i, ma in enumerate(mods):
            for j, mb in enumerate(mods):
                key = f'{ma}-{mb}'
                if key in per_pair:
                    mat[i, j] = per_pair[key]['SI-SDR']
        im = ax.imshow(mat, cmap='RdYlGn', vmin=-2, vmax=1, aspect='auto')
        ax.set_xticks(range(4)); ax.set_xticklabels(mods)
        ax.set_yticks(range(4)); ax.set_yticklabels(mods)
        ax.set_xlabel('Source 2')
        ax.set_ylabel('Source 1')
        ax.set_title(f'{label}\nSI-SDR (dB)', fontsize=11)
        for i in range(4):
            for j in range(4):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                            color='black', fontsize=9)
    fig.suptitle('Per-Modulation-Pair SI-SDR (dB) — Diagonal = Same Modulation',
                 fontsize=14, y=1.02)
    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label('SI-SDR (dB)')
    plt.savefig(f'{CHARTS}/02_per_mod_heatmap.png', dpi=140, bbox_inches='tight')
    plt.close()


# =============================================================================
# Chart 3: Same vs Diff modulation aggregated bar chart
# =============================================================================
def chart_same_vs_diff(per_mod):
    mods = ['BPSK', 'QPSK', '8PSK', '16QAM']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.25
    x = np.arange(len(mods))
    for ax_idx, metric in enumerate(['SI-SDR', 'SDR']):
        ax = axes[ax_idx]
        for i, m in enumerate(MODELS):
            same_mod = per_mod[m]['same_mod']
            label, color, _ = MODEL_INFO[m]
            vals = [same_mod[mod][metric] for mod in mods]
            ax.bar(x + (i-1)*width, vals, width, label=label, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(mods)
        ax.set_xlabel('Modulation')
        ax.set_ylabel(f'{metric} (dB)')
        ax.set_title(f'Same-Modulation {metric} (Source1 = Source2)')
        ax.axhline(0, color='black', linewidth=0.6)
        ax.legend(loc='best', fontsize=10)
    fig.suptitle('Same-Modulation Separation Difficulty (BPSK easiest, 16QAM hardest)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{CHARTS}/03_same_modulation.png', dpi=140, bbox_inches='tight')
    plt.close()


# =============================================================================
# Chart 4: Frequency-offset robustness
# =============================================================================
def chart_freq_offset(freq_data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    seps = sorted(freq_data['se'].keys())
    for ax_idx, metric in enumerate(['SI-SDR', 'SDR']):
        ax = axes[ax_idx]
        for m in MODELS:
            label, color, marker = MODEL_INFO[m]
            ys = [freq_data[m][s][metric] for s in seps]
            ax.plot(seps, ys, marker=marker, color=color, label=label)
        ax.set_xscale('log')
        ax.set_xlabel('Carrier-frequency separation (Hz)')
        ax.set_ylabel(f'{metric} (dB)')
        ax.set_title(f'{metric} vs Frequency Separation (SNR=10 dB)')
        ax.set_xticks(seps)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.legend(loc='best', fontsize=10)
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
    fig.suptitle('Frequency-Offset Robustness (Models trained at 5 Hz, tested at wider gaps)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{CHARTS}/04_freq_offset.png', dpi=140, bbox_inches='tight')
    plt.close()


# =============================================================================
# Chart 5: Training curves (loss + SDR) — 3 subplots, one per model
# =============================================================================
def chart_training_curves():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
    for col, m in enumerate(MODELS):
        path = f'{LOGS}/{m}.log'
        if not os.path.exists(path):
            continue
        rows = parse_training_log(path)
        if not rows:
            continue
        epochs = [r['epoch'] for r in rows]
        train_loss = [r['train_loss'] for r in rows]
        val_loss = [r['val_loss'] for r in rows]
        sdr = [r['sdr'] for r in rows]
        si_sdr = [r['si_sdr'] for r in rows]
        label, color, _ = MODEL_INFO[m]

        # Top row: loss
        ax = axes[0, col]
        ax.plot(epochs, train_loss, color=color, linestyle='--', alpha=0.6, label='Train')
        ax.plot(epochs, val_loss, color=color, linewidth=2.2, label='Val')
        ax.set_title(f'{label}\nLoss', fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Combined loss')
        ax.legend(loc='best', fontsize=9)

        # Bottom row: SDR/SI-SDR
        ax = axes[1, col]
        ax.plot(epochs, sdr, color=color, marker='o', markersize=4, label='SDR')
        ax.plot(epochs, si_sdr, color=color, marker='s', markersize=4, linestyle='--', label='SI-SDR')
        ax.set_title(f'{label}\nSDR / SI-SDR', fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('dB')
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.legend(loc='best', fontsize=9)
    fig.suptitle('Training Dynamics (Combined loss = 0.5·MSE + 0.5·(-SI-SDR))',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{CHARTS}/05_training_curves.png', dpi=140, bbox_inches='tight')
    plt.close()


# =============================================================================
# Chart 6: Model size vs Performance scatter
# =============================================================================
def chart_params_vs_perf(snr_data):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    params = {'se': 235525, 'nose': 202757, 'real': 77765}
    for m in MODELS:
        df = snr_data[m]
        ovr = df[df['SNR'] == 'Overall'].iloc[0]
        label, color, marker = MODEL_INFO[m]
        ax.scatter(params[m], ovr['SDR'], s=180, color=color, marker=marker,
                   label=label, edgecolors='black', linewidth=1.2, zorder=3)
        # Annotate
        ax.annotate(f'{ovr["SDR"]:.2f} dB', (params[m], ovr['SDR']),
                    textcoords='offset points', xytext=(8, -3), fontsize=10)
    ax.set_xscale('log')
    ax.set_xlabel('Model parameters (log scale)')
    ax.set_ylabel('Overall SDR (dB)')
    ax.set_title('Parameter Count vs SDR Performance (Overall, 5 Hz gap)')
    ax.set_xticks(list(params.values()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{CHARTS}/06_params_vs_perf.png', dpi=140, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print('Loading data...')
    snr_data = load_snr_csvs()
    per_mod = load_per_mod()
    freq_data = load_freq_offset()

    print('Chart 1: SNR sweep (4 metrics)...')
    chart_snr_sweep(snr_data)

    print('Chart 2: Per-modulation heatmap...')
    chart_per_mod_heatmap(per_mod)

    print('Chart 3: Same-modulation bar chart...')
    chart_same_vs_diff(per_mod)

    print('Chart 4: Freq offset robustness...')
    chart_freq_offset(freq_data)

    print('Chart 5: Training curves...')
    chart_training_curves()

    print('Chart 6: Params vs performance...')
    chart_params_vs_perf(snr_data)

    print(f'\nAll charts saved to {CHARTS}/')
    for f in sorted(os.listdir(CHARTS)):
        if f.endswith('.png'):
            size = os.path.getsize(f'{CHARTS}/{f}') / 1024
            print(f'  {f}  ({size:.0f} KB)')
