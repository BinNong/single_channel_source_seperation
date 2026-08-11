"""
Generate publication-quality charts from multi-seed experimental results.

Reads:
  results/pub_2026/per_mod/<model>_s<seed>/<model>_per_mod.json   (per-mod, per-seed)
  results/pub_2026/freq_offset/<model>_s<seed>/<model>_freq_offset.json
  results/pub_2026/freq_offset/gap<N>/complex_cnn_se_freq_offset.json

Outputs 8 PNGs into results/charts/.
"""
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Style
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
    'errorbar.capsize': 4,
})

MODELS = {
    'complex_cnn_se':    ('Complex CNN + SE (Proposed)',  '#1f77b4', 'o'),
    'complex_cnn_no_se': ('Complex CNN no SE (Ablation)', '#ff7f0e', 's'),
    'real_baseline':     ('Real-Valued CNN (Baseline)',   '#2ca02c', '^'),
    'conv_tasnet':       ('Conv-TasNet (SOTA baseline)',  '#d62728', 'D'),
}
SNR_TEST = [-10, -5, 0, 5, 10, 15, 20]
_BASE = os.path.dirname(os.path.abspath(__file__))
CHARTS = os.path.join(_BASE, '..', 'results', 'charts')
# Use fixed-metric directory if it exists; fall back to original.
FIXED = os.path.join(_BASE, '..', 'results', 'pub_2026', 'per_mod_fixed')
ORIG = os.path.join(_BASE, '..', 'results', 'pub_2026', 'per_mod')
if os.path.isdir(FIXED) and len(glob.glob(f'{FIXED}/*/')) > 0:
    PER_MOD_DIR = FIXED
    print(f'[charts] Using FIXED per_mod directory: {FIXED}')
else:
    PER_MOD_DIR = ORIG
    print(f'[charts] Using ORIGINAL per_mod directory: {ORIG}')
RESULTS = os.path.join(_BASE, '..', 'results', 'pub_2026')
os.makedirs(CHARTS, exist_ok=True)


# ============================================================================
# Load data: aggregate 3 seeds for each model
# ============================================================================
def load_per_mod_all_seeds():
    """Returns dict: {model_key: list of per_mod dicts across seeds}"""
    data = {}
    for m in MODELS:
        files = sorted(glob.glob(f'{PER_MOD_DIR}/{m}_s*/*per_mod*.json'))
        data[m] = [json.load(open(f)) for f in files]
    return data


def load_freq_offset(model_key, seeds=(42, 43, 44)):
    """Returns: {sep_hz: [metric_dict_for_each_seed]}"""
    out = {}
    for s in seeds:
        f = f'{RESULTS}/freq_offset/{model_key}_s{s}/{model_key}_freq_offset.json'
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for sep_str, metrics in d['results'].items():
            sep = int(sep_str)
            out.setdefault(sep, []).append(metrics)
    return out


def load_gap_experiments():
    """Returns: {gap_hz: metric_dict} from gap-trained models"""
    out = {}
    for gap in [10, 50, 100, 200, 500]:
        f = f'{RESULTS}/freq_offset/gap{gap}/complex_cnn_se_freq_offset.json'
        if os.path.exists(f):
            d = json.load(open(f))
            # Eval the gap-trained model at the SAME gap it was trained on
            sep_str = str(gap)
            if sep_str in d['results']:
                out[gap] = d['results'][sep_str]
    return out


def aggregate_per_pair(per_mod_dicts):
    """Aggregate per_pair values across seeds. Returns dict {pair_key: mean_dict}"""
    pair_data = {}
    for d in per_mod_dicts:
        for k, v in d['per_pair'].items():
            pair_data.setdefault(k, []).append(v)
    out = {}
    for k, lst in pair_data.items():
        out[k] = {m: np.mean([x[m] for x in lst]) for m in ['SI-SDR','SDR','SIR','NMSE']}
        out[k]['_std'] = {m: np.std([x[m] for x in lst]) for m in ['SI-SDR','SDR','SIR','NMSE']}
        out[k]['_n'] = len(lst)
    return out


# ============================================================================
# CHART 1: Headline 4-metric × 4-model with error bars
# ============================================================================
def chart_headline():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics = ['SI-SDR', 'SDR', 'SIR', 'NMSE']

    for ax, metric in zip(axes.flat, metrics):
        keys, vals, stds, colors, labels = [], [], [], [], []
        for m, (label, color, marker) in MODELS.items():
            files = sorted(glob.glob(f'{PER_MOD_DIR}/{m}_s*/*per_mod*.json'))
            seed_metric = []
            for f in files:
                d = json.load(open(f))
                avg_per_pair = np.mean([v[metric] for v in d['per_pair'].values()])
                seed_metric.append(avg_per_pair)
            if seed_metric:
                keys.append(m); vals.append(np.mean(seed_metric))
                stds.append(np.std(seed_metric)); colors.append(color); labels.append(label)
        x = np.arange(len(keys))
        ax.bar(x, vals, yerr=stds, color=colors, alpha=0.85, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace(' (', '\n(') for l in labels], fontsize=9)
        ax.set_ylabel(f'{metric} (dB)')
        ax.set_title(f'{metric}: 4 Models × 3 Seeds')
        if metric == 'SDR':
            for i, v in enumerate(vals):
                ax.text(i, v + max(stds)*1.5 + 0.05, f'{v:.2f}', ha='center', fontsize=9)

    fig.suptitle('Headline: Complex CNN + SE vs Three Baselines (mean ± std, 3 seeds)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{CHARTS}/01_headline_4models.png', dpi=140, bbox_inches='tight')
    plt.savefig(f'{CHARTS}/01_headline_4models.pdf', bbox_inches='tight')
    plt.close()


# ============================================================================
# CHART 2: Per-modulation SI-SDR heatmap (3 models × 4×4)
# ============================================================================
def chart_per_mod_heatmap():
    mods = ['BPSK', 'QPSK', '8PSK', '16QAM']
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    vmin, vmax = -3, 4
    im = None

    for ax, (m_key, (label, _, _)) in zip(axes, MODELS.items()):
        files = sorted(glob.glob(f'{PER_MOD_DIR}/{m_key}_s*/*per_mod*.json'))
        if not files:
            ax.text(0.5, 0.5, 'no data', ha='center', transform=ax.transAxes)
            continue
        pair_data = {}
        for f in files:
            d = json.load(open(f))
            for k, v in d['per_pair'].items():
                pair_data.setdefault(k, []).append(v['SI-SDR'])
        mat = np.full((4, 4), np.nan)
        for i, ma in enumerate(mods):
            for j, mb in enumerate(mods):
                key = f'{ma}-{mb}'
                if key in pair_data:
                    mat[i, j] = np.mean(pair_data[key])
        im = ax.imshow(mat, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(4)); ax.set_xticklabels(mods)
        ax.set_yticks(range(4)); ax.set_yticklabels(mods)
        ax.set_xlabel('Source 2')
        ax.set_ylabel('Source 1')
        ax.set_title(f'{label}\n(mean SI-SDR dB)', fontsize=10)
        for i in range(4):
            for j in range(4):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                            color='black', fontsize=9)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label('SI-SDR (dB)')
    fig.suptitle('Per-Modulation-Pair SI-SDR (3-seed mean)', fontsize=14, y=1.02)
    plt.savefig(f'{CHARTS}/02_per_mod_heatmap.png', dpi=140, bbox_inches='tight')
    plt.close()


# ============================================================================
# CHART 3: Same-modulation SI-SDR (bars by model)
# ============================================================================
def chart_same_mod():
    mods = ['BPSK', 'QPSK', '8PSK', '16QAM']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.2
    x = np.arange(len(mods))

    for ax_idx, metric in enumerate(['SI-SDR', 'SDR']):
        ax = axes[ax_idx]
        for i, (m, (label, color, _)) in enumerate(MODELS.items()):
            files = sorted(glob.glob(f'{PER_MOD_DIR}/{m}_s*/*per_mod*.json'))
            same_mod_vals = {mod: [] for mod in mods}
            for f in files:
                d = json.load(open(f))
                same_mod = d.get('same_mod', {})
                for mod in mods:
                    if mod in same_mod:
                        same_mod_vals[mod].append(same_mod[mod][metric])
            vals = [np.mean(same_mod_vals[mod]) if same_mod_vals[mod] else 0 for mod in mods]
            stds = [np.std(same_mod_vals[mod]) if same_mod_vals[mod] else 0 for mod in mods]
            ax.bar(x + (i-1.5)*width, vals, width, yerr=stds, label=label,
                   color=color, alpha=0.85, capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(mods)
        ax.set_xlabel('Modulation (both sources)')
        ax.set_ylabel(f'{metric} (dB)')
        ax.set_title(f'Same-Modulation {metric}')
        ax.axhline(0, color='black', linewidth=0.6)
        ax.legend(loc='best', fontsize=8)
    fig.suptitle('Same-Modulation Pair Difficulty (BPSK easiest, 16QAM hardest)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{CHARTS}/03_same_modulation.png', dpi=140, bbox_inches='tight')
    plt.close()


# ============================================================================
# CHART 4: Frequency-offset robustness (4 models, log-x)
# ============================================================================
def chart_freq_offset():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    seps = [5, 10, 50, 100, 200, 500]

    for ax_idx, metric in enumerate(['SI-SDR', 'SDR']):
        ax = axes[ax_idx]
        for m, (label, color, marker) in MODELS.items():
            d = load_freq_offset(m)
            ys, xs, yerr = [], [], []
            for s in seps:
                if s in d and d[s]:
                    ys.append(np.mean([m_[metric] for m_ in d[s]]))
                    yerr.append(np.std([m_[metric] for m_ in d[s]]))
                    xs.append(s)
            ax.errorbar(xs, ys, yerr=yerr, marker=marker, color=color, label=label,
                        capsize=4, linewidth=2, markersize=7)
        ax.set_xscale('log')
        ax.set_xlabel('Carrier-frequency separation (Hz)')
        ax.set_ylabel(f'{metric} (dB)')
        ax.set_title(f'{metric} vs Frequency Separation (trained at 5 Hz, mean ± std)')
        ax.set_xticks(seps)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.legend(loc='best', fontsize=8)
    fig.suptitle('Frequency-Offset Robustness (all 4 models, 3 seeds)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{CHARTS}/04_freq_offset.png', dpi=140, bbox_inches='tight')
    plt.savefig(f'{CHARTS}/04_freq_offset.pdf', bbox_inches='tight')
    plt.close()


# ============================================================================
# CHART 5: Gap-training experiment (SE at 5/10/50/100/200/500 Hz, evaluated at matching gap)
# ============================================================================
def chart_gap_training():
    gaps = [5, 10, 50, 100, 200, 500]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Load gap training results (model trained at gap X, evaluated at gap X)
    gap_data = load_gap_experiments()
    # Also need to add the 5Hz result (from baseline SE training)
    files_5hz = sorted(glob.glob(f'{RESULTS}/freq_offset/complex_cnn_se_s*/*'))
    if files_5hz:
        seps_seen = set()
        ys_sdr = []; ys_si = []
        for f in files_5hz:
            d = json.load(open(f))
            for sep_str, metrics in d['results'].items():
                if int(sep_str) == 5:
                    ys_sdr.append(metrics['SDR'])
                    ys_si.append(metrics['SI-SDR'])
                    seps_seen.add(5)
        if 5 in seps_seen:
            gap_data[5] = {'SDR': np.mean(ys_sdr), 'SI-SDR': np.mean(ys_si)}

    # Plot
    sdr_means = [gap_data.get(g, {}).get('SDR', np.nan) for g in gaps]
    si_means = [gap_data.get(g, {}).get('SI-SDR', np.nan) for g in gaps]

    ax.plot(gaps, sdr_means, marker='o', linewidth=2.5, markersize=10,
            color='#1f77b4', label='SDR')
    ax.plot(gaps, si_means, marker='s', linewidth=2.5, markersize=10,
            color='#d62728', label='SI-SDR')
    for g, sdr, si in zip(gaps, sdr_means, si_means):
        if not np.isnan(sdr):
            ax.text(g, sdr + 0.2, f'{sdr:.2f}', ha='center', fontsize=9, color='#1f77b4')
        if not np.isnan(si):
            ax.text(g, si - 0.4, f'{si:.2f}', ha='center', fontsize=9, color='#d62728')

    ax.set_xscale('log')
    ax.set_xlabel('Training frequency gap (Hz)')
    ax.set_ylabel('Metric (dB)')
    ax.set_title('SE Model: Train at gap → Evaluate at matching gap')
    ax.set_xticks(gaps)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{CHARTS}/05_gap_training.png', dpi=140, bbox_inches='tight')
    plt.close()


# ============================================================================
# CHART 6: Parameter count vs SDR (scatter)
# ============================================================================
def chart_params_vs_perf():
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    params = {'complex_cnn_se': 235525, 'complex_cnn_no_se': 202757,
              'real_baseline': 77765, 'conv_tasnet': 816643}
    for m, (label, color, marker) in MODELS.items():
        files = sorted(glob.glob(f'{PER_MOD_DIR}/{m}_s*/*per_mod*.json'))
        seed_sdr = []
        for f in files:
            d = json.load(open(f))
            seed_sdr.append(np.mean([v['SDR'] for v in d['per_pair'].values()]))
        if not seed_sdr:
            continue
        mean_sdr = np.mean(seed_sdr)
        std_sdr = np.std(seed_sdr)
        ax.scatter(params[m], mean_sdr, s=200, color=color, marker=marker,
                   edgecolors='black', linewidth=1.5, zorder=3, label=label)
        ax.errorbar(params[m], mean_sdr, yerr=std_sdr, color=color,
                    capsize=5, linewidth=2, zorder=2)
        ax.annotate(f'{mean_sdr:.2f}±{std_sdr:.2f}',
                    (params[m], mean_sdr), textcoords='offset points',
                    xytext=(10, -5), fontsize=10)
    ax.set_xscale('log')
    ax.set_xlabel('Model parameters (log scale)')
    ax.set_ylabel('Overall SDR (dB, mean ± std across 3 seeds)')
    ax.set_title('Parameter Count vs SDR Performance')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{CHARTS}/06_params_vs_perf.png', dpi=140, bbox_inches='tight')
    plt.savefig(f'{CHARTS}/06_params_vs_perf.pdf', bbox_inches='tight')
    plt.close()


# ============================================================================
# CHART 7: SE block ablation effect (3 models, clear comparison)
# ============================================================================
def chart_se_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # SDR comparison
    for ax, metric in zip(axes, ['SDR', 'SI-SDR']):
        names, vals, stds = [], [], []
        for m in ['complex_cnn_se', 'complex_cnn_no_se']:
            files = sorted(glob.glob(f'{PER_MOD_DIR}/{m}_s*/*per_mod*.json'))
            data = []
            for f in files:
                d = json.load(open(f))
                data.append(np.mean([v[metric] for v in d['per_pair'].values()]))
            names.append(MODELS[m][0])
            vals.append(np.mean(data))
            stds.append(np.std(data))
        x = np.arange(len(names))
        bars = ax.bar(x, vals, yerr=stds,
                      color=[MODELS['complex_cnn_se'][1], MODELS['complex_cnn_no_se'][1]],
                      alpha=0.85, capsize=8, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel(f'{metric} (dB)')
        ax.set_title(f'{metric}: SE Ablation Effect')
        # Annotate
        for i, (v, s) in enumerate(zip(vals, stds)):
            ax.text(i, v + s + 0.05, f'{v:.2f}±{s:.2f}', ha='center', fontsize=10, fontweight='bold')
        # Improvement annotation
        if metric == 'SDR':
            imp = vals[0] - vals[1]
            ax.annotate(f'+{imp:.2f} dB', xy=(0.5, max(vals)+0.6),
                        ha='center', fontsize=12, fontweight='bold', color='green',
                        arrowprops=dict(arrowstyle='->', color='green'))

    fig.suptitle('Complex SE Block Contribution (same arch, 3 seeds each)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{CHARTS}/07_se_ablation.png', dpi=140, bbox_inches='tight')
    plt.savefig(f'{CHARTS}/07_se_ablation.pdf', bbox_inches='tight')
    plt.close()


# ============================================================================
# CHART 8: ConvTasNet vs SE comparison (per-mod pair)
# ============================================================================
def chart_se_vs_tasnet():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pairs = ['BPSK-BPSK', 'QPSK-QPSK', '8PSK-8PSK', '16QAM-16QAM']

    for ax, metric in zip(axes, ['SI-SDR', 'SDR']):
        # Compute mean across seeds for both models
        def get_pair_metric(model, pair, metric):
            files = sorted(glob.glob(f'{PER_MOD_DIR}/{model}_s*/*per_mod*.json'))
            vals = []
            for f in files:
                d = json.load(open(f))
                if pair in d['per_pair']:
                    vals.append(d['per_pair'][pair][metric])
            return np.mean(vals) if vals else np.nan, np.std(vals) if vals else np.nan

        x = np.arange(len(pairs))
        width = 0.35
        for i, (m, label, color) in enumerate([
            ('complex_cnn_se', 'Complex CNN + SE\n(Proposed, 235K)', '#1f77b4'),
            ('conv_tasnet',     'Conv-TasNet\n(SOTA baseline, 817K)', '#d62728')
        ]):
            means, stds = [], []
            for p in pairs:
                mu, sd = get_pair_metric(m, p, metric)
                means.append(mu); stds.append(sd)
            ax.bar(x + (i-0.5)*width, means, width, yerr=stds, label=label,
                   color=color, alpha=0.85, capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(pairs)
        ax.set_xlabel('Same-modulation pair')
        ax.set_ylabel(f'{metric} (dB)')
        ax.set_title(f'{metric}: SE vs Conv-TasNet')
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.legend(loc='best', fontsize=10)
    fig.suptitle('Direct Comparison: Proposed SE Model vs Conv-TasNet SOTA Baseline',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{CHARTS}/08_se_vs_tasnet.png', dpi=140, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print('Generating publication charts...')
    chart_headline()
    print('  ✓ 01_headline_4models.png')
    chart_per_mod_heatmap()
    print('  ✓ 02_per_mod_heatmap.png')
    chart_same_mod()
    print('  ✓ 03_same_modulation.png')
    chart_freq_offset()
    print('  ✓ 04_freq_offset.png')
    chart_gap_training()
    print('  ✓ 05_gap_training.png')
    chart_params_vs_perf()
    print('  ✓ 06_params_vs_perf.png')
    chart_se_ablation()
    print('  ✓ 07_se_ablation.png')
    chart_se_vs_tasnet()
    print('  ✓ 08_se_vs_tasnet.png')
    print('\nAll 8 charts saved to results/charts/')
    import os as _os
    for f in sorted(_os.listdir('results/charts')):
        if f.endswith('.png'):
            sz = _os.path.getsize(f'results/charts/{f}') / 1024
            print(f'  {f}  ({sz:.0f} KB)')