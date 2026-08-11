"""
Evaluation script for trained models.

Evaluates on test sets with different SNR points and modulation types,
generates comparison tables and visualizations.

Usage:
    python evaluate.py --model complex_cnn_se --checkpoint checkpoints/xxx_best.pt
    python evaluate.py --model real_baseline --checkpoint checkpoints/yyy_best.pt --compare
"""

import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from config import SignalConfig, DataConfig, ModelConfig, DEVICE
from data_generator import CommBSSTestDataset
from models import (ComplexLightweightSepNet, RealValuedBaseline,
                    SimpleComplexCNN, ComplexConvTasNet, CNSE, S4UNET)
from utils import evaluate_batch, visualize_separation, compute_sdr, compute_sir, compute_nmse


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate Communication Signal BSS Model')
    parser.add_argument('--model', type=str, required=True,
                        choices=['complex_cnn_se', 'complex_cnn_no_se', 'real_baseline', 'simple_complex', 'conv_tasnet', 'cnse', 's4unet'])
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--baseline_hidden', type=int, default=64,
                        help='Hidden channels for real_baseline (must match checkpoint)')
    parser.add_argument('--baseline_layers', type=int, default=6,
                        help='Number of middle layers for real_baseline (must match checkpoint)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--compare', action='store_true', help='Compare multiple models')
    parser.add_argument('--viz', action='store_true', default=True, help='Generate visualizations')
    parser.add_argument('--no_viz', dest='viz', action='store_false', help='Skip visualizations')
    parser.add_argument('--per_mod', action='store_true', help='Also report per-modulation-pair breakdown')
    parser.add_argument('--se_scale_mode', type=str, default='real',
                        choices=['real', 'complex_mean', 'separate'],
                        help='C-SE scale mode (only for complex_cnn_se, must match checkpoint)')
    parser.add_argument('--se_pooling_mode', type=str, default='mean',
                        choices=['mean', 'power', 'magnitude', 'mean+power'],
                        help='C-SE squeeze pooling mode (only for complex_cnn_se, must match checkpoint)')
    parser.add_argument('--cnse_hidden', type=int, default=256,
                        help='CNSE hidden channels (must match checkpoint)')
    parser.add_argument('--s4_base_channels', type=int, default=16,
                        help='S4-UNET base channels (must match checkpoint)')
    parser.add_argument('--s4_state_dim', type=int, default=16,
                        help='S4-UNET SSM state dimension (must match checkpoint)')
    return parser.parse_args()


def _si_sdr_per_sample(estimate, reference, eps=1e-8):
    """Per-sample SI-SDR (shape [B])."""
    from utils import si_sdr as _si_sdr_full
    return _si_sdr_full(estimate, reference, eps=eps, return_per_sample=True)


def _ser_per_sample(est_t, ref_t, mod_type,
                    sample_rate=16000, n_symbols=256, num_taps=64,
                    roll_off=0.35, carrier_freq=2000.0):
    """SER per sample (scalar 0..1). Wrapper around utils.compute_ser_from_signal."""
    from utils import compute_ser_from_signal
    return compute_ser_from_signal(
        est_t, ref_t, mod_type=mod_type,
        sample_rate=sample_rate, n_symbols=n_symbols,
        num_taps=num_taps, roll_off=roll_off, carrier_freq=carrier_freq)


@torch.no_grad()
def _per_sample_metrics(est1, est2, orig1, orig2, mod1_list=None, mod2_list=None):
    """Returns dict of [B]-shaped tensors / numpy arrays.

    Uses the SAME permutation (best SI-SDR) for every metric so that
    SI-SDR >= SDR per-sample by construction.

    Args:
        mod1_list, mod2_list: optional lists of strings (length B) — mod types
                              per sample, used to compute SER.
    """
    # 1. Determine optimal permutation by SI-SDR
    sisdr_11 = _si_sdr_per_sample(est1, orig1)
    sisdr_22 = _si_sdr_per_sample(est2, orig2)
    sisdr_12 = _si_sdr_per_sample(est1, orig2)
    sisdr_21 = _si_sdr_per_sample(est2, orig1)

    perm1 = (sisdr_11 + sisdr_22) / 2
    perm2 = (sisdr_12 + sisdr_21) / 2
    use_swap = perm2 > perm1  # [B]
    swap_mask = use_swap.view(-1, 1, 1)

    o1_aligned = torch.where(swap_mask, orig2, orig1)
    o2_aligned = torch.where(swap_mask, orig1, orig2)

    # 2. Compute all metrics on the aligned references
    si_best = torch.where(use_swap, perm2, perm1)            # [B]
    sdr_vals = compute_sdr((est1, est2), (o1_aligned, o2_aligned),
                           return_per_sample=True)
    sir_vals = compute_sir((est1, est2), (o1_aligned, o2_aligned))
    nmse1 = compute_nmse(est1, o1_aligned)
    nmse2 = compute_nmse(est2, o2_aligned)

    result = {
        'SI-SDR': si_best.detach().cpu().numpy(),
        'SDR':    sdr_vals.detach().cpu().numpy(),
        'SIR':    sir_vals.detach().cpu().numpy(),
        'NMSE':   ((nmse1 + nmse2) / 2).detach().cpu().numpy(),
    }

    # 3. Optional SER (slow: matched-filter + min-distance demodulation)
    if mod1_list is not None and mod2_list is not None:
        B = est1.shape[0]
        ser = np.zeros(B, dtype=np.float32)
        for i in range(B):
            try:
                ser[i] = _ser_per_sample(est1[i, 0], o1_aligned[i, 0],
                                         mod1_list[i])
                ser[i] = 0.5 * (ser[i] +
                                _ser_per_sample(est2[i, 0], o2_aligned[i, 0],
                                                mod2_list[i]))
            except Exception:
                ser[i] = np.nan
        result['SER'] = ser

    return result


@torch.no_grad()
def evaluate_model(model, test_loader, device, per_mod=False):
    """Evaluate model on test set, grouped by SNR (and optionally by mod pair)."""
    model.eval()

    snr_results = {snr: {'SI-SDR': [], 'SDR': [], 'SIR': [], 'NMSE': []}
                   for snr in SignalConfig.snr_test_points}
    all_results = {'SI-SDR': [], 'SDR': [], 'SIR': [], 'NMSE': []}
    mod_results = {} if per_mod else None

    for mixture, source1, source2, snr, mod1, mod2 in test_loader:
        mixture = mixture.to(device)
        source1 = source1.to(device)
        source2 = source2.to(device)

        est1, est2 = model(mixture)

        if per_mod:
            ps = _per_sample_metrics(est1, est2, source1, source2,
                                     mod1_list=list(mod1), mod2_list=list(mod2))
            for k, arr in ps.items():
                if k not in all_results:
                    all_results[k] = []
                all_results[k].extend(arr.tolist())
            # Group by SNR
            for i, s in enumerate(snr.numpy()):
                s = int(s)
                if s in snr_results:
                    for k, arr in ps.items():
                        snr_results[s].setdefault(k, []).append(arr[i])
            # Group by (mod1, mod2) per sample
            for i in range(len(mod1)):
                key = (str(mod1[i]), str(mod2[i]))
                if key not in mod_results:
                    mod_results[key] = {k: [] for k in ps}
                for k, arr in ps.items():
                    mod_results[key][k].append(arr[i])
        else:
            metrics = evaluate_batch((est1, est2), (source1, source2))
            for i, s in enumerate(snr.numpy()):
                s = int(s)
                if s in snr_results:
                    for k in metrics:
                        snr_results[s][k].append(metrics[k])
            for k in metrics:
                all_results[k].append(metrics[k])

    summary = {}
    for snr in SignalConfig.snr_test_points:
        summary[snr] = {k: float(np.mean(v)) if v else 0.0
                        for k, v in snr_results[snr].items()}

    overall = {k: float(np.mean(v)) if v else 0.0 for k, v in all_results.items()}

    mod_summary = None
    if per_mod and mod_results:
        same_mod_summary = {}
        diff_mod_accum = {}
        per_pair_summary = {}
        for (m1, m2), vals in mod_results.items():
            per_pair_summary[f"{m1}-{m2}"] = {k: float(np.mean(v)) for k, v in vals.items()}
            if m1 == m2:
                same_mod_summary[m1] = {k: float(np.mean(v)) for k, v in vals.items()}
            else:
                key = f"{m1}-{m2}" if f"{m1}-{m2}" < f"{m2}-{m1}" else f"{m2}-{m1}"
                if key not in diff_mod_accum:
                    diff_mod_accum[key] = {k: [] for k in vals}
                for k in vals:
                    diff_mod_accum[key][k].extend(vals[k])
        diff_mod_summary = {k: {kk: float(np.mean(vv)) for kk, vv in v.items()}
                            for k, v in diff_mod_accum.items()}
        mod_summary = {'per_pair': per_pair_summary,
                       'same_mod': same_mod_summary,
                       'diff_mod': diff_mod_summary}

    return summary, overall, mod_summary


def print_results_table(model_name, summary, overall):
    """Print results in a formatted table."""
    print(f"\n{'='*70}")
    print(f"Results: {model_name}")
    print(f"{'='*70}")

    # SNR-dependent results
    has_ser = 'SER' in overall
    if has_ser:
        header = f"\n{'SNR (dB)':>10} | {'SI-SDR':>9} | {'SDR':>9} | {'SIR':>9} | {'NMSE':>9} | {'SER':>9}"
    else:
        header = f"\n{'SNR (dB)':>10} | {'SI-SDR':>10} | {'SDR':>10} | {'SIR':>10} | {'NMSE':>10}"
    print(header)
    print("-" * len(header))
    for snr in sorted(summary.keys()):
        r = summary[snr]
        row = (f"{snr:>10} | {r['SI-SDR']:>9.2f} | {r['SDR']:>9.2f} | "
               f"{r['SIR']:>9.2f} | {r['NMSE']:>9.2f}")
        if has_ser:
            ser_val = r.get('SER', 0.0)
            row += f" | {ser_val:>9.4f}"
        print(row)

    print("-" * len(header))
    row = (f"{'Overall':>10} | {overall['SI-SDR']:>9.2f} | {overall['SDR']:>9.2f} | "
           f"{overall['SIR']:>9.2f} | {overall['NMSE']:>9.2f}")
    if has_ser:
        row += f" | {overall['SER']:>9.4f}"
    print(row)

    # Save as CSV
    rows = []
    for snr in sorted(summary.keys()):
        row = summary[snr].copy()
        row['SNR'] = snr
        row['Model'] = model_name
        rows.append(row)
    row = overall.copy()
    row['SNR'] = 'Overall'
    row['Model'] = model_name
    rows.append(row)

    return pd.DataFrame(rows)


def compare_models(results_dict, output_dir):
    """Compare multiple models and save comparison table."""
    print(f"\n{'='*70}")
    print("Model Comparison")
    print(f"{'='*70}")

    all_dfs = []
    for name, (_, overall) in results_dict.items():
        row = overall.copy()
        row['Model'] = name
        all_dfs.append(row)

    df = pd.DataFrame(all_dfs)
    print(f"\n{df.to_string(index=False)}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f'{output_dir}/comparison.csv', index=False)
    print(f"\nComparison saved: {output_dir}/comparison.csv")

    # LaTeX table for paper
    latex = df.to_latex(index=False, float_format='%.2f')
    with open(f'{output_dir}/comparison_table.tex', 'w') as f:
        f.write(latex)
    print(f"LaTeX table saved: {output_dir}/comparison_table.tex")


def visualize_samples(model, test_dataset, device, output_dir, n_samples=5):
    """Generate visualization figures."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    for i in range(n_samples):
        mixture, source1, source2, snr, mod1, mod2 = test_dataset[i]
        mixture_b = mixture.unsqueeze(0).to(device)
        source1_b = source1.unsqueeze(0).to(device)
        source2_b = source2.unsqueeze(0).to(device)

        est1, est2 = model(mixture_b)

        save_path = f'{output_dir}/figures/sample_{i}_SNR{int(snr)}_{mod1}_{mod2}.png'
        visualize_separation(mixture_b[0], source1_b[0], source2_b[0],
                             est1[0], est2[0], save_path)

    print(f"Visualizations saved to {output_dir}/figures/")


def main():
    args = get_args()

    # Build model
    if args.model == 'complex_cnn_se':
        model = ComplexLightweightSepNet(
            hidden_channels=args.hidden, n_layers=args.layers, use_se=True,
            se_scale_mode=args.se_scale_mode,
            se_pooling_mode=args.se_pooling_mode
        ).to(DEVICE)
    elif args.model == 'complex_cnn_no_se':
        model = ComplexLightweightSepNet(
            hidden_channels=args.hidden, n_layers=args.layers, use_se=False
        ).to(DEVICE)
    elif args.model == 'real_baseline':
        model = RealValuedBaseline(
            hidden=args.baseline_hidden, n_layers=args.baseline_layers
        ).to(DEVICE)
    elif args.model == 'simple_complex':
        model = SimpleComplexCNN().to(DEVICE)
    elif args.model == 'conv_tasnet':
        model = ComplexConvTasNet(N=64, B=64, Sc=64, H=128, P=3, X=5, R=3, L=16).to(DEVICE)
    elif args.model == 'cnse':
        model = CNSE(hidden=args.cnse_hidden, n_stacks=3).to(DEVICE)
    elif args.model == 's4unet':
        model = S4UNET(base_channels=args.s4_base_channels,
                       state_dim=args.s4_state_dim).to(DEVICE)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Epoch: {checkpoint['epoch']}, Best metric: {checkpoint['best_metric']:.4f}")

    # Test dataset
    print("\nLoading test dataset...")
    test_dataset = CommBSSTestDataset(
        n_per_snr=500,
        snr_points=SignalConfig.snr_test_points,
        mod_types=SignalConfig.mod_types_test,
        signal_length=SignalConfig.signal_length,
        sample_rate=SignalConfig.sample_rate,
        seed=99999
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"  Test samples: {len(test_dataset):,}")

    # Evaluate
    print("\nEvaluating...")
    summary, overall, mod_summary = evaluate_model(model, test_loader, DEVICE, per_mod=args.per_mod)

    # Print results
    df = print_results_table(args.model, summary, overall)

    # Print per-modulation breakdown
    if args.per_mod and mod_summary:
        os.makedirs(args.output_dir, exist_ok=True)
        print("\n" + "=" * 70)
        print(f"Per-Modulation-Pair Breakdown: {args.model}")
        print("=" * 70)
        for key, vals in mod_summary['per_pair'].items():
            print(f"  {key:>14}: SI-SDR={vals['SI-SDR']:6.2f}  SDR={vals['SDR']:6.2f}  "
                  f"SIR={vals['SIR']:6.2f}  NMSE={vals['NMSE']:6.2f}")
        print("\nSame-modulation pairs:")
        for mod, vals in mod_summary['same_mod'].items():
            print(f"  {mod:>10}: SI-SDR={vals['SI-SDR']:6.2f}  SDR={vals['SDR']:6.2f}  "
                  f"SIR={vals['SIR']:6.2f}  NMSE={vals['NMSE']:6.2f}")
        print("\nDifferent-modulation pairs (averaged over both orderings):")
        for key, vals in mod_summary['diff_mod'].items():
            print(f"  {key:>14}: SI-SDR={vals['SI-SDR']:6.2f}  SDR={vals['SDR']:6.2f}  "
                  f"SIR={vals['SIR']:6.2f}  NMSE={vals['NMSE']:6.2f}")
        # Save as JSON
        with open(f'{args.output_dir}/{args.model}_per_mod.json', 'w') as f:
            json.dump(mod_summary, f, indent=2)
        print(f"\nPer-mod breakdown saved: {args.output_dir}/{args.model}_per_mod.json")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(f'{args.output_dir}/{args.model}_results.csv', index=False)
    print(f"\nResults saved: {args.output_dir}/{args.model}_results.csv")

    # Visualizations
    if args.viz:
        print("\nGenerating visualizations...")
        visualize_samples(model, test_dataset, DEVICE, args.output_dir, n_samples=5)

    # Save overall metrics as JSON
    with open(f'{args.output_dir}/{args.model}_overall.json', 'w') as f:
        json.dump(overall, f, indent=2)

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
