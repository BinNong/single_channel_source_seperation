"""
Frequency-offset robustness evaluation.

Tests how well a trained model generalizes across different carrier-frequency
separations between the two co-channel sources. Larger separation = easier
separation task. Default training uses 2000 vs 2005 Hz (5 Hz gap).

Usage:
    python eval_freq_offset.py --model complex_cnn_se --checkpoint <path>
    python eval_freq_offset.py --model complex_cnn_se --checkpoint <path> \
        --separations 5 10 50 100 500 1000
"""
import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import SignalConfig, DEVICE
from data_generator import generate_mixture
from models import ComplexLightweightSepNet, RealValuedBaseline, SimpleComplexCNN, ComplexConvTasNet
from utils import evaluate_batch


def get_args():
    p = argparse.ArgumentParser(description='Frequency-offset robustness eval')
    p.add_argument('--model', type=str, required=True,
                   choices=['complex_cnn_se', 'complex_cnn_no_se', 'real_baseline', 'simple_complex', 'conv_tasnet'])
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--separations', type=float, nargs='+',
                   default=[5, 10, 50, 100, 500],
                   help='Carrier-frequency separations to test (Hz)')
    p.add_argument('--n_per_sep', type=int, default=200,
                   help='Number of test mixtures per separation')
    p.add_argument('--snr_db', type=float, default=10.0,
                   help='Fixed SNR for the robustness test (dB)')
    p.add_argument('--output_dir', type=str, default='./results/freq_offset')
    return p.parse_args()


def build_model(name, hidden, layers):
    if name == 'complex_cnn_se':
        return ComplexLightweightSepNet(hidden_channels=hidden, n_layers=layers, use_se=True)
    if name == 'complex_cnn_no_se':
        return ComplexLightweightSepNet(hidden_channels=hidden, n_layers=layers, use_se=False)
    if name == 'real_baseline':
        return RealValuedBaseline()
    if name == 'simple_complex':
        return SimpleComplexCNN()
    if name == 'conv_tasnet':
        return ComplexConvTasNet(N=64, B=64, Sc=64, H=128, P=3, X=5, R=3, L=16)
    raise ValueError(name)


@torch.no_grad()
def eval_at_separation(model, sep_hz, n_samples, snr_db, device):
    """Evaluate model at a given carrier-frequency separation."""
    model.eval()
    accum = {'SI-SDR': [], 'SDR': [], 'SIR': [], 'NMSE': []}
    cf1 = SignalConfig.carrier_freq_1
    cf2 = cf1 + sep_hz
    for i in range(n_samples):
        # Use a deterministic-but-varied config: random modulation pair
        mod1 = SignalConfig.mod_types_test[i % len(SignalConfig.mod_types_test)]
        mod2 = SignalConfig.mod_types_test[(i + 1) % len(SignalConfig.mod_types_test)]
        mix, s1, s2, _, _ = generate_mixture(
            signal_length=SignalConfig.signal_length,
            sample_rate=SignalConfig.sample_rate,
            snr_db=snr_db,
            mod_type_1=mod1, mod_type_2=mod2,
            carrier_freq_1=cf1, carrier_freq_2=cf2,
            n_symbols=SignalConfig.n_symbols,
            roll_off=SignalConfig.roll_off, num_taps=SignalConfig.num_taps,
            apply_fading=SignalConfig.apply_fading, fading_taps=SignalConfig.fading_taps,
        )
        m_t = torch.from_numpy(mix).unsqueeze(0).unsqueeze(0).to(torch.complex64).to(device)
        s1_t = torch.from_numpy(s1).unsqueeze(0).unsqueeze(0).to(torch.complex64).to(device)
        s2_t = torch.from_numpy(s2).unsqueeze(0).unsqueeze(0).to(torch.complex64).to(device)
        e1, e2 = model(m_t)
        m = evaluate_batch((e1, e2), (s1_t, s2_t))
        for k in accum:
            accum[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in accum.items()}


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = build_model(args.model, args.hidden, args.layers).to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded: {args.checkpoint}  (epoch={ckpt['epoch']}, best={ckpt['best_metric']:.4f})")
    print(f"Device: {DEVICE}")
    print(f"Test SNR: {args.snr_db} dB, N per separation: {args.n_per_sep}")
    print(f"Separations (Hz): {args.separations}")
    print("-" * 70)
    print(f"{'Sep (Hz)':>10} | {'SI-SDR':>8} | {'SDR':>8} | {'SIR':>8} | {'NMSE':>8}")
    print("-" * 70)

    results = {}
    for sep in args.separations:
        r = eval_at_separation(model, sep, args.n_per_sep, args.snr_db, DEVICE)
        results[sep] = r
        print(f"{sep:>10} | {r['SI-SDR']:>8.2f} | {r['SDR']:>8.2f} | "
              f"{r['SIR']:>8.2f} | {r['NMSE']:>8.2f}")

    out_path = f'{args.output_dir}/{args.model}_freq_offset.json'
    with open(out_path, 'w') as f:
        json.dump({'model': args.model, 'snr_db': args.snr_db,
                   'n_per_sep': args.n_per_sep, 'results': results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()