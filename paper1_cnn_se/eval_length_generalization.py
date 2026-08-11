"""
Length-generalization test: evaluate the proposed model on signal lengths
T = 2048, 4096 (training), 8192. Shows that the model is not overfit to T=4096.

For each length:
  - 200 samples per modulation pair (10 mod-pair types, both orderings, plus 4 same-mod)
  - SNR = 10 dB
  - 3 random seeds (different seed than training)
"""
import os
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from config import SignalConfig, DEVICE
from data_generator import CommBSSTestDataset
from models import ComplexLightweightSepNet
from utils import evaluate_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--se_scale_mode', type=str, default='real')
    ap.add_argument('--lengths', type=int, nargs='+', default=[2048, 4096, 8192])
    ap.add_argument('--n_per', type=int, default=50, help='samples per (snr, mod-pair)')
    ap.add_argument('--snr', type=float, default=10.0)
    ap.add_argument('--output', type=str, default='../results/length_gen.json')
    args = ap.parse_args()

    model = ComplexLightweightSepNet(
        hidden_channels=args.hidden, n_layers=args.layers,
        use_se=True, se_scale_mode=args.se_scale_mode
    ).to(DEVICE)
    ck = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    print(f'Loaded {args.checkpoint} (epoch={ck["epoch"]})')

    out = {'checkpoint': args.checkpoint,
           'snr_db': args.snr, 'n_per_pair': args.n_per, 'lengths': {}}

    mod_types = SignalConfig.mod_types_test
    # Build ordered pairs (mod1, mod2), mod1<=mod2 to keep modest set
    pairs = [(m1, m2) for i, m1 in enumerate(mod_types)
                       for j, m2 in enumerate(mod_types) if i <= j]

    for T in args.lengths:
        print(f'\n=== Signal length T={T} ===')
        ds = CommBSSTestDataset(
            n_per_snr=args.n_per, snr_points=[args.snr],
            mod_types=mod_types, signal_length=T,
            sample_rate=SignalConfig.sample_rate, seed=88888,
        )
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        per_pair_metrics = {f'{a}-{b}': [] for a, b in pairs}
        all_metrics = {'SI-SDR': [], 'SDR': [], 'SIR': [], 'NMSE': []}
        with torch.no_grad():
            for mixture, s1, s2, snr, mod1, mod2 in loader:
                mixture = mixture.to(DEVICE); s1 = s1.to(DEVICE); s2 = s2.to(DEVICE)
                e1, e2 = model(mixture)
                m = evaluate_batch((e1, e2), (s1, s2))
                for k in m: all_metrics[k].append(m[k])
                for i in range(len(mod1)):
                    key = f'{mod1[i]}-{mod2[i]}'
                    if key not in per_pair_metrics:
                        key = f'{mod2[i]}-{mod1[i]}'
                    per_pair_metrics[key].append(m['SDR'])
        overall = {k: float(np.mean(v)) for k, v in all_metrics.items()}
        # pairwise mean SDR (only include pairs that actually appeared)
        pair_sdr = {k: float(np.mean(v)) if v else None
                    for k, v in per_pair_metrics.items()}
        print(f'  overall SDR={overall["SDR"]:.3f}, SI-SDR={overall["SI-SDR"]:.3f}, '
              f'SIR={overall["SIR"]:.3f}, NMSE={overall["NMSE"]:.3f}')
        out['lengths'][str(T)] = {
            'overall': overall,
            'per_pair_sdr': pair_sdr,
            'n_samples': len(ds),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()