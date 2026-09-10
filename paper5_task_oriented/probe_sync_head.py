"""Paper 5 — controlled probe: is blind burst-level sync the blocker for
the joint demod head?  (verifies the S3 v2 root-cause HYPOTHESIS)

Design (A/B, no separation, no PIT, fixed K=1):
  variant P (passband) : head input = the PASSBAND received signal
      (source through fading + AWGN; carrier 2000-2010 Hz, ±5 Hz jitter,
      arbitrary initial phase) — the head must do blind carrier/phase
      recovery, exactly what we claim it cannot do.
  variant S (synced)   : head input = the SAME signal after ORACLE
      down-conversion at the true carrier AND phase alignment against the
      true source (receiver-style, whole-burst LS phase fit) — the
      ceiling: sync solved by an oracle.

Same head architecture for both (the v2 JointLLRHead tail: 1x1 -> learned
strided conv to symbol rate -> 2 symbol-rate blocks -> readout), same
data, same seeds.  If S learns and P does not, the S3 v2 failure is
causally attributable to blind sync, not to head capacity.

Usage: python probe_sync_head.py [--n_train 2000] [--epochs 30]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

import config as C
from data_generator_vark import (generate_vark_mixture, MOD_TYPES,
                                 MOD_TO_IDX)
from soft_demod import ref_hard_labels, CONST_MASK
from models import ComplexConv1d, ComplexBatchNorm1d, ComplexReLU


class ProbeHead(nn.Module):
    """v2 JointLLRHead tail, single complex input channel."""

    def __init__(self, sps: int = 16, n_out: int = 16):
        super().__init__()
        self.in_proj = nn.Sequential(
            ComplexConv1d(1, 32, kernel_size=1),
            ComplexBatchNorm1d(32),
            ComplexReLU(),
        )
        self.to_symrate = nn.Sequential(
            ComplexConv1d(32, 32, kernel_size=2 * sps, stride=sps,
                          padding=sps // 2),
            ComplexBatchNorm1d(32),
            ComplexReLU(),
        )
        self.sym_blocks = nn.Sequential(
            ComplexConv1d(32, 32, kernel_size=5, padding=2),
            ComplexBatchNorm1d(32),
            ComplexReLU(),
            ComplexConv1d(32, 32, kernel_size=5, padding=2),
            ComplexBatchNorm1d(32),
            ComplexReLU(),
        )
        self.to_logits = nn.Conv1d(64, n_out, kernel_size=1)

    def forward(self, x):                # x: [B, 1, T] complex
        h = self.in_proj(x)
        h = self.to_symrate(h)
        h = self.sym_blocks(h)
        h = torch.cat([h.real, h.imag], dim=1)
        return self.to_logits(h).transpose(1, 2)   # [B, N, 16]


def make_data(n, snr_range, seed):
    """K=1 samples: returns mix[T], src[T], mod int, true_carrier float."""
    rng = np.random.RandomState(seed)
    data = []
    st = np.random.get_state()
    np.random.seed(seed)
    try:
        for _ in range(n):
            snr = float(rng.uniform(*snr_range))
            mix, srcs, midx, cars = generate_vark_mixture(
                C.SignalConfig.signal_length, C.SignalConfig.sample_rate,
                snr, MOD_TYPES, k=1,
                carrier_base=C.SignalConfig.carrier_base,
                freq_gap_range=C.SignalConfig.freq_gap_range,
                n_symbols=C.SignalConfig.n_symbols,
                roll_off=C.SignalConfig.roll_off,
                num_taps=C.SignalConfig.num_taps,
                apply_fading=C.SignalConfig.apply_fading,
                fading_taps=C.SignalConfig.fading_taps,
                return_carriers=True)
            data.append((mix, srcs[0], midx[0], cars[0]))
    finally:
        np.random.set_state(st)
    return data


def oracle_sync(mix, src, carrier, sample_rate):
    """Down-convert at the TRUE carrier + whole-burst LS phase alignment
    against the TRUE source (the evaluation receiver's sync, oracle level)."""
    T = len(mix)
    t = np.arange(T) / sample_rate
    mix_bb = mix * np.exp(-1j * 2 * np.pi * carrier * t)
    src_bb = src * np.exp(-1j * 2 * np.pi * carrier * t)
    z = np.sum(src_bb * np.conj(mix_bb)) / (np.sum(np.abs(mix_bb) ** 2) + 1e-10)
    return mix_bb * np.exp(1j * np.angle(z))


def evaluate(model, data, variant, device):
    model.eval()
    per_mod_err: dict[int, list[float]] = {}
    with torch.no_grad():
        for mix, src, midx, car in data:
            x_np = mix if variant == 'P' else oracle_sync(
                mix, src, car, C.SignalConfig.sample_rate)
            x = torch.from_numpy(x_np).view(1, 1, -1).to(torch.complex64) \
                .to(device)
            logits = model(x)[0]                       # [N, 16]
            lab = ref_hard_labels(
                torch.from_numpy(src).view(1, -1).to(torch.complex64).to(device),
                torch.tensor([car]), torch.tensor([midx]),
                sample_rate=C.SignalConfig.sample_rate,
                n_symbols=C.SignalConfig.n_symbols,
                roll_off=C.SignalConfig.roll_off,
                num_taps=C.SignalConfig.num_taps)[0]
            n_pts = int(CONST_MASK[midx].sum())
            est = logits[:, :n_pts].argmax(dim=-1)
            n = min(len(est), len(lab))
            per_mod_err.setdefault(midx, []).append(
                float((est[:n] != lab[:n]).float().mean()))
    return {MOD_TYPES[m]: float(np.mean(v)) for m, v in per_mod_err.items()}


def train_variant(variant, train_data, val_data, device, epochs, lr):
    torch.manual_seed(42)
    model = ProbeHead().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.RandomState(0)
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_data))
        tot, cnt = 0.0, 0
        for i in range(0, len(order), 16):
            idx = order[i:i + 16]
            xs, labs, ms, cs = [], [], [], []
            for j in idx:
                mix, src, midx, car = train_data[j]
                x_np = mix if variant == 'P' else oracle_sync(
                    mix, src, car, C.SignalConfig.sample_rate)
                xs.append(torch.from_numpy(x_np))
                ms.append(midx)
                cs.append(car)
            x = torch.stack(xs).unsqueeze(1).to(torch.complex64).to(device)
            src_t = torch.stack([
                torch.from_numpy(train_data[j][1]) for j in idx]
            ).to(torch.complex64).to(device)
            car_t = torch.tensor(cs, dtype=torch.float32, device=device)
            mod_t = torch.tensor(ms, dtype=torch.long, device=device)
            lab = ref_hard_labels(
                src_t, car_t, mod_t,
                sample_rate=C.SignalConfig.sample_rate,
                n_symbols=C.SignalConfig.n_symbols,
                roll_off=C.SignalConfig.roll_off,
                num_taps=C.SignalConfig.num_taps)       # [B, N]
            logits = model(x)                            # [B, N, 16]
            vmask = CONST_MASK.to(device)[mod_t]         # [B, 16]
            B, N, _ = logits.shape
            lg = logits.reshape(B * N, -1).masked_fill(
                ~vmask.unsqueeze(1).expand(B, N, 16).reshape(B * N, 16),
                -1e9)
            loss = nn.functional.cross_entropy(lg, lab.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            cnt += 1
        if ep % 5 == 0 or ep == 1:
            ser = evaluate(model, val_data, variant, device)
            overall = float(np.mean(list(ser.values())))
            print(f"  [{variant}] ep{ep:>2d}  loss={tot / cnt:.4f}  "
                  f"val SER={overall:.4f}  per-mod: "
                  + '  '.join(f'{m}:{v:.3f}' for m, v in sorted(ser.items())))
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_train', type=int, default=2000)
    p.add_argument('--n_val', type=int, default=400)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()
    device = C.DEVICE

    print(f"Building K=1 probe data (train {args.n_train}, val {args.n_val})")
    train_data = make_data(args.n_train, C.SignalConfig.snr_range_train, 42)
    val_data = make_data(args.n_val, C.SignalConfig.snr_range_train, 1042)

    print("\n=== Variant P (passband — blind sync required) ===")
    train_variant('P', train_data, val_data, device, args.epochs, args.lr)
    print("\n=== Variant S (oracle sync — ceiling) ===")
    train_variant('S', train_data, val_data, device, args.epochs, args.lr)

    print("\nVerdict rule: S learns & P fails  => blind burst-level sync is "
          "the blocker (S3 v2 diagnosis CONFIRMED);"
          " P learns too => diagnosis REJECTED.")


if __name__ == '__main__':
    main()
