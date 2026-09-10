"""Paper 5 — reviewer experiment (2): K=2 oracle-sync probe (extension of
probe_sync_head.py, which was fixed K=1).

Question: the K=1 probe showed a synced head reaches val SER 0.150 while
the blind head stays at chance (0.68) — blind burst sync is the blocker
for a joint demod head.  Does an ORACLE-synced head also beat the S2
waveform-routing pipeline at K=2 (the interference-limited regime)?
Reference points (S2, K=2 cells):
    waveform-routing SER (config b, mse) : 0.5475
    mixture baseline K=2                 : 0.5611
    chance (blind head, K=1 probe)       : ~0.68

Variants (same head, same data, same seeds as the K=1 probe):
  variant P (passband)      : input = raw passband K=2 mixture, 1 channel —
                              blind separation AND blind sync.
  variant S (freq-oracle)   : input = 2 channels, channel j = mixture
                              down-converted at the TRUE carrier f_j
                              (mix * exp(-j2 pi f_j t)).  NOTE: for a K=2
                              mixture, per-source PHASE alignment against
                              the true isolated source is impossible (the
                              phases are entangled in the mixture) — the
                              oracle is frequency-only.  This is stated in
                              the paper as an inherent constraint of the
                              K=2 oracle.

Head: K=1 ProbeHead trunk (1x1 -> strided-to-symbol-rate -> 2 symbol-rate
blocks), input channels = variant-dependent (1 or 2); the 1x1 readout is
widened to 2*16=32 channels giving TWO independent per-symbol logit maps
[B, 2, N, 16] — capacity is identical to the K=1 head except the readout.

Loss: 2! PIT over slot<->source assignments + per-modulation-masked CE
(CONST_MASK) against ref_hard_labels (same pipeline as the S3 LLR term).

Usage: python probe_sync_head_k2.py [--n_train 2000] [--epochs 30]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

import config as C
from data_generator_vark import generate_vark_mixture, MOD_TYPES
from soft_demod import ref_hard_labels, CONST_MASK
from models import ComplexConv1d, ComplexBatchNorm1d, ComplexReLU

# Reference points for the verdict (S2, K=2, 5-seed means).
REF_WAVEFORM_ROUTED_SER = 0.5475
REF_MIXTURE_BASELINE_SER = 0.5611
REF_CHANCE_SER = 0.68


class ProbeHeadK2(nn.Module):
    """K=1 ProbeHead trunk + 2-slot readout.  in_ch: 1 (P) or 2 (S)."""

    def __init__(self, in_ch: int = 1, sps: int = 16, n_out: int = 16,
                 n_slots: int = 2):
        super().__init__()
        self.n_slots = n_slots
        self.n_out = n_out
        self.in_proj = nn.Sequential(
            ComplexConv1d(in_ch, 32, kernel_size=1),
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
        self.to_logits = nn.Conv1d(64, n_slots * n_out, kernel_size=1)

    def forward(self, x):                # x: [B, in_ch, T] complex
        B = x.shape[0]
        h = self.in_proj(x)
        h = self.to_symrate(h)
        h = self.sym_blocks(h)
        h = torch.cat([h.real, h.imag], dim=1)          # [B, 64, N]
        lg = self.to_logits(h)                          # [B, 2*16, N]
        lg = lg.view(B, self.n_slots, self.n_out, -1)   # [B, 2, 16, N]
        return lg.permute(0, 1, 3, 2)                   # [B, 2, N, 16]


def make_data_k2(n, snr_range, seed):
    """K=2 samples (weights ~ U(0.4,0.6), the training distribution).
    Returns list of (mix[T], srcs[2,T], mods[2], carriers[2])."""
    rng = np.random.RandomState(seed)
    data = []
    st = np.random.get_state()
    np.random.seed(seed)
    try:
        for _ in range(n):
            snr = float(rng.uniform(*snr_range))
            mix, srcs, midx, cars = generate_vark_mixture(
                C.SignalConfig.signal_length, C.SignalConfig.sample_rate,
                snr, MOD_TYPES, k=2,
                carrier_base=C.SignalConfig.carrier_base,
                freq_gap_range=C.SignalConfig.freq_gap_range,
                n_symbols=C.SignalConfig.n_symbols,
                roll_off=C.SignalConfig.roll_off,
                num_taps=C.SignalConfig.num_taps,
                apply_fading=C.SignalConfig.apply_fading,
                fading_taps=C.SignalConfig.fading_taps,
                return_carriers=True)
            data.append((mix, np.stack(srcs), list(midx), list(cars)))
    finally:
        np.random.set_state(st)
    return data


def make_input(mix, carriers, variant, sample_rate):
    """Head input [in_ch, T]: P = raw mixture; S = per-carrier down-converted
    channels (frequency oracle only — no phase alignment possible at K=2)."""
    if variant == 'P':
        return mix[None, :]
    t = np.arange(len(mix)) / sample_rate
    return np.stack([mix * np.exp(-1j * 2 * np.pi * f * t)
                     for f in carriers])


def pit_masked_ce(logits, labels, mods):
    """2! PIT + constellation-masked CE.

    logits : [B, 2, N, 16] real
    labels : [B, 2, N] long
    mods   : [B, 2] long
    Returns (per-sample best loss [B], per-sample best assignment [B,2],
             per-sample symbol-error [B,2] under the best assignment).
    """
    B, S, N, _ = logits.shape
    vmask = CONST_MASK.to(logits.device)[mods]          # [B, 2, 16]
    losses, errs = [], []
    for perm in ((0, 1), (1, 0)):
        # slot j is scored against source perm[j]
        lg = logits.reshape(B * S * N, 16)
        vm = vmask[:, perm]                             # [B, 2, 16]
        vm = vm.unsqueeze(2).expand(B, S, N, 16).reshape(B * S * N, 16)
        lab = labels[:, perm].reshape(-1)
        ce = nn.functional.cross_entropy(
            lg.masked_fill(~vm, -1e9), lab, reduction='none')
        losses.append(ce.view(B, S, N).mean(dim=2))     # [B, 2] per slot
        with torch.no_grad():
            est = lg.masked_fill(~vm, -1e9).argmax(dim=-1)
            errs.append((est != lab).float().view(B, S, N).mean(dim=2))
    l = torch.stack(losses, dim=0)                      # [2, B, 2]
    e = torch.stack(errs, dim=0)
    tot = l.sum(dim=2)                                  # [2, B]
    best = tot.argmin(dim=0)                            # [B]
    b_idx = torch.arange(B, device=logits.device)
    return (l[best, b_idx].sum(dim=1),                  # [B] best loss
            best,                                       # [B] winning perm
            e[best, b_idx])                             # [B, 2] per-slot SER


def evaluate(model, data, variant, device):
    model.eval()
    per_mod_err: dict[int, list[float]] = {}
    all_err = []
    with torch.no_grad():
        for mix, srcs, midx, cars in data:
            x = torch.from_numpy(make_input(
                mix, cars, variant, C.SignalConfig.sample_rate)
            ).unsqueeze(0).to(torch.complex64).to(device)
            logits = model(x)                            # [1, 2, N, 16]
            src_t = torch.from_numpy(srcs).to(torch.complex64).to(device)
            car_t = torch.tensor(cars, dtype=torch.float32, device=device)
            mod_t = torch.tensor(midx, dtype=torch.long, device=device)
            lab = ref_hard_labels(
                src_t, car_t, mod_t,
                sample_rate=C.SignalConfig.sample_rate,
                n_symbols=C.SignalConfig.n_symbols,
                roll_off=C.SignalConfig.roll_off,
                num_taps=C.SignalConfig.num_taps).unsqueeze(0)   # [1, 2, N]
            _, _, ser = pit_masked_ce(logits, lab,
                                      mod_t.unsqueeze(0))
            for j in range(2):
                per_mod_err.setdefault(midx[j], []).append(
                    float(ser[0, j]))
                all_err.append(float(ser[0, j]))
    overall = float(np.mean(all_err))
    return overall, {MOD_TYPES[m]: float(np.mean(v))
                     for m, v in per_mod_err.items()}


def train_variant(variant, train_data, val_data, device, epochs, lr):
    torch.manual_seed(42)
    in_ch = 1 if variant == 'P' else 2
    model = ProbeHeadK2(in_ch=in_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.RandomState(0)
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_data))
        tot, cnt = 0.0, 0
        for i in range(0, len(order), 16):
            idx = order[i:i + 16]
            xs = [torch.from_numpy(make_input(
                train_data[j][0], train_data[j][3], variant,
                C.SignalConfig.sample_rate)) for j in idx]
            x = torch.stack(xs).to(torch.complex64).to(device)  # [B, in_ch, T]
            src_t = torch.stack([
                torch.from_numpy(train_data[j][1]) for j in idx]
            ).to(torch.complex64).to(device)             # [B, 2, T]
            car_t = torch.tensor([train_data[j][3] for j in idx],
                                 dtype=torch.float32, device=device)
            mod_t = torch.tensor([train_data[j][2] for j in idx],
                                 dtype=torch.long, device=device)
            lab = ref_hard_labels(
                src_t.reshape(-1, src_t.shape[-1]),
                car_t.reshape(-1), mod_t.reshape(-1),
                sample_rate=C.SignalConfig.sample_rate,
                n_symbols=C.SignalConfig.n_symbols,
                roll_off=C.SignalConfig.roll_off,
                num_taps=C.SignalConfig.num_taps).view(len(idx), 2, -1)
            logits = model(x)                            # [B, 2, N, 16]
            loss, _, _ = pit_masked_ce(logits, lab, mod_t)
            loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            cnt += 1
        if ep % 5 == 0 or ep == 1:
            overall, per_mod = evaluate(model, val_data, variant, device)
            print(f"  [{variant}] ep{ep:>2d}  loss={tot / cnt:.4f}  "
                  f"val SER={overall:.4f}  per-mod: "
                  + '  '.join(f'{m}:{v:.3f}' for m, v in sorted(per_mod.items())),
                  flush=True)
    # Final report for this variant.
    overall, per_mod = evaluate(model, val_data, variant, device)
    print(f"  [{variant}] FINAL val SER={overall:.4f}  per-mod: "
          + '  '.join(f'{m}:{v:.4f}' for m, v in sorted(per_mod.items())),
          flush=True)
    return overall


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_train', type=int, default=2000)
    p.add_argument('--n_val', type=int, default=400)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()
    device = C.DEVICE

    print(f"Building K=2 probe data (train {args.n_train}, val {args.n_val}; "
          f"same sizes/seeds as the K=1 probe)", flush=True)
    train_data = make_data_k2(args.n_train, C.SignalConfig.snr_range_train, 42)
    val_data = make_data_k2(args.n_val, C.SignalConfig.snr_range_train, 1042)

    print(f"\nReferences (S2, K=2): waveform-routed SER "
          f"{REF_WAVEFORM_ROUTED_SER}, mixture baseline "
          f"{REF_MIXTURE_BASELINE_SER}, chance ~{REF_CHANCE_SER}")
    print("\n=== Variant P (passband — blind sync + blind separation) ===",
          flush=True)
    ser_p = train_variant('P', train_data, val_data, device,
                          args.epochs, args.lr)
    print("\n=== Variant S (frequency-oracle down-conversion, 2 channels) ===",
          flush=True)
    ser_s = train_variant('S', train_data, val_data, device,
                          args.epochs, args.lr)

    print("\n" + "=" * 72)
    print(f"K=2 probe verdict: P={ser_p:.4f}  S={ser_s:.4f}  |  refs: "
          f"waveform-routed {REF_WAVEFORM_ROUTED_SER}, baseline "
          f"{REF_MIXTURE_BASELINE_SER}, chance ~{REF_CHANCE_SER}")
    if ser_s < REF_WAVEFORM_ROUTED_SER:
        print("  S < 0.5475 => an oracle-synced direct demod head BEATS "
              "waveform routing at K=2 (positive result).")
    else:
        print("  S >= 0.5475 => even frequency-oracle sync does NOT beat "
              "waveform routing at K=2 (negative result).")


if __name__ == '__main__':
    main()
