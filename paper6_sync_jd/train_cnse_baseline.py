"""Paper 6 — CNSE strong-baseline training (Hou & Gao 2022, K=2 only).

Mirrors train_baseline.py's recipe exactly (epochs 100, batch 16, lr 1e-3,
grad_clip 1.0, train_samples 2000, val_samples 400, early_stop 20,
snr_range [-5, 20], Adam + ReduceLROnPlateau(mode='max', factor=0.5,
patience=5) on val SI-SDR), but trains the vendored CNSE separator
(cnse.py) on K=2 mixtures only — CNSE always outputs exactly 2 sources.

Loss: 2-source PIT — for both permutations compute the mean SI-SDR via
losses_baseline._si_sdr_per_sample; loss = -(best SI-SDR)
+ lambda_mse x MSE of the best permutation.

Usage:
    python train_cnse_baseline.py --epochs 100 --seed 42
    python train_cnse_baseline.py --smoke          # 1 epoch, 64 samples (CPU ok)
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import config as C
from cnse import CNSE
from data_generator import CommBSSVarKDataset, MOD_TYPES
from losses_baseline import _si_sdr_per_sample


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train the CNSE strong baseline '
                                            '(Hou & Gao 2022) at K=2')
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--n_stacks', type=int, default=3)
    p.add_argument('--lambda_mse', type=float, default=1.0)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--train_samples', type=int, default=2000)
    p.add_argument('--val_samples', type=int, default=400)
    p.add_argument('--early_stop', type=int, default=20)
    p.add_argument('--snr_range', type=float, nargs=2, default=[-5, 20])
    p.add_argument('--seed', type=int, default=C.SEED)
    p.add_argument('--name', type=str, default='')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--smoke', action='store_true',
                   help='1 epoch, 64 train / 32 val samples (CPU sanity)')
    return p.parse_args()


def build_dataset(args, n_samples, seed):
    # Identical to train_baseline.build_dataset except k_min=k_max=2:
    # CNSE always emits exactly 2 sources, so train on K=2 mixtures only.
    return CommBSSVarKDataset(
        n_samples=n_samples, snr_range=tuple(args.snr_range),
        mod_types=MOD_TYPES,
        k_min=2, k_max=2,
        k_slots=C.VarKConfig.k_slots,
        signal_length=C.SignalConfig.signal_length,
        sample_rate=C.SignalConfig.sample_rate,
        carrier_base=C.SignalConfig.carrier_base,
        freq_gap_range=C.SignalConfig.freq_gap_range,
        n_symbols=C.SignalConfig.n_symbols,
        roll_off=C.SignalConfig.roll_off,
        num_taps=C.SignalConfig.num_taps,
        apply_fading=C.SignalConfig.apply_fading,
        fading_taps=C.SignalConfig.fading_taps,
        seed=seed)


def compute_loss(model, args, mix, sources):
    """2-source PIT: -(best mean SI-SDR) + lambda_mse x MSE(best perm).

    sources: [B, k_slots, 1, T] complex; only the first 2 slots are used
    (K=2 always).
    """
    s1, s2 = model(mix)                                   # [B, 1, T] complex
    est = torch.stack([s1.squeeze(1), s2.squeeze(1)], dim=1)   # [B, 2, T]
    ref = sources[:, :2, 0]                               # [B, 2, T]

    # SI-SDR pair matrix [B, 2(est), 2(ref)]
    b, T = est.shape[0], est.shape[-1]
    pair = _si_sdr_per_sample(
        est.unsqueeze(2).expand(b, 2, 2, T).reshape(b * 4, T),
        ref.unsqueeze(1).expand(b, 2, 2, T).reshape(b * 4, T)).view(b, 2, 2)

    score_id = pair[:, 0, 0] + pair[:, 1, 1]              # identity perm
    score_sw = pair[:, 0, 1] + pair[:, 1, 0]              # swapped perm
    best_sisdr = torch.maximum(score_id, score_sw) / 2    # mean over 2 pairs

    # MSE of each permutation, then select the PIT-best one per sample
    mse_id = ((est - ref).abs().pow(2)).mean(dim=(-2, -1))
    mse_sw = ((est - ref.flip(1)).abs().pow(2)).mean(dim=(-2, -1))
    use_sw = score_sw > score_id
    best_mse = torch.where(use_sw, mse_sw, mse_id)

    per_sample = -best_sisdr + args.lambda_mse * best_mse
    loss = per_sample.mean()
    aux = {'si_sdr': best_sisdr.detach().mean(),
           'mse': best_mse.detach().mean()}
    return loss, aux


def train_one_epoch(model, loader, optimizer, device, args):
    model.train()
    sum_loss = sum_sisdr = 0.0
    n_batches = 0
    t0 = time.time()
    for mix, sources, _occ, _k in loader:
        mix = mix.to(device)
        sources = sources.to(device)
        loss, aux = compute_loss(model, args, mix, sources)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.grad_clip)
        optimizer.step()
        sum_loss += loss.item()
        sum_sisdr += aux['si_sdr'].item()
        n_batches += 1
    return {'loss': sum_loss / max(n_batches, 1),
            'si_sdr': sum_sisdr / max(n_batches, 1),
            'time': time.time() - t0}


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    sum_loss = sum_sisdr = 0.0
    n_batches = 0
    for mix, sources, _occ, _k in loader:
        mix = mix.to(device)
        sources = sources.to(device)
        loss, aux = compute_loss(model, args, mix, sources)
        sum_loss += loss.item()
        sum_sisdr += aux['si_sdr'].item()
        n_batches += 1
    return {'loss': sum_loss / max(n_batches, 1),
            'si_sdr': sum_sisdr / max(n_batches, 1)}


def main():
    args = get_args()
    if args.smoke:
        args.epochs = 1
        args.train_samples = 64
        args.val_samples = 32
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = C.DEVICE
    print(f"Device: {device}  seed: {args.seed}")
    print(f"CNSE K=2 recipe: lambda_mse={args.lambda_mse} "
          f"(2-source PIT SI-SDR + MSE anchor)")

    train_ds = build_dataset(args, args.train_samples, args.seed)
    val_ds = build_dataset(args, args.val_samples, args.seed + 1000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    model = CNSE(hidden=args.hidden, n_stacks=args.n_stacks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)

    name = (f"cnse_h{args.hidden}_s{args.n_stacks}"
            f"_bs{args.batch_size}_lr{args.lr}_mse"
            f"{('_' + args.name) if args.name else ''}_s{args.seed}")

    start_epoch = 0
    best_val = float('-inf')
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val = ckpt.get('best_val', best_val)

    no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===", flush=True)
        tm = train_one_epoch(model, train_loader, optimizer, device, args)
        vm = validate(model, val_loader, device, args)
        scheduler.step(vm['si_sdr'])
        print(f"  TRAIN loss={tm['loss']:.4f}  si_sdr={tm['si_sdr']:.3f} dB "
              f"({tm['time']:.0f}s)", flush=True)
        print(f"  VAL   loss={vm['loss']:.4f}  si_sdr={vm['si_sdr']:.3f} dB",
              flush=True)

        ckpt_payload = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch, 'best_val': best_val,
                        'args': vars(args)}
        if vm['si_sdr'] > best_val:
            best_val = vm['si_sdr']
            no_improve = 0
            path = os.path.join(C.CHECKPOINT_DIR, f"{name}_best.pt")
            torch.save(ckpt_payload, path)
            print(f"  -> saved best to {path}", flush=True)
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                print(f"Early stopping after {no_improve} epochs")
                break
        torch.save(ckpt_payload,
                   os.path.join(C.CHECKPOINT_DIR, f"{name}_last.pt"))

    print(f"\nDone. Best val SI-SDR = {best_val:.3f} dB")


if __name__ == '__main__':
    main()
