"""Paper 6 — baseline separator training (self-contained reproduction path).

Trimmed from paper5_task_oriented/train.py on 2026-09-19: slot arch only,
the paper-6 baseline recipe — variable-K SI-SDR PIT + MSE anchor +
occupancy BCE + count CE (lambda_sep=1.0, lambda_mse=1.0, lambda_occ=1.0,
lambda_cnt=0.1; the paper-5 task-oriented terms are not vendored).

NOTE: the canonical paper-6 evaluation checkpoints are the paper-5
config-(b) ones (trained by paper5's train.py on an identical recipe);
this script exists so the pipeline is reproducible end-to-end inside
paper6_sync_jd/.  Do not retrain for headline numbers.

Usage:
    python train_baseline.py --epochs 100 --seed 42
    python train_baseline.py --smoke          # 1 epoch, 64 samples (CPU ok)
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
from data_generator import CommBSSVarKDataset, MOD_TYPES
from losses_baseline import variable_k_pit_loss
from models import SlotSepNet


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train the paper-6 baseline '
                                            'SlotSepNet (self-contained)')
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--k_slots', type=int, default=C.VarKConfig.k_slots)
    p.add_argument('--lambda_occ', type=float, default=1.0)
    p.add_argument('--lambda_cnt', type=float, default=0.1)
    p.add_argument('--lambda_mse', type=float, default=1.0)
    p.add_argument('--lambda_sep', type=float, default=1.0)
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
    return CommBSSVarKDataset(
        n_samples=n_samples, snr_range=tuple(args.snr_range),
        mod_types=MOD_TYPES,
        k_min=C.VarKConfig.k_min, k_max=C.VarKConfig.k_max,
        k_slots=args.k_slots,
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


def compute_loss(model, args, mix, sources, occ, k):
    slots, occ_logits, count_logits = model(mix)
    return variable_k_pit_loss(
        slots, occ_logits, count_logits, sources, k,
        lambda_occ=args.lambda_occ, lambda_cnt=args.lambda_cnt,
        lambda_mse=args.lambda_mse, lambda_sep=args.lambda_sep)


def train_one_epoch(model, loader, optimizer, device, args, epoch):
    model.train()
    sum_loss = sum_sisdr = 0.0
    n_batches = 0
    t0 = time.time()
    for mix, sources, occ, k in loader:
        mix = mix.to(device)
        sources = sources.to(device)
        occ = occ.to(device)
        k = k.to(device)
        loss, aux = compute_loss(model, args, mix, sources, occ, k)
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
    cnt_correct = n_samples = n_batches = 0
    for mix, sources, occ, k in loader:
        mix = mix.to(device)
        sources = sources.to(device)
        occ = occ.to(device)
        k = k.to(device)
        loss, aux = compute_loss(model, args, mix, sources, occ, k)
        sum_loss += loss.item()
        sum_sisdr += aux['si_sdr'].item()
        n_batches += 1
        out = model(mix)
        k_hat = (torch.sigmoid(out[1]) > C.VarKConfig.occ_threshold).sum(dim=1)
        cnt_correct += (k_hat == k).sum().item()
        n_samples += k.numel()
    return {'loss': sum_loss / max(n_batches, 1),
            'si_sdr': sum_sisdr / max(n_batches, 1),
            'count_acc': cnt_correct / max(n_samples, 1)}


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
    print(f"Baseline recipe: lambda_sep={args.lambda_sep} "
          f"lambda_mse={args.lambda_mse} lambda_occ={args.lambda_occ} "
          f"lambda_cnt={args.lambda_cnt} (paper-5 config (b))")

    train_ds = build_dataset(args, args.train_samples, args.seed)
    val_ds = build_dataset(args, args.val_samples, args.seed + 1000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    model = SlotSepNet(hidden_channels=args.hidden, n_layers=args.layers,
                       k_max=args.k_slots).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)

    name = (f"slot_h{args.hidden}_l{args.layers}"
            f"_k{C.VarKConfig.k_min}{C.VarKConfig.k_max}"
            f"_bs{args.batch_size}_lr{args.lr}"
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
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        tm = train_one_epoch(model, train_loader, optimizer, device, args,
                             epoch + 1)
        vm = validate(model, val_loader, device, args)
        scheduler.step(vm['si_sdr'])
        print(f"  TRAIN loss={tm['loss']:.4f}  si_sdr={tm['si_sdr']:.3f} dB")
        print(f"  VAL   loss={vm['loss']:.4f}  si_sdr={vm['si_sdr']:.3f} dB  "
              f"count_acc={vm['count_acc']:.3f}")

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
            print(f"  -> saved best to {path}")
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
