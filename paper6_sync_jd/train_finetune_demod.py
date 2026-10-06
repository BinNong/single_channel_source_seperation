"""Paper 6 — E-C (review 3): demodulation-aware fine-tuning control.

Reviewer question: is the separate-then-detect detection floor structural,
or would a DEMODULATION-AWARE loss remove it?  This script fine-tunes the
existing MSE-anchored SlotSepNet baseline checkpoints (the E3/E4 evaluation
checkpoints, checkpoints/slot_h64_l4_k13_bs16_lr0.001_mse_s{seed}_best.pt)
with the paper-5 soft-demodulation term ADDED to the unchanged baseline
recipe:

    loss = baseline PIT loss (lambda_sep=1.0 SI-SDR + lambda_mse=1.0 anchor
           + lambda_occ=1.0 BCE + lambda_cnt=0.1 CE, via
           losses_baseline.variable_k_pit_loss)
         + lambda_ser * soft-SER CE on the SAME PIT-assigned pairs
           (soft_demod.soft_ser_pairs, reusing aux['assigned'] — no second
           assignment enumeration, same construction as paper5 losses.py)

Same architecture (identical parameter budget), same generator and RNG
convention (CommBSSVarKDataset, return_carriers=True consumes no extra
draws — the mixture stream is identical to baseline training), only the
objective changes.  Checkpoint selection stays by val SI-SDR (lambda_sep
= 1.0, so this is the paper-5 config-(d) regime, not the reward-hacked
pure-task config (e)).

Pair with eval_finetune_demod.py for the K=2 separate -> oracle-sync ->
detect SER/BER comparison against the unmodified baseline.

Usage:
    python train_finetune_demod.py --smoke                  # CPU sanity
    python train_finetune_demod.py --seed 42 --lambda_ser 1.0 \
        --epochs 15 --lr 1e-4                               # server run
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
from soft_demod import soft_ser_pairs

BASE_CKPT_PATTERN = 'slot_h{hidden}_l{layers}_k13_bs16_lr0.001_mse_s{seed}_best.pt'


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Demodulation-aware fine-tuning of the paper-6 baseline '
                    'SlotSepNet (E-C, review 3)')
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--k_slots', type=int, default=C.VarKConfig.k_slots)
    p.add_argument('--lambda_occ', type=float, default=1.0)
    p.add_argument('--lambda_cnt', type=float, default=0.1)
    p.add_argument('--lambda_mse', type=float, default=1.0)
    p.add_argument('--lambda_sep', type=float, default=1.0)
    p.add_argument('--lambda_ser', type=float, default=1.0,
                   help='Weight on the soft-demodulation term '
                        '(soft_demod.soft_ser_pairs on PIT-assigned pairs)')
    p.add_argument('--ser_sigma2', type=float, default=0.1,
                   help='Gaussian softness of the soft log-likelihood '
                        '(paper5 TaskConfig.ser_sigma2)')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--train_samples', type=int, default=2000)
    p.add_argument('--val_samples', type=int, default=400)
    p.add_argument('--early_stop', type=int, default=20)
    p.add_argument('--snr_range', type=float, nargs=2, default=[-5, 20])
    p.add_argument('--seed', type=int, default=C.SEED)
    p.add_argument('--name', type=str, default='')
    p.add_argument('--init_from', type=str, default='',
                   help='Baseline checkpoint to fine-tune from; default: '
                        'checkpoints/' + BASE_CKPT_PATTERN)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--smoke', action='store_true',
                   help='1 epoch, 64 train / 32 val samples (CPU sanity)')
    return p.parse_args()


def build_dataset(args, n_samples, seed):
    # return_carriers=True: SAME RNG stream as baseline training (the
    # generator draws the carriers either way; the flag only exposes them).
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
        seed=seed, return_carriers=True)


def soft_ser_term(slots, sources, k, assigned, carriers, mods, args):
    """Soft-SER CE on the PIT-assigned pairs (reuses aux['assigned'] from
    the baseline loss, so no second assignment enumeration — same
    construction as paper5_task_oriented/losses.py)."""
    B, S, T = slots.shape
    device = slots.device
    src = sources.squeeze(2)                              # [B, S, T]
    per_sample = torch.zeros(B, device=device)
    for k_val in torch.unique(k):
        kv = int(k_val.item())
        idx = (k == k_val).nonzero(as_tuple=True)[0]
        b = idx.numel()
        chosen = assigned[idx, :kv]                       # [b, kv]
        b_idx = torch.arange(b, device=device).unsqueeze(1)
        est_pairs = slots[idx][b_idx, chosen]             # [b, kv, T]
        ref_pairs = src[idx, :kv]                         # [b, kv, T]
        idx_cpu = idx.cpu()
        car = carriers[idx_cpu, :kv].to(device).float().reshape(-1)
        mod = mods[idx_cpu, :kv].to(device).reshape(-1)
        ce = soft_ser_pairs(
            est_pairs.reshape(b * kv, T), ref_pairs.reshape(b * kv, T),
            car, mod, sigma2=args.ser_sigma2,
            sample_rate=C.SignalConfig.sample_rate,
            n_symbols=C.SignalConfig.n_symbols,
            roll_off=C.SignalConfig.roll_off,
            num_taps=C.SignalConfig.num_taps)
        per_sample[idx] = ce.view(b, kv).mean(dim=-1)
    return per_sample.mean()


def compute_loss(model, args, mix, sources, occ, k, mods, carriers):
    slots, occ_logits, count_logits = model(mix)
    # Baseline recipe unchanged (lambda_ser is NOT passed into
    # losses_baseline — the soft-SER term is added here on top).
    loss, aux = variable_k_pit_loss(
        slots, occ_logits, count_logits, sources, k,
        lambda_occ=args.lambda_occ, lambda_cnt=args.lambda_cnt,
        lambda_mse=args.lambda_mse, lambda_sep=args.lambda_sep)
    ser = soft_ser_term(slots, sources, k, aux['assigned'], carriers, mods,
                        args)
    aux['ser_soft'] = ser.detach()
    return loss + args.lambda_ser * ser, aux


def train_one_epoch(model, loader, optimizer, device, args):
    model.train()
    sum_loss = sum_sisdr = sum_ser = 0.0
    n_batches = 0
    t0 = time.time()
    for mix, sources, occ, k, mods, carriers in loader:
        mix = mix.to(device)
        sources = sources.to(device)
        occ = occ.to(device)
        k = k.to(device)
        loss, aux = compute_loss(model, args, mix, sources, occ, k,
                                 mods, carriers)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.grad_clip)
        optimizer.step()
        sum_loss += loss.item()
        sum_sisdr += aux['si_sdr'].item()
        sum_ser += aux['ser_soft'].item()
        n_batches += 1
    n = max(n_batches, 1)
    return {'loss': sum_loss / n, 'si_sdr': sum_sisdr / n,
            'ser_soft': sum_ser / n, 'time': time.time() - t0}


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    sum_loss = sum_sisdr = sum_ser = 0.0
    cnt_correct = n_samples = n_batches = 0
    for mix, sources, occ, k, mods, carriers in loader:
        mix = mix.to(device)
        sources = sources.to(device)
        occ = occ.to(device)
        k = k.to(device)
        loss, aux = compute_loss(model, args, mix, sources, occ, k,
                                 mods, carriers)
        sum_loss += loss.item()
        sum_sisdr += aux['si_sdr'].item()
        sum_ser += aux['ser_soft'].item()
        n_batches += 1
        out = model(mix)
        k_hat = (torch.sigmoid(out[1]) > C.VarKConfig.occ_threshold).sum(dim=1)
        cnt_correct += (k_hat == k).sum().item()
        n_samples += k.numel()
    n = max(n_batches, 1)
    return {'loss': sum_loss / n, 'si_sdr': sum_sisdr / n,
            'ser_soft': sum_ser / n,
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
    print(f"Fine-tune recipe: baseline (lambda_sep={args.lambda_sep} "
          f"lambda_mse={args.lambda_mse} lambda_occ={args.lambda_occ} "
          f"lambda_cnt={args.lambda_cnt}) + lambda_ser={args.lambda_ser} "
          f"(ser_sigma2={args.ser_sigma2}), lr={args.lr}, "
          f"epochs={args.epochs}")

    init_from = args.init_from or os.path.join(
        C.CHECKPOINT_DIR,
        BASE_CKPT_PATTERN.format(hidden=args.hidden, layers=args.layers,
                                 seed=args.seed))
    assert os.path.exists(init_from), f"baseline checkpoint not found: {init_from}"
    print(f"Initialising from baseline checkpoint: {init_from}")

    train_ds = build_dataset(args, args.train_samples, args.seed)
    val_ds = build_dataset(args, args.val_samples, args.seed + 1000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    model = SlotSepNet(hidden_channels=args.hidden, n_layers=args.layers,
                       k_max=args.k_slots).to(device)
    ckpt = torch.load(init_from, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    print(f"  loaded baseline weights (epoch {ckpt.get('epoch')}, "
          f"best_val {ckpt.get('best_val', float('nan')):.3f} dB)")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)

    name = (f"slot_h{args.hidden}_l{args.layers}"
            f"_k{C.VarKConfig.k_min}{C.VarKConfig.k_max}"
            f"_bs{args.batch_size}_lr{args.lr}"
            f"_mse_ftser{args.lambda_ser}"
            f"{('_' + args.name) if args.name else ''}_s{args.seed}")

    best_val = float('-inf')
    no_improve = 0
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        tm = train_one_epoch(model, train_loader, optimizer, device, args)
        vm = validate(model, val_loader, device, args)
        scheduler.step(vm['si_sdr'])
        print(f"  TRAIN loss={tm['loss']:.4f}  si_sdr={tm['si_sdr']:.3f} dB  "
              f"ser_soft={tm['ser_soft']:.4f}  ({tm['time']:.1f}s)")
        print(f"  VAL   loss={vm['loss']:.4f}  si_sdr={vm['si_sdr']:.3f} dB  "
              f"ser_soft={vm['ser_soft']:.4f}  count_acc={vm['count_acc']:.3f}")

        ckpt_payload = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch, 'best_val': best_val,
                        'init_from': init_from,
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
