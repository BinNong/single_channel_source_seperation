"""
Paper 3 — Open-Set SC-BSS: Training script.

Trains OpenSetCSE on the open-set training set (only the 4 known
modulations; unknown modulations never appear in training, per the
design decision: "只用已知 4 类训练 head").

Usage:
    python train.py --epochs 100 --batch_size 16 --seed 42
    python train.py --resume checkpoints/openset_cse_h32_l4_..._best.pt
    python train.py --hidden 16 --layers 3 --name smaller_cse
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure paper3_open_set/ is first on sys.path so `from models import
# OpenSetCSE` resolves to OUR models.py (not paper1's via the
# paper1_cnn_se/ directory).
_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

import config as C
from data_generator_extended import (
    CommBSSOpenSetDataset,
    MOD_KNOWN,
    MOD_TO_IDX,
)
from losses import pit_multi_task_loss, CenterLoss
from models import OpenSetCSE


# ============================================================================
# Argument parsing
# ============================================================================
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train OpenSetCSE for SC-BSS')

    # Model hyperparameters
    p.add_argument('--hidden', type=int, default=C.BACKBONE_HIDDEN_CHANNELS,
                   help='C-SE hidden channels')
    p.add_argument('--layers', type=int, default=C.BACKBONE_N_LAYERS,
                   help='Number of C-SE residual blocks')
    p.add_argument('--no_se', action='store_true', help='Disable SE block (ablation)')
    p.add_argument('--embed_dim', type=int, default=C.EMBED_DIM,
                   help='Per-source embedding dimension')
    p.add_argument('--loss_alpha', type=float, default=C.LOSS_ALPHA_CLS,
                   help='Weight on classification CE loss vs SI-SDR')
    p.add_argument('--loss_lambda_center', type=float, default=0.0,
                   help='Weight on Center Loss (0 disables it)')

    # Training hyperparameters
    p.add_argument('--epochs', type=int, default=C.NUM_EPOCHS)
    p.add_argument('--batch_size', type=int, default=C.BATCH_SIZE)
    p.add_argument('--lr', type=float, default=C.LEARNING_RATE)
    p.add_argument('--weight_decay', type=float, default=C.WEIGHT_DECAY)
    p.add_argument('--grad_clip', type=float, default=C.GRAD_CLIP_NORM)
    p.add_argument('--train_samples', type=int, default=2000)
    p.add_argument('--val_samples', type=int, default=400)
    p.add_argument('--early_stop', type=int, default=C.EARLY_STOP_PATIENCE)

    # Misc
    p.add_argument('--seed', type=int, default=C.SEED)
    p.add_argument('--name', type=str, default='', help='Experiment name suffix')
    p.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    p.add_argument('--no_tensorboard', action='store_true')
    p.add_argument('--log_every', type=int, default=C.LOG_EVERY)
    return p.parse_args()


# ============================================================================
# Train / validation routines
# ============================================================================
def train_one_epoch(model, loader, optimizer, device, args, epoch: int,
                     center_loss: CenterLoss | None = None) -> dict:
    model.train()
    sum_loss = 0.0
    sum_sep = 0.0
    sum_cls = 0.0
    sum_cen = 0.0
    n_batches = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        mix, src1, src2, mod1, mod2 = batch
        mix  = mix.to(device, non_blocking=True)
        src1 = src1.to(device, non_blocking=True)
        src2 = src2.to(device, non_blocking=True)
        mod1 = mod1.to(device, non_blocking=True)
        mod2 = mod2.to(device, non_blocking=True)

        s1_hat, s2_hat, emb1, emb2, logits1, logits2 = model(mix)
        loss, use_swap, perm = pit_multi_task_loss(
            s1_hat, s2_hat, logits1, logits2,
            src1, src2, mod1, mod2, alpha=args.loss_alpha,
        )
        s1p, s2p, l1p, l2p = perm
        # PIT-align embeddings and labels to the chosen permutation.
        # Note: embeddings are [B, D] (no channel dim); use_swap is [B].
        # Use a [B, 1] broadcast shape for the embedding where (NOT [B, 1, 1]
        # which would add an unwanted batch broadcast on dim 1).
        swap_e = use_swap.view(-1, 1)
        emb1_p = torch.where(swap_e, emb2, emb1)
        emb2_p = torch.where(swap_e, emb1, emb2)
        mod1_p = torch.where(use_swap, mod2, mod1)
        mod2_p = torch.where(use_swap, mod1, mod2)

        # Re-derive the per-term losses for logging
        sep_only = losses.si_sdr_loss(s1p, src1) + losses.si_sdr_loss(s2p, src2)
        cls_only = nn.functional.cross_entropy(l1p, mod1_p) + \
                   nn.functional.cross_entropy(l2p, mod2_p)
        # Center loss on the PIT-aligned embeddings
        if center_loss is not None and args.loss_lambda_center > 0:
            emb_all = torch.cat([emb1_p, emb2_p], dim=0)
            mod_all = torch.cat([mod1_p, mod2_p], dim=0)
            if step == 0:
                print(f"DEBUG step0: emb1_p.shape={emb1_p.shape}  emb2_p.shape={emb2_p.shape}  "
                      f"emb_all.shape={emb_all.shape}  mod_all.shape={mod_all.shape}")
            c_loss = center_loss(emb_all, mod_all)
            loss = loss + c_loss
        else:
            c_loss = torch.tensor(0.0, device=device)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        sum_loss += loss.item()
        sum_sep  += sep_only.item()
        sum_cls  += cls_only.item()
        sum_cen  += c_loss.item() if isinstance(c_loss, torch.Tensor) else 0.0
        n_batches += 1

        if (step + 1) % args.log_every == 0:
            print(f"  ep{epoch:>3d} step {step+1:>4d}/{len(loader)}  "
                  f"loss={sum_loss/n_batches:.4f}  sep={sum_sep/n_batches:.4f}  "
                  f"cls={sum_cls/n_batches:.4f}  cen={sum_cen/n_batches:.4f}  "
                  f"({time.time()-t0:.1f}s)")

    return {
        'loss': sum_loss / max(n_batches, 1),
        'sep':  sum_sep  / max(n_batches, 1),
        'cls':  sum_cls  / max(n_batches, 1),
        'cen':  sum_cen  / max(n_batches, 1),
        'time': time.time() - t0,
    }


@torch.no_grad()
def validate(model, loader, device, args) -> dict:
    model.eval()
    sum_loss = 0.0
    sum_si_sdr = 0.0
    sum_correct = 0
    sum_total = 0
    n_batches = 0

    for batch in loader:
        mix, src1, src2, mod1, mod2 = batch
        mix  = mix.to(device, non_blocking=True)
        src1 = src1.to(device, non_blocking=True)
        src2 = src2.to(device, non_blocking=True)
        mod1 = mod1.to(device, non_blocking=True)
        mod2 = mod2.to(device, non_blocking=True)

        s1_hat, s2_hat, _, _, logits1, logits2 = model(mix)
        loss, _swap, perm = pit_multi_task_loss(
            s1_hat, s2_hat, logits1, logits2,
            src1, src2, mod1, mod2, alpha=args.loss_alpha,
        )
        s1p, s2p, l1p, l2p = perm
        # SI-SDR for the PIT-aligned predictions
        si_sdr_1 = -losses.si_sdr_loss(s1p, src1)
        si_sdr_2 = -losses.si_sdr_loss(s2p, src2)
        sum_si_sdr += (si_sdr_1.item() + si_sdr_2.item()) / 2

        # Classification accuracy
        pred1 = l1p.argmax(dim=-1)
        pred2 = l2p.argmax(dim=-1)
        sum_correct += (pred1 == mod1).sum().item() + (pred2 == mod2).sum().item()
        sum_total += 2 * mix.shape[0]

        sum_loss += loss.item()
        n_batches += 1

    return {
        'loss': sum_loss / max(n_batches, 1),
        'si_sdr': sum_si_sdr / max(n_batches, 1),
        'cls_acc': sum_correct / max(sum_total, 1),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  seed: {args.seed}")

    # Datasets — train uses ONLY the 4 known modulations
    train_ds = CommBSSOpenSetDataset(
        n_samples=args.train_samples,
        snr_range=C.SNR_TRAIN_RANGE,
        mod_types=MOD_KNOWN,
        seed=args.seed,
    )
    val_ds = CommBSSOpenSetDataset(
        n_samples=args.val_samples,
        snr_range=C.SNR_TRAIN_RANGE,
        mod_types=MOD_KNOWN,
        seed=args.seed + 1000,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)

    # Model
    model = OpenSetCSE(
        hidden_channels=args.hidden,
        n_layers=args.layers,
        use_se=not args.no_se,
        embed_dim=args.embed_dim,
        num_known_classes=C.NUM_KNOWN_CLASSES,
    ).to(device)
    print(f"DEBUG: args.embed_dim={args.embed_dim}  model.head.embed_dim={model.head.embed_dim}")

    # Center loss (optional, off by default to keep baseline runs comparable)
    center_loss = None
    if args.loss_lambda_center > 0:
        center_loss = CenterLoss(
            num_classes=C.NUM_KNOWN_CLASSES,
            feat_dim=args.embed_dim,
            lambda_c=args.loss_lambda_center,
        ).to(device)
        print(f"DEBUG: center_loss.centers.shape={center_loss.centers.shape}")
        print(f"Center Loss enabled: lambda_c={args.loss_lambda_center}")

    # Optimizer (includes CenterLoss centres if enabled)
    opt_params = list(model.parameters())
    if center_loss is not None:
        opt_params += list(center_loss.parameters())
    optimizer = optim.Adam(opt_params, lr=args.lr,
                             weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume
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

    # TensorBoard
    run_name = (f"openset_cse_h{args.hidden}_l{args.layers}"
                f"_bs{args.batch_size}_lr{args.lr}"
                f"_alpha{args.loss_alpha}"
                f"{('_lc' + str(args.loss_lambda_center)) if args.loss_lambda_center > 0 else ''}"
                f"_seed{args.seed}"
                f"{('_' + args.name) if args.name else ''}")
    writer = None
    if HAS_TENSORBOARD and not args.no_tensorboard:
        writer = SummaryWriter(os.path.join(C.RUN_DIR, run_name))

    # Training loop
    no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        train_metrics = train_one_epoch(model, train_loader, optimizer,
                                          device, args, epoch + 1,
                                          center_loss=center_loss)
        val_metrics = validate(model, val_loader, device, args)
        scheduler.step()

        print(f"  TRAIN  loss={train_metrics['loss']:.4f}  "
              f"sep={train_metrics['sep']:.4f}  cls={train_metrics['cls']:.4f}  "
              f"cen={train_metrics.get('cen', 0.0):.4f}")
        print(f"  VAL    loss={val_metrics['loss']:.4f}  "
              f"si_sdr={val_metrics['si_sdr']:.3f} dB  "
              f"cls_acc={val_metrics['cls_acc']:.3f}")

        if writer is not None:
            writer.add_scalar('train/loss', train_metrics['loss'], epoch)
            writer.add_scalar('train/sep_loss', train_metrics['sep'], epoch)
            writer.add_scalar('train/cls_loss', train_metrics['cls'], epoch)
            writer.add_scalar('train/cen_loss', train_metrics.get('cen', 0.0), epoch)
            writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            writer.add_scalar('val/si_sdr', val_metrics['si_sdr'], epoch)
            writer.add_scalar('val/cls_acc', val_metrics['cls_acc'], epoch)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

        # Save best by val/si_sdr
        ckpt_payload = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_val': best_val,
            'args': vars(args),
        }
        if center_loss is not None:
            ckpt_payload['center_loss'] = center_loss.state_dict()
        if val_metrics['si_sdr'] > best_val:
            best_val = val_metrics['si_sdr']
            no_improve = 0
            ckpt_path = os.path.join(C.CHECKPOINT_DIR, f"{run_name}_best.pt")
            torch.save(ckpt_payload, ckpt_path)
            print(f"  -> saved best to {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                print(f"Early stopping after {no_improve} epochs without improvement")
                break

        # Always save last
        torch.save(ckpt_payload, os.path.join(C.CHECKPOINT_DIR, f"{run_name}_last.pt"))

    if writer is not None:
        writer.close()
    print(f"\nDone. Best val SI-SDR = {best_val:.3f} dB")


if __name__ == '__main__':
    # Late import so that --help works without torch installed (rare).
    import losses  # noqa: E402
    main()