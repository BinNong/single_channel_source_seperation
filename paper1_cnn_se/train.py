"""
Training script for Communication Signal Blind Source Separation.

Supports:
  - Proposed ComplexLightweightSepNet
  - RealValuedBaseline
  - SimpleComplexCNN

Usage:
    python train.py --model complex_cnn_se --epochs 100 --batch_size 16
    python train.py --model real_baseline --epochs 100

Checkpoints are saved to ./checkpoints/
TensorBoard logs to ./runs/
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("[Warning] tensorboard not installed. Logging to console only.")
import numpy as np

from config import SignalConfig, DataConfig, ModelConfig, TrainConfig, DEVICE, SEED
from data_generator import CommBSSDataset
from models import (ComplexLightweightSepNet, RealValuedBaseline,
                    SimpleComplexCNN, ComplexConvTasNet, CNSE, S4UNET,
                    model_summary)
from utils import evaluate_batch, save_checkpoint, load_checkpoint


# =============================================================================
# Parse Arguments
# =============================================================================
def get_args():
    parser = argparse.ArgumentParser(description='Train Communication Signal BSS Model')
    parser.add_argument('--model', type=str, default='complex_cnn_se',
                        choices=['complex_cnn_se', 'complex_cnn_no_se', 'real_baseline', 'simple_complex', 'conv_tasnet', 'cnse', 's4unet'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=TrainConfig.epochs, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=DataConfig.batch_size_train, help='Batch size')
    parser.add_argument('--lr', type=float, default=TrainConfig.lr, help='Learning rate')
    parser.add_argument('--train_samples', type=int, default=DataConfig.train_samples,
                        help='Number of training samples (override DataConfig)')
    parser.add_argument('--val_samples', type=int, default=DataConfig.val_samples,
                        help='Number of validation samples (override DataConfig)')
    parser.add_argument('--hidden', type=int, default=32, help='Hidden channels (for proposed model)')
    parser.add_argument('--layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--baseline_hidden', type=int, default=64,
                        help='Hidden channels for real_baseline (overrides ModelConfig.baseline_hidden)')
    parser.add_argument('--baseline_layers', type=int, default=6,
                        help='Number of middle layers for real_baseline (overrides ModelConfig.baseline_layers)')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'si_sdr', 'combined'],
                        help='Loss function type')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    parser.add_argument('--name', type=str, default='', help='Experiment name suffix')
    parser.add_argument('--freq_gap', type=float, default=5.0,
                        help='Carrier-frequency gap between two sources (Hz)')
    parser.add_argument('--freq_gap_min', type=float, default=None,
                        help='Lower bound for random frequency gap (Hz). If set with '
                             '--freq_gap_max, training samples draw gaps uniformly from '
                             '[min, max] Hz.')
    parser.add_argument('--freq_gap_max', type=float, default=None,
                        help='Upper bound for random frequency gap (Hz).')
    parser.add_argument('--n_seeds', type=int, default=1,
                        help='Number of random seeds to run (sequential multi-run)')
    parser.add_argument('--se_scale_mode', type=str, default='real',
                        choices=['real', 'complex_mean', 'separate'],
                        help='C-SE scale mode (only for complex_cnn_se)')
    parser.add_argument('--se_pooling_mode', type=str, default='mean',
                        choices=['mean', 'power', 'magnitude', 'mean+power'],
                        help='C-SE squeeze pooling mode (only for complex_cnn_se)')
    parser.add_argument('--cnse_hidden', type=int, default=256,
                        help='CNSE hidden channels (paper: 512, scaled-down here for 8GB GPU)')
    parser.add_argument('--s4_base_channels', type=int, default=16,
                        help='S4-UNET base channels (paper: 32, scaled-down here for 8GB GPU)')
    parser.add_argument('--s4_state_dim', type=int, default=16,
                        help='S4-UNET SSM state dimension (paper: 64, scaled-down here)')
    return parser.parse_args()


# =============================================================================
# Loss Functions
# =============================================================================
def mse_loss(estimated, original):
    """Complex MSE loss."""
    est1, est2 = estimated
    orig1, orig2 = original
    loss = ((est1 - orig1).abs() ** 2).mean() + ((est2 - orig2).abs() ** 2).mean()
    return loss


def si_sdr_loss(estimated, original, eps=1e-8):
    """SI-SDR loss (negative SI-SDR to minimize)."""
    def _si_sdr_single(estimate, reference):
        if torch.is_complex(estimate):
            est = torch.view_as_real(estimate).flatten(-2)
            ref = torch.view_as_real(reference).flatten(-2)
        else:
            est, ref = estimate, reference

        est_zm = est - est.mean(dim=-1, keepdim=True)
        ref_zm = ref - ref.mean(dim=-1, keepdim=True)

        alpha = (ref_zm * est_zm).sum(dim=-1, keepdim=True) / \
                ((ref_zm ** 2).sum(dim=-1, keepdim=True) + eps)
        target = alpha * ref_zm
        noise = est_zm - target

        si_sdr_val = 10 * torch.log10(
            (target ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps) + eps
        )
        return si_sdr_val

    est1, est2 = estimated
    orig1, orig2 = original

    sdr_11 = _si_sdr_single(est1, orig1)
    sdr_22 = _si_sdr_single(est2, orig2)
    sdr_12 = _si_sdr_single(est1, orig2)
    sdr_21 = _si_sdr_single(est2, orig1)

    loss_perm1 = -(sdr_11 + sdr_22) / 2
    loss_perm2 = -(sdr_12 + sdr_21) / 2

    return torch.min(loss_perm1, loss_perm2).mean()


def combined_loss(estimated, original, alpha=0.5):
    """Combined MSE + SI-SDR loss."""
    return alpha * mse_loss(estimated, original) + \
           (1 - alpha) * si_sdr_loss(estimated, original)


def get_loss_fn(loss_type):
    if loss_type == 'mse':
        return mse_loss
    elif loss_type == 'si_sdr':
        return si_sdr_loss
    elif loss_type == 'combined':
        return lambda est, orig: combined_loss(est, orig, alpha=TrainConfig.loss_alpha)


# =============================================================================
# Training Epoch
# =============================================================================
def train_epoch(model, dataloader, optimizer, loss_fn, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for mixture, source1, source2 in dataloader:
        mixture = mixture.to(device)
        source1 = source1.to(device)
        source2 = source2.to(device)

        optimizer.zero_grad()

        est1, est2 = model(mixture)
        loss = loss_fn((est1, est2), (source1, source2))

        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


# =============================================================================
# Validation
# =============================================================================
@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_metrics = {'SI-SDR': 0, 'SDR': 0, 'SIR': 0, 'NMSE': 0}
    n_batches = 0

    for mixture, source1, source2 in dataloader:
        mixture = mixture.to(device)
        source1 = source1.to(device)
        source2 = source2.to(device)

        est1, est2 = model(mixture)
        loss = loss_fn((est1, est2), (source1, source2))
        total_loss += loss.item()

        metrics = evaluate_batch((est1, est2), (source1, source2))
        for k in all_metrics:
            all_metrics[k] += metrics[k]
        n_batches += 1

    avg_loss = total_loss / n_batches
    for k in all_metrics:
        all_metrics[k] /= n_batches

    return avg_loss, all_metrics


# =============================================================================
# Single Experiment Run
# =============================================================================
def run_experiment(args, seed, run_id=0):
    """Run a single training experiment with given seed. Returns best_val_loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Experiment name
    exp_name = f"{args.model}"
    if args.model == 'real_baseline':
        exp_name += f"_h{args.baseline_hidden}_l{args.baseline_layers}"
    elif args.model == 'cnse':
        exp_name += f"_h{args.cnse_hidden}"
    elif args.model == 's4unet':
        exp_name += f"_b{args.s4_base_channels}_s{args.s4_state_dim}"
    else:
        exp_name += f"_h{args.hidden}_l{args.layers}"
    exp_name += f"_bs{args.batch_size}_lr{args.lr}"
    if args.loss != 'mse':
        exp_name += f"_{args.loss}"
    if args.model == 'complex_cnn_se' and args.se_scale_mode != 'real':
        exp_name += f"_sc{args.se_scale_mode}"
    if args.model == 'complex_cnn_se' and args.se_pooling_mode != 'mean':
        exp_name += f"_po{args.se_pooling_mode.replace('+', '_')}"
    if args.name:
        exp_name += f"_{args.name}"
    if args.n_seeds > 1 or (hasattr(args, 'seed') and args.seed != SEED):
        exp_name += f"_s{seed}"

    os.makedirs(TrainConfig.checkpoint_dir, exist_ok=True)
    os.makedirs('./runs', exist_ok=True)

    print("=" * 70)
    print(f"Experiment: {exp_name}")
    print(f"Device: {DEVICE}")
    print(f"Model: {args.model}")
    print(f"Seed: {seed}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"Loss: {args.loss}")
    print("=" * 70)

    # TensorBoard
    writer = SummaryWriter(f'./runs/{exp_name}') if HAS_TENSORBOARD else None

    # Dataset
    print("\n[1/5] Loading datasets...")
    carrier_f2 = SignalConfig.carrier_freq_1 + args.freq_gap
    freq_gap_range = None
    if args.freq_gap_min is not None and args.freq_gap_max is not None:
        freq_gap_range = (args.freq_gap_min, args.freq_gap_max)
    train_dataset = CommBSSDataset(
        n_samples=args.train_samples,
        snr_range=SignalConfig.snr_range_train,
        mod_types=SignalConfig.mod_types_train,
        signal_length=SignalConfig.signal_length,
        sample_rate=SignalConfig.sample_rate,
        carrier_freq_1=SignalConfig.carrier_freq_1,
        carrier_freq_2=carrier_f2,
        seed=args.seed,
        freq_gap_range=freq_gap_range,
    )
    val_dataset = CommBSSDataset(
        n_samples=args.val_samples,
        snr_range=SignalConfig.snr_range_train,
        mod_types=SignalConfig.mod_types_train,
        signal_length=SignalConfig.signal_length,
        sample_rate=SignalConfig.sample_rate,
        carrier_freq_1=SignalConfig.carrier_freq_1,
        carrier_freq_2=carrier_f2,
        seed=args.seed + 1,
        freq_gap_range=freq_gap_range,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=DataConfig.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=DataConfig.batch_size_val,
                            shuffle=False, num_workers=DataConfig.num_workers,
                            pin_memory=True)

    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val:   {len(val_dataset):,} samples")

    # Model
    print("\n[2/5] Building model...")
    if args.model == 'complex_cnn_se':
        model = ComplexLightweightSepNet(
            hidden_channels=args.hidden, n_layers=args.layers,
            use_se=True, se_reduction=ModelConfig.se_reduction,
            se_scale_mode=args.se_scale_mode,
            se_pooling_mode=args.se_pooling_mode
        ).to(DEVICE)
    elif args.model == 'complex_cnn_no_se':
        model = ComplexLightweightSepNet(
            hidden_channels=args.hidden, n_layers=args.layers,
            use_se=False
        ).to(DEVICE)
    elif args.model == 'real_baseline':
        model = RealValuedBaseline(
            hidden=args.baseline_hidden,
            n_layers=args.baseline_layers
        ).to(DEVICE)
    elif args.model == 'simple_complex':
        model = SimpleComplexCNN(
            hidden=48, n_layers=6
        ).to(DEVICE)
    elif args.model == 'conv_tasnet':
        model = ComplexConvTasNet(
            N=64, B=64, Sc=64, H=128, P=3, X=5, R=3, L=16
        ).to(DEVICE)
    elif args.model == 'cnse':
        model = CNSE(hidden=args.cnse_hidden, n_stacks=3).to(DEVICE)
    elif args.model == 's4unet':
        model = S4UNET(base_channels=args.s4_base_channels,
                       state_dim=args.s4_state_dim).to(DEVICE)

    model_summary(model, input_shape=(1, 1, SignalConfig.signal_length))

    # Optimizer & Scheduler
    print("\n[3/5] Setting up optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=TrainConfig.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=TrainConfig.scheduler_patience,
        factor=TrainConfig.scheduler_factor
    )
    loss_fn = get_loss_fn(args.loss)

    # Resume
    start_epoch = 0
    best_metric = float('inf')
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch, best_metric = load_checkpoint(model, optimizer, args.resume)
        start_epoch += 1

    # Training
    print("\n[4/5] Starting training...")
    print("-" * 70)
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | "
          f"{'SI-SDR':>10} | {'SDR':>10} | {'SIR':>10} | {'Time':>8}")
    print("-" * 70)

    early_stop_counter = 0

    for epoch in range(start_epoch, args.epochs):
        t_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn,
                                 DEVICE, TrainConfig.grad_clip)

        # Validate
        val_loss, val_metrics = validate(model, val_loader, loss_fn, DEVICE)

        # Scheduler
        scheduler.step(val_loss)

        # Logging
        epoch_time = time.time() - t_start
        print(f"{epoch+1:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
              f"{val_metrics['SI-SDR']:>10.2f} | {val_metrics['SDR']:>10.2f} | "
              f"{val_metrics['SIR']:>10.2f} | {epoch_time:>7.1f}s")

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/SI-SDR', val_metrics['SI-SDR'], epoch)
            writer.add_scalar('Metrics/SDR', val_metrics['SDR'], epoch)
            writer.add_scalar('Metrics/SIR', val_metrics['SIR'], epoch)
            writer.add_scalar('Metrics/NMSE', val_metrics['NMSE'], epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Save best
        if val_loss < best_metric:
            best_metric = val_loss
            save_checkpoint(model, optimizer, epoch, best_metric,
                            f'{TrainConfig.checkpoint_dir}/{exp_name}_best.pt')
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Periodic save every 5 epochs (safety net)
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, val_loss,
                            f'{TrainConfig.checkpoint_dir}/{exp_name}_epoch{epoch+1}.pt')

        # Early stopping
        if early_stop_counter >= TrainConfig.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    if writer:
        writer.close()

    print("\n" + "=" * 70)
    print(f"[Seed {seed}] Training completed!")
    print(f"Best validation loss: {best_metric:.6f}")
    print(f"Checkpoint: {TrainConfig.checkpoint_dir}/{exp_name}_best.pt")
    print("=" * 70)
    return best_metric


# =============================================================================
# Main Entry Point (with optional multi-seed support)
# =============================================================================
def main():
    args = get_args()

    all_best = []
    for run_id in range(args.n_seeds):
        current_seed = args.seed + run_id
        best = run_experiment(args, current_seed, run_id)
        all_best.append(best)

    if args.n_seeds > 1:
        print("\n" + "=" * 70)
        print("Multi-Seed Summary:")
        for i, (s, b) in enumerate(zip(
            range(args.seed, args.seed + args.n_seeds), all_best)):
            print(f"  Seed {s}: best_val_loss = {b:.6f}")
        print(f"  Mean ± Std: {np.mean(all_best):.6f} ± {np.std(all_best):.6f}")
        print("=" * 70)


if __name__ == '__main__':
    main()
