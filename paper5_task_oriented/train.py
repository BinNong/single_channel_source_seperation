"""
Paper 5 — Task-Oriented SC-BSS: Training script.

Trains SlotSepNet (--arch slot; the only arch wired for the paper-5
soft-SER term) or the paper4 baselines (recursive / specialist, unchanged)
on the variable-K training set (per-sample K ~ Uniform{k_min..k_max}).

Usage:
    python train.py --arch slot --epochs 100 --batch_size 16 --seed 42
    python train.py --arch slot --lambda_ser 1.0 --name ser --seed 42
    python train.py --arch slot --lambda_ser 1.0 --lambda_mse 1.0 --name ser_mse
    python train.py --arch slot --resume checkpoints/<name>_best.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure paper4_open_world/ is first on sys.path so `from models import
# SlotSepNet` resolves to OUR models.py (not paper1's via the
# paper1_cnn_se/ directory appended by data_generator_vark).
_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

import config as C
from data_generator_vark import CommBSSVarKDataset, MOD_TYPES
from losses import variable_k_pit_loss, recursive_pit_loss, specialist_loss
from models import SlotSepNet, OneAndRestNet, SpecialistBankNet


# ============================================================================
# Argument parsing
# ============================================================================
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train variable-K SC-BSS models')

    # Architecture
    p.add_argument('--arch', type=str, default='slot',
                   choices=['slot', 'recursive', 'specialist'],
                   help='slot = SlotSepNet, recursive = OneAndRestNet, '
                        'specialist = SpecialistBankNet (A6 baseline)')
    p.add_argument('--hidden', type=int, default=C.ModelConfig.hidden_channels,
                   help='C-SE hidden channels')
    p.add_argument('--layers', type=int, default=C.ModelConfig.n_layers,
                   help='Number of C-SE residual blocks')
    p.add_argument('--no_se', action='store_true', help='Disable SE block (ablation)')

    # Variable-K problem
    p.add_argument('--k_min', type=int, default=C.VarKConfig.k_min,
                   help='Minimum source count in training')
    p.add_argument('--k_max', type=int, default=C.VarKConfig.k_max,
                   help='Maximum source count in training')
    p.add_argument('--k_slots', type=int, default=C.VarKConfig.k_slots,
                   help='Model slot budget (>= k_max+1 for the K=4 probe)')

    # Loss weights
    p.add_argument('--lambda_occ', type=float, default=C.VarKConfig.lambda_occ,
                   help='Weight on occupancy BCE (slot arch)')
    p.add_argument('--lambda_cnt', type=float, default=C.VarKConfig.lambda_cnt,
                   help='Weight on count-head CE (slot arch)')
    p.add_argument('--lambda_mse', type=float, default=0.0,
                   help='Weight on the MSE anchor term (all archs; 0.0 '
                        'disables it). Anchors the scale-invariant -SI-SDR '
                        'term so output magnitudes match the sources '
                        '(paper1 used 0.5*MSE + 0.5*(-SI-SDR)).')
    p.add_argument('--lambda_ser', type=float, default=C.TaskConfig.lambda_ser,
                   help='Weight on the differentiable soft-SER term (paper 5, '
                        'slot arch only; 0.0 disables it). Requires the '
                        'dataset return_carriers pipeline (auto-enabled).')
    p.add_argument('--lambda_sep', type=float, default=1.0,
                   help='Weight on the -SI-SDR separation term (slot arch). '
                        '1.0 = paper4 behavior; 0.0 = pure task-oriented '
                        'config (e) — PIT assignment still uses SI-SDR, but '
                        'no waveform gradient. With lambda_sep=0 and '
                        'lambda_ser>0, checkpoint selection switches from '
                        'val SI-SDR to val soft-SER.')
    p.add_argument('--ser_sigma2', type=float, default=C.TaskConfig.ser_sigma2,
                   help='Gaussian softness of the soft-demod log-likelihood '
                        'on the unit-power grid (ablation: 0.05 / 0.2)')
    p.add_argument('--lambda_llr', type=float, default=C.TaskConfig.lambda_llr,
                   help='Weight on the JointLLRHead symbol-CE term (paper 5, '
                        'S3; slot arch only, builds the joint head). 0.0 '
                        'disables it.')
    p.add_argument('--no_count_head', action='store_true',
                   help='Disable the explicit count head (slot arch)')

    # Curriculum
    p.add_argument('--curriculum_k2', type=int, default=0, metavar='EPOCHS',
                   help='Train on K=2 only for the first EPOCHS epochs, '
                        'then open to the full K range (ablation)')

    # Training hyperparameters
    p.add_argument('--epochs', type=int, default=C.TrainConfig.epochs)
    p.add_argument('--batch_size', type=int, default=C.DataConfig.batch_size)
    p.add_argument('--lr', type=float, default=C.TrainConfig.lr)
    p.add_argument('--weight_decay', type=float, default=C.TrainConfig.weight_decay)
    p.add_argument('--grad_clip', type=float, default=C.TrainConfig.grad_clip)
    p.add_argument('--train_samples', type=int, default=C.DataConfig.train_samples)
    p.add_argument('--val_samples', type=int, default=C.DataConfig.val_samples)
    p.add_argument('--early_stop', type=int, default=C.TrainConfig.early_stop_patience)

    # Data
    p.add_argument('--freq_gap_range', type=float, nargs=2,
                   default=list(C.SignalConfig.freq_gap_range),
                   metavar=('LOW', 'HIGH'),
                   help='Per-source carrier offset range (Hz) around carrier_base')
    p.add_argument('--snr_range', type=float, nargs=2,
                   default=list(C.SignalConfig.snr_range_train),
                   metavar=('LOW', 'HIGH'),
                   help='Training/validation SNR range (dB). Default '
                        '(-5, 20) from config; use -10 20 for the '
                        'extended-range ablation.')
    p.add_argument('--signal_length', type=int,
                   default=C.SignalConfig.signal_length,
                   help=argparse.SUPPRESS)  # debug-only; default must stay 4096
    p.add_argument('--num_workers', type=int, default=C.DataConfig.num_workers)

    # Misc
    p.add_argument('--seed', type=int, default=C.SEED)
    p.add_argument('--name', type=str, default='', help='Experiment name suffix')
    p.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    p.add_argument('--no_tensorboard', action='store_true')
    p.add_argument('--log_every', type=int, default=50)
    return p.parse_args()


# ============================================================================
# Model / dataset builders
# ============================================================================
def build_model(args) -> torch.nn.Module:
    common = dict(
        hidden_channels=args.hidden,
        n_layers=args.layers,
        k_max=args.k_slots,
        use_se=not args.no_se,
        head_embed_dim=C.ModelConfig.head_embed_dim,
    )
    if args.arch == 'slot':
        return SlotSepNet(use_count_head=not args.no_count_head,
                          use_joint_head=(args.lambda_llr > 0), **common)
    if args.arch == 'specialist':
        return SpecialistBankNet(**common)
    return OneAndRestNet(**common)


def build_dataset(args, n_samples: int, seed: int,
                  k_range: tuple[int, int] | None = None) -> CommBSSVarKDataset:
    k_min, k_max = k_range if k_range is not None else (args.k_min, args.k_max)
    return CommBSSVarKDataset(
        n_samples=n_samples,
        snr_range=tuple(args.snr_range),
        mod_types=MOD_TYPES,
        k_min=k_min, k_max=k_max, k_slots=args.k_slots,
        signal_length=args.signal_length,
        sample_rate=C.SignalConfig.sample_rate,
        carrier_base=C.SignalConfig.carrier_base,
        freq_gap_range=tuple(args.freq_gap_range),
        n_symbols=C.SignalConfig.n_symbols,
        roll_off=C.SignalConfig.roll_off,
        num_taps=C.SignalConfig.num_taps,
        apply_fading=C.SignalConfig.apply_fading,
        fading_taps=C.SignalConfig.fading_taps,
        seed=seed,
        # paper 5: soft-SER / joint-LLR losses need true carriers + mods
        return_carriers=(args.lambda_ser > 0 or args.lambda_llr > 0),
    )


def run_name(args) -> str:
    return (f"{args.arch}_h{args.hidden}_l{args.layers}"
            f"_k{args.k_min}{args.k_max}"
            f"_bs{args.batch_size}_lr{args.lr}"
            f"{('_' + args.name) if args.name else ''}"
            f"_s{args.seed}")


# ============================================================================
# Loss dispatch
# ============================================================================
def compute_loss(model, args, mix, sources, occ, k, mods=None, carriers=None):
    """Returns (loss, aux).  aux carries 'si_sdr' for all archs."""
    if args.arch == 'slot':
        out = model(mix)
        if len(out) == 4:
            slots, occ_logits, count_logits, sym_logits = out
        else:
            slots, occ_logits, count_logits = out
            sym_logits = None
        return variable_k_pit_loss(
            slots, occ_logits, count_logits, sources, k,
            lambda_occ=args.lambda_occ, lambda_cnt=args.lambda_cnt,
            lambda_mse=args.lambda_mse,
            lambda_ser=args.lambda_ser, lambda_sep=args.lambda_sep,
            lambda_llr=args.lambda_llr, sym_logits=sym_logits,
            ser_sigma2=args.ser_sigma2,
            carriers=carriers, mods=mods,
            sample_rate=C.SignalConfig.sample_rate,
            n_symbols=C.SignalConfig.n_symbols,
            roll_off=C.SignalConfig.roll_off,
            num_taps=C.SignalConfig.num_taps,
        )
    if args.arch == 'specialist':
        # Teacher routing: route each sample through the specialist of its
        # TRUE K (standard uPIT training for specialist banks).
        slots, count_logits = model(mix, k_route=k)
        return specialist_loss(slots, count_logits, sources, occ, k,
                               lambda_cnt=args.lambda_cnt,
                               lambda_mse=args.lambda_mse)
    return recursive_pit_loss(model, mix, sources, k,
                              lambda_mse=args.lambda_mse)


# ============================================================================
# Train / validation routines
# ============================================================================
def train_one_epoch(model, loader, optimizer, device, args, epoch: int) -> dict:
    model.train()
    sum_loss = 0.0
    sum_si_sdr = 0.0
    sum_ser = 0.0
    sum_llr = 0.0
    n_batches = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        # The training dataset yields a 6-tuple (with mods + TRUE carriers)
        # only when built with return_carriers=True (--lambda_ser > 0).
        if len(batch) == 6:
            mix, sources, occ, k, mods, carriers = batch
            mods = mods.to(device, non_blocking=True)
            carriers = carriers.to(device, non_blocking=True)
        else:
            mix, sources, occ, k = batch
            mods = carriers = None
        mix = mix.to(device, non_blocking=True)
        sources = sources.to(device, non_blocking=True)
        occ = occ.to(device, non_blocking=True)
        k = k.to(device, non_blocking=True)

        loss, aux = compute_loss(model, args, mix, sources, occ, k,
                                 mods, carriers)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        sum_loss += loss.item()
        sum_si_sdr += aux['si_sdr'].item()
        sum_ser += aux['ser_soft'].item()
        sum_llr += aux['llr'].item()
        n_batches += 1

        if (step + 1) % args.log_every == 0:
            msg = (f"  ep{epoch:>3d} step {step+1:>4d}/{len(loader)}  "
                   f"loss={sum_loss/n_batches:.4f}  "
                   f"si_sdr={sum_si_sdr/n_batches:.3f} dB  "
                   f"({time.time()-t0:.1f}s)")
            if args.lambda_ser > 0:
                msg += f"  ser_soft={sum_ser/n_batches:.4f}"
            if args.lambda_llr > 0:
                msg += f"  llr={sum_llr/n_batches:.4f}"
            print(msg)

    return {
        'loss': sum_loss / max(n_batches, 1),
        'si_sdr': sum_si_sdr / max(n_batches, 1),
        'ser_soft': sum_ser / max(n_batches, 1),
        'llr': sum_llr / max(n_batches, 1),
        'time': time.time() - t0,
    }


@torch.no_grad()
def validate(model, loader, device, args) -> dict:
    model.eval()
    sum_loss = 0.0
    sum_si_sdr = 0.0
    sum_ser = 0.0
    sum_llr = 0.0
    cnt_correct = 0
    n_samples = 0
    n_batches = 0

    for batch in loader:
        if len(batch) == 6:
            mix, sources, occ, k, mods, carriers = batch
            mods = mods.to(device, non_blocking=True)
            carriers = carriers.to(device, non_blocking=True)
        else:
            mix, sources, occ, k = batch
            mods = carriers = None
        mix = mix.to(device, non_blocking=True)
        sources = sources.to(device, non_blocking=True)
        occ = occ.to(device, non_blocking=True)
        k = k.to(device, non_blocking=True)

        # Note (specialist): the loss uses teacher routing (true K), so
        # val si_sdr is an oracle-K figure; honest end-to-end numbers come
        # from evaluate.py.
        loss, aux = compute_loss(model, args, mix, sources, occ, k,
                                 mods, carriers)
        sum_loss += loss.item()
        sum_si_sdr += aux['si_sdr'].item()
        sum_ser += aux['ser_soft'].item()
        sum_llr += aux['llr'].item()
        n_batches += 1

        # Counting accuracy of the inference-time route.
        if args.arch == 'slot':
            out = model(mix)
            occ_logits = out[1]
            k_hat = (torch.sigmoid(occ_logits) > C.VarKConfig.occ_threshold
                     ).sum(dim=1)
        elif args.arch == 'specialist':
            _, count_logits = model(mix)      # self-routed (inference rule)
            k_hat = count_logits.argmax(dim=1) + 1
        else:
            _, stop_logits = model(mix, max_steps=args.k_slots)
            stop_fired = torch.sigmoid(stop_logits) < 0.5    # "no more"
            # K_hat = first step where the stop head fires, +1; if it never
            # fires within the budget, K_hat = k_slots.
            k_hat = torch.full_like(k, args.k_slots)
            has_fired = torch.zeros_like(k, dtype=torch.bool)
            for i in range(args.k_slots):
                newly = stop_fired[:, i] & ~has_fired
                k_hat[newly] = i + 1
                has_fired |= stop_fired[:, i]
        cnt_correct += (k_hat == k).sum().item()
        n_samples += k.numel()

    return {
        'loss': sum_loss / max(n_batches, 1),
        'si_sdr': sum_si_sdr / max(n_batches, 1),
        'ser_soft': sum_ser / max(n_batches, 1),
        'llr': sum_llr / max(n_batches, 1),
        'count_acc': cnt_correct / max(n_samples, 1),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    args = get_args()
    assert (args.lambda_ser == 0.0 and args.lambda_llr == 0.0) \
        or args.arch == 'slot', \
        '--lambda_ser/--lambda_llr are wired for the slot arch only (plan v2 O3)'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = C.DEVICE
    print(f"Device: {device}  seed: {args.seed}")
    print(f"Arch: {args.arch}  K ~ U{{{args.k_min}..{args.k_max}}}  "
          f"k_slots={args.k_slots}  lambda_occ={args.lambda_occ}  "
          f"lambda_cnt={args.lambda_cnt}  lambda_mse={args.lambda_mse}  "
          f"lambda_ser={args.lambda_ser}  lambda_sep={args.lambda_sep}  "
          f"lambda_llr={args.lambda_llr}  ser_sigma2={args.ser_sigma2}  "
          f"count_head={not args.no_count_head}")
    if args.curriculum_k2 > 0:
        print(f"Curriculum: K=2 only for the first {args.curriculum_k2} epochs")

    # Datasets
    train_ds = build_dataset(args, args.train_samples, args.seed)
    val_ds = build_dataset(args, args.val_samples, args.seed + 1000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Curriculum: K=2-only warm-up dataset (re-instantiated, spec §7)
    curr_loader = None
    if args.curriculum_k2 > 0:
        curr_ds = build_dataset(args, args.train_samples, args.seed,
                                k_range=(2, 2))
        curr_loader = DataLoader(curr_ds, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)

    # Model
    model = build_model(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    # Checkpoint selection / LR-scheduling metric: val SI-SDR normally.
    # For the PURE task-oriented config (lambda_sep=0, lambda_ser>0) the
    # waveform metric is meaningless by construction — select by val
    # soft-SER (minimise) instead.
    select_by_ser = (args.lambda_sep == 0.0 and args.lambda_ser > 0.0)
    sel_mode = 'min' if select_by_ser else 'max'
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=sel_mode, factor=C.TrainConfig.scheduler_factor,
        patience=C.TrainConfig.scheduler_patience)

    # Resume
    start_epoch = 0
    best_val = float('inf') if select_by_ser else float('-inf')
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val = ckpt.get('best_val', best_val)

    # TensorBoard
    name = run_name(args)
    writer = None
    if HAS_TENSORBOARD and not args.no_tensorboard:
        writer = SummaryWriter(os.path.join(C.RUN_DIR, name))

    # Training loop
    no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        # Curriculum switch: K=2-only loader for the warm-up phase
        if curr_loader is not None and epoch < args.curriculum_k2:
            if epoch == 0:
                print(f"[curriculum] epochs 0..{args.curriculum_k2-1} use K=2 only")
            active_loader = curr_loader
        else:
            if curr_loader is not None and epoch == args.curriculum_k2:
                print(f"[curriculum] opening to full K range "
                      f"{{{args.k_min}..{args.k_max}}} from epoch {epoch}")
            active_loader = train_loader

        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        train_metrics = train_one_epoch(model, active_loader, optimizer,
                                        device, args, epoch + 1)
        val_metrics = validate(model, val_loader, device, args)
        val_sel = (val_metrics['ser_soft'] if select_by_ser
                   else val_metrics['si_sdr'])
        scheduler.step(val_sel)

        print(f"  TRAIN  loss={train_metrics['loss']:.4f}  "
              f"si_sdr={train_metrics['si_sdr']:.3f} dB")
        print(f"  VAL    loss={val_metrics['loss']:.4f}  "
              f"si_sdr={val_metrics['si_sdr']:.3f} dB  "
              f"count_acc={val_metrics['count_acc']:.3f}"
              + (f"  ser_soft={val_metrics['ser_soft']:.4f}"
                 if args.lambda_ser > 0 else "")
              + (f"  llr={val_metrics['llr']:.4f}"
                 if args.lambda_llr > 0 else ""))

        if writer is not None:
            writer.add_scalar('train/loss', train_metrics['loss'], epoch)
            writer.add_scalar('train/si_sdr', train_metrics['si_sdr'], epoch)
            writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            writer.add_scalar('val/si_sdr', val_metrics['si_sdr'], epoch)
            writer.add_scalar('val/count_acc', val_metrics['count_acc'], epoch)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            if args.lambda_ser > 0:
                writer.add_scalar('train/ser_soft',
                                  train_metrics['ser_soft'], epoch)

        # Save best by the selection metric (val SI-SDR, or val soft-SER
        # for the pure task-oriented config).
        ckpt_payload = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_val': best_val,
            'args': vars(args),
        }
        improved = (val_sel < best_val) if select_by_ser else (val_sel > best_val)
        if improved:
            best_val = val_sel
            no_improve = 0
            ckpt_path = os.path.join(C.CHECKPOINT_DIR, f"{name}_best.pt")
            torch.save(ckpt_payload, ckpt_path)
            print(f"  -> saved best to {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                print(f"Early stopping after {no_improve} epochs without improvement")
                break

        # Periodic checkpoint every save_every epochs + always save last
        if (epoch + 1) % C.TrainConfig.save_every == 0:
            torch.save(ckpt_payload,
                       os.path.join(C.CHECKPOINT_DIR, f"{name}_ep{epoch+1}.pt"))
        torch.save(ckpt_payload, os.path.join(C.CHECKPOINT_DIR, f"{name}_last.pt"))

    if writer is not None:
        writer.close()
    if select_by_ser:
        print(f"\nDone. Best val soft-SER = {best_val:.4f}")
    else:
        print(f"\nDone. Best val SI-SDR = {best_val:.3f} dB")


if __name__ == '__main__':
    main()
