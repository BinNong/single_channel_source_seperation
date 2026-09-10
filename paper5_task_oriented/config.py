"""
Paper 5 — Task-Oriented SC-BSS: Central configuration.

Single source of truth for all hyperparameters.  Style follows
paper1_cnn_se/config.py (config classes + device banner at import);
variable-K knobs live in VarKConfig (paper4-identical), task-oriented
soft-demodulation knobs live in TaskConfig (paper 5 addition).

Read at runtime by:
  - train.py          : training loop + checkpointing
  - evaluate.py       : deterministic test-grid construction
  - data_generator_vark.py / models.py / losses.py / soft_demod.py : defaults
"""

from __future__ import annotations

import torch

# =============================================================================
# Device Configuration
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[Config] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Config] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =============================================================================
# Signal Generation Parameters
# =============================================================================
class SignalConfig:
    """Communication signal generation parameters (identical to paper1-3)."""
    sample_rate: int = 16000          # Sampling rate (Hz)
    signal_length: int = 4096         # Number of samples per signal
    n_symbols: int = 256              # Number of symbols per signal
    carrier_base: float = 2000.0      # Per-source carrier = base + U(freq_gap_range)
    freq_gap_range: tuple = (0.0, 5.0)  # Per-source carrier offset range (Hz);
                                        # paper1's ±5 Hz jitter stays on top
    roll_off: float = 0.35            # RRC filter roll-off factor
    num_taps: int = 64                # RRC filter length

    # SNR range for training (dB); SNR is defined on TOTAL mixture power.
    snr_range_train: tuple = (-5, 20)
    # SNR values for testing
    snr_test_points: list = [-10, -5, 0, 5, 10, 15, 20]

    # Modulation types (same closed set as paper1)
    mod_types: list = ['BPSK', 'QPSK', '8PSK', '16QAM']

    # Channel effects
    apply_fading: bool = True         # Apply multipath fading
    fading_taps: int = 3              # Number of fading channel taps


# =============================================================================
# Variable-K problem definition
# =============================================================================
class VarKConfig:
    """Open-world variable-source-count configuration."""
    k_min: int = 1                    # Training range: K sampled from
    k_max: int = 3                    #   Uniform{k_min .. k_max}
    k_slots: int = 4                  # Model slot budget K_max (>= k_max + 1
                                      #   so the K=4 zero-shot probe is representable)
    k_extrap: int = 4                 # Zero-shot extrapolation cell in the test grid
    mix_weight_range: tuple = (0.4, 0.6)  # w_k ~ U(0.4, 0.6) i.i.d., NOT normalized

    # Loss weights (slot model)
    lambda_occ: float = 1.0           # Weight on occupancy BCE
    lambda_cnt: float = 0.1           # Weight on count-head CE (when enabled)

    # Inference
    occ_threshold: float = 0.5        # Occupancy threshold for counting/gating


# =============================================================================
# Task-oriented training (paper 5 addition)
# =============================================================================
class TaskConfig:
    """Differentiable soft-demodulation (task-oriented) loss configuration.

    See docs/PAPER5_TASK_ORIENTED_PLAN.md (v2) — the soft-SER term is a
    cross-entropy over constellation points computed on the SAME receiver
    pipeline as ser_comp.py (oracle carrier, matched filter, unit-power
    grid, phase-only alignment), so the training proxy and the evaluation
    truth agree by construction.
    """
    lambda_ser: float = 0.0       # Weight on the soft-SER term (0.0 disables;
                                  #   byte-identical to paper4 behavior)
    ser_sigma2: float = 0.1       # Gaussian softness of the soft-demod
                                  #   log-likelihood on the unit-power grid
    lambda_llr: float = 0.0       # S3: weight on the JointLLRHead symbol-CE
                                  #   term (0.0 disables)


# =============================================================================
# Dataset Parameters
# =============================================================================
class DataConfig:
    """Dataset configuration."""
    train_samples: int = 2000
    val_samples: int = 400
    test_n_per_cell: int = 100        # Per (SNR, K) cell of the test grid (all paper runs use 100)
    batch_size: int = 16              # Safe for 8GB RTX 4060
    num_workers: int = 2
    test_seed: int = 99999            # Deterministic test seed (paper3 convention)


# =============================================================================
# Model Parameters
# =============================================================================
class ModelConfig:
    """Model architecture configuration (C-SE backbone, paper3-identical)."""
    in_channels: int = 1              # Single complex channel (I+jQ)
    hidden_channels: int = 64         # Hidden dimension
    n_layers: int = 4                 # Number of ComplexResidualBlocks
    kernel_size_enc: int = 7          # Encoder kernel size
    kernel_size_hidden: int = 3       # Hidden layer kernel size
    kernel_size_dec: int = 7          # Decoder kernel size
    use_se: bool = True               # Use Complex Squeeze-and-Excitation
    se_reduction: int = 4             # SE channel reduction factor
    head_embed_dim: int = 64          # Occupancy / count / stop head MLP width
    use_count_head: bool = True       # SlotSepNet: explicit K-classifier head


# =============================================================================
# Training Parameters
# =============================================================================
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    scheduler_patience: int = 5       # ReduceLROnPlateau patience (on val SI-SDR)
    scheduler_factor: float = 0.5
    early_stop_patience: int = 20
    grad_clip: float = 1.0
    save_every: int = 5               # Periodic checkpoint every N epochs


# =============================================================================
# Reproducibility
# =============================================================================
SEED: int = 42


# =============================================================================
# Paths
# =============================================================================
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
RUN_DIR = os.path.join(ROOT_DIR, 'runs')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

for _d in (CHECKPOINT_DIR, RUN_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)
