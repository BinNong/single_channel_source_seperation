"""
Configuration for Single-Channel Blind Source Separation of Communication Signals
using Complex-Valued Lightweight CNN.

All hyperparameters are centralized here for easy tuning.
"""

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
    """Communication signal generation parameters."""
    sample_rate: int = 16000          # Sampling rate (Hz)
    signal_length: int = 4096         # Number of samples per signal
    n_symbols: int = 256              # Number of symbols per signal
    carrier_freq_1: float = 2000.0    # Carrier frequency of source 1 (Hz)
    carrier_freq_2: float = 2005.0    # Carrier frequency of source 2 (Hz) - co-frequency overlap
    freq_offset_range: tuple = (-10.0, 10.0)  # Random frequency offset range (Hz)
    roll_off: float = 0.35            # RRC filter roll-off factor
    num_taps: int = 64                # RRC filter length

    # SNR range for training (dB)
    snr_range_train: tuple = (-5, 20)
    # SNR values for testing
    snr_test_points: list = [-10, -5, 0, 5, 10, 15, 20]

    # Modulation types
    mod_types_train: list = ['BPSK', 'QPSK', '8PSK', '16QAM']
    mod_types_test: list = ['BPSK', 'QPSK', '8PSK', '16QAM']

    # Channel effects
    apply_fading: bool = True         # Apply multipath fading
    fading_taps: int = 3              # Number of fading channel taps
    apply_timing_offset: bool = True  # Apply random timing offset

# =============================================================================
# Dataset Parameters
# =============================================================================
class DataConfig:
    """Dataset configuration."""
    train_samples: int = 50000
    val_samples: int = 5000
    test_samples: int = 5000
    batch_size_train: int = 16        # Safe for 8GB RTX 4060
    batch_size_val: int = 32
    batch_size_test: int = 32
    num_workers: int = 4

# =============================================================================
# Model Parameters (Complex Lightweight CNN)
# =============================================================================
class ModelConfig:
    """Model architecture configuration."""
    # --- Complex CNN (Proposed) ---
    in_channels: int = 1              # Single complex channel (I+jQ)
    hidden_channels: int = 32         # Hidden dimension (small for lightweight)
    n_layers: int = 4                 # Number of ComplexConv + SE blocks
    kernel_size_enc: int = 7          # Encoder kernel size
    kernel_size_hidden: int = 3       # Hidden layer kernel size
    kernel_size_dec: int = 7          # Decoder kernel size
    use_se: bool = True               # Use Complex Squeeze-and-Excitation
    se_reduction: int = 4             # SE channel reduction factor

    # --- Baseline: Real-valued CNN ---
    baseline_hidden: int = 64         # Larger hidden dim for real-valued baseline
    baseline_layers: int = 6

# =============================================================================
# Training Parameters
# =============================================================================
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    early_stop_patience: int = 20
    grad_clip: float = 1.0

    # Loss function: 'mse' | 'si_sdr' | 'combined'
    loss_type: str = 'mse'
    # For combined loss: loss = alpha * mse + (1-alpha) * si_sdr
    loss_alpha: float = 0.5

    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_best_only: bool = True

# =============================================================================
# Evaluation Parameters
# =============================================================================
class EvalConfig:
    """Evaluation configuration."""
    metrics: list = ['SDR', 'SIR', 'SER', 'NMSE']
    save_results: bool = True
    results_dir: str = './results'
    visualize: bool = True
    n_viz_samples: int = 5

# =============================================================================
# Reproducibility
# =============================================================================
SEED: int = 42
