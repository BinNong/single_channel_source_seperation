"""
Paper 6 — Sync-Aware Joint Detection: Central configuration.

Single source of truth for all hyperparameters.  Style follows
paper5_task_oriented/config.py (config classes + device banner at import);
SignalConfig values are identical to paper 5 so the benchmark is unchanged.
Paper-6 additions: SyncConfig (BlindCarrierSync, plan §3.1) and JointConfig
(JointPairDetector, plan §3.3).
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
# Signal Generation Parameters (identical to paper 5)
# =============================================================================
class SignalConfig:
    """Communication signal generation parameters (paper1-5 unchanged)."""
    sample_rate: int = 16000          # Sampling rate (Hz)
    signal_length: int = 4096         # Number of samples per signal
    n_symbols: int = 256              # Number of symbols per signal
    carrier_base: float = 2000.0      # Per-source carrier = base + U(freq_gap_range)
    freq_gap_range: tuple = (0.0, 5.0)  # Per-source carrier offset range (Hz);
                                      # paper1's ±5 Hz jitter stays on top
    roll_off: float = 0.35            # RRC filter roll-off factor
    num_taps: int = 64                # RRC filter length

    # SNR values for testing
    snr_test_points: list = [-10, -5, 0, 5, 10, 15, 20]

    # Modulation types (same closed set as paper1)
    mod_types: list = ['BPSK', 'QPSK', '8PSK', '16QAM']

    # Channel effects
    apply_fading: bool = True         # Apply multipath fading
    fading_taps: int = 3              # Number of fading channel taps


# =============================================================================
# Blind carrier synchronisation (plan §3.1 — BlindCarrierSync)
# =============================================================================
class SyncConfig:
    """Classical non-data-aided sync front-end (no learning).

    After coarse down-conversion at the nominal carrier, the residual
    offset is Δf = U(0,5) + U(-5,5) ∈ [-5, +10] Hz (carrier_base +
    freq_gap_range + paper1's in-generator ±5 Hz jitter).
    """
    nominal_carrier: float = 2000.0   # Coarse down-conversion frequency (Hz)
    freq_search_lo: float = -5.0      # Residual-offset search window (Hz)
    freq_search_hi: float = 10.0
    fft_zeropad: int = 8              # Zero-padding factor of the M-th-power FFT
    dd_iterations: int = 0            # Plan §3.1 step 4 DD phase line-fit passes.
                                    # DEFAULT 0 — measured fold-biased (hurts SER;
                                    # see sync.py docstring).  >0 enables the
                                    # spec'd DD refinement for ablation.
    # Symbol-level refinement / wrong-peak rescue (sync.py step 3):
    refine_half_win: float = 2.0      # Candidate A: narrow window around the
                                    #   sample-level coarse estimate (Hz)
    wide_half_win: float = 13.0       # Candidate B: full-window symbol-level
                                    #   estimate from the nominal carrier (Hz)
    # Rotational symmetry order per modulation (M-th power spectral line):
    sym_order: dict = None            # set below (class-body dict default)
    # Amplitude handling of the M-th-power line estimator per modulation
    # (measured 2026-09-13 on K=1 bursts, 20 trials x 2 seeds per cell):
    # 'ampnorm' (phase-only M-th power) is best for the constant-modulus
    # PSKs; 'raw' (amplitude-weighted x^M) is best for 16QAM, whose
    # amplitude normalisation injects transition noise.  All modes run on
    # the RRC-matched-filtered signal (10.7 dB noise-bandwidth reduction
    # before the nonlinearity; plain 'ampnorm' on the raw burst fails for
    # 8PSK/16QAM even at 0 dB).
    power_mode: dict = None           # set below
    # eq knobs (plan §3.2; NOT used by E1, reserved for E3+)
    eq_taps: int = 11                 # Complex FIR length, init = identity
    eq_mu: float = 1e-2               # Decision-directed LMS step size
    eq_passes: int = 8                # Adaptation passes per burst


SyncConfig.sym_order = {'BPSK': 2, 'QPSK': 4, '8PSK': 8, '16QAM': 4}
SyncConfig.power_mode = {'BPSK': 'ampnorm', 'QPSK': 'ampnorm',
                         '8PSK': 'ampnorm', '16QAM': 'raw'}


# =============================================================================
# Joint pair detection (plan §3.3 — JointPairDetector)
# =============================================================================
class JointConfig:
    """Joint ML detection over the K=2 slot pair + EM coupling estimation."""
    sigma2: float = 0.1               # Noise softness of the joint
                                      # log-likelihoods on the unit-power grid
    em_rounds: int = 3                # EM rounds for blind A estimation
    max_const: int = 16               # Largest constellation (16QAM); the
                                      # joint hypothesis grid is <= 16x16


# =============================================================================
# Dataset Parameters
# =============================================================================
class DataConfig:
    """Dataset configuration."""
    test_seed: int = 99999            # Deterministic test seed (paper3 convention)


# =============================================================================
# Variable-K problem definition (vendored from paper5 for the evaluation
# pipeline's dataset construction — values identical to paper4/5)
# =============================================================================
class VarKConfig:
    """Open-world variable-source-count configuration."""
    k_min: int = 1                    # Training range: K sampled from
    k_max: int = 3                    #   Uniform{k_min .. k_max}
    k_slots: int = 4                  # Model slot budget K_max
    k_extrap: int = 4                 # Zero-shot extrapolation cell
    mix_weight_range: tuple = (0.4, 0.6)  # w_k ~ U(0.4, 0.6) i.i.d.
    occ_threshold: float = 0.5        # Occupancy threshold for gating


# =============================================================================
# Theory validation (E2 — theory_validation.py)
# =============================================================================
class TheoryConfig:
    """Monte-Carlo / bound evaluation knobs for the §2 theory validation."""
    snr_points: list = (-5, 0, 5, 10, 15, 20)   # SNR grid for SER-vs-SNR
    sir_points: list = (0, 5, 10)               # SIR grid for SER-vs-SNR
    sir_sweep: list = (-5, -2.5, 0, 2.5, 5, 7.5, 10, 15, 20)  # SER-vs-SIR
    sir_sweep_snr: float = 20.0                 # fixed SNR for the SIR sweep
    n_mc_qpsk: int = 1_000_000    # MC symbols per point (QPSK x QPSK)
    n_mc_qam: int = 500_000       # MC symbols per point (16QAM x 16QAM)
    n_phi_mc: int = 12            # fixed-phase grid for the MC (offset from
                                #   the zero-contact rotations)
    n_phi_bound: int = 64         # phase grid for the union bound
    # d_min(phi) landscape
    dmin_phi_points: int = 1440   # phi grid over [0, 2pi)
    # measured-vs-CRB frequency error sweep (K=1, blind_sync_known_mod)
    crb_snr_points: list = (-5, 0, 5, 10, 15, 20)
    n_freq_bursts: int = 100      # bursts per (SNR, modulation)


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
