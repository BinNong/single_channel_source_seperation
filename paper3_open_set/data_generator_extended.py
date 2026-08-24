"""
Paper 3 — Open-Set SC-BSS: Extended data generator.

This file EXTENDS paper1_cnn_se/data_generator.py without modifying it
(because paper2_dp_mamba/data_generator.py is a symlink to paper1's).

Adds 3 modulation types not present in paper1's data_generator:
  - 64QAM          : 8×8 grid constellation
  - PI4_DQPSK      : π/4-shifted differential QPSK
  - OFDM_QPSK      : OFDM with QPSK subcarriers + cyclic prefix

Reuses MSK from paper1's data_generator (it exists but is unused by
paper1's training set, hence it is "unknown" for paper3).

Public API:
  - MOD_KNOWN         : ['BPSK', 'QPSK', '8PSK', '16QAM']
  - MOD_UNKNOWN       : ['64QAM', 'PI4_DQPSK', 'MSK', 'OFDM_QPSK']
  - MOD_ALL           : known + unknown
  - MOD_TO_IDX        : modulation name -> integer index
  - generate_open_set_signal(...)  : like paper1's generate_single_signal but
                                     routes OFDM through a special path
  - generate_open_set_mixture(...) : like paper1's generate_mixture but with
                                     per-source modulation labels in the
                                     return tuple
  - CommBSSOpenSetDataset          : training-style dataset (random
                                     per-sample, exposes per-source labels)
  - CommBSSOpenSetTestDataset      : fixed deterministic test set with
                                     controllable (known/known, known/unknown,
                                     unknown/unknown) protocols
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

# torch is a soft dependency: needed for the dataset classes and downstream
# training, but the signal-generation functions are pure numpy and can be
# smoke-tested on a machine without torch installed (e.g. local dev).
try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    Dataset = object  # fallback so class definition does not error
    _TORCH_AVAILABLE = False

# Pull paper1's data generator as the base; do NOT modify paper1's file.
# Use sys.path.append (not insert) so paper3_open_set/ stays ahead of
# paper1_cnn_se/ in sys.path — otherwise train.py's later `from models
# import OpenSetCSE` would resolve to paper1's models.py.
_PAPER1_DIR = Path(__file__).resolve().parent.parent / "paper1_cnn_se"
if str(_PAPER1_DIR) not in sys.path:
    sys.path.append(str(_PAPER1_DIR))

from data_generator import (  # noqa: E402
    rrc_filter,
    generate_symbols as _paper1_generate_symbols,
    generate_single_signal as _paper1_generate_single_signal,
)


# ============================================================================
# Modulation vocabulary
# ============================================================================
MOD_KNOWN   = ['BPSK', 'QPSK', '8PSK', '16QAM']
MOD_UNKNOWN = ['64QAM', 'PI4_DQPSK', 'MSK', 'OFDM_QPSK']
MOD_ALL     = MOD_KNOWN + MOD_UNKNOWN
MOD_TO_IDX  = {m: i for i, m in enumerate(MOD_ALL)}
IDX_TO_MOD  = {i: m for m, i in MOD_TO_IDX.items()}
NUM_MOD     = len(MOD_ALL)


# ============================================================================
# New symbol generators for unknown modulations
# ============================================================================
def _generate_64qam_symbols(n_symbols: int) -> np.ndarray:
    """64QAM: 8x8 grid constellation, normalized to unit average power."""
    re = 2 * np.random.randint(0, 8, n_symbols) - 7  # {±1, ±3, ±5, ±7}
    im = 2 * np.random.randint(0, 8, n_symbols) - 7
    sym = (re + 1j * im).astype(np.complex64)
    power = np.mean(np.abs(sym) ** 2)
    if power > 0:
        sym = sym / np.sqrt(power)
    return sym


def _generate_pi4_dqpsk_symbols(n_symbols: int) -> np.ndarray:
    """π/4-DQPSK: differential QPSK with ±π/4 or ±3π/4 phase jumps.

    Each output sample is the cumulative phase (mod 2π) of a random walk
    over the 4-DQPSK differential constellation.
    """
    delta = np.random.choice(
        [np.pi / 4, -np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4],
        size=n_symbols,
    )
    phase = np.cumsum(delta) + np.random.uniform(0.0, 2 * np.pi)
    return np.exp(1j * phase).astype(np.complex64)


def _generate_ofdm_qpsk_symbols(n_symbols: int,
                                n_subcarriers: int = 64,
                                cp_length: int = 16) -> np.ndarray:
    """OFDM-QPSK: IFFT of random QPSK subcarriers, plus cyclic prefix.

    Returns a time-domain complex signal whose total length is
    (n_ofdm_symbols * (n_subcarriers + cp_length)). Length is then
    truncated/padded to n_symbols at the caller.
    """
    qpsk_map = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64)
    qpsk_map /= np.sqrt(2.0)

    n_ofdm_sym = max(1, (n_symbols + n_subcarriers - 1) // n_subcarriers)
    total_data = n_ofdm_sym * n_subcarriers

    bits = np.random.randint(0, 4, size=total_data)
    freq = qpsk_map[bits].reshape(n_ofdm_sym, n_subcarriers)

    time = np.fft.ifft(freq, axis=1)              # [n_ofdm_sym, n_subcarriers]
    cp = time[:, -cp_length:]                    # last cp_length samples
    with_cp = np.concatenate([cp, time], axis=1) # [n_ofdm_sym, n_subcarriers + cp]

    signal = with_cp.flatten().astype(np.complex64)
    power = np.mean(np.abs(signal) ** 2)
    if power > 0:
        signal = signal / np.sqrt(power)
    return signal


# ============================================================================
# Dispatchers
# ============================================================================
def generate_open_set_symbols(n_symbols: int, mod_type: str) -> np.ndarray:
    """Generate `n_symbols` random constellation points for any supported mod.

    For OFDM the returned length can be larger than n_symbols because each
    "OFDM symbol" contributes n_subcarriers + cp_length samples. Callers
    that need an exact-length symbol stream (e.g. for upsampling) should
    resample/truncate downstream.
    """
    if mod_type in MOD_KNOWN or mod_type == 'MSK':
        return _paper1_generate_symbols(n_symbols, mod_type)
    if mod_type == '64QAM':
        return _generate_64qam_symbols(n_symbols)
    if mod_type == 'PI4_DQPSK':
        return _generate_pi4_dqpsk_symbols(n_symbols)
    if mod_type == 'OFDM_QPSK':
        return _generate_ofdm_qpsk_symbols(n_symbols)
    raise ValueError(f"Unknown modulation type: {mod_type}")


def generate_open_set_signal(n_symbols: int,
                             carrier_freq: float,
                             sample_rate: float,
                             signal_length: int,
                             mod_type: str,
                             roll_off: float,
                             num_taps: int,
                             apply_fading: bool = True,
                             fading_taps: int = 3):
    """Generate a single modulated signal for paper3's modulation set.

    Returns
    -------
    signal  : complex ndarray [signal_length], unit power.
    symbols : complex ndarray (raw constellation); for OFDM this is the
              IFFT-with-CP time-domain vector (its length may differ
              from `n_symbols`).
    """
    if mod_type == 'OFDM_QPSK':
        # OFDM is already a shaped baseband signal; bypass the standard
        # upsample + RRC pipeline and apply only fading + carrier offset.
        symbols = generate_open_set_symbols(n_symbols, 'OFDM_QPSK')
        return _apply_ofdm_tail(symbols, carrier_freq, sample_rate,
                                  signal_length, apply_fading, fading_taps), symbols

    if mod_type in ('64QAM', 'PI4_DQPSK'):
        # Custom constellations that paper1 does not know about.  Generate
        # the symbols ourselves, then run them through paper1's upsample +
        # RRC + upconvert + fading pipeline (copied locally because paper1's
        # generate_single_signal hard-codes its own generate_symbols).
        symbols = generate_open_set_symbols(n_symbols, mod_type)
        signal = _apply_paper1_pipeline(symbols, n_symbols, carrier_freq,
                                          sample_rate, signal_length,
                                          roll_off, num_taps,
                                          apply_fading, fading_taps)
        return signal.astype(np.complex64), symbols

    # Known 4 + MSK: paper1 handles everything.
    return _paper1_generate_single_signal(
        n_symbols, carrier_freq, sample_rate, signal_length,
        mod_type, roll_off, num_taps, apply_fading, fading_taps,
    )


# ----------------------------------------------------------------------------
# Local copies of paper1's per-signal pipeline.  We re-implement the steps
# that consume `symbols` so we can drive them with custom constellations
# (64QAM, π/4-DQPSK) without modifying paper1's source file.
# ----------------------------------------------------------------------------
def _apply_paper1_pipeline(symbols: np.ndarray,
                           n_symbols: int,
                           carrier_freq: float,
                           sample_rate: float,
                           signal_length: int,
                           roll_off: float,
                           num_taps: int,
                           apply_fading: bool = True,
                           fading_taps: int = 3) -> np.ndarray:
    """Mirror of paper1.generate_single_signal steps 2-7.

    Accepts custom `symbols` (already-generated constellation points) and
    runs them through the standard upsample + RRC + upconvert + fading
    pipeline.  Used for 64QAM and π/4-DQPSK.
    """
    # 1. Upsample (zero-insertion)
    sps = max(4, signal_length // n_symbols)
    upsampled = np.zeros(n_symbols * sps, dtype=complex)
    upsampled[::sps] = symbols

    # 2. RRC pulse shaping
    rrc = rrc_filter(num_taps, roll_off, sps)
    shaped = np.convolve(upsampled, rrc, mode='same')

    # 3. Resample to exact signal_length
    if len(shaped) != signal_length:
        shaped = resample_poly(shaped, signal_length, len(shaped))
    shaped = shaped[:signal_length]

    # 4. Upconvert to carrier frequency
    t = np.arange(signal_length) / sample_rate
    freq_offset = carrier_freq + np.random.uniform(-5, 5)
    signal = shaped * np.exp(1j * 2 * np.pi * freq_offset * t)

    # 5. Multipath fading
    if apply_fading:
        fading = np.random.randn(fading_taps) + 1j * np.random.randn(fading_taps)
        fading = fading / np.linalg.norm(fading)
        signal = np.convolve(signal, fading, mode='same')[:signal_length]

    # 6. Normalize power
    signal = signal / (np.sqrt(np.mean(np.abs(signal) ** 2)) + 1e-10)
    return signal


def _apply_ofdm_tail(symbols: np.ndarray,
                     carrier_freq: float,
                     sample_rate: float,
                     signal_length: int,
                     apply_fading: bool,
                     fading_taps: int) -> np.ndarray:
    """Mirror of paper1.generate_single_signal steps 4-7, but for OFDM
    (which has already been shaped in the time domain by the IFFT).
    """
    if len(symbols) > signal_length:
        signal = symbols[:signal_length]
    elif len(symbols) < signal_length:
        signal = resample_poly(symbols, signal_length, len(symbols))[:signal_length]
    else:
        signal = symbols

    t = np.arange(signal_length) / sample_rate
    freq_offset = carrier_freq + np.random.uniform(-5, 5)
    signal = signal * np.exp(1j * 2 * np.pi * freq_offset * t)

    if apply_fading:
        fading = np.random.randn(fading_taps) + 1j * np.random.randn(fading_taps)
        fading = fading / np.linalg.norm(fading)
        signal = np.convolve(signal, fading, mode='same')[:signal_length]

    signal = signal / (np.sqrt(np.mean(np.abs(signal) ** 2)) + 1e-10)
    return signal.astype(np.complex64)


def generate_open_set_mixture(signal_length: int,
                              sample_rate: float,
                              snr_db: float,
                              mod_type_1: str,
                              mod_type_2: str,
                              carrier_freq_1: float = 2000.0,
                              carrier_freq_2: float = 2005.0,
                              n_symbols: int = 256,
                              roll_off: float = 0.35,
                              num_taps: int = 64,
                              apply_fading: bool = True,
                              fading_taps: int = 3):
    """Generate a 2-source mixture; returns modulation labels too.

    Returns
    -------
    mixture  : complex ndarray [signal_length]
    source1  : complex ndarray [signal_length]
    source2  : complex ndarray [signal_length]
    mod1_idx : int in [0, NUM_MOD)
    mod2_idx : int in [0, NUM_MOD)
    """
    src1, _ = generate_open_set_signal(
        n_symbols, carrier_freq_1, sample_rate, signal_length,
        mod_type_1, roll_off, num_taps, apply_fading, fading_taps,
    )
    src2, _ = generate_open_set_signal(
        n_symbols, carrier_freq_2, sample_rate, signal_length,
        mod_type_2, roll_off, num_taps, apply_fading, fading_taps,
    )

    alpha = np.random.uniform(0.4, 0.6)
    mix_clean = alpha * src1 + (1 - alpha) * src2

    sig_power = np.mean(np.abs(mix_clean) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(signal_length) + 1j * np.random.randn(signal_length)
    )
    mixture = mix_clean + noise

    return (
        mixture.astype(np.complex64),
        src1.astype(np.complex64),
        src2.astype(np.complex64),
        MOD_TO_IDX[mod_type_1],
        MOD_TO_IDX[mod_type_2],
    )


# ============================================================================
# Datasets
# ============================================================================
class CommBSSOpenSetDataset(Dataset):
    """Training-style dataset; both sources come from `mod_types`.

    Returns
    -------
    mixture   : torch.complex64 [1, signal_length]
    source1   : torch.complex64 [1, signal_length]
    source2   : torch.complex64 [1, signal_length]
    mod1_idx  : int64 scalar
    mod2_idx  : int64 scalar
    """

    def __init__(self,
                 n_samples: int,
                 snr_range: tuple[float, float],
                 mod_types: list[str],
                 signal_length: int = 4096,
                 sample_rate: int = 16000,
                 carrier_freq_1: float = 2000.0,
                 carrier_freq_2: float = 2005.0,
                 n_symbols: int = 256,
                 roll_off: float = 0.35,
                 num_taps: int = 64,
                 apply_fading: bool = True,
                 fading_taps: int = 3,
                 seed: int | None = None,
                 freq_gap_range: tuple[float, float] | None = None) -> None:
        self.n_samples = n_samples
        self.snr_range = snr_range
        self.mod_types = list(mod_types)
        self.signal_length = signal_length
        self.sample_rate = sample_rate
        self.carrier_freq_1 = carrier_freq_1
        self.carrier_freq_2 = carrier_freq_2
        self.n_symbols = n_symbols
        self.roll_off = roll_off
        self.num_taps = num_taps
        self.apply_fading = apply_fading
        self.fading_taps = fading_taps
        self.seed = seed
        self.freq_gap_range = freq_gap_range

        if seed is not None:
            np.random.seed(seed)
        self.configs = []
        for _ in range(n_samples):
            snr = np.random.uniform(snr_range[0], snr_range[1])
            mod1 = str(np.random.choice(mod_types))
            mod2 = str(np.random.choice(mod_types))
            if freq_gap_range is not None:
                gap = float(np.random.uniform(freq_gap_range[0], freq_gap_range[1]))
                cf2 = carrier_freq_1 + gap
            else:
                cf2 = carrier_freq_2
            self.configs.append((snr, mod1, mod2, cf2))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        snr, mod1, mod2, cf2 = self.configs[idx]
        mix, s1, s2, m1_idx, m2_idx = generate_open_set_mixture(
            self.signal_length, self.sample_rate, snr, mod1, mod2,
            self.carrier_freq_1, cf2,
            self.n_symbols, self.roll_off, self.num_taps,
            self.apply_fading, self.fading_taps,
        )
        return (
            torch.from_numpy(mix).unsqueeze(0).to(torch.complex64),
            torch.from_numpy(s1).unsqueeze(0).to(torch.complex64),
            torch.from_numpy(s2).unsqueeze(0).to(torch.complex64),
            torch.tensor(m1_idx, dtype=torch.long),
            torch.tensor(m2_idx, dtype=torch.long),
        )


class CommBSSOpenSetTestDataset(Dataset):
    """Fixed test dataset that supports per-pair modulation protocols.

    protocol ∈ {'kk', 'ku', 'uu'}
        kk : both sources from mod_known_pool (closed-set)
        ku : one source from mod_known_pool, one from mod_unknown_pool
        uu : both sources from mod_unknown_pool (extreme OOD)

    Returns
    -------
    mixture      : torch.complex64 [1, signal_length]
    source1      : torch.complex64 [1, signal_length]
    source2      : torch.complex64 [1, signal_length]
    mod1_idx     : int64
    mod2_idx     : int64
    mod1_is_ood  : bool (True if mod1 is from MOD_UNKNOWN)
    mod2_is_ood  : bool
    """

    def __init__(self,
                 n_per_snr: int = 500,
                 snr_points: list[int] | None = None,
                 mod_known_pool: list[str] | None = None,
                 mod_unknown_pool: list[str] | None = None,
                 signal_length: int = 4096,
                 sample_rate: int = 16000,
                 seed: int = 12345,
                 carrier_freq_1: float = 2000.0,
                 carrier_freq_2: float = 2005.0,
                 protocol: str = 'kk') -> None:
        if snr_points is None:
            snr_points = [-10, -5, 0, 5, 10, 15, 20]
        if mod_known_pool is None:
            mod_known_pool = MOD_KNOWN
        if mod_unknown_pool is None:
            mod_unknown_pool = MOD_UNKNOWN
        assert protocol in ('kk', 'ku', 'uu'), f"Unknown protocol {protocol}"

        self.samples = []
        rng = np.random.RandomState(seed)

        # Save the RNG state, restore at the end to keep behaviour deterministic
        # across instances if multiple are created with overlapping seeds.
        _stashed = np.random.get_state()
        try:
            np.random.seed(seed)

            for snr in snr_points:
                # For each (mod1, mod2) pair from the appropriate pools, generate
                # n_per_snr // |pool|^2 samples.
                if protocol == 'kk':
                    pairs = [(a, b) for a in mod_known_pool for b in mod_known_pool]
                elif protocol == 'ku':
                    # Include (known, known), (known, unknown), (unknown, known)
                    # but tag them so we can split later if needed.
                    pairs = (
                        [(a, b) for a in mod_known_pool for b in mod_unknown_pool]
                        + [(a, b) for a in mod_unknown_pool for b in mod_known_pool]
                    )
                else:  # uu
                    pairs = [(a, b) for a in mod_unknown_pool for b in mod_unknown_pool]

                n_per_pair = max(1, n_per_snr // len(pairs))

                for mod1, mod2 in pairs:
                    for _ in range(n_per_pair):
                        mix, s1, s2, m1_idx, m2_idx = generate_open_set_mixture(
                            signal_length, sample_rate, snr, mod1, mod2,
                            carrier_freq_1=carrier_freq_1,
                            carrier_freq_2=carrier_freq_2,
                        )
                        self.samples.append({
                            'mixture': torch.from_numpy(mix).unsqueeze(0).to(torch.complex64),
                            'source1': torch.from_numpy(s1).unsqueeze(0).to(torch.complex64),
                            'source2': torch.from_numpy(s2).unsqueeze(0).to(torch.complex64),
                            'mod1_idx': int(m1_idx),
                            'mod2_idx': int(m2_idx),
                            'mod1_is_ood': mod1 in mod_unknown_pool,
                            'mod2_is_ood': mod2 in mod_unknown_pool,
                            'snr': float(snr),
                        })
        finally:
            np.random.set_state(_stashed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            s['mixture'], s['source1'], s['source2'],
            torch.tensor(s['mod1_idx'], dtype=torch.long),
            torch.tensor(s['mod2_idx'], dtype=torch.long),
            torch.tensor(s['mod1_is_ood'], dtype=torch.bool),
            torch.tensor(s['mod2_is_ood'], dtype=torch.bool),
            float(s['snr']),
        )


# ============================================================================
# Quick smoke test
# ============================================================================
if __name__ == '__main__':
    print("Testing paper3 data_generator_extended ...")

    # 1. Each modulation produces a valid signal (numpy-only)
    for mod in MOD_ALL:
        sig, sym = generate_open_set_signal(
            n_symbols=256, carrier_freq=2000.0, sample_rate=16000,
            signal_length=4096, mod_type=mod, roll_off=0.35, num_taps=64,
        )
        power = float(np.mean(np.abs(sig) ** 2))
        print(f"  {mod:12s}  signal shape={sig.shape}  unit_power={power:.4f}")

    # 2. Mixture generation with per-source labels
    mix, s1, s2, m1, m2 = generate_open_set_mixture(
        signal_length=4096, sample_rate=16000, snr_db=10,
        mod_type_1='QPSK', mod_type_2='64QAM',
    )
    print(f"\nMixture (QPSK + 64QAM) shapes: mix={mix.shape}, s1={s1.shape}, s2={s2.shape}")
    print(f"  mod1_idx={m1} ({IDX_TO_MOD[m1]}), mod2_idx={m2} ({IDX_TO_MOD[m2]})")

    # 3. Dataset-level smoke (requires torch)
    try:
        import torch  # noqa: F401
        from torch.utils.data import DataLoader
    except ImportError:
        print("\n[torch not installed locally — skipping dataset tests; "
              "training runs on remote server]")
    else:
        train_ds = CommBSSOpenSetDataset(
            n_samples=8, snr_range=(0, 20),
            mod_types=MOD_KNOWN, seed=42,
        )
        loader = DataLoader(train_ds, batch_size=4)
        batch = next(iter(loader))
        mix_b, s1_b, s2_b, m1_b, m2_b = batch
        print(f"\nTrain batch shapes:")
        print(f"  mix={tuple(mix_b.shape)}  s1={tuple(s1_b.shape)}  "
              f"m1={tuple(m1_b.shape)}  m1[0]={m1_b[0].item()}")

        for proto in ('kk', 'ku', 'uu'):
            ds = CommBSSOpenSetTestDataset(
                n_per_snr=10, snr_points=[0, 10],
                seed=12345, protocol=proto,
            )
            print(f"\nTest set protocol={proto}: {len(ds)} samples")
            m, s1, s2, mi1, mi2, ood1, ood2, snr = ds[0]
            print(f"  [0] mod1={IDX_TO_MOD[int(mi1)]} (ood={bool(ood1)}), "
                  f"mod2={IDX_TO_MOD[int(mi2)]} (ood={bool(ood2)}), snr={snr}")

    print("\ndata_generator_extended smoke test passed!")