"""
Communication Signal Dataset Generator

Generates co-frequency overlapped communication signals for single-channel
blind source separation. Supports BPSK, QPSK, 8PSK, 16QAM modulation with
RRC pulse shaping, multipath fading, and AWGN.

Output format: Complex-valued baseband signals (I + jQ).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import resample_poly
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# RRC Filter
# =============================================================================
def rrc_filter(num_taps, roll_off, sps):
    """Generate Root-Raised Cosine (RRC) filter coefficients."""
    t = np.arange(-num_taps // 2, num_taps // 2 + 1) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1.0 + roll_off * (4 / np.pi - 1)
        elif abs(abs(4 * roll_off * ti) - 1.0) < 1e-10:
            h[i] = (roll_off / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * roll_off)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * roll_off))
            )
        else:
            num = np.sin(np.pi * ti * (1 - roll_off)) + 4 * roll_off * ti * np.cos(np.pi * ti * (1 + roll_off))
            den = np.pi * ti * (1 - (4 * roll_off * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))


# =============================================================================
# Modulation Constellations
# =============================================================================
def generate_symbols(n_symbols, mod_type):
    """Generate random symbol sequence for given modulation type."""
    if mod_type == 'BPSK':
        return 2 * np.random.randint(0, 2, n_symbols) - 1 + 0j
    elif mod_type == 'QPSK':
        return (2 * np.random.randint(0, 2, n_symbols) - 1) + 1j * (2 * np.random.randint(0, 2, n_symbols) - 1)
    elif mod_type == '8PSK':
        m = np.random.randint(0, 8, n_symbols)
        return np.exp(1j * (2 * np.pi * m / 8 + np.pi / 8))
    elif mod_type == '16QAM':
        re = 2 * np.random.randint(0, 4, n_symbols) - 3
        im = 2 * np.random.randint(0, 4, n_symbols) - 3
        return re + 1j * im
    elif mod_type == 'MSK':
        # MSK: continuous-phase FSK with modulation index 0.5
        bits = np.random.randint(0, 2, n_symbols)
        symbols = 2 * bits - 1  # +1 or -1 for each bit period
        return symbols.astype(np.complex64)
    elif mod_type == 'GMSK':
        # GMSK: same as MSK for symbol generation (Gaussian filtering in pulse shaping)
        bits = np.random.randint(0, 2, n_symbols)
        symbols = 2 * bits - 1
        return symbols.astype(np.complex64)
    else:
        raise ValueError(f"Unknown modulation type: {mod_type}")


# =============================================================================
# Single Signal Generation
# =============================================================================
def generate_single_signal(n_symbols, carrier_freq, sample_rate, signal_length,
                           mod_type, roll_off, num_taps, apply_fading=True,
                           fading_taps=3):
    """Generate a single modulated communication signal with channel effects."""
    # 1. Generate random symbols
    symbols = generate_symbols(n_symbols, mod_type)

    # 2. Upsample (zero-insertion)
    sps = int(sample_rate / (n_symbols / (signal_length / sample_rate)))
    # Adaptive samples per symbol
    sps = max(4, signal_length // n_symbols)
    upsampled = np.zeros(n_symbols * sps, dtype=complex)
    upsampled[::sps] = symbols

    # 3. RRC pulse shaping
    rrc = rrc_filter(num_taps, roll_off, sps)
    shaped = np.convolve(upsampled, rrc, mode='same')

    # 4. Resample to exact signal_length
    if len(shaped) != signal_length:
        g = signal_length // len(shaped) + 1
        shaped = resample_poly(shaped, signal_length, len(shaped))

    shaped = shaped[:signal_length]

    # 5. Up-convert to carrier frequency
    t = np.arange(signal_length) / sample_rate
    freq_offset = carrier_freq + np.random.uniform(-5, 5)  # Small random offset
    signal = shaped * np.exp(1j * 2 * np.pi * freq_offset * t)

    # 6. Apply multipath fading channel
    if apply_fading:
        fading = np.random.randn(fading_taps) + 1j * np.random.randn(fading_taps)
        fading = fading / np.linalg.norm(fading)
        signal = np.convolve(signal, fading, mode='same')
        signal = signal[:signal_length]

    # 7. Normalize power
    signal = signal / (np.sqrt(np.mean(np.abs(signal) ** 2)) + 1e-10)

    return signal, symbols


# =============================================================================
# Mixture Generation
# =============================================================================
def generate_mixture(signal_length, sample_rate, snr_db, mod_type_1='QPSK',
                     mod_type_2='QPSK', carrier_freq_1=2000.0, carrier_freq_2=2005.0,
                     n_symbols=256, roll_off=0.35, num_taps=64,
                     apply_fading=True, fading_taps=3):
    """
    Generate a mixture of two co-frequency communication signals.

    Returns:
        mixture: complex array [signal_length]
        source1: complex array [signal_length]
        source2: complex array [signal_length]
        symbols1: original symbols of source 1
        symbols2: original symbols of source 2
    """
    # Generate two source signals (nearly co-frequency)
    source1, symbols1 = generate_single_signal(
        n_symbols, carrier_freq_1, sample_rate, signal_length,
        mod_type_1, roll_off, num_taps, apply_fading, fading_taps
    )
    source2, symbols2 = generate_single_signal(
        n_symbols, carrier_freq_2, sample_rate, signal_length,
        mod_type_2, roll_off, num_taps, apply_fading, fading_taps
    )

    # Mix with random amplitude ratio
    alpha = np.random.uniform(0.4, 0.6)  # Roughly balanced
    mixture_clean = alpha * source1 + (1 - alpha) * source2

    # Add AWGN
    signal_power = np.mean(np.abs(mixture_clean) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(signal_length) + 1j * np.random.randn(signal_length))
    mixture = mixture_clean + noise

    return mixture, source1, source2, symbols1, symbols2


# =============================================================================
# PyTorch Dataset
# =============================================================================
class CommBSSDataset(Dataset):
    """
    PyTorch Dataset for communication signal blind source separation.
    Generates data on-the-fly for memory efficiency.
    """

    def __init__(self, n_samples, snr_range, mod_types, signal_length=4096,
                 sample_rate=16000, carrier_freq_1=2000.0, carrier_freq_2=2005.0,
                 n_symbols=256, roll_off=0.35, num_taps=64,
                 apply_fading=True, fading_taps=3, seed=None,
                 freq_gap_range=None):
        """
        Args:
            freq_gap_range: if not None, a tuple (low, high) in Hz. Each sample draws
                its carrier-frequency gap uniformly from [low, high]. Used to train
                models that need to generalise across a range of co-frequency
                separations, e.g. freq_gap_range=(0.0, 5.0). If None, the fixed
                carrier_freq_2 - carrier_freq_1 gap is used.
        """
        self.n_samples = n_samples
        self.snr_range = snr_range
        self.mod_types = mod_types
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

        # Pre-generate configurations for reproducibility
        if seed is not None:
            np.random.seed(seed)
        self.configs = []
        for i in range(n_samples):
            snr = np.random.uniform(snr_range[0], snr_range[1])
            mod1 = np.random.choice(mod_types)
            mod2 = np.random.choice(mod_types)
            if freq_gap_range is not None:
                gap = float(np.random.uniform(freq_gap_range[0], freq_gap_range[1]))
                cf2 = carrier_freq_1 + gap
            else:
                cf2 = carrier_freq_2
            self.configs.append((snr, mod1, mod2, cf2))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        snr, mod1, mod2, cf2 = self.configs[idx]

        mixture, source1, source2, _, _ = generate_mixture(
            self.signal_length, self.sample_rate, snr,
            mod1, mod2, self.carrier_freq_1, cf2,
            self.n_symbols, self.roll_off, self.num_taps,
            self.apply_fading, self.fading_taps
        )

        # Convert to PyTorch complex tensors
        # Shape: [1, signal_length] - channel-first format
        mixture_t = torch.from_numpy(mixture).unsqueeze(0).to(torch.complex64)
        source1_t = torch.from_numpy(source1).unsqueeze(0).to(torch.complex64)
        source2_t = torch.from_numpy(source2).unsqueeze(0).to(torch.complex64)

        return mixture_t, source1_t, source2_t


# =============================================================================
# Test Dataset (Fixed SNR points for evaluation)
# =============================================================================
class CommBSSTestDataset(Dataset):
    """Fixed test dataset with specific SNR points for systematic evaluation."""

    def __init__(self, n_per_snr=500, snr_points=None, mod_types=None,
                 signal_length=4096, sample_rate=16000, seed=12345,
                 carrier_freq_1=2000.0, carrier_freq_2=2005.0):
        """If carrier_freq_1 != carrier_freq_2, a fixed frequency gap is used."""
        if snr_points is None:
            snr_points = [-10, -5, 0, 5, 10, 15, 20]
        if mod_types is None:
            mod_types = ['BPSK', 'QPSK', '8PSK', '16QAM']

        self.samples = []
        np.random.seed(seed)

        for snr in snr_points:
            for mod1 in mod_types:
                for mod2 in mod_types:
                    for _ in range(n_per_snr // (len(mod_types) ** 2)):
                        mixture, source1, source2, _, _ = generate_mixture(
                            signal_length, sample_rate, snr, mod1, mod2,
                            carrier_freq_1=carrier_freq_1,
                            carrier_freq_2=carrier_freq_2
                        )
                        self.samples.append({
                            'mixture': torch.from_numpy(mixture).unsqueeze(0).to(torch.complex64),
                            'source1': torch.from_numpy(source1).unsqueeze(0).to(torch.complex64),
                            'source2': torch.from_numpy(source2).unsqueeze(0).to(torch.complex64),
                            'snr': snr,
                            'mod1': mod1,
                            'mod2': mod2
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['mixture'], s['source1'], s['source2'], s['snr'], s['mod1'], s['mod2']


# =============================================================================
# Quick Test
# =============================================================================
if __name__ == '__main__':
    print("Testing data generator...")

    # Test single mixture generation
    mix, s1, s2, sym1, sym2 = generate_mixture(
        signal_length=4096, sample_rate=16000, snr_db=10,
        mod_type_1='QPSK', mod_type_2='16QAM'
    )
    print(f"Mixture shape: {mix.shape}, dtype: {mix.dtype}")
    print(f"Source 1 power: {np.mean(np.abs(s1)**2):.4f}")
    print(f"Mixture SNR (estimated): {10*np.log10(np.mean(np.abs(s1+s2)**2)/np.mean(np.abs(mix-s1-s2)**2)):.1f} dB")

    # Test PyTorch dataset
    from torch.utils.data import DataLoader
    dataset = CommBSSDataset(n_samples=100, snr_range=(0, 20),
                             mod_types=['BPSK', 'QPSK'], seed=42)
    loader = DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))
    mix_batch, s1_batch, s2_batch = batch
    print(f"\nDataset batch shapes:")
    print(f"  Mixture: {mix_batch.shape}, dtype: {mix_batch.dtype}")
    print(f"  Source1: {s1_batch.shape}, dtype: {s1_batch.dtype}")
    print(f"  Source2: {s2_batch.shape}, dtype: {s2_batch.dtype}")
    print("\nData generator test passed!")
