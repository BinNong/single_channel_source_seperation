"""Paper 6 — vendored signal-generation helpers (self-contained).

Copied from paper1_cnn_se/data_generator.py (rrc_filter, generate_symbols,
generate_single_signal) and paper1_cnn_se/utils.py (CONSTELLATIONS,
_get_constellation) on 2026-09-19; the soft_demod.py constellation tables
(CONST_MAT / CONST_MASK, MOD_TYPES ordering) were rebuilt as numpy from the
same source lists.  Only the numpy parts are vendored — the paper-1 dataset
classes and torch-dependent utilities are not needed by paper 6.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# RRC Filter (copied from paper1_cnn_se/data_generator.py on 2026-09-19)
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
    """Copied from paper1_cnn_se/data_generator.py on 2026-09-19 (closed-set
    modulations only; MSK/GMSK dropped — paper 6 never uses them)."""
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
    else:
        raise ValueError(f"Unknown modulation type: {mod_type}")


# =============================================================================
# Single Signal Generation
# =============================================================================
def generate_single_signal(n_symbols, carrier_freq, sample_rate, signal_length,
                           mod_type, roll_off, num_taps, apply_fading=True,
                           fading_taps=3, return_freq=False, freq_jitter=None):
    """Copied from paper1_cnn_se/data_generator.py on 2026-09-19.

    return_freq=False (default) keeps the historical 2-tuple return.
    return_freq=True returns (signal, symbols, freq_offset) where
    freq_offset is the ACTUAL carrier (carrier_freq + jitter).  The RNG
    stream is identical either way (no draws added, removed, or reordered).

    freq_jitter (2026-09-22, robustness sweeps): when a float is given it
    is used as the in-generator jitter INSTEAD of drawing
    U(-5, 5) — the draw is skipped, so the RNG stream is unchanged for
    all pre-existing callers (None default) and controlled true carrier
    offsets become possible (eval_robustness_sweeps.py).
    """
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
    if freq_jitter is None:
        freq_offset = carrier_freq + np.random.uniform(-5, 5)  # Small random offset
    else:
        freq_offset = carrier_freq + float(freq_jitter)
    signal = shaped * np.exp(1j * 2 * np.pi * freq_offset * t)

    # 6. Apply multipath fading channel
    if apply_fading:
        fading = np.random.randn(fading_taps) + 1j * np.random.randn(fading_taps)
        fading = fading / np.linalg.norm(fading)
        signal = np.convolve(signal, fading, mode='same')
        signal = signal[:signal_length]

    # 7. Normalize power
    signal = signal / (np.sqrt(np.mean(np.abs(signal) ** 2)) + 1e-10)

    if return_freq:
        return signal, symbols, freq_offset
    return signal, symbols


# =============================================================================
# Constellation tables (copied from paper1_cnn_se/utils.py on 2026-09-19)
# =============================================================================
CONSTELLATIONS = {
    'BPSK': np.array([-1+0j, 1+0j]),
    'QPSK': np.array([-1-1j, -1+1j, 1-1j, 1+1j]) / np.sqrt(2),
    '8PSK': np.exp(1j * (2*np.pi*np.arange(8)/8 + np.pi/8)),
    '16QAM': (lambda: np.array([
        -3-3j, -3-1j, -3+1j, -3+3j,
        -1-3j, -1-1j, -1+1j, -1+3j,
         1-3j,  1-1j,  1+1j,  1+3j,
         3-3j,  3-1j,  3+1j,  3+3j
    ]) / np.sqrt(10))(),
}


def _get_constellation(mod_type):
    """Return normalized constellation points as [K] complex numpy array."""
    if mod_type in CONSTELLATIONS:
        return CONSTELLATIONS[mod_type].copy()
    # Fallback: treat as QPSK
    return CONSTELLATIONS['QPSK'].copy()


# =============================================================================
# Padded constellation tables (rebuilt as numpy from paper5_task_oriented/
# soft_demod.py's _CONST_LISTS on 2026-09-19 — the torch versions there are
# the same values as torch.from_numpy of these)
# =============================================================================
MOD_TYPES = ['BPSK', 'QPSK', '8PSK', '16QAM']
_M_MAX = 16

_CONST_LISTS = {
    'BPSK': [-1 + 0j, 1 + 0j],
    'QPSK': [(-1 - 1j) / np.sqrt(2), (-1 + 1j) / np.sqrt(2),
             (1 - 1j) / np.sqrt(2), (1 + 1j) / np.sqrt(2)],
    '8PSK': list(np.exp(1j * (2 * np.pi * np.arange(8) / 8 + np.pi / 8))),
    '16QAM': [(i + 1j * q) / np.sqrt(10)
              for i in (-3, -1, 1, 3) for q in (-3, -1, 1, 3)],
}


def _build_tables():
    mat = np.zeros((len(MOD_TYPES), _M_MAX), dtype=np.complex64)
    mask = np.zeros((len(MOD_TYPES), _M_MAX), dtype=bool)
    for m, name in enumerate(MOD_TYPES):
        pts = _CONST_LISTS[name]
        mat[m, :len(pts)] = pts
        mask[m, :len(pts)] = True
    return mat, mask


CONST_MAT, CONST_MASK = _build_tables()      # [4, 16] complex64 / bool (numpy)


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == '__main__':
    print("Testing paper6 signal_utils ...")
    # RRC sanity + unit-energy
    h = rrc_filter(64, 0.35, 16)
    assert len(h) == 65 and abs(np.sum(h ** 2) - 1.0) < 1e-6
    # signal generation sanity + determinism
    np.random.seed(0)
    s1, sy1, fc1 = generate_single_signal(256, 2003.0, 16000, 4096, 'QPSK',
                                          0.35, 64, True, 3,
                                          return_freq=True)
    np.random.seed(0)
    s2, sy2, fc2 = generate_single_signal(256, 2003.0, 16000, 4096, 'QPSK',
                                          0.35, 64, True, 3,
                                          return_freq=True)
    assert np.array_equal(np.asarray(s1), np.asarray(s2)) and fc1 == fc2
    assert abs(np.mean(np.abs(s1) ** 2) - 1.0) < 1e-6
    # constellation table parity: CONST_MAT rows == CONSTELLATIONS entries
    for m, name in enumerate(MOD_TYPES):
        n = int(CONST_MASK[m].sum())
        assert np.allclose(CONST_MAT[m, :n], CONSTELLATIONS[name],
                           atol=1e-6), name
    print("  RRC / determinism / constellation-table parity: OK")
    print("\nsignal_utils smoke test passed!")
