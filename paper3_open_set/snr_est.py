"""
Paper 3 — Open-Set SC-BSS: Blind SNR estimation from the received mixture.

Implements the estimator required by the R1-4 reviewer comment: the SNR-routed
OOD ensemble (ensemble_analysis.py) must be driven by a *real* blind SNR
estimate obtained from the received complex mixture waveform y[t] (T = 4096,
complex baseband/passband, fs = 16 kHz), not by the ground-truth SNR and not by
a synthetic Gaussian perturbation of it.

Two estimators are provided:

1. M2M4 (primary) — classic second-/fourth-moment estimator
   (Matzner & Englberger 1994; Pauluzzi & Beaulieu, IEEE Trans. Commun. 2000).
   For y = s + n with s the (two-source) signal part and n circular complex
   AWGN,

       M2 = E|y|^2 = S + N
       M4 = E|y|^4 = kappa * S^2 + 4 S N + 2 N^2

   where kappa = E|s|^4 / (E|s|^2)^2 is the signal kurtosis at the observation
   rate.  Solving the quadratic in S gives

       S_hat = sqrt( (M4 - 2 M2^2) / (kappa - 2) ),   N_hat = M2 - S_hat.

   Choice of kappa (calibrated, documented): the "signal" here is the *sum of
   two* RRC-shaped, 3-tap-faded co-frequency sources with a random mixing
   weight alpha in [0.4, 0.6].  Fading plus two-source mixing drives the
   mixture kurtosis well above the textbook constellation values (1.0-1.4):
   measured on noise-free mixtures from data_generator_extended over all
   64 modulation pairs (8 reps each, seed 0), kappa ranges 1.58-2.01 with mean
   1.71 (OFDM-containing pairs approach the Gaussian limit 2.0).  We use
   kappa = 1.5, *below the observed minimum*, so the discriminant
   (M4 - 2 M2^2) stays positive and the estimator never diverges.  The price
   is high-SNR compression (a 20 dB mixture reads ~+5 dB); the estimator is
   biased but strictly monotone, which is all the 0 dB routing threshold needs.

   No decimation to symbol rate: the moment relations above are sampling-rate
   agnostic (they only need the kurtosis at the observation rate), and blind
   symbol-timing/carrier recovery for two overlapping sources is out of scope.
   The moments are computed on |y| directly, so the 2 kHz carrier offsets are
   irrelevant.

2. Subspace / eigenvalue (comparison) — sample the L=64 biased autocorrelation
   of y, form the Hermitian Toeplitz matrix R, eigendecompose.  White noise
   contributes N to every eigenvalue; the narrowband signal (RRC bandwidth
   ~1.35 kHz around ~2 kHz carriers, i.e. ~8 % of the 16 kHz band) occupies
   only the few largest eigenvalues, so

       N_hat = median(eig(R)),   S_hat = M2 - N_hat.

   Assumptions: white noise; signal occupies < 50 % of the band.  No kurtosis
   calibration needed.  Included as a comparison point (the classic
   eigenvalue/subspace family, cf. Wax & Kailath 1985).

Both estimators clip the output to [-40, +40] dB.

Public API
----------
    estimate_snr_db(waveform, method='m2m4') -> float   # primary entry point
    estimate_snr_m2m4_db(waveform, kappa=1.5) -> float
    estimate_snr_subspace_db(waveform, n_lags=64) -> float

Pure NumPy; no torch dependency.
"""

from __future__ import annotations

import numpy as np

# Calibrated signal kurtosis for the two-source clean mixture (see module
# docstring): empirical range over all modulation pairs is 1.58-2.01; 1.5 is
# deliberately conservative so M2M4 stays finite for every pair.
KAPPA_DEFAULT = 1.5

# Autocorrelation order for the subspace estimator.  Must exceed the signal
# subspace dimension (~6-10 here) while staying well below T.
N_LAGS_DEFAULT = 64

# Output clipping range (dB).  Only the neighbourhood of the 0 dB routing
# threshold matters; the extremes are numerically irrelevant to the router.
SNR_CLIP_DB = (-40.0, 40.0)


def _as_complex_array(waveform) -> np.ndarray:
    """Accept numpy arrays or torch tensors, any shape; return 1-D complex128."""
    if hasattr(waveform, 'detach'):        # torch tensor
        waveform = waveform.detach().cpu().numpy()
    elif hasattr(waveform, 'numpy') and not isinstance(waveform, np.ndarray):
        waveform = waveform.numpy()
    y = np.asarray(waveform).ravel()
    if not np.iscomplexobj(y):
        y = y.astype(np.float64)
    return y.astype(np.complex128, copy=False)


def _clip(snr_db: float) -> float:
    return float(np.clip(snr_db, SNR_CLIP_DB[0], SNR_CLIP_DB[1]))


def estimate_snr_m2m4_db(waveform, kappa: float = KAPPA_DEFAULT) -> float:
    """M2M4 moment-based blind SNR estimate (dB).

    Parameters
    ----------
    waveform : complex array-like, shape [T]
        Received mixture (any SNR); only |y| moments are used.
    kappa : float
        Assumed signal kurtosis E|s|^4 / (E|s|^2)^2 at the observation rate.
        Must be < 2 (circular Gaussian limit); see module docstring for the
        calibrated default.
    """
    y = _as_complex_array(waveform)
    a2 = np.abs(y) ** 2
    m2 = float(np.mean(a2))
    m4 = float(np.mean(a2 ** 2))
    # S^2 = (M4 - 2 M2^2) / (kappa - 2); both numerator and denominator are
    # negative in the operating regime, positive fluctuations are clipped.
    s2 = (m4 - 2.0 * m2 * m2) / (kappa - 2.0)
    s2 = max(s2, 1e-12)
    s_hat = np.sqrt(s2)
    n_hat = max(m2 - s_hat, 1e-12)
    return _clip(10.0 * np.log10(s_hat / n_hat))


def estimate_snr_subspace_db(waveform, n_lags: int = N_LAGS_DEFAULT) -> float:
    """Eigenvalue/subspace blind SNR estimate (dB).

    N_hat = median eigenvalue of the n_lags x n_lags Toeplitz autocorrelation
    matrix (valid while the signal occupies fewer than n_lags/2 eigenvalues);
    S_hat = total power - N_hat.
    """
    y = _as_complex_array(waveform)
    n = y.shape[0]
    if n_lags >= n:
        raise ValueError(f"n_lags={n_lags} must be < T={n}")
    ac = np.array([np.vdot(y[: n - l], y[l:]) / n for l in range(n_lags)])
    # Hermitian Toeplitz with first column ac (R[i, j] = r[i - j]).
    from scipy.linalg import toeplitz
    r_mat = toeplitz(ac)
    eig = np.linalg.eigvalsh(r_mat)
    n_hat = max(float(np.median(eig)), 1e-12)
    s_hat = max(float(np.mean(np.abs(y) ** 2)) - n_hat, 1e-12)
    return _clip(10.0 * np.log10(s_hat / n_hat))


def estimate_snr_db(waveform, method: str = 'm2m4', **kwargs) -> float:
    """Primary API: blind SNR estimate (dB) of a received mixture waveform.

    method : 'm2m4' (default; classic fourth-moment estimator) or 'subspace'
    (eigenvalue method, comparison baseline).
    """
    if method == 'm2m4':
        return estimate_snr_m2m4_db(waveform, **kwargs)
    if method == 'subspace':
        return estimate_snr_subspace_db(waveform, **kwargs)
    raise ValueError(f"unknown method {method!r}; use 'm2m4' or 'subspace'")


# ============================================================================
# Smoke test
# ============================================================================
if __name__ == '__main__':
    print("snr_est smoke test")
    try:
        from data_generator_extended import generate_open_set_mixture

        print(f"  {'true':>5s} | {'m2m4':>8s} | {'subspace':>8s}")
        for snr in (-10, -5, 0, 5, 10, 20):
            mix, _, _, _, _ = generate_open_set_mixture(
                4096, 16000, snr, 'QPSK', '64QAM')
            e1 = estimate_snr_db(mix, method='m2m4')
            e2 = estimate_snr_db(mix, method='subspace')
            print(f"  {snr:+5d} | {e1:+8.2f} | {e2:+8.2f}")
    except ImportError:
        # Pure-synthetic fallback: unit-power QPSK-like tone in AWGN.
        rng = np.random.RandomState(0)
        t = np.arange(4096) / 16000.0
        sym = rng.choice([1, -1, 1j, -1j], 256)
        s = np.repeat(sym, 16) * np.exp(1j * 2 * np.pi * 2000 * t)
        s = s / np.sqrt(np.mean(np.abs(s) ** 2))
        for snr in (-5, 0, 10):
            npow = 10 ** (-snr / 10)
            noise = np.sqrt(npow / 2) * (
                rng.randn(4096) + 1j * rng.randn(4096))
            y = s + noise
            print(f"  true={snr:+5d}  m2m4={estimate_snr_db(y):+8.2f}  "
                  f"sub={estimate_snr_db(y, method='subspace'):+8.2f}")
    print("snr_est smoke test passed!")
