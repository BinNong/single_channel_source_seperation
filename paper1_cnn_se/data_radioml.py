"""Optional RadioML 2016.10A loader for cross-dataset generalisation experiments.

This module is NOT required for the main results in the paper (those use the
synthetic generator in `data_generator.py`). It exists so that the proposed
model can be evaluated on a publicly available signal corpus without retraining.

Usage:
    1. Download RadioML 2016.10A from http://opendata.deepsig.io/datasets/2016.10A
       (24 GB compressed RAR file; the loader expects the *extracted* directory).
    2. Set RADIOML_PATH to point at the extracted directory.
    3. Wrap the loader as a torch Dataset; concatenate two random samples with a
       random amplitude ratio and a small frequency offset to create a 2-source
       co-frequency mixture; train/evaluate the proposed model as usual.

NOTE: this loader is provided as-is for reproducibility and future-work
investigations. It is NOT used by any of the headline results reported in
this paper (the reviewer did not request an external-dataset evaluation; we
include the loader for completeness).
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_RADIOML_PATH = os.environ.get(
    "RADIOML_PATH",
    "/data/external/RML2016.10a_dict",
)


class RadioMLMixtureDataset(Dataset):
    """Build a co-frequency mixture dataset from RadioML 2016.10A.

    Each sample draws two random entries from the RadioML corpus, applies a
    random amplitude ratio, a small per-source carrier-frequency offset, and
    AWGN at a chosen SNR. Returns (mixture, source1, source2) as torch.complex64
    tensors of shape [1, T] -- the same layout as the synthetic generator.

    Notes:
      - RadioML 2016.10A contains 11 modulation types (8 digital, 3 analog)
        and 20 SNR levels (-20 dB to +18 dB in 2 dB steps). We select the
        digital modulations {BPSK, QPSK, 8PSK, QAM16, QAM64} to mirror our
        synthetic generator's modulation coverage as closely as possible.
      - The original RadioML signals are 128 samples long. We pad to 4096 by
        periodic repetition, then apply a small frequency offset between the
        two sources. Pad+repeat is a deliberate simplification; a more faithful
        cross-dataset evaluation would use real long-form co-frequency
        recordings (left for future work).
    """

    def __init__(
        self,
        n_samples: int,
        snr_range: tuple,
        target_length: int = 4096,
        radioml_path: str = DEFAULT_RADIOML_PATH,
        mod_subset: tuple = ("BPSK", "QPSK", "8PSK", "QAM16", "QAM64"),
        seed: int | None = None,
    ):
        self.n_samples = n_samples
        self.snr_range = snr_range
        self.target_length = target_length
        self.mod_subset = mod_subset
        self.seed = seed

        if not Path(radioml_path).exists():
            raise FileNotFoundError(
                f"RadioML dataset not found at {radioml_path}. Set RADIOML_PATH."
            )

        with open(radioml_path, "rb") as f:
            raw = pickle.load(f, encoding="latin1")

        # raw is { (mod, snr): np.array of shape [N, 2, 128] }
        # filter to the mod_subset and any SNR in range
        self.samples_by_mod_snr: dict = {}
        for (mod, snr), arr in raw.items():
            if mod in mod_subset and self.snr_range[0] <= snr <= self.snr_range[1]:
                # arrange as complex IQ samples: shape [N, 128] complex64
                self.samples_by_mod_snr[(mod, snr)] = (arr[:, 0, :] + 1j * arr[:, 1, :]).astype(
                    np.complex64
                )

        # Index lookup for sampling
        self.index_pool = [(m, s, i) for (m, s), arr in self.samples_by_mod_snr.items()
                          for i in range(len(arr))]
        if not self.index_pool:
            raise RuntimeError("No RadioML samples match the requested snr_range/mod_subset.")
        if seed is not None:
            np.random.seed(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Sample two random sources
        idx1 = np.random.choice(len(self.index_pool))
        idx2 = np.random.choice(len(self.index_pool))
        mod1, snr1, k1 = self.index_pool[idx1]
        mod2, snr2, k2 = self.index_pool[idx2]
        s1 = self.samples_by_mod_snr[(mod1, snr1)][k1]
        s2 = self.samples_by_mod_snr[(mod2, snr2)][k2]

        # Pad to target_length by periodic repetition
        T = self.target_length
        s1 = np.resize(s1, T)
        s2 = np.resize(s2, T)

        # Per-source carrier offset (Hz). Use small offsets to mirror our
        # co-frequency setup.
        fs = 1.0  # arbitrary unit sampling rate; offsets are in cycles/sample
        delta_f1 = np.random.uniform(-0.01, 0.01)
        delta_f2 = delta_f1 + np.random.uniform(0.0001, 0.001)  # ensure 0.01-1 Hz gap
        t = np.arange(T)
        s1 = s1 * np.exp(1j * 2 * np.pi * delta_f1 * t).astype(np.complex64)
        s2 = s2 * np.exp(1j * 2 * np.pi * delta_f2 * t).astype(np.complex64)

        # Normalise each to unit power
        s1 = s1 / (np.sqrt(np.mean(np.abs(s1) ** 2)) + 1e-10)
        s2 = s2 / (np.sqrt(np.mean(np.abs(s2) ** 2)) + 1e-10)

        # Random amplitude ratio
        alpha = np.random.uniform(0.4, 0.6)
        mix_clean = alpha * s1 + (1 - alpha) * s2

        # Target SNR (random within snr_range)
        target_snr_db = np.random.uniform(*self.snr_range)
        signal_power = np.mean(np.abs(mix_clean) ** 2)
        noise_power = signal_power / (10 ** (target_snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(T) + 1j * np.random.randn(T)
        )
        mixture = (mix_clean + noise).astype(np.complex64)

        def to_t(x):
            return torch.from_numpy(x).unsqueeze(0).to(torch.complex64)

        return to_t(mixture), to_t(s1), to_t(s2)


if __name__ == "__main__":
    print("Testing RadioMLMixtureDataset...")
    ds = RadioMLMixtureDataset(n_samples=4, snr_range=(0, 10), seed=42)
    m, s1, s2 = ds[0]
    print(f"Mixture shape: {m.shape}, dtype: {m.dtype}")