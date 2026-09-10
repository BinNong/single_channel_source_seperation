"""
Paper 4 — Open-World SC-BSS: Variable-K data generator.

This file EXTENDS paper1_cnn_se/data_generator.py without modifying it
(because paper2_dp_mamba/data_generator.py is a symlink to paper1's).

Problem (spec: docs/PAPER4_VARIABLE_K_PLAN.md §3-4):
  x = sum_{k=1..K} w_k s_k + n,   K in {1,2,3} unknown at test time,
  w_k ~ U(0.4, 0.6) i.i.d. (NOT normalized), n = AWGN.
  Per-source carrier = carrier_base + U(freq_gap_range) (paper1's
  generate_single_signal adds its own ±5 Hz jitter internally — kept).
  SNR is defined on the TOTAL mixture power: noise power scales with K,
  so the mixture-level operating point is comparable across K, while the
  per-source effective SNR is SNR - 10*log10(K) (4.77 dB lower at K=3).

Public API:
  - MOD_TYPES / MOD_TO_IDX     : ['BPSK', 'QPSK', '8PSK', '16QAM']
  - generate_vark_mixture(...) : mix k sources -> (mixture, sources, mod_idx)
  - CommBSSVarKDataset         : training-style dataset, per-sample
                                 k ~ Uniform{k_min..k_max}
  - CommBSSVarKTestDataset     : deterministic test grid
                                 SNR x K{1,2,3} + K=4 extrapolation cell
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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
# Use sys.path.append (not insert) so paper4_open_world/ stays ahead of
# paper1_cnn_se/ in sys.path — otherwise train.py's later `from models
# import SlotSepNet` would resolve to paper1's models.py.
_PAPER1_DIR = Path(__file__).resolve().parent.parent / "paper1_cnn_se"
if str(_PAPER1_DIR) not in sys.path:
    sys.path.append(str(_PAPER1_DIR))

from data_generator import generate_single_signal  # noqa: E402


# ============================================================================
# Modulation vocabulary
# ============================================================================
MOD_TYPES  = ['BPSK', 'QPSK', '8PSK', '16QAM']
MOD_TO_IDX = {m: i for i, m in enumerate(MOD_TYPES)}
IDX_TO_MOD = {i: m for m, i in MOD_TO_IDX.items()}


# ============================================================================
# Mixture generation
# ============================================================================
def generate_vark_mixture(signal_length, sample_rate, snr_db, mod_types,
                          k, carrier_base=2000.0, freq_gap_range=(0.0, 5.0),
                          n_symbols=256, roll_off=0.35, num_taps=64,
                          apply_fading=True, fading_taps=3,
                          mods=None, carriers=None, weights=None,
                          return_carriers=False):
    """Mix k co-frequency sources; returns (mixture, sources, mod_idx).

    With return_carriers=True returns a 4-tuple whose last element is the
    list of TRUE per-source carriers (carrier + in-generator jitter) —
    used by the offset-compensated SER diagnostic (ser_comp.py).  The
    RNG stream is identical either way.

    Per source k:
      - modulation sampled from `mod_types`
      - carrier = carrier_base + U(freq_gap_range)  [Hz]
        (paper1's generate_single_signal adds its own ±5 Hz jitter on top)
      - mixing weight w_k ~ U(0.4, 0.6) i.i.d., NOT normalized (design
        decision: mixture power grows with K; SNR is defined on the total
        mixture power, so noise power scales with K and the per-source
        effective SNR is SNR - 10*log10(K))
      - sources are unit-power before weighting (paper1 pipeline), so w_k
        is the relative amplitude.

    Args:
        mods / carriers / weights: optional pre-drawn per-source values
            (used by the dataset classes to make per-sample configs
            reproducible).  When None they are drawn internally.

    Returns
    -------
    mixture  : complex ndarray [signal_length]
    sources  : list of k complex ndarrays [signal_length] (UNWEIGHTED,
               unit power — the ground-truth separation targets)
    mod_idx  : list of k ints (index into MOD_TYPES)
    """
    mod_types = list(mod_types)
    if mods is None:
        mods = [str(np.random.choice(mod_types)) for _ in range(k)]
    if carriers is None:
        carriers = carrier_base + np.random.uniform(
            freq_gap_range[0], freq_gap_range[1], size=k)
    if weights is None:
        weights = np.random.uniform(0.4, 0.6, size=k)

    sources = []
    mod_idx = []
    true_carriers = []
    for j in range(k):
        sig, _, fc = generate_single_signal(
            n_symbols, float(carriers[j]), sample_rate, signal_length,
            mods[j], roll_off, num_taps, apply_fading, fading_taps,
            return_freq=True,   # same RNG stream; exposes carrier+jitter
        )
        sources.append(sig.astype(np.complex64))
        mod_idx.append(MOD_TO_IDX[mods[j]])
        true_carriers.append(float(fc))

    mix_clean = np.zeros(signal_length, dtype=np.complex64)
    for w, s in zip(weights, sources):
        mix_clean = mix_clean + w * s

    # AWGN: SNR defined on the TOTAL clean-mixture power.
    sig_power = np.mean(np.abs(mix_clean) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(signal_length) + 1j * np.random.randn(signal_length)
    )
    mixture = (mix_clean + noise).astype(np.complex64)

    if return_carriers:
        return mixture, sources, mod_idx, true_carriers
    return mixture, sources, mod_idx


# ============================================================================
# Datasets
# ============================================================================
class CommBSSVarKDataset(Dataset):
    """Training-style dataset; per-sample K ~ Uniform{k_min..k_max}.

    Per-sample configs (snr, k, mods, carriers) are pre-generated under a
    fixed seed for reproducibility; mixing weights and noise draw from the
    global RNG at __getitem__ time (same convention as paper1/paper3).

    __getitem__ returns
    -------------------
    mixture   : torch.complex64 [1, T]
    sources   : torch.complex64 [K_max, 1, T] zero-padded (first k are real)
    occ_mask  : torch.float32   [K_max]  (1.0 for the first k slots)
    k         : torch.int64     scalar

    With return_carriers=True (paper 5, plan §5 改动 1) two more fields are
    appended — same format as CommBSSVarKTestDataset:
    mods      : torch.int64     [K_max]  (index into MOD_TYPES, -1 padded)
    carriers  : torch.float64   [K_max]  (TRUE carriers incl. the
                in-generator ±5 Hz jitter, NaN padded)

    The RNG stream is IDENTICAL either way: generate_vark_mixture always
    calls generate_single_signal with return_freq=True internally, so
    capturing the true carriers consumes no extra random draws.  NOTE: the
    carriers stashed in self.configs are the PRE-jitter values; the true
    (post-jitter) carriers can only be captured at generation time, which
    is what the flag does.
    """

    def __init__(self,
                 n_samples: int,
                 snr_range: tuple[float, float],
                 mod_types: list[str],
                 k_min: int = 1,
                 k_max: int = 3,
                 k_slots: int = 4,
                 signal_length: int = 4096,
                 sample_rate: int = 16000,
                 carrier_base: float = 2000.0,
                 freq_gap_range: tuple[float, float] = (0.0, 5.0),
                 n_symbols: int = 256,
                 roll_off: float = 0.35,
                 num_taps: int = 64,
                 apply_fading: bool = True,
                 fading_taps: int = 3,
                 seed: int | None = None,
                 return_carriers: bool = False) -> None:
        assert 1 <= k_min <= k_max <= k_slots
        self.n_samples = n_samples
        self.snr_range = snr_range
        self.mod_types = list(mod_types)
        self.k_min = k_min
        self.k_max = k_max
        self.k_slots = k_slots
        self.signal_length = signal_length
        self.sample_rate = sample_rate
        self.carrier_base = carrier_base
        self.freq_gap_range = freq_gap_range
        self.n_symbols = n_symbols
        self.roll_off = roll_off
        self.num_taps = num_taps
        self.apply_fading = apply_fading
        self.fading_taps = fading_taps
        self.seed = seed
        self.return_carriers = return_carriers

        # Pre-generate configs for reproducibility (stash/restore the global
        # RNG so constructing a dataset does not perturb the training stream).
        _stashed = np.random.get_state()
        try:
            if seed is not None:
                np.random.seed(seed)
            self.configs = []
            for _ in range(n_samples):
                snr = float(np.random.uniform(snr_range[0], snr_range[1]))
                k = int(np.random.randint(k_min, k_max + 1))
                mods = [str(np.random.choice(self.mod_types)) for _ in range(k)]
                carriers = carrier_base + np.random.uniform(
                    freq_gap_range[0], freq_gap_range[1], size=k)
                self.configs.append((snr, k, mods, carriers))
        finally:
            np.random.set_state(_stashed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        snr, k, mods, carriers = self.configs[idx]
        mix, srcs, _mod_idx, true_carriers = generate_vark_mixture(
            self.signal_length, self.sample_rate, snr, self.mod_types, k,
            carrier_base=self.carrier_base, freq_gap_range=self.freq_gap_range,
            n_symbols=self.n_symbols, roll_off=self.roll_off,
            num_taps=self.num_taps, apply_fading=self.apply_fading,
            fading_taps=self.fading_taps, mods=mods, carriers=carriers,
            return_carriers=True,   # same RNG stream; exposes carrier+jitter
        )

        sources = np.zeros((self.k_slots, self.signal_length), dtype=np.complex64)
        for j, s in enumerate(srcs):
            sources[j] = s
        occ_mask = np.zeros(self.k_slots, dtype=np.float32)
        occ_mask[:k] = 1.0

        if self.return_carriers:
            mods_pad = np.full(self.k_slots, -1, dtype=np.int64)
            mods_pad[:k] = [MOD_TO_IDX[m] for m in mods]
            carriers_pad = np.full(self.k_slots, np.nan)
            carriers_pad[:k] = true_carriers
            return (
                torch.from_numpy(mix).unsqueeze(0).to(torch.complex64),
                torch.from_numpy(sources).unsqueeze(1).to(torch.complex64),
                torch.from_numpy(occ_mask),
                torch.tensor(k, dtype=torch.long),
                torch.from_numpy(mods_pad),
                torch.from_numpy(carriers_pad),
            )
        return (
            torch.from_numpy(mix).unsqueeze(0).to(torch.complex64),
            torch.from_numpy(sources).unsqueeze(1).to(torch.complex64),
            torch.from_numpy(occ_mask),
            torch.tensor(k, dtype=torch.long),
        )


class CommBSSVarKTestDataset(Dataset):
    """Deterministic test dataset: SNR grid x K grid + K-extrapolation cell.

    Grid: snr_points x K in {k_lo..k_hi} (default 1..3), n_per_cell samples
    each, PLUS a zero-shot extrapolation cell at K = k_extrap (default 4),
    n_per_cell samples per SNR.  Modulation combinations are sampled (not
    enumerated) under the fixed seed.

    Deterministic seed=99999 (paper3 convention): all samples are generated
    eagerly in __init__ under a stashed/restored global seed, so the set is
    identical across runs, processes and loader orders.

    __getitem__ returns
    -------------------
    mixture   : torch.complex64 [1, T]
    sources   : torch.complex64 [K_max, 1, T] zero-padded
    occ_mask  : torch.float32   [K_max]
    k         : torch.int64     scalar
    snr       : float scalar
    mods      : torch.int64     [K_max] padded with -1
    """

    def __init__(self,
                 n_per_cell: int = 200,
                 snr_points: list[int] | None = None,
                 mod_types: list[str] | None = None,
                 k_lo: int = 1,
                 k_hi: int = 3,
                 k_extrap: int = 4,
                 k_slots: int = 4,
                 signal_length: int = 4096,
                 sample_rate: int = 16000,
                 carrier_base: float = 2000.0,
                 freq_gap_range: tuple[float, float] = (0.0, 5.0),
                 n_symbols: int = 256,
                 roll_off: float = 0.35,
                 num_taps: int = 64,
                 apply_fading: bool = True,
                 fading_taps: int = 3,
                 seed: int = 99999,
                 return_carriers: bool = False) -> None:
        if snr_points is None:
            snr_points = [-10, -5, 0, 5, 10, 15, 20]
        if mod_types is None:
            mod_types = MOD_TYPES
        # K_extrap must be representable in the sources tensor.
        self.k_slots = max(k_slots, k_extrap)
        self.return_carriers = return_carriers
        self.samples = []

        _stashed = np.random.get_state()
        try:
            np.random.seed(seed)
            k_values = list(range(k_lo, k_hi + 1))
            if k_extrap is not None and k_extrap not in k_values:
                k_values.append(k_extrap)

            for snr in snr_points:
                for k in k_values:
                    for _ in range(n_per_cell):
                        mix, srcs, mod_idx, carriers = generate_vark_mixture(
                            signal_length, sample_rate, float(snr),
                            mod_types, k,
                            carrier_base=carrier_base,
                            freq_gap_range=freq_gap_range,
                            n_symbols=n_symbols, roll_off=roll_off,
                            num_taps=num_taps, apply_fading=apply_fading,
                            fading_taps=fading_taps,
                            return_carriers=True,
                        )
                        sources = np.zeros((self.k_slots, signal_length),
                                           dtype=np.complex64)
                        for j, s in enumerate(srcs):
                            sources[j] = s
                        occ_mask = np.zeros(self.k_slots, dtype=np.float32)
                        occ_mask[:k] = 1.0
                        mods = np.full(self.k_slots, -1, dtype=np.int64)
                        mods[:k] = mod_idx
                        carriers_pad = np.full(self.k_slots, np.nan)
                        carriers_pad[:k] = carriers
                        self.samples.append({
                            'mixture': torch.from_numpy(mix).unsqueeze(0).to(torch.complex64),
                            'sources': torch.from_numpy(sources).unsqueeze(1).to(torch.complex64),
                            'occ_mask': torch.from_numpy(occ_mask),
                            'k': int(k),
                            'snr': float(snr),
                            'mods': torch.from_numpy(mods),
                            'carriers': carriers_pad,
                        })
        finally:
            np.random.set_state(_stashed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        if self.return_carriers:
            return (
                s['mixture'], s['sources'], s['occ_mask'],
                torch.tensor(s['k'], dtype=torch.long),
                float(s['snr']), s['mods'],
                torch.from_numpy(s['carriers']),
            )
        return (
            s['mixture'], s['sources'], s['occ_mask'],
            torch.tensor(s['k'], dtype=torch.long),
            float(s['snr']), s['mods'],
        )


# ============================================================================
# Quick smoke test
# ============================================================================
if __name__ == '__main__':
    print("Testing paper4 data_generator_vark ...")

    # 1. Numpy-level mixture generation for each K
    for k in (1, 2, 3, 4):
        mix, srcs, midx = generate_vark_mixture(
            signal_length=4096, sample_rate=16000, snr_db=10,
            mod_types=MOD_TYPES, k=k,
        )
        p_src = [float(np.mean(np.abs(s) ** 2)) for s in srcs]
        print(f"  K={k}: mix={mix.shape} dtype={mix.dtype} "
              f"n_src={len(srcs)} unit_power={[f'{p:.3f}' for p in p_src]} "
              f"mods={[IDX_TO_MOD[m] for m in midx]}")
        assert mix.shape == (4096,) and len(srcs) == k
        assert all(abs(p - 1.0) < 0.05 for p in p_src)

    # SNR sanity: at 20 dB the residual noise power should be ~1% of the mix
    mix, srcs, _ = generate_vark_mixture(4096, 16000, 20.0, MOD_TYPES, k=2)
    print(f"  K=2 mix power={np.mean(np.abs(mix)**2):.3f} "
          f"(clean ~ sum w^2 in [0.32, 0.72] + 1% noise)")

    # 2. Dataset-level smoke (requires torch)
    if not _TORCH_AVAILABLE:
        print("\n[torch not installed locally — skipping dataset tests]")
    else:
        from torch.utils.data import DataLoader

        train_ds = CommBSSVarKDataset(
            n_samples=8, snr_range=(0, 20), mod_types=MOD_TYPES,
            k_min=1, k_max=3, k_slots=4, seed=42,
        )
        k_counts = [int(train_ds.configs[i][1]) for i in range(8)]
        print(f"\nTrain configs K values: {k_counts}")
        loader = DataLoader(train_ds, batch_size=4)
        mix_b, src_b, occ_b, k_b = next(iter(loader))
        print(f"Train batch shapes: mix={tuple(mix_b.shape)} "
              f"src={tuple(src_b.shape)} occ={tuple(occ_b.shape)} "
              f"k={tuple(k_b.shape)}")
        print(f"  dtypes: mix={mix_b.dtype} src={src_b.dtype} "
              f"occ={occ_b.dtype} k={k_b.dtype}")
        assert mix_b.shape == (4, 1, 4096) and src_b.shape == (4, 4, 1, 4096)
        assert occ_b.shape == (4, 4) and k_b.shape == (4,)
        assert mix_b.is_complex() and src_b.is_complex()
        for i in range(4):
            assert int(occ_b[i].sum().item()) == int(k_b[i].item())
            # zero-padded slots must be exactly zero
            kk = int(k_b[i].item())
            if kk < 4:
                assert src_b[i, kk:].abs().sum().item() == 0.0

        # Determinism of configs
        train_ds2 = CommBSSVarKDataset(
            n_samples=8, snr_range=(0, 20), mod_types=MOD_TYPES,
            k_min=1, k_max=3, k_slots=4, seed=42,
        )
        same_cfg = all(
            c1[0] == c2[0] and c1[1] == c2[1] and c1[2] == c2[2]
            and np.array_equal(c1[3], c2[3])
            for c1, c2 in zip(train_ds.configs, train_ds2.configs)
        )
        assert same_cfg
        print("  config determinism (seed=42 twice): OK")

        # ---- paper 5: return_carriers option (plan §5 改动 1) ----
        train_rc = CommBSSVarKDataset(
            n_samples=8, snr_range=(0, 20), mod_types=MOD_TYPES,
            k_min=1, k_max=3, k_slots=4, seed=42, return_carriers=True,
        )
        mix1, src1, occ1, k1, mods1, car1 = train_rc[0]
        print(f"\nTrain return_carriers [0]: k={int(k1)} mods={mods1.tolist()} "
              f"carriers={np.round(car1.numpy(), 3).tolist()}")
        assert mods1.shape == (4,) and mods1.dtype == torch.int64
        assert car1.shape == (4,)
        kk = int(k1)
        assert (mods1[:kk] >= 0).all() and (mods1[kk:] == -1).all()
        assert np.isfinite(car1[:kk].numpy()).all()
        assert np.isnan(car1[kk:].numpy()).all()
        # True carriers = config carrier + in-generator ±5 Hz jitter
        cfg_car = train_rc.configs[0][3]
        assert np.all(np.abs(car1[:kk].numpy() - cfg_car) <= 5.0 + 1e-6)

        # RNG-stream conservation: flag on/off must yield bit-identical
        # mixture/sources for the same global RNG state at __getitem__ time.
        np.random.seed(123)
        mix_off, src_off, _, _ = train_ds[0]
        np.random.seed(123)
        mix_on, src_on, _, _, _, _ = train_rc[0]
        assert np.array_equal(mix_off.numpy(), mix_on.numpy())
        assert np.array_equal(src_off.numpy(), src_on.numpy())
        print("  RNG-stream conservation (return_carriers on/off): OK")

        test_ds = CommBSSVarKTestDataset(n_per_cell=2, snr_points=[0, 10],
                                          seed=99999)
        print(f"\nTest set: {len(test_ds)} samples "
              f"(2 SNR x K{{1,2,3,4}} x 2 per cell)")
        m, s, occ, k, snr, mods = test_ds[0]
        print(f"  [0] mix={tuple(m.shape)} src={tuple(s.shape)} "
              f"occ={occ.tolist()} k={int(k)} snr={snr} mods={mods.tolist()}")
        ks = sorted({test_ds.samples[i]['k'] for i in range(len(test_ds))})
        print(f"  K values present: {ks}")
        assert ks == [1, 2, 3, 4]
        # K=4 cell: all 4 slots occupied
        for i in range(len(test_ds)):
            if test_ds.samples[i]['k'] == 4:
                assert test_ds.samples[i]['occ_mask'].sum().item() == 4.0
                assert test_ds.samples[i]['sources'].shape[0] == 4
                break
        # Test-set determinism
        test_ds2 = CommBSSVarKTestDataset(n_per_cell=2, snr_points=[0, 10],
                                           seed=99999)
        same = np.array_equal(test_ds.samples[0]['mixture'].numpy(),
                              test_ds2.samples[0]['mixture'].numpy())
        print(f"  test-set determinism (seed=99999 twice): {'OK' if same else 'FAIL'}")
        assert same

    print("\ndata_generator_vark smoke test passed!")
