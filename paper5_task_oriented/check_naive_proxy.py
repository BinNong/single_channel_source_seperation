"""Paper 5 — verify the Sec. III-B claim: a naive demodulation proxy
(nominal-carrier down-conversion, no timing or scale correction) scores
SER ~ 0.68 even when fed the TRUE source waveform on 16QAM.

Method: deterministic test set (seed 99999), take the clean per-slot
source waveforms, demodulate each 16QAM source three ways:
  oracle  : ser_comp.ref_labels_only path (oracle carrier, matched
            filter, 0::sps grid, unit-power) -- the clean-source floor;
  naive   : same pipeline but carrier fixed at carrier_base (2000 Hz)
            and NO unit-power normalisation (the Sec. III-B proxy);
  naive+N : naive but WITH unit-power normalisation (isolates the
            carrier-offset effect from the scale effect).
Truth labels always come from the oracle path.
Run:  python check_naive_proxy.py
"""
import numpy as np
import torch

import config as C
from data_generator_vark import CommBSSVarKTestDataset  # appends paper1 dir
from data_generator import rrc_filter  # resolves via paper1 path (above)
from ser_comp import MOD_TYPES, ref_labels_only
from utils import _get_constellation


def naive_demod(source, carrier_freq, normalize):
    x = np.asarray(source, dtype=np.complex64).squeeze()
    T = len(x)
    fs = C.SignalConfig.sample_rate
    n_symbols = C.SignalConfig.n_symbols
    sps = T // n_symbols
    t = np.arange(T) / fs
    bb = x * np.exp(-1j * 2 * np.pi * carrier_freq * t)
    rrc = rrc_filter(C.SignalConfig.num_taps, C.SignalConfig.roll_off, sps)
    mf = np.convolve(bb, rrc, mode='same')
    syms = mf[0::sps][:n_symbols]
    if normalize:
        syms = syms / (np.sqrt(np.mean(np.abs(syms) ** 2)) + 1e-10)
    const = _get_constellation('16QAM')
    return np.argmin(np.abs(syms[:, None] - const[None, :]), axis=1)


def main():
    ds = CommBSSVarKTestDataset(
        n_per_cell=100, snr_points=[20], mod_types=MOD_TYPES,
        k_lo=C.VarKConfig.k_min, k_hi=C.VarKConfig.k_max,
        k_extrap=C.VarKConfig.k_extrap, k_slots=C.VarKConfig.k_slots,
        signal_length=C.SignalConfig.signal_length,
        sample_rate=C.SignalConfig.sample_rate,
        carrier_base=C.SignalConfig.carrier_base,
        freq_gap_range=C.SignalConfig.freq_gap_range,
        seed=C.DataConfig.test_seed, return_carriers=True)

    mod16 = MOD_TYPES.index('16QAM')
    errs = {'oracle': [], 'naive': [], 'naive+N': []}
    n_src = 0
    for i in range(len(ds)):
        _, sources, occ, k, _snr, mods, carriers = ds[i]
        for j in range(int(k)):
            if int(mods[j]) != mod16:
                continue
            src = sources[j].numpy()
            fc = float(carriers[j])
            truth = ref_labels_only(
                sources[j], mod_type='16QAM',
                sample_rate=C.SignalConfig.sample_rate,
                n_symbols=C.SignalConfig.n_symbols,
                roll_off=C.SignalConfig.roll_off,
                num_taps=C.SignalConfig.num_taps, carrier_freq=fc)
            errs['oracle'].append(np.mean(
                naive_demod(src, fc, True) != truth))
            errs['naive'].append(np.mean(
                naive_demod(src, C.SignalConfig.carrier_base, False) != truth))
            errs['naive+N'].append(np.mean(
                naive_demod(src, C.SignalConfig.carrier_base, True) != truth))
            n_src += 1
    print(f"16QAM sources scored: {n_src} (test seed {C.DataConfig.test_seed})")
    for k, v in errs.items():
        print(f"  {k:8s}: SER = {np.mean(v):.4f} "
              f"(per-source min {np.min(v):.3f} / max {np.max(v):.3f})")


if __name__ == '__main__':
    main()
