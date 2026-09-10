"""Paper 5 — offset-compensated SER + Gray-mapped BER (evaluation truth).

The proxy demodulator used for the paper's SER table
(paper1_cnn_se/utils.py:compute_ser_from_signal) down-converts at the
NOMINAL carrier (carrier_base) and performs no carrier or timing
recovery, while each source actually sits at carrier_base + offset with
offset in (-5, 10) Hz.  This module implements the SAME pipeline with
one change: both the estimate and the reference are down-converted at
the source's TRUE carrier (exposed by the generator, i.e. oracle
carrier synchronisation).  Timing is fixed on the (zero-delay) symbol
grid with no per-pair timing search, so what remains in the metric is
waveform fidelity alone.

Paper 5 addition: BER.  The generator has no bit stream (symbols are
drawn directly as constellation points), so BER is DEFINED post-hoc via
a Gray labelling of the constellation indices (GRAY_BITS below): a
symbol's bit string is its Gray label, and BER is the fraction of
mismatched bit positions between the estimate's and the reference's
decisions.  SER and BER come from the SAME decisions (one pipeline).

__main__ computes the compensated MIXTURE BASELINE (no separation) on
the deterministic test grid (same seed, K<=3 cells, per-modulation +
overall SER and BER), mirroring the baseline of the raw-SER run.

Debug history (2026-08-31): two successor designs were tried before
this one.  (1) Fitting/removing the est-vs-ref phase ramp is wrong for
this AGREEMENT-based metric: the reference labels are generated from
the same spinning constellation, so de-spinning the estimate only
decorrelates it from the reference decisions.  (2) While validating
(1), the RAW proxy itself turned out to be defective: identity
(est == reference) scored SER 0.68 on 16QAM.  Root causes: symbol
timing at sps//2 — half a symbol off the eye opening (every
convolution in the pipeline uses mode='same', so the net symbol
delay is ZERO) — and the estimate being demodulated at the LS-fit
scale (~3.7x the unit
grid) while the reference was demodulated at unit power.  The corrected
receiver below fixes carrier, timing, and grid scale together; identity
now scores 0.000 for all four modulations and clean-source decisions
match the true transmitted symbols.

Usage:
    python ser_comp.py [--n_per_cell 50]
Output:
    results/ser/baseline_ser_comp.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import config as C
from data_generator_vark import CommBSSVarKTestDataset, MOD_TYPES, IDX_TO_MOD


# ---------------------------------------------------------------------------
# Gray labelling of the constellation indices (paper 5, plan v2 O5).
# Index order MUST match utils.CONSTELLATIONS / soft_demod.CONST_MAT.
#   BPSK : [-1, 1]                       -> 0 / 1
#   QPSK : [-1-1j, -1+1j, 1-1j, 1+1j]    -> (sign(I), sign(Q)) bits
#   8PSK : angle pi/8 + m*pi/4, m=0..7   -> binary-reflected Gray on m
#   16QAM: idx = i*4+q, levels -3,-1,1,3 -> Gray(-3,-1,1,3)=00,01,11,10
#          per axis, bits = gray(I) ++ gray(Q)
# Every nearest-neighbour pair differs in exactly one bit (checked in
# soft_demod.py's __main__).
# ---------------------------------------------------------------------------
def _gray_code(n_bits: int) -> list[int]:
    """Binary-reflected Gray code sequence of 2**n_bits codewords."""
    g = [0, 1]
    for _ in range(n_bits - 1):
        g = g + [x | (1 << (len(bin(max(g))) - 2)) for x in reversed(g)]
    return g


def _bits_of(code: int, n_bits: int) -> list[int]:
    return [(code >> b) & 1 for b in range(n_bits - 1, -1, -1)]


def _build_gray_bits() -> dict[str, np.ndarray]:
    g1 = _gray_code(1)                       # [0, 1]
    g2 = _gray_code(2)                       # [0, 1, 3, 2]
    g3 = _gray_code(3)                       # [0,1,3,2,6,7,5,4]
    qpsk = [(_bits_of(i, 1) + _bits_of(q, 1)) for i in range(2) for q in range(2)]
    # QPSK index order in CONSTELLATIONS is [-1-1j, -1+1j, 1-1j, 1+1j]:
    # i = I sign (0:-1, 1:+1), q = Q sign -> [(-1,-1),(-1,+1),(+1,-1),(+1,+1)]
    qam16 = [(_bits_of(g2[i], 2) + _bits_of(g2[q], 2))
             for i in range(4) for q in range(4)]
    return {
        'BPSK': np.array([_bits_of(v, 1) for v in g1], dtype=np.uint8),
        'QPSK': np.array(qpsk, dtype=np.uint8),
        '8PSK': np.array([_bits_of(v, 3) for v in g3], dtype=np.uint8),
        '16QAM': np.array(qam16, dtype=np.uint8),
    }


GRAY_BITS = _build_gray_bits()


def _demod_pair_labels(estimated_signal, source_signal,
                       mod_type='QPSK', sample_rate=16000,
                       n_symbols=256, roll_off=0.35, num_taps=64,
                       carrier_freq=2000.0):
    """Shared receiver front-end: demodulate BOTH signals to hard labels.

    Returns (est_labels, ref_labels): int numpy arrays [n_use].  This is
    exactly the pipeline of the former compute_ser_compensated up to the
    min-distance decisions, factored out so SER and BER share one front-end.
    """
    from utils import _get_constellation
    # resolves to paper1's data_generator via the path appended by
    # data_generator_vark (same trick as utils.compute_ser_from_signal)
    from data_generator import rrc_filter

    est = estimated_signal.detach().cpu().numpy().squeeze()
    ref = source_signal.detach().cpu().numpy().squeeze()

    T = len(est)
    sps = T // n_symbols

    # 1. Down-convert at the TRUE carrier (oracle carrier sync)
    t = np.arange(T) / sample_rate
    est_bb = est * np.exp(-1j * 2 * np.pi * carrier_freq * t)
    ref_bb = ref * np.exp(-1j * 2 * np.pi * carrier_freq * t)

    # 2. Matched filter (RRC)
    rrc = rrc_filter(num_taps, roll_off, sps)
    est_mf = np.convolve(est_bb, rrc, mode='same')
    ref_mf = np.convolve(ref_bb, rrc, mode='same')

    # 3. Sample at the symbol peaks: all convolutions in the pipeline
    #    (generation RRC, channel, demod RRC) use mode='same' and are
    #    therefore centred, so the net symbol delay is zero.
    sym0 = 0
    n_use = min(n_symbols, T // sps)
    est_syms = est_mf[sym0::sps][:n_use]
    ref_syms = ref_mf[sym0::sps][:n_use]

    # 4. Both on the SAME unit-power grid; phase-only alignment of est
    est_u = est_syms / (np.sqrt(np.mean(np.abs(est_syms) ** 2)) + 1e-10)
    ref_u = ref_syms / (np.sqrt(np.mean(np.abs(ref_syms) ** 2)) + 1e-10)
    z = np.sum(ref_u * np.conj(est_u)) / (np.sum(np.abs(est_u) ** 2) + 1e-10)
    est_u = est_u * np.exp(1j * np.angle(z))

    # 5. Min-distance demodulation on the shared grid
    const = _get_constellation(mod_type)
    est_labels = np.argmin(np.abs(est_u[:, None] - const[None, :]), axis=1)
    ref_labels = np.argmin(np.abs(ref_u[:, None] - const[None, :]), axis=1)
    return est_labels, ref_labels


def compute_ser_compensated(estimated_signal, source_signal,
                            mod_type='QPSK', sample_rate=16000,
                            n_symbols=256, roll_off=0.35, num_taps=64,
                            carrier_freq=2000.0):
    """Proxy SER with ORACLE carrier synchronisation AND corrected
    timing / symbol scaling.

    Differs from utils.compute_ser_from_signal in three ways, each of
    which fixes a defect of the raw proxy (verified 2026-08-31: the raw
    proxy returns SER ~0.68 even for est == reference on 16QAM):

    1. Down-conversion at the source's TRUE carrier (this argument), not
       the nominal carrier_base — no residual constellation rotation.
    2. Symbol timing at the true pipeline delay: every convolution in
       the generator and here uses mode='same', so each is centred and
       the NET symbol delay is zero (verified empirically: clean-source
       decisions at offset 0 match the true transmitted symbols for all
       four modulations; the only exception is a small multipath-ISI
       residual on 16QAM).  The raw proxy sampled at sps//2 = 8, i.e.
       HALF A SYMBOL away from the eye opening.
    3. Both estimate and reference are demodulated on the SAME
       unit-power grid: the raw proxy compared the reference at unit
       power against the estimate at the LS-fit scale (≈3.7x off), which
       alone breaks high-order constellations.

    Timing stays fixed at the nominal grid (no per-pair timing search),
    so the metric still measures waveform fidelity, not a production
    receiver.
    """
    est_labels, ref_labels = _demod_pair_labels(
        estimated_signal, source_signal, mod_type, sample_rate,
        n_symbols, roll_off, num_taps, carrier_freq)
    return float(np.mean(est_labels != ref_labels))


def compute_ser_ber_compensated(estimated_signal, source_signal,
                                mod_type='QPSK', sample_rate=16000,
                                n_symbols=256, roll_off=0.35, num_taps=64,
                                carrier_freq=2000.0):
    """(SER, BER) from the SAME compensated-receiver decisions.

    BER is defined post-hoc via GRAY_BITS: each decided symbol index maps
    to its Gray bit string; BER = fraction of mismatched bit positions
    between the estimate's and the reference's decision bits.
    """
    est_labels, ref_labels = _demod_pair_labels(
        estimated_signal, source_signal, mod_type, sample_rate,
        n_symbols, roll_off, num_taps, carrier_freq)
    ser = float(np.mean(est_labels != ref_labels))
    bits = GRAY_BITS[mod_type]
    ber = float(np.mean(bits[est_labels] != bits[ref_labels]))
    return ser, ber


def ref_labels_only(source_signal, mod_type='QPSK', sample_rate=16000,
                    n_symbols=256, roll_off=0.35, num_taps=64,
                    carrier_freq=2000.0):
    """Reference hard labels only (numpy), mirroring the ref side of
    _demod_pair_labels step for step (oracle carrier -> matched filter ->
    0::sps grid -> unit-power grid -> masked argmin; the phase alignment
    there applies to the ESTIMATE only, so the reference side is exactly
    this).  Used by evaluate.py to score JointLLRHead decisions (S3).
    Returns int labels [n_use]."""
    from utils import _get_constellation
    from data_generator import rrc_filter

    ref = source_signal.detach().cpu().numpy().squeeze()
    T = len(ref)
    sps = T // n_symbols
    t = np.arange(T) / sample_rate
    ref_bb = ref * np.exp(-1j * 2 * np.pi * carrier_freq * t)
    rrc = rrc_filter(num_taps, roll_off, sps)
    ref_mf = np.convolve(ref_bb, rrc, mode='same')
    n_use = min(n_symbols, T // sps)
    ref_syms = ref_mf[0::sps][:n_use]
    ref_u = ref_syms / (np.sqrt(np.mean(np.abs(ref_syms) ** 2)) + 1e-10)
    const = _get_constellation(mod_type)
    return np.argmin(np.abs(ref_u[:, None] - const[None, :]), axis=1)


def ser_comp_one(est, ref, mod_id, true_carrier):
    """Returns (ser, ber) for one estimate/reference pair."""
    return compute_ser_ber_compensated(
        est, ref, mod_type=IDX_TO_MOD[mod_id],
        sample_rate=C.SignalConfig.sample_rate,
        n_symbols=C.SignalConfig.n_symbols,
        roll_off=C.SignalConfig.roll_off,
        num_taps=C.SignalConfig.num_taps,
        carrier_freq=float(true_carrier))


def main():
    p = argparse.ArgumentParser(description='Compensated-SER mixture baseline')
    p.add_argument('--n_per_cell', type=int, default=50)
    p.add_argument('--out', type=str,
                   default='results/ser/baseline_ser_comp.json')
    args = p.parse_args()

    ds = CommBSSVarKTestDataset(
        n_per_cell=args.n_per_cell,
        snr_points=C.SignalConfig.snr_test_points,
        mod_types=MOD_TYPES,
        k_lo=C.VarKConfig.k_min, k_hi=C.VarKConfig.k_max,
        k_extrap=C.VarKConfig.k_extrap,
        k_slots=C.VarKConfig.k_slots,
        signal_length=C.SignalConfig.signal_length,
        sample_rate=C.SignalConfig.sample_rate,
        carrier_base=C.SignalConfig.carrier_base,
        freq_gap_range=C.SignalConfig.freq_gap_range,
        seed=C.DataConfig.test_seed,
        return_carriers=True)

    per_mod: dict[str, list[float]] = {}
    per_k: dict[int, list[float]] = {}
    per_mod_ber: dict[str, list[float]] = {}
    per_k_ber: dict[int, list[float]] = {}
    per_snr: dict[float, list[float]] = {}
    per_snr_ber: dict[float, list[float]] = {}
    n_pairs = 0
    for mix, sources, _occ, k, snr, mods, carriers in ds:
        kt = int(k)
        if kt > 3:
            continue                      # K=4 probe cell excluded (as before)
        for j in range(kt):
            ser_v, ber_v = ser_comp_one(mix[0], sources[j, 0], int(mods[j]),
                                        float(carriers[j]))
            per_mod.setdefault(IDX_TO_MOD[int(mods[j])], []).append(ser_v)
            per_k.setdefault(kt, []).append(ser_v)
            per_mod_ber.setdefault(IDX_TO_MOD[int(mods[j])], []).append(ber_v)
            per_k_ber.setdefault(kt, []).append(ber_v)
            per_snr.setdefault(float(snr), []).append(ser_v)
            per_snr_ber.setdefault(float(snr), []).append(ber_v)
            n_pairs += 1

    out = {
        'n_pairs': n_pairs,
        'n_per_cell': args.n_per_cell,
        'per_mod': {m: float(np.mean(v)) for m, v in per_mod.items()},
        'per_k': {kk: float(np.mean(v)) for kk, v in per_k.items()},
        'overall': float(np.mean([x for v in per_mod.values() for x in v])),
        'per_mod_ber': {m: float(np.mean(v)) for m, v in per_mod_ber.items()},
        'per_k_ber': {kk: float(np.mean(v)) for kk, v in per_k_ber.items()},
        'overall_ber': float(np.mean(
            [x for v in per_mod_ber.values() for x in v])),
        'per_snr': {s: float(np.mean(v)) for s, v in per_snr.items()},
        'per_snr_ber': {s: float(np.mean(v)) for s, v in per_snr_ber.items()},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
