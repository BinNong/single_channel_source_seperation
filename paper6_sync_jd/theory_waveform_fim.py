"""Paper 6 — communication-waveform frequency Fisher analysis (review-2
bridge for Proposition 3).

Reviewer 2 (Major Concern 4): the two-tone CRB of Proposition 3 is a
PURE-TONE surrogate — it bounds the information in the unmodulated
carrier lines, not in the modulated, pulse-shaped, faded communication
waveform.  This script computes the Fisher information of the ACTUAL
benchmark waveform under the minimal data-aided model the reviewer
suggests (known symbols, known modulation, known pulse shaping; unknown
CFOs and unknown complex channels):

    s_n = sum_k e^{j 2 pi Df_k n / fs} (h_k conv x_k)[n],
    x_k  = RRC-shaped KNOWN symbol stream (256 symbols, sps=16, 65 taps),
    h_k  = unknown 3-tap complex fading channel,
    theta = [Df_1, Df_2, Re h_1(0..2), Im h_1(0..2), Re h_2, Im h_2].

Three references are compared per |Df|:
  (a) single-source known-symbol CRB (same model, K=1) — the decoupled
      "sync after separation" ideal;
  (b) pure-tone surrogate inflation (theory_validation.two_tone_inflation,
      phases-only and gains-unknown nuisance) — the information a
      non-data-aided spectral-line synchroniser can reach;
  (c) known-symbols + known-channel FIM (only Df unknown) — the
      data-aided genie bound.

The scientific question: does the co-frequency frequency-estimation
coupling survive once the modulation structure is modelled?  If the
known-symbol inflation is mild at small |Df| while the pure-tone one
diverges, the modulation structure carries the disambiguating
information — but a NON-data-aided receiver cannot access it, which is
exactly the regime our blind synchroniser is confined to.  Either
outcome bridges Proposition 3 to the communication waveform.

sigma2 = 1 throughout (cancels in all inflation ratios).
Output: results/waveform_fim.json  (+ printed summary).
Deterministic (seeded).  CPU, seconds.
"""
from __future__ import annotations

import json
import os

import numpy as np

import config as C  # noqa: E402
from signal_utils import rrc_filter, generate_symbols  # noqa: E402
from theory_validation import (two_tone_inflation,  # noqa: E402
                               crb_single_tone_hz2)

FS = C.SignalConfig.sample_rate          # 16000
T = C.SignalConfig.signal_length         # 4096
NSYM = C.SignalConfig.n_symbols          # 256
SPS = T // NSYM                          # 16
RRC = rrc_filter(C.SignalConfig.num_taps, C.SignalConfig.roll_off, SPS)
MODS = ['QPSK', 'QPSK']                  # the benchmark's headline pair;
                                         # conclusions are modulation-agnostic
                                         # (PSK); QAM checked in __main__.

DF_GRID = np.array([0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5,
                    3.0, 3.906, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0])
N_TRIALS = 40                            # symbol+channel realisations per Df
SEED = 20260922


def shaped_stream(symbols: np.ndarray) -> np.ndarray:
    """Upsample x16 + RRC ('same'), exactly the generator's pulse chain
    (256*16 = 4096 = signal_length, so the generator's resample step is
    a no-op and is omitted).  Normalised to unit mean power, matching the
    generator's unit-power sources (the post-fading normalisation is
    absorbed into the free channel taps)."""
    up = np.zeros(NSYM * SPS, dtype=complex)
    up[::SPS] = symbols
    x = np.convolve(up, RRC, mode='same')
    return x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)


def _shift(x, k):
    """x[n-k] with zero fill (matches np.convolve(..., 'same') boundary
    semantics; the burst edges are pulse peaks, so circular wrap — as in
    np.roll — would be badly wrong)."""
    if k == 0:
        return x.copy()
    if k > 0:
        return np.concatenate([np.zeros(k, dtype=x.dtype), x[:-k]])
    return np.concatenate([x[-k:], np.zeros(-k, dtype=x.dtype)])


def _fisher_waveform(x1, x2, h1, h2, df1, df2, unknown_channel=True):
    """Fisher matrix of the two-source communication waveform.
    Returns (J, n_params).  sigma2 = 1.

    Generator-faithful model (signal_utils.generate_single_signal): the
    fading channel acts on the PASSBAND stream, i.e.
        s_k[n] = sum_l h_k[l] * (u_k x_k)[n+1-l],
        u_k[n] = e^{j 2 pi Df_k n / fs},
    so d s_k / d Df_k = convolve(j 2 pi (n/fs) u_k x_k, h_k, 'same') and
    d s_k / d h_k[l] = (u_k x_k)[n+1-l]."""
    n = np.arange(T) / FS
    u1 = np.exp(1j * 2 * np.pi * df1 * n)
    u2 = np.exp(1j * 2 * np.pi * df2 * n)
    v1, v2 = u1 * x1, u2 * x2                    # passband known streams
    y1 = np.convolve(v1, h1, mode='same')
    y2 = np.convolve(v2, h2, mode='same')
    derivs = [np.convolve(1j * 2 * np.pi * n * v1, h1, mode='same'),
              np.convolve(1j * 2 * np.pi * n * v2, h2, mode='same')]
    if unknown_channel:
        for l in range(len(h1)):
            derivs.append(_shift(v1, l - 1))
            derivs.append(1j * _shift(v1, l - 1))
        for l in range(len(h2)):
            derivs.append(_shift(v2, l - 1))
            derivs.append(1j * _shift(v2, l - 1))
    D = np.stack(derivs)                         # [P, T]
    J = 2.0 * np.real(D @ D.conj().T)
    return J, len(derivs)


def crb_two_source(x1, x2, h1, h2, df, unknown_channel=True):
    """[J^{-1}]_{11} of the two-source waveform model at Df1=0, Df2=df."""
    J, _ = _fisher_waveform(x1, x2, h1, h2, 0.0, float(df),
                            unknown_channel)
    return float(np.linalg.inv(J)[0, 0])


def crb_one_source(x, h, unknown_channel=True):
    """[J^{-1}]_{11} of the single-source waveform model (Df = 0)."""
    n = np.arange(T) / FS
    y = np.convolve(x, h, mode='same')
    derivs = [1j * 2 * np.pi * n * y]
    if unknown_channel:
        for l in range(len(h)):
            xl = _shift(x, l - 1)
            derivs.append(xl)
            derivs.append(1j * xl)
    D = np.stack(derivs)
    J = 2.0 * np.real(D @ D.conj().T)
    return float(np.linalg.inv(J)[0, 0])
def main():
    rng = np.random.default_rng(SEED)
    mods = MODS

    # ---- single-source references (once per trial) ----
    ref_unk_ch, ref_known_ch = [], []
    trials = []
    for _ in range(N_TRIALS):
        h1 = rng.standard_normal(3) + 1j * rng.standard_normal(3)
        h1 /= np.linalg.norm(h1)
        h2 = rng.standard_normal(3) + 1j * rng.standard_normal(3)
        h2 /= np.linalg.norm(h2)
        x1 = shaped_stream(generate_symbols(NSYM, mods[0]))
        x2 = shaped_stream(generate_symbols(NSYM, mods[1]))
        ref_unk_ch.append(crb_one_source(x1, h1, True))
        ref_known_ch.append(crb_one_source(x1, h1, False))
        trials.append((x1, x2, h1, h2))
    ref_unk_ch = np.array(ref_unk_ch)
    ref_known_ch = np.array(ref_known_ch)

    # ---- two-source inflation vs |Df| ----
    infl_unk, infl_known = [], []
    for df in DF_GRID:
        row_u, row_k = [], []
        for x1, x2, h1, h2 in trials:
            row_u.append(crb_two_source(x1, x2, h1, h2, df, True)
                         / ref_unk_ch[len(row_u)])
            row_k.append(crb_two_source(x1, x2, h1, h2, df, False)
                         / ref_known_ch[len(row_k)])
        infl_unk.append(row_u)
        infl_known.append(row_k)
    infl_unk = np.array(infl_unk)      # [n_df, n_trials]
    infl_known = np.array(infl_known)

    # ---- pure-tone surrogate (existing exact analysis; floored at 0.02 Hz
    # like theory_validation.py — the two-tone FIM is exactly singular at
    # Df = 0, i.e. the pure-tone inflation diverges there) ----
    df_tone = np.maximum(DF_GRID, 0.02)
    infl_tone = two_tone_inflation(df_tone, T, FS)
    infl_tone_ua = two_tone_inflation(df_tone, T, FS, unknown_amplitudes=True)

    out = {
        'df_grid_hz': DF_GRID.tolist(),
        'n_trials': N_TRIALS,
        'mods': mods,
        'waveform_unk_channel': {
            'median': np.median(infl_unk, axis=1).tolist(),
            'mean': np.mean(infl_unk, axis=1).tolist(),
            'p90': np.percentile(infl_unk, 90, axis=1).tolist(),
        },
        'waveform_known_channel': {
            'median': np.median(infl_known, axis=1).tolist(),
            'p90': np.percentile(infl_known, 90, axis=1).tolist(),
        },
        'pure_tone_phase_nuisance': infl_tone.tolist(),
        'pure_tone_gains_unknown': infl_tone_ua.tolist(),
        'crb_single_source_median_hz2': float(np.median(ref_unk_ch)),
    }
    path = os.path.join(C.RESULTS_DIR, 'waveform_fim.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)

    # ---- summary ----
    print(f"saved {path}")
    print(f"single-source known-symbol CRB (median): "
          f"{np.sqrt(np.median(ref_unk_ch)) * 1e3:.3f} mHz (std, sigma2=1)")
    print(f"\n{'|Df| Hz':>8s} | {'waveform unk-ch':>15s} | "
          f"{'waveform kn-ch':>15s} | {'pure tone':>10s} | {'tone+gains':>10s}")
    for i, df in enumerate(DF_GRID):
        print(f"{df:8.3f} | {np.median(infl_unk[i]):15.2f} | "
              f"{np.median(infl_known[i]):15.2f} | "
              f"{infl_tone[i]:10.2f} | {infl_tone_ua[i]:10.2f}")
    i1 = int(np.argmin(np.abs(DF_GRID - 3.906)))
    print(f"\nat |Df| = 1/T_burst = 3.906 Hz: waveform(unk-ch) median "
          f"{np.median(infl_unk[i1]):.2f}x  vs pure-tone {infl_tone[i1]:.2f}x")
    print(f"at |Df| = 0: waveform(unk-ch) median {np.median(infl_unk[0]):.2f}x"
          f"  vs pure-tone {infl_tone[0]:.1f}x")


if __name__ == '__main__':
    # sanity: the tap-derivative convention must reproduce 'same' convolution
    rng = np.random.default_rng(0)
    h = rng.standard_normal(3) + 1j * rng.standard_normal(3)
    h /= np.linalg.norm(h)
    x = shaped_stream(generate_symbols(NSYM, 'QPSK'))
    y = np.convolve(x, h, mode='same')
    y2 = sum(h[l] * _shift(x, l - 1) for l in range(3))
    assert np.allclose(y, y2, atol=1e-10), 'tap-shift convention mismatch'
    print("tap-shift / 'same'-convolution convention: OK")
    # sanity: known-channel single-source CRB ~ Rife-Boorstyn
    v = crb_one_source(x, h, False)
    rb = crb_single_tone_hz2(0.0, T, FS)   # eta = 1 (sigma2=1, unit power)
    print(f"known-channel waveform CRB {v:.3e} vs Rife-Boorstyn {rb:.3e} "
          f"(ratio {v / rb:.2f})")
    main()
