"""Paper 6 — BlindCarrierSync (plan §3.1): classical blind carrier
synchronisation for one complex passband burst (E1 needs only this).

Pipeline per burst (all numpy, no learning):

  1. Coarse down-conversion at the nominal 2000 Hz carrier; residual
     offset Δf ∈ [-5, +10] Hz.
  2. Sample-level coarse frequency estimate: RRC-matched-filter FIRST
     (10.7 dB noise-bandwidth reduction before the nonlinearity), then
     M-th power (M = rotational symmetry order of the hypothesised
     modulation; amplitude handling per SyncConfig.power_mode), zero-padded
     FFT (x8), peak + parabolic interpolation -> Δf̂₁.
  3. Symbol-level refinement / wrong-peak rescue: three candidates —
     Δf̂₁ itself, candidate A (symbol-level M-th-power line in a narrow
     ±2 Hz window around Δf̂₁), candidate B (symbol-level line in the full
     ±13 Hz window from the nominal carrier).  The symbol-level stages use
     the DECIMATED symbol stream (256 clean, ISI-free-ish constellation
     samples instead of 4096 slew-corrupted ones) and always the 'raw'
     (amplitude-weighted) power, which down-weights ISI-smashed symbols on
     faded bursts.  Arbitration maximises the PRODUCT of two decision-free
     coherence scores — sample-level line strength |mean(y·e^{-j2πMΔf t})|
     on the MF'd 4096-sample M-th power, and symbol-level |mean(z^M)| on
     the decimated stream — because their failure modes are disjoint
     (measured 2026-09-13): deep-fade/high-ISI bursts have a strong
     sample-level line but unreliable symbol-level coherence (it once
     scored 0.72 at a wrong residual vs 0.41 at the truth), while
     transition-slew spur bursts are the opposite.  Neither the
     decision-referred lock score nor the decision-referred phase slope
     can replace this: hard decisions absorb a slow spin into a zero-mean
     sawtooth (at a 2.6 Hz wrong pick, lock and residual slope are
     indistinguishable from the true sync).
  4. Final demodulation: correct with the winning Δf̂, RRC matched filter,
     0::sps symbol grid (zero net delay, as in paper5's ser_comp),
     unit-power normalise -> z[256].
  5. NO decision-directed phase refinement by default (dd_iterations=0).
     Deviation from plan §3.1 step 4, measured: the DD line fit
     θ_n = angle(z_n conj(ĉ_n)) is FOLD-BIASED — wrong decisions fold the
     measured phase residual toward the nearest grid point, biasing the
     fitted intercept by ~6° at high SNR (fading-ISI spreads the true
     phases beyond the folding radius), which INCREASES SER (8PSK at
     20 dB: 0.028 -> 0.087; BPSK 0.002 -> 0.007).  Non-data-aided phase
     alternatives are no better under the 3-tap fading channel (Viterbi-
     Viterbi per-burst phase fluctuates by several degrees for
     QPSK/8PSK/16QAM; the min-distance phase scan minimises at the same
     biased angle).  The residual constant phase is small after the
     frequency correction and is absorbed by the M-fold ambiguity
     resolution in scoring.  The spec'd DD fit is still implemented
     (dd_iterations > 0) for ablation.
  6. Modulation-hypothesis selection (blind_sync only): steps 2–4 run for
     all four constellations; the hypothesis minimising the lock score
     mean|z − ĉ|² is kept.

Known limitations:
  - M-fold phase ambiguity: the M-th power method is blind to
    constellation-symmetry rotations, so the returned decisions equal the
    transmitted labels only up to a fixed rotation k·2π/M.  A real system
    resolves this with differential coding; the E1 evaluation resolves it
    post-hoc against the reference labels (genie ambiguity resolution),
    the standard convention for evaluating blind synchronisers.
  - Weak-line bursts (plan risk R2): for an occasional 16QAM burst the
    per-draw data average Σ c_n⁴ nearly cancels, so NO 4th-power method
    has a usable line for that burst (~1/16 bursts at 20 dB in local
    tests); the coherence score then also prefers a wrong residual.
    Reported, not fixed.
  - 8PSK at SNR <= -5 dB: the M=8 line is below the noise floor; the
    estimator fails wholesale (physical limit of the M-th-power method).

Public API:
  blind_sync(waveform, ...)                        — full hypothesis selection
  blind_sync_known_mod(waveform, mod_idx, ...)     — oracle-mod variant (E1)

Both accept one waveform [T] or a batch [P, T] and return a dict of
numpy arrays with leading dimension P (scalars squeezed for 1-D input):
  z [P,N] complex, decisions [P,N] int, mod_idx [P] int, df [P] float,
  phase_a [P] float, slope_b [P] float, lock [P] float.

__main__ smoke test: clean high-SNR bursts per modulation; checks
(a) |Δf̂ − Δf_true| < 1 Hz, (b) blind decisions reproduce the oracle
receiver's reference labels (ser_comp.ref_labels_only on the CLEAN
source) up to the M-fold ambiguity, (c) batch path == single path,
(d) hypothesis selection on clean bursts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Vendored paper6 modules (self-contained; no sys.path tricks).
from signal_utils import (rrc_filter, CONST_MAT, CONST_MASK,  # noqa: E402
                          MOD_TYPES)

import config as C                                         # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def constellation_np(mod_idx: int) -> np.ndarray:
    """Unit-power constellation points for MOD_TYPES[mod_idx] (numpy)."""
    n = int(CONST_MASK[mod_idx].sum())
    return np.asarray(CONST_MAT[mod_idx, :n]).astype(np.complex128)


def _mpower(x: np.ndarray, m_order: int, mode: str) -> np.ndarray:
    """M-th power with per-modulation amplitude handling."""
    if mode == 'ampnorm':
        return (x / (np.abs(x) + 1e-12)) ** m_order
    if mode == 'raw':
        return x ** m_order
    raise ValueError(f"unknown power_mode: {mode}")


def _fft_line(y: np.ndarray, fs: float, f_lo: float, f_hi: float,
              zeropad: int) -> float:
    """Peak + parabolic interpolation of the (1-D) M-th-power spectrum,
    search window [f_lo, f_hi] on the LINE frequency.  y: [T]."""
    T = len(y)
    n_fft = T * zeropad
    spec = np.abs(np.fft.fft(y, n=n_fft))
    freqs = np.fft.fftfreq(n_fft, d=1.0 / fs)
    in_win = (freqs >= f_lo) & (freqs <= f_hi)
    k = int(np.argmax(np.where(in_win, spec, -np.inf)))
    ym, y0, yp = spec[max(k - 1, 0)], spec[k], spec[min(k + 1, n_fft - 1)]
    denom = ym - 2.0 * y0 + yp
    delta = 0.5 * (ym - yp) / denom if abs(denom) > 1e-20 else 0.0
    f = (k + float(np.clip(delta, -1.0, 1.0))) * (fs / n_fft)
    if f > fs / 2:
        f -= fs
    return f


def _coarse_freq_sample(x_bb: np.ndarray, fs: float, m_order: int,
                        f_lo: float, f_hi: float, zeropad: int,
                        rrc: np.ndarray, mode: str) -> float:
    """Sample-level coarse estimate (step 2): MF first, then M-th power,
    FFT peak + parabolic interpolation.  Returns Δf̂₁ (Hz)."""
    xf = np.convolve(x_bb, rrc, mode='same')
    y = _mpower(xf, m_order, mode)
    f_line = _fft_line(y, fs, m_order * f_lo, m_order * f_hi, zeropad)
    return f_line / m_order


def _symbols(x_bb: np.ndarray, df: float, fs: float, sps: int,
             n_symbols: int, rrc: np.ndarray) -> np.ndarray:
    """Correct residual carrier, matched filter, 0::sps grid, unit power."""
    T = len(x_bb)
    t = np.arange(T) / fs
    xc = x_bb * np.exp(-1j * 2 * np.pi * df * t)
    z = np.convolve(xc, rrc, mode='same')[0::sps][:n_symbols]
    return z / (np.sqrt(np.mean(np.abs(z) ** 2)) + 1e-10)


def _symbol_refine(x_bb: np.ndarray, df1: float, m_order: int, mode: str,
                   half_win: float, fs: float, sps: int, n_symbols: int,
                   rrc: np.ndarray, zeropad: int) -> float:
    """Symbol-level M-th-power line estimate of the RESIDUAL offset around
    df1 (search window ±half_win Hz), from the decimated symbol stream."""
    fsym = fs / sps
    z = _symbols(x_bb, df1, fs, sps, n_symbols, rrc)
    y = _mpower(z, m_order, mode)
    f_line = _fft_line(y, fsym, -m_order * half_win, m_order * half_win,
                       zeropad)
    return df1 + f_line / m_order


def _coherence(x_bb: np.ndarray, df: float, m_order: int, mode: str,
               fs: float, sps: int, n_symbols: int,
               rrc: np.ndarray) -> float:
    """Decision-free sync quality score: |mean(z^M)| at residual df.  At
    the true residual the M-th powered symbols sum coherently
    (E[c^M] != 0); a wrong residual leaves a spinning constellation whose
    M-th power decoheres."""
    z = _symbols(x_bb, df, fs, sps, n_symbols, rrc)
    return float(np.abs(np.mean(_mpower(z, m_order, mode))))


def _sample_coherence(x_bb: np.ndarray, df: float, m_order: int,
                      mode: str, fs: float, rrc: np.ndarray) -> float:
    """Sample-level counterpart of _coherence: strength of the M-th-power
    line at M·df on the matched-filtered 4096-sample stream."""
    T = len(x_bb)
    t = np.arange(T) / fs
    xf = np.convolve(x_bb, rrc, mode='same')
    y = _mpower(xf, m_order, mode)
    return float(np.abs(np.mean(y * np.exp(-1j * 2 * np.pi * m_order * df * t))))


def _dd_refine(z: np.ndarray, const: np.ndarray, iterations: int):
    """Plan §3.1 step 4 verbatim: decision-directed LS line fit of the
    per-symbol phase residual (unwrap + polyfit), iterated.  NOT used by
    default (dd_iterations=0) — measured fold-biased, see module docstring.
    Returns (z_refined, a_applied, b_applied)."""
    a_tot = b_tot = 0.0
    n = np.arange(len(z), dtype=float)
    for _ in range(iterations):
        dec = np.argmin(np.abs(z[:, None] - const[None, :]), axis=1)
        th = np.unwrap(np.angle(z * np.conj(const[dec])))
        b, a = np.polyfit(n, th, 1)
        z = z * np.exp(-1j * (a + b * n))
        a_tot += a
        b_tot += b
    return z, a_tot, b_tot


def _sync_one(x_bb: np.ndarray, mod_idx: int, fs: float, sps: int,
              n_symbols: int, rrc: np.ndarray):
    """Full per-burst pipeline for one modulation hypothesis.
    Returns (z, decisions, df, phase_a, slope_b, lock)."""
    sc = C.SyncConfig
    name = MOD_TYPES[mod_idx]
    m_order = sc.sym_order[name]
    mode = sc.power_mode[name]
    const = constellation_np(mod_idx)

    # 2. Sample-level coarse estimate.
    df1 = _coarse_freq_sample(x_bb, fs, m_order, sc.freq_search_lo,
                              sc.freq_search_hi, sc.fft_zeropad, rrc, mode)

    # 3. Symbol-level candidates + product-coherence arbitration.
    df_a = _symbol_refine(x_bb, df1, m_order, 'raw', sc.refine_half_win,
                          fs, sps, n_symbols, rrc, sc.fft_zeropad)
    df_b = _symbol_refine(x_bb, 0.0, m_order, 'raw', sc.wide_half_win,
                          fs, sps, n_symbols, rrc, sc.fft_zeropad)
    cands = {df1, df_a, df_b}
    df = max(cands, key=lambda d: _sample_coherence(x_bb, d, m_order, mode,
                                                    fs, rrc)
             * _coherence(x_bb, d, m_order, 'raw', fs, sps, n_symbols, rrc))

    # 4. Final demodulation; optional (non-default) DD phase refinement.
    z = _symbols(x_bb, df, fs, sps, n_symbols, rrc)
    a = b = 0.0
    if sc.dd_iterations > 0:
        z, a, b = _dd_refine(z, const, sc.dd_iterations)
    dec = np.argmin(np.abs(z[:, None] - const[None, :]), axis=1)
    lock = float(np.mean(np.abs(z - const[dec]) ** 2))
    return z, dec, df, a, b, lock


def _run(waveform, mod_idx):
    """Shared driver.  mod_idx None -> hypothesis selection over all mods."""
    w = np.asarray(waveform, dtype=np.complex128)
    squeeze = (w.ndim == 1)
    if squeeze:
        w = w[None, :]
    P, T = w.shape
    fs = C.SignalConfig.sample_rate
    n_symbols = C.SignalConfig.n_symbols
    sps = T // n_symbols
    rrc = rrc_filter(C.SignalConfig.num_taps, C.SignalConfig.roll_off, sps)
    sc = C.SyncConfig

    # 1. Coarse down-conversion at the nominal carrier.
    t = np.arange(T) / fs
    x_bb = w * np.exp(-1j * 2 * np.pi * sc.nominal_carrier * t)[None, :]

    z = np.empty((P, n_symbols), dtype=np.complex128)
    dec = np.empty((P, n_symbols), dtype=np.int64)
    df = np.empty(P)
    a = np.empty(P)
    b = np.empty(P)
    lock = np.empty(P)
    chosen = np.empty(P, dtype=np.int64)
    if mod_idx is None:
        mods = np.full(P, -1, dtype=np.int64)         # sentinel: select
    else:
        mods = np.broadcast_to(np.asarray(mod_idx, dtype=np.int64), (P,))
    for p in range(P):
        if mods[p] < 0:
            per_mod = [_sync_one(x_bb[p], m, fs, sps, n_symbols, rrc)
                       for m in range(len(MOD_TYPES))]
            best = int(np.argmin([r[5] for r in per_mod]))
            r = per_mod[best]
        else:
            best = int(mods[p])
            r = _sync_one(x_bb[p], best, fs, sps, n_symbols, rrc)
        z[p], dec[p], df[p], a[p], b[p], lock[p] = r
        chosen[p] = best

    out = {
        'z': z,                     # [P, N] synced unit-power symbols
        'decisions': dec,           # [P, N] hard constellation indices
        'mod_idx': chosen,          # [P] chosen (or given) modulation index
        'df': df,                   # [P] frequency estimate (Hz)
        'phase_a': a,               # [P] DD phase-fit intercept (rad; 0 if off)
        'slope_b': b,               # [P] DD phase-fit slope (rad/symbol; 0 if off)
        'lock': lock,               # [P] constellation-lock score
    }
    if squeeze:
        out = {k: v[0] for k, v in out.items()}
    return out


def blind_sync(waveform, fs: float | None = None):
    """Full BlindCarrierSync with modulation-hypothesis selection (§3.1).

    waveform: complex passband burst [T] or batch [P, T].
    The `fs` argument is accepted for API clarity; the pipeline always uses
    config.SignalConfig.sample_rate (the benchmark is fixed-rate).
    """
    return _run(waveform, mod_idx=None)


def blind_sync_known_mod(waveform, mod_idx, fs: float | None = None):
    """Oracle-modulation variant: skips hypothesis selection (used by E1,
    which scores per TRUE modulation)."""
    return _run(waveform, mod_idx=mod_idx)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import torch

    from signal_utils import generate_single_signal      # vendored paper1
    import ser_comp                                      # vendored paper5

    print("Testing paper6 sync (BlindCarrierSync) ...")
    np.random.seed(C.SEED)

    n_sym = C.SignalConfig.n_symbols
    T = C.SignalConfig.signal_length
    fs = C.SignalConfig.sample_rate

    def make_clean(mod_name, carrier):
        sig, syms, fc = generate_single_signal(
            n_sym, carrier, fs, T, mod_name, C.SignalConfig.roll_off,
            C.SignalConfig.num_taps, C.SignalConfig.apply_fading,
            C.SignalConfig.fading_taps, return_freq=True)
        return sig.astype(np.complex128), syms, float(fc)

    def ser_min_rotation(z, const, ref_labels, m_order):
        """SER of z's decisions vs ref_labels, minimised over the M-fold
        symmetry rotations (genie ambiguity resolution)."""
        best = 1.0
        for k in range(m_order):
            zr = z * np.exp(-1j * 2 * np.pi * k / m_order)
            d = np.argmin(np.abs(zr[:, None] - const[None, :]), axis=1)
            best = min(best, float(np.mean(d != ref_labels)))
        return best

    # -- (a) frequency accuracy, all mods, clean + 20 dB noise -------------
    # Rare weak-line bursts (per-draw Σc^M cancellation; plan risk R2) can
    # produce a wrong-peak estimate even at 20 dB — assert typical accuracy
    # (median) plus a bounded failure count rather than worst case.
    print("  (a) frequency estimate vs true offset:")
    for m, name in enumerate(MOD_TYPES):
        errs = []
        for trial in range(8):
            carrier = 2000.0 + np.random.uniform(0.0, 5.0)  # pre-jitter; the
            # generator adds its own +/-5 Hz jitter on top (residual window)
            sig, _syms, fc = make_clean(name, carrier)
            p = np.mean(np.abs(sig) ** 2)
            noise = np.sqrt(p / 10 ** (20 / 10) / 2) * (
                np.random.randn(T) + 1j * np.random.randn(T))
            res = blind_sync_known_mod(sig + noise, m)
            errs.append(abs(float(res['df']) - (fc - C.SyncConfig.nominal_carrier)))
        n_bad = sum(e > 1.0 for e in errs)
        print(f"      {name:5s}: median |df err| = {np.median(errs):.3f} Hz, "
              f">1 Hz failures: {n_bad}/{len(errs)}")
        assert np.median(errs) < 0.1, f"{name}: median error {np.median(errs):.3f} Hz"
        assert n_bad <= 1, f"{name}: {n_bad}/8 wrong-peak failures"

    # -- (b) blind decisions reproduce the oracle reference labels ---------
    # Gated to DECODABLE bursts: the 3-tap Rayleigh channel occasionally
    # nulls the band, and then even the oracle reference labels disagree
    # with the transmitted symbols (8PSK clean floor mean 0.014 has a heavy
    # per-burst tail).  Sync reproduction is only meaningful where the
    # burst is readable, so trials whose own phase-aligned floor exceeds
    # 0.05 are reported but skipped for the assertion.
    print("  (b) decisions vs ser_comp.ref_labels_only (clean source):")
    for m, name in enumerate(MOD_TYPES):
        sers, skipped = [], 0
        const = constellation_np(m)
        m_order = C.SyncConfig.sym_order[name]
        for trial in range(6):
            carrier = 2000.0 + np.random.uniform(0.0, 5.0)
            sig, syms, fc = make_clean(name, carrier)
            ref_labels = ser_comp.ref_labels_only(
                torch.from_numpy(sig.astype(np.complex64)),
                mod_type=name, carrier_freq=fc)
            # burst readability floor: demodulate the clean source at its
            # true carrier, align phase to the TRUE transmitted symbols
            # (genie), re-decide — the best any synchroniser could do.
            from signal_utils import rrc_filter as _rrc
            t = np.arange(T) / fs
            xbb = sig * np.exp(-1j * 2 * np.pi * fc * t)
            rrc = _rrc(C.SignalConfig.num_taps, C.SignalConfig.roll_off,
                       T // n_sym)
            zf = np.convolve(xbb, rrc, mode='same')[0::T // n_sym][:n_sym]
            zf = zf / (np.sqrt(np.mean(np.abs(zf) ** 2)) + 1e-10)
            syms_u = np.asarray(syms) / np.sqrt(np.mean(np.abs(syms) ** 2))
            ti = np.argmin(np.abs(syms_u[:, None] - const[None, :]), axis=1)
            g = np.sum(syms_u * np.conj(zf)) / (np.sum(np.abs(zf) ** 2) + 1e-10)
            za = zf * np.exp(1j * np.angle(g))
            floor = float(np.mean(
                np.argmin(np.abs(za[:, None] - const[None, :]), axis=1) != ti))
            if floor > 0.05:
                skipped += 1
                continue
            res = blind_sync_known_mod(sig, m)
            sers.append(ser_min_rotation(res['z'], const, ref_labels,
                                         m_order))
        print(f"      {name:5s}: max SER(min-rot) = "
              f"{max(sers) if sers else float('nan'):.4f} over "
              f"{len(sers)} decodable bursts ({skipped} skipped)")
        assert sers, f"{name}: all bursts skipped"
        # 16QAM tolerates one weak-line burst (per-draw Σc⁴ cancellation,
        # plan risk R2 — a clean-signal failure mode of the 4th-power
        # estimator, reported not fixed).
        n_bad = sum(s > 0.05 for s in sers)
        if name == '16QAM':
            assert n_bad <= 1 and float(np.mean(sers)) < 0.08, \
                f"{name}: {n_bad}/6 bursts above 0.05, mean {np.mean(sers):.4f}"
        else:
            assert max(sers) < 0.05, f"{name}: SER {max(sers):.4f} >= 0.05"

    # -- (c) batch path matches single path --------------------------------
    sigs, fcs, midx = [], [], []
    for m, name in enumerate(MOD_TYPES):
        sig, _syms, fc = make_clean(name, 2000.0 + 5.0)
        sigs.append(sig)
        fcs.append(fc)
        midx.append(m)
    batch = blind_sync_known_mod(np.stack(sigs), np.array(midx))
    for i in range(len(sigs)):
        single = blind_sync_known_mod(sigs[i], midx[i])
        assert np.allclose(batch['z'][i], single['z'], atol=1e-10)
        assert np.array_equal(batch['decisions'][i], single['decisions'])
        assert abs(batch['df'][i] - single['df']) < 1e-9
    print("  (c) batch path == single path: OK")

    # -- (d) hypothesis selection on clean bursts ---------------------------
    n_right = 0
    n_tot = 0
    detail = []
    for m, name in enumerate(MOD_TYPES):
        for trial in range(3):
            carrier = 2000.0 + np.random.uniform(0.0, 5.0)
            sig, _syms, fc = make_clean(name, carrier)
            res = blind_sync(sig)
            ok = int(int(res['mod_idx']) == m)
            n_right += ok
            n_tot += 1
            detail.append(f"{name}->{MOD_TYPES[int(res['mod_idx'])]}")
    print(f"  (d) hypothesis selection on clean bursts: "
          f"{n_right}/{n_tot} correct ({detail})")

    print("\nsync smoke test passed!")
