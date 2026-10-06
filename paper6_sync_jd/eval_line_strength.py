"""Paper 6 — symbol-level line-strength vs |E[c^4] * sum_l h_l^4|
(reviewer-requested artifact for the claim "the measured symbol-level line
strength tracks |E[c^4] sum_l h_l^4| with correlation 0.65", main.tex
E1 limitations paragraph).

The original number came from an interactive probe during the E1
development (2026-09-13/14); no script survived.  This file fixes the
measurement protocol:

  * K=1 bursts from the standard generator (signal_utils.
    generate_single_signal, fading on, carrier = 2000 + U(0,5) + in-generator
    +/-5 Hz jitter), n_bursts per modulation, seed 99999.
  * predictor  = |E[c^4]| * |sum_l h_l^4| with h the burst's TRUE unit-norm
    3-tap fading channel (exposed via generate_single_signal(
    return_channel=True)) and E[c^4] over the unit-power constellation.
  * measured   = symbol-level line strength of the burst demodulated at its
    TRUE carrier (down-convert at fc, RRC matched filter, 0::sps grid,
    unit power — the sync.py symbol-level front-end):
      - 'M4':  |mean(z^4)|  — the literal fourth-power strength that the
        predictor refers to (this is the M-th-power line for the M=4
        modulations; for 8PSK E[c^4] = 0 and mean(z^4) ~ 0 too);
      - 'MpM': |mean(z^M)| with M the modulation's symmetry order — the
        actual decision-free coherence score sync.py arbitrates with.
  * reported at 20 dB (paper operating point of the E1 discussion) and
    noiseless (inf) as a control.  At 20 dB a third measurement uses the
    ACTUAL BlindCarrierSync output z (blind_sync_known_mod — estimated
    carrier, not the true one): this is the literal "measured symbol-level
    line strength" of the deployed pipeline.

Pearson and Spearman correlations are reported over all bursts pooled and
per modulation (within a modulation |E[c^4]| is constant, so the
per-modulation correlation isolates the channel factor |sum_l h_l^4|).

Output: results/line_strength_corr.json.  Local CPU, seconds.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

import config as C                                          # noqa: E402
from signal_utils import (generate_single_signal, rrc_filter,  # noqa: E402
                          CONSTELLATIONS, MOD_TYPES)
from sync import blind_sync_known_mod                        # noqa: E402

SYM_ORDER = C.SyncConfig.sym_order

N_PER_MOD = 100
SEED = 99999
SNRS = [20.0, None]                    # None = noiseless control


def demod_symbols(sig, fc):
    """Demodulate at the TRUE carrier: baseband + RRC MF + 0::sps + unit
    power (the sync.py symbol-level front-end at residual df = 0)."""
    T = C.SignalConfig.signal_length
    n_sym = C.SignalConfig.n_symbols
    sps = T // n_sym
    t = np.arange(T) / C.SignalConfig.sample_rate
    rrc = rrc_filter(C.SignalConfig.num_taps, C.SignalConfig.roll_off, sps)
    xbb = np.asarray(sig, dtype=np.complex128) * np.exp(-1j * 2 * np.pi * fc * t)
    z = np.convolve(xbb, rrc, mode='same')[0::sps][:n_sym]
    return z / (np.sqrt(np.mean(np.abs(z) ** 2)) + 1e-10)


def main():
    np.random.seed(SEED)
    E4 = {m: complex(np.mean(CONSTELLATIONS[m] ** 4)) for m in MOD_TYPES}
    print('E[c^4] per modulation:', {m: np.round(v, 4) for m, v in E4.items()})

    records = []
    for mod in MOD_TYPES:
        for b in range(N_PER_MOD):
            carrier = (C.SignalConfig.carrier_base
                       + np.random.uniform(*C.SignalConfig.freq_gap_range))
            sig, _syms, fc, h = generate_single_signal(
                C.SignalConfig.n_symbols, carrier, C.SignalConfig.sample_rate,
                C.SignalConfig.signal_length, mod, C.SignalConfig.roll_off,
                C.SignalConfig.num_taps, C.SignalConfig.apply_fading,
                C.SignalConfig.fading_taps,
                return_freq=True, return_channel=True)
            pred = abs(E4[mod]) * abs(np.sum(h ** 4))
            rec = {'mod': mod, 'burst': b,
                   'pred_absEc4_sumh4': float(pred),
                   'abs_sum_h4': float(abs(np.sum(h ** 4)))}
            for snr in SNRS:
                if snr is None:
                    x = np.asarray(sig, dtype=np.complex128)
                    tag = 'clean'
                else:
                    p = np.mean(np.abs(sig) ** 2)
                    nstd = np.sqrt(p / 10 ** (snr / 10) / 2)
                    x = np.asarray(sig, dtype=np.complex128) + nstd * (
                        np.random.randn(len(sig))
                        + 1j * np.random.randn(len(sig)))
                    tag = f'{snr:g}dB'
                z = demod_symbols(x, fc)
                rec[f'line_M4_{tag}'] = float(abs(np.mean(z ** 4)))
                rec[f'line_MpM_{tag}'] = float(
                    abs(np.mean(z ** SYM_ORDER[mod])))
                if snr == 20.0:
                    # deployed pipeline: blind sync output (estimated carrier)
                    zb = blind_sync_known_mod(x, MOD_TYPES.index(mod))['z']
                    rec['line_M4_blindsync_20dB'] = float(
                        abs(np.mean(zb ** 4)))
                    rec['line_MpM_blindsync_20dB'] = float(
                        abs(np.mean(zb ** SYM_ORDER[mod])))
            records.append(rec)
        print(f'  {mod}: {N_PER_MOD} bursts done', flush=True)

    out = {'experiment': 'symbol-level line strength vs |E[c^4] sum h_l^4|',
           'seed': SEED, 'n_per_mod': N_PER_MOD,
           'paper_claim': {'correlation': 0.65,
                           'quote': 'symbol-level line strength tracks '
                                    '|E[c^4] sum_l h_l^4|'},
           'abs_Ec4': {m: float(abs(E4[m])) for m in MOD_TYPES},
           'correlations': {}}

    def corr_block(rs, key):
        x = np.array([r['pred_absEc4_sumh4'] for r in rs])
        y = np.array([r[key] for r in rs])
        return {'n': len(rs),
                'pearson': float(pearsonr(x, y)[0]),
                'pearson_p': float(pearsonr(x, y)[1]),
                'spearman': float(spearmanr(x, y)[0]),
                'spearman_p': float(spearmanr(x, y)[1])}

    for snr in SNRS:
        tag = 'clean' if snr is None else f'{snr:g}dB'
        meass = ['line_M4', 'line_MpM']
        if snr == 20.0:
            meass += ['line_M4_blindsync', 'line_MpM_blindsync']
        for meas in meass:
            key = f'{meas}_{tag}'
            if not all(key in r for r in records):
                continue
            blk = {'pooled_all_mods': corr_block(records, key)}
            # within-modulation: predictor variation is only |sum h_l^4|
            for mod in MOD_TYPES:
                rs = [r for r in records if r['mod'] == mod]
                blk[f'within_{mod}'] = corr_block(rs, key)
            out['correlations'][key] = blk

    path = os.path.join(C.RESULTS_DIR, 'line_strength_corr.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nsaved {path}')

    print(f"\n{'measurement':<22s}{'scope':<18s}{'pearson':>9s}"
          f"{'spearman':>10s}")
    for key, blk in out['correlations'].items():
        for scope, v in blk.items():
            print(f"{key:<22s}{scope:<18s}{v['pearson']:>9.3f}"
                  f"{v['spearman']:>10.3f}")


if __name__ == '__main__':
    main()
