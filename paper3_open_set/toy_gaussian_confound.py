"""
Paper 3 — Generic Gaussian toy for Corollary 1 (review-3 follow-up).

Demonstrates that the indexing-permutation confound of Proposition 1 is
not specific to SC-BSS: take a score that is a noisy linear readout of a
condition C and carries NO OOD information whatsoever,

    S = gamma * C + eps,   eps ~ N(0, sigma^2),   S _||_ O | C   (trivially),

on the letter's 7-bin grid C in {-10, ..., 20} dB, n_per_bin mixtures of
two slots each.  Half the mixtures form the known pool, half the unknown
pool; as in the letter's pipeline the KNOWN labels pass through the
storage layout under test while the UNKNOWN pool is exactly labeled.
Label layouts for the known pool:

  correct    : labels match the scores' true bin        -> A_b = 1/2 exactly
  repeat/tile: scores stacked tile-wise (all slot-1, then all slot-2),
               labels written repeat-wise (interleaved) — the letter's bug
  reversed   : labels of the reversed mixture order      -> extreme case

Per-bin AUROCs are computed BOTH analytically from Corollary 1,
    A_b = E_{c ~ P(C_k | l_k = b)}[ Phi( (mu(b) - mu(c)) / (sqrt(2) sigma) )],
and by Monte Carlo (rank statistic on the simulated scores).

Usage:  python toy_gaussian_confound.py
"""

import numpy as np
from math import erf

BINS = np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
N_PER_BIN = 1000          # mixtures per bin (2 slots each)
GAMMA = 1.0               # score drift per dB of condition
SIGMA = 7.0               # score noise (dB-equivalent)
SEED = 0


def Phi(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def auroc(k, u):
    """Rank-based AUROC P(S_u > S_k) with tie correction."""
    k = np.asarray(k)
    u = np.asarray(u)
    diff = u[None, :] - k[:, None]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)


def analytic_auroc(mu_b, cond_dist):
    """Corollary 1: E_c[Phi((mu(b) - mu(c)) / (sqrt(2) sigma))]."""
    return float(sum(p * Phi((mu_b - GAMMA * c) / (np.sqrt(2.0) * SIGMA))
                     for c, p in cond_dist))


def main():
    rng = np.random.default_rng(SEED)
    n_mix = N_PER_BIN * len(BINS)
    true_mix_snr = np.repeat(BINS, N_PER_BIN)            # block order

    # known half / unknown half (both 2 slots per mixture, tile-stacked)
    scores_k = GAMMA * np.tile(true_mix_snr, 2) \
        + SIGMA * rng.standard_normal(2 * n_mix)
    scores_u = GAMMA * np.tile(true_mix_snr, 2) \
        + SIGMA * rng.standard_normal(2 * n_mix)
    labels_u = np.tile(true_mix_snr, 2)     # unknown pool exactly labeled

    layouts = {
        'correct': np.tile(true_mix_snr, 2),
        'repeat/tile': np.repeat(true_mix_snr, 2),
        'reversed': np.tile(true_mix_snr[::-1], 2),
    }

    print(f'toy: S = {GAMMA}*C + N(0,{SIGMA}^2), 2x{n_mix} mixtures x 2 slots; '
          'S carries NO OOD information (S _||_ O | C)')
    print(f'{"layout":<12} per-bin AUROC (analytic | MC), '
          'bins -10..20'.ljust(66) + 'wavg(MC)')
    for name, labels_k in layouts.items():
        per_bin_txt, wavg_num, wavg_den = [], 0.0, 0.0
        for b in BINS:
            k_idx = labels_k == b
            u_idx = labels_u == b
            # analytic: true-condition distribution of the known scores
            # assigned to stored bin b
            src_mix = np.tile(np.arange(n_mix), 2)[k_idx]
            conds, cnts = np.unique(true_mix_snr[src_mix], return_counts=True)
            a_ana = analytic_auroc(GAMMA * b, list(zip(conds, cnts / cnts.sum())))
            a_mc = auroc(scores_k[k_idx], scores_u[u_idx])
            per_bin_txt.append(f'{a_ana:.3f}|{a_mc:.3f}')
            w = k_idx.sum() * u_idx.sum()
            wavg_num += a_mc * w
            wavg_den += w
        print(f'{name:<12} {" ".join(per_bin_txt):<66} {wavg_num / wavg_den:.3f}')

    print('\nclosed-form spot check (two-group composition, half mass each):')
    print(f'  stored bin -5 dB, true C_k in {{-5,+10}} 50/50: '
          f'A = 0.5*Phi(0) + 0.5*Phi(-15/(sqrt2*{SIGMA})) = '
          f'{analytic_auroc(GAMMA * -5.0, [(-5.0, 0.5), (10.0, 0.5)]):.3f}')
    print(f'  stored bin +10 dB, same composition:            A = '
          f'{analytic_auroc(GAMMA * 10.0, [(-5.0, 0.5), (10.0, 0.5)]):.3f}')


if __name__ == '__main__':
    main()
