"""Paper 6 — JointPairDetector (plan §3.3): joint ML detection over a
synced K=2 slot pair, with EM estimation of the complex coupling matrix A.

Symbol-rate model (plan §2.1 Proposition 0), after per-slot sync:

    z_n = A [c_{1,n}; c_{2,n}] + w_n,     w_n ~ CN(0, sigma2)

with z_n the stacked unit-power symbol streams of the two slots and A the
2x2 complex coupling matrix (post-equalisation per-source gains; the
off-diagonals are the residual cross-slot leakage the separator left
behind).  Joint ML decisions:

    (ĉ_1, ĉ_2) = argmin_{(c_i,c_j)} || z_n − A [c_i; c_j] ||^2

over the <= 16x16 hypothesis grid (vectorised).

A estimation (blind): EM from A = I — E-step joint decisions with the
current A, M-step closed-form complex LS refit
A = Z C^H (C C^H)^{-1} on the current decisions, em_rounds rounds.
Because EM from a single I-init can land in swapped/degenerate minima at
SIR ≈ 0 (plan risk R3), estimate_A_em additionally runs a small restart
grid over off-diagonal magnitudes/phases (the known operating point:
mixing weights w_k ~ U(0.4,0.6), so |leakage| ~ 1 at SIR 0 dB) and keeps
the max-likelihood solution.

All numpy.  Constellation tables are a local copy of soft_demod's
_CONST_LISTS (copy-not-share keeps this module torch-free); __main__
asserts parity against soft_demod.CONST_MAT.

__main__ smoke test: synthetic z_n = A_true [c1;c2] + noise at several
SNR/SIR operating points; checks (i) exact recovery with A=I and no
interference/noise, (ii) oracle-A joint detection beats separate (A=I)
detection at SIR 0 dB, (iii) EM-A beats separate at SIR 0 dB.
"""
from __future__ import annotations

import numpy as np

import config as C

# ---------------------------------------------------------------------------
# Constellation tables — local numpy copy of paper5 soft_demod._CONST_LISTS
# (index order MUST match utils.CONSTELLATIONS / soft_demod.CONST_MAT;
# parity asserted in __main__).
# ---------------------------------------------------------------------------
MOD_TYPES = ['BPSK', 'QPSK', '8PSK', '16QAM']
_CONST_LISTS = {
    'BPSK': [-1 + 0j, 1 + 0j],
    'QPSK': [(-1 - 1j) / np.sqrt(2), (-1 + 1j) / np.sqrt(2),
             (1 - 1j) / np.sqrt(2), (1 + 1j) / np.sqrt(2)],
    '8PSK': list(np.exp(1j * (2 * np.pi * np.arange(8) / 8 + np.pi / 8))),
    '16QAM': [(i + 1j * q) / np.sqrt(10)
              for i in (-3, -1, 1, 3) for q in (-3, -1, 1, 3)],
}


def constellation_np(mod_idx: int) -> np.ndarray:
    """Unit-power constellation points for MOD_TYPES[mod_idx]."""
    return np.array(_CONST_LISTS[MOD_TYPES[int(mod_idx)]],
                    dtype=np.complex128)


# ---------------------------------------------------------------------------
# Joint ML detection
# ---------------------------------------------------------------------------
def joint_detect(z1: np.ndarray, z2: np.ndarray,
                 const1: np.ndarray, const2: np.ndarray,
                 A: np.ndarray | None = None,
                 sigma2: float | None = None):
    """Joint ML decisions over the M1 x M2 hypothesis grid (vectorised).

    Args:
        z1, z2 : [N] complex — synced unit-power symbol streams
        const1, const2 : [M1] / [M2] complex — slot constellations
        A      : [2, 2] complex coupling matrix (default: identity)
        sigma2 : noise softness (only needed for soft log-likelihoods;
                 hard decisions are invariant to it)

    Returns
    -------
    idx1, idx2 : [N] int — joint hard decisions (constellation indices)
    """
    if A is None:
        A = np.eye(2, dtype=np.complex128)
    A = np.asarray(A, dtype=np.complex128)
    Z = np.stack([z1, z2], axis=0)                          # [2, N]
    m2 = len(const2)
    G = np.array([[c1, c2] for c1 in const1 for c2 in const2],
                 dtype=np.complex128)                       # [H, 2]
    pred = G @ A.T                                          # [H, 2]
    d2 = np.sum(np.abs(Z.T[:, None, :] - pred[None, :, :]) ** 2, axis=-1)
    h = np.argmin(d2, axis=-1)                              # [N]
    return h // m2, h % m2


def build_oracle_A(a11: complex, a12: complex,
                   a21: complex, a22: complex) -> np.ndarray:
    """Oracle-A helper: assemble the coupling matrix from known per-slot
    gains/leakages (A[row=slot_obs, col=source])."""
    return np.array([[a11, a12], [a21, a22]], dtype=np.complex128)


# ---------------------------------------------------------------------------
# EM estimation of A
# ---------------------------------------------------------------------------
def _fit_A(Z: np.ndarray, idx1: np.ndarray, idx2: np.ndarray,
           const1: np.ndarray, const2: np.ndarray) -> np.ndarray:
    """M-step: closed-form complex LS A = argmin_A ||Z − A C||_F^2."""
    Cd = np.stack([const1[idx1], const2[idx2]], axis=0)     # [2, N]
    G = Cd @ Cd.conj().T                                    # [2, 2]
    return Z @ Cd.conj().T @ np.linalg.inv(G + 1e-9 * np.eye(2))


def _residual_energy(z1, z2, const1, const2, A) -> float:
    """Joint fit residual sum_n min_{i,j} ||z_n − A[c_i;c_j]||^2 (the EM
    objective; lower = better likelihood)."""
    idx1, idx2 = joint_detect(z1, z2, const1, const2, A)
    Z = np.stack([z1, z2], axis=0)
    Cd = np.stack([const1[idx1], const2[idx2]], axis=0)
    return float(np.sum(np.abs(Z - A @ Cd) ** 2))


def _em_from(z1, z2, const1, const2, A0, rounds):
    A = np.asarray(A0, dtype=np.complex128).copy()
    idx1 = idx2 = None
    for _ in range(rounds):
        idx1, idx2 = joint_detect(z1, z2, const1, const2, A)   # E-step
        A = _fit_A(np.stack([z1, z2], axis=0), idx1, idx2,
                   const1, const2)                             # M-step
    idx1, idx2 = joint_detect(z1, z2, const1, const2, A)
    return A, idx1, idx2


def estimate_A_em(z1: np.ndarray, z2: np.ndarray,
                  const1: np.ndarray, const2: np.ndarray,
                  rounds: int | None = None,
                  restart_mags=(0.5, 1.0), restart_phases: int = 8,
                  rng: np.random.RandomState | None = None):
    """Blind EM estimation of the coupling matrix A (plan §3.3).

    Primary init is A = I (per spec); a small restart grid over
    off-diagonal magnitude/phase guards against swapped/degenerate minima
    at SIR ≈ 0 (plan risk R3 mitigation).  The max-likelihood (minimum
    joint residual) solution is kept.

    Returns (A_hat [2,2], idx1 [N], idx2 [N]).
    """
    if rounds is None:
        rounds = C.JointConfig.em_rounds
    inits = [np.eye(2, dtype=np.complex128)]
    for mag in restart_mags:
        for k1 in range(restart_phases):
            for k2 in range(restart_phases):
                inits.append(np.array(
                    [[1.0, mag * np.exp(1j * 2 * np.pi * k1 / restart_phases)],
                     [mag * np.exp(1j * 2 * np.pi * k2 / restart_phases), 1.0]],
                    dtype=np.complex128))
    best = None
    for A0 in inits:
        A, idx1, idx2 = _em_from(z1, z2, const1, const2, A0, rounds)
        res = _residual_energy(z1, z2, const1, const2, A)
        if best is None or res < best[0]:
            best = (res, A, idx1, idx2)
    return best[1], best[2], best[3]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Testing paper6 joint_detect (JointPairDetector) ...")
    rng = np.random.RandomState(C.SEED)
    N = 256

    def draw(const, n):
        return const[rng.randint(0, len(const), n)]

    def synth(const1, const2, A_true, snr_db):
        c1 = draw(const1, N)
        c2 = draw(const2, N)
        Z = A_true @ np.stack([c1, c2], axis=0)
        p = np.mean(np.abs(Z) ** 2)
        noise = np.sqrt(p / 10 ** (snr_db / 10) / 2) * (
            rng.randn(2, N) + 1j * rng.randn(2, N))
        return Z + noise, c1, c2

    def ser_of(idx1, idx2, c1, c2, const1, const2):
        return 0.5 * (np.mean(const1[idx1] != c1) + np.mean(const2[idx2] != c2))

    # -- parity of the local constellation copy with the vendored table --
    from signal_utils import CONST_MAT, CONST_MASK   # vendored paper5
    for m, name in enumerate(MOD_TYPES):
        n = int(CONST_MASK[m].sum())
        assert np.allclose(constellation_np(m), np.asarray(CONST_MAT[m, :n]),
                           atol=1e-12), name
    print("  constellation parity with the vendored CONST_MAT: OK")

    # -- (i) A = I, no interference, no noise -> exact decisions -----------
    const_q = constellation_np(1)   # QPSK
    c1 = draw(const_q, N)
    c2 = draw(const_q, N)
    i1, i2 = joint_detect(c1, c2, const_q, const_q, np.eye(2))
    assert np.array_equal(const_q[i1], c1) and np.array_equal(const_q[i2], c2)
    print("  (i) A=I, clean: decisions exact: OK")

    # -- (ii)/(iii) SIR = 0 dB operating point, QPSK x QPSK ----------------
    print("  (ii)/(iii) SIR 0 dB, QPSKxQPSK, mean SER over trials:")
    for snr in (10, 20, 30):
        s_sep, s_ora, s_em = [], [], []
        for _trial in range(10):
            phi1 = rng.uniform(0, 2 * np.pi)
            phi2 = rng.uniform(0, 2 * np.pi)
            A_true = build_oracle_A(1.0, np.exp(1j * phi1),
                                    np.exp(1j * phi2), 1.0)
            Z, c1, c2 = synth(const_q, const_q, A_true, snr)
            z1, z2 = Z[0], Z[1]
            i1, i2 = joint_detect(z1, z2, const_q, const_q, np.eye(2))
            s_sep.append(ser_of(i1, i2, c1, c2, const_q, const_q))
            i1, i2 = joint_detect(z1, z2, const_q, const_q, A_true)
            s_ora.append(ser_of(i1, i2, c1, c2, const_q, const_q))
            A_hat, i1, i2 = estimate_A_em(z1, z2, const_q, const_q, rng=rng)
            s_em.append(ser_of(i1, i2, c1, c2, const_q, const_q))
        m_sep, m_ora, m_em = map(np.mean, (s_sep, s_ora, s_em))
        print(f"      SNR {snr:>2d} dB: separate(A=I)={m_sep:.4f}  "
              f"oracle-A={m_ora:.4f}  EM-A={m_em:.4f}")
        assert m_ora < m_sep, "oracle-A joint must beat separate detection"
        assert m_em < m_sep, "EM-A joint must beat separate detection"

    # -- (iii-b) mixed constellations (QPSK x 16QAM), SIR 0 dB -------------
    const_qam = constellation_np(3)
    s_sep, s_em = [], []
    for _trial in range(6):
        A_true = build_oracle_A(1.0, np.exp(1j * rng.uniform(0, 2 * np.pi)),
                                np.exp(1j * rng.uniform(0, 2 * np.pi)), 1.0)
        Z, c1, c2 = synth(const_q, const_qam, A_true, 20)
        i1, i2 = joint_detect(Z[0], Z[1], const_q, const_qam, np.eye(2))
        s_sep.append(ser_of(i1, i2, c1, c2, const_q, const_qam))
        A_hat, i1, i2 = estimate_A_em(Z[0], Z[1], const_q, const_qam, rng=rng)
        s_em.append(ser_of(i1, i2, c1, c2, const_q, const_qam))
    print(f"      QPSKx16QAM SIR 0 dB SNR 20: separate={np.mean(s_sep):.4f} "
          f" EM-A={np.mean(s_em):.4f}")
    assert np.mean(s_em) < np.mean(s_sep)

    print("\njoint_detect smoke test passed!")
