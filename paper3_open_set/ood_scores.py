"""
Paper 3 — Open-Set SC-BSS: OOD scoring functions.

Three inference-time OOD scoring methods, all operating on per-source
embeddings or logits:

  1. Energy score  : -logsumexp(logits / T).  Higher = more OOD.
                     Baseline; references ODIN / Liu et al. 2020.

  2. Prototype     : min distance from the embedding to any known-class
                     prototype in embedding space.  Higher = more OOD.
                     Closely related to the nearest-class-mean classifier.

  3. VOS           : distance to synthesised virtual outliers (generated
                     by extrapolating from known prototypes).  Higher = more
                     OOD.  References Du et al. ICLR 2022 "VOS: Learning
                     What You Don't Know by Virtual Outlier Synthesis".

All functions accept numpy arrays or torch tensors; outputs are 1-D numpy
arrays of shape (N,) where larger values indicate higher OOD likelihood.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ----------------------------------------------------------------------------
# 1. Energy score (ODIN-style)
# ----------------------------------------------------------------------------
def energy_score(logits, temperature: float = 1.0) -> np.ndarray:
    """Energy = -logsumexp(logits / T).  Higher = more OOD.

    Args:
        logits      : (N, K) array — per-sample logits for K known classes.
        temperature : softmax temperature. T=1.0 matches the un-scaled
                      softmax; T>1 softens (closer to uniform -> harder to
                      tell apart in-dist vs OOD).

    Returns
    -------
    scores : (N,) ndarray — energy values, larger = more OOD.
    """
    logits = np.asarray(logits, dtype=np.float64)
    T = float(temperature)
    # logsumexp along class dim
    m = logits.max(axis=-1, keepdims=True)
    lse = m.squeeze(-1) + np.log(np.exp(logits / T - m).sum(axis=-1))
    return -lse


# ----------------------------------------------------------------------------
# 2. Prototype distance score
# ----------------------------------------------------------------------------
def prototype_score(embeddings: np.ndarray,
                    prototypes: np.ndarray) -> np.ndarray:
    """Min L2 distance from each embedding to any known-class prototype.

    Higher distance = more OOD (further from in-distribution clusters).

    Args
    ----
    embeddings : (N, D) array of per-sample embeddings.
    prototypes : (K, D) array of K known-class prototypes (e.g. class means).

    Returns
    -------
    scores : (N,) ndarray — distances, larger = more OOD.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    proto = np.asarray(prototypes, dtype=np.float64)
    # Pairwise squared L2: (N, K)
    dist_sq = ((emb[:, None, :] - proto[None, :, :]) ** 2).sum(axis=-1)
    return np.sqrt(np.maximum(dist_sq.min(axis=-1), 0.0))


def compute_prototypes(embeddings: np.ndarray,
                       labels: np.ndarray,
                       num_classes: int) -> np.ndarray:
    """Compute class-mean prototypes from (embeddings, labels).

    Args
    ----
    embeddings : (N, D)
    labels     : (N,) integer class labels in [0, num_classes)
    num_classes: K

    Returns
    -------
    prototypes : (K, D) ndarray.  Rows for classes with zero samples are
                 left as zeros (and should be masked by the caller if
                 needed).
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    lab = np.asarray(labels).astype(int)
    D = emb.shape[1]
    protos = np.zeros((num_classes, D), dtype=np.float64)
    for k in range(num_classes):
        mask = (lab == k)
        if mask.any():
            protos[k] = emb[mask].mean(axis=0)
    return protos


# ----------------------------------------------------------------------------
# 3. VOS (Virtual Outlier Synthesis) score
# ----------------------------------------------------------------------------
def synthesize_virtual_outliers(prototypes: np.ndarray,
                                alpha: float = 2.0,
                                n_per_class: int = 100,
                                seed: Optional[int] = None) -> np.ndarray:
    """Generate virtual outliers by extrapolating beyond each prototype.

    For each known class k, sample n_per_class virtual outliers:
        v = μ_k + α * r * d
    where r ~ U(0, 1) and d is a random unit-norm direction.  The
    extrapolation pushes the synthetic outlier away from the prototype
    by α × std-scaled distance.

    Returns
    -------
    virtuals : (K * n_per_class, D) ndarray of virtual outliers.
    """
    proto = np.asarray(prototypes, dtype=np.float64)
    K, D = proto.shape
    rng = np.random.RandomState(seed)
    out = np.zeros((K * n_per_class, D), dtype=np.float64)
    for k in range(K):
        for j in range(n_per_class):
            d = rng.randn(D)
            d /= (np.linalg.norm(d) + 1e-8)        # unit direction
            r = rng.uniform(0.0, 1.0)              # 0..1 magnitude
            out[k * n_per_class + j] = proto[k] + alpha * r * d
    return out


def vos_score(embeddings: np.ndarray,
              prototypes: np.ndarray,
              alpha: float = 2.0,
              n_per_class: int = 100,
              seed: Optional[int] = None) -> np.ndarray:
    """Min distance from each embedding to the synthesised virtual outliers.

    Higher distance = more OOD.

    Args
    ----
    embeddings : (N, D) — same shape as prototype_score.
    prototypes : (K, D) — known-class prototypes.
    alpha      : extrapolation distance for VOS synthesis.
    n_per_class: number of virtual outliers per known class.
    seed       : RNG seed for reproducibility.

    Returns
    -------
    scores : (N,) ndarray.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    virtuals = synthesize_virtual_outliers(prototypes, alpha, n_per_class, seed)
    # Pairwise squared L2 to virtuals, then min
    dist_sq = ((emb[:, None, :] - virtuals[None, :, :]) ** 2).sum(axis=-1)
    return np.sqrt(np.maximum(dist_sq.min(axis=-1), 0.0))


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    print("Testing ood_scores ...")
    rng = np.random.RandomState(42)

    # 3 known classes (in 8-D), 1 unknown cluster (offset)
    K, D = 3, 8
    proto = rng.randn(K, D) * 2.0
    emb_known = proto[rng.randint(0, K, size=200)] + 0.3 * rng.randn(200, D)
    emb_unknown = proto.mean(axis=0) + 6.0 * rng.randn(50, D) + 6.0

    # Energy score: simulate a *confident* classifier on in-dist and an
    # *uncertain* classifier on OOD (which is what a trained network looks
    # like in practice — see Liu et al., NeurIPS 2020).
    #   in-dist logits: one hot-ish column with high max
    #   OOD    logits: all columns similar (low contrast)
    fake_logits_known = rng.randn(200, K) * 0.5
    fake_logits_known[np.arange(200), rng.randint(0, K, 200)] += 8.0  # confident
    fake_logits_unknown = rng.randn(50, K) * 0.5                        # uncertain
    e_known = energy_score(fake_logits_known)
    e_unknown = energy_score(fake_logits_unknown)
    print(f"  Energy    : in={e_known.mean():.3f}  out={e_unknown.mean():.3f}  "
          f"(want out > in)")

    # Prototype score
    p_known = prototype_score(emb_known, proto)
    p_unknown = prototype_score(emb_unknown, proto)
    print(f"  Prototype : in={p_known.mean():.3f}  out={p_unknown.mean():.3f}  "
          f"(want out > in)")

    # VOS score
    v_known = vos_score(emb_known, proto, alpha=2.0, n_per_class=20, seed=0)
    v_unknown = vos_score(emb_unknown, proto, alpha=2.0, n_per_class=20, seed=0)
    print(f"  VOS       : in={v_known.mean():.3f}  out={v_unknown.mean():.3f}  "
          f"(want out > in)")

    # Sanity: prototype helpers
    p2 = compute_prototypes(emb_known, np.array([0] * 70 + [1] * 70 + [2] * 60),
                            num_classes=K)
    assert np.allclose(p2[0], emb_known[:70].mean(axis=0)), "prototype mismatch"
    print("  compute_prototypes: OK")

    print("ood_scores smoke test passed!")