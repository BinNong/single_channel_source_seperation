"""
Paper 3 — Open-Set SC-BSS: Evaluation metrics for open-set detection.

Standard open-set recognition metrics, all returning a scalar in [0, 1]:

  - auroc       : Area under ROC curve.  Treats in-dist as positive.
  - aupr_in     : Area under PR curve, in-dist as positive.
  - fpr_at_95_tpr: False positive rate of OOD samples when TPR=95%.
  - oscr        : Open-Set Classification Rate (area under the OSCR curve).

The OSCR curve (Dhamija et al., 2018) measures the trade-off between
correct classification of known samples and correct rejection of unknown
samples at various confidence thresholds.  Higher OSCR is better; the
random baseline is the closed-set accuracy * (1 - prior_ood).
"""

from __future__ import annotations

import numpy as np

# NumPy 2.0 removed `np.trapz` in favour of `np.trapezoid`.  Detect once.
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
if _trapz is None:
    raise ImportError("NumPy version has neither trapezoid nor trapz.")


# ----------------------------------------------------------------------------
# AUROC
# ----------------------------------------------------------------------------
def auroc(scores_known: np.ndarray, scores_unknown: np.ndarray) -> float:
    """Area under the ROC curve for OOD detection.

    Convention: higher score = more OOD.  Higher AUROC is better.
    Uses the Mann-Whitney U formulation:

        AUROC = P(score_unknown > score_known)

    for a randomly drawn (unknown, known) pair (with 0.5 weight on ties).
    """
    s_known = np.asarray(scores_known, dtype=np.float64).ravel()
    s_unknown = np.asarray(scores_unknown, dtype=np.float64).ravel()
    n_pos = len(s_unknown)
    n_neg = len(s_known)
    if n_pos == 0 or n_neg == 0:
        return float('nan')

    # All pairwise comparisons
    diff = s_unknown[:, None] - s_known[None, :]    # (n_pos, n_neg)
    wins = (diff > 0).sum()
    ties = (diff == 0).sum()
    return float((wins + 0.5 * ties) / (n_pos * n_neg))


# ----------------------------------------------------------------------------
# AUPR (in-dist positive)
# ----------------------------------------------------------------------------
def aupr_in(scores_known: np.ndarray, scores_unknown: np.ndarray) -> float:
    """AUPR with in-dist as positive class."""
    s_known = np.asarray(scores_known, dtype=np.float64).ravel()
    s_unknown = np.asarray(scores_unknown, dtype=np.float64).ravel()
    if len(s_known) == 0 or len(s_unknown) == 0:
        return float('nan')

    y = np.concatenate([np.ones(len(s_known)), np.zeros(len(s_unknown))])
    s = np.concatenate([s_known, s_unknown])

    # Sort by ascending score
    order = np.argsort(s, kind='mergesort')
    y_sorted = y[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    recalls = tp / max(tp[-1], 1)        # = tp / P
    precisions = tp / np.maximum(tp + fp, 1)

    # Append endpoints and integrate via trapezoidal rule on recall
    recalls = np.concatenate(([0.0], recalls))
    precisions = np.concatenate(([1.0], precisions))
    return float(_trapz(precisions, recalls))


# ----------------------------------------------------------------------------
# FPR @ 95% TPR
# ----------------------------------------------------------------------------
def fpr_at_95_tpr(scores_known: np.ndarray, scores_unknown: np.ndarray) -> float:
    """False positive rate of OOD samples when TPR (of OOD) = 95%.

    Lower is better.
    """
    s_known = np.asarray(scores_known, dtype=np.float64).ravel()
    s_unknown = np.asarray(scores_unknown, dtype=np.float64).ravel()
    if len(s_unknown) == 0 or len(s_known) == 0:
        return float('nan')

    # Threshold at the 5th percentile of OOD scores (so 95% of OOD have higher
    # score -> classified as OOD).  FPR = P(score > t | known).
    threshold = np.percentile(s_unknown, 5.0)
    return float((s_known > threshold).mean())


# ----------------------------------------------------------------------------
# OSCR (Open-Set Classification Rate)
# ----------------------------------------------------------------------------
def oscr(correct_known_mask: np.ndarray,
         scores_known: np.ndarray,
         scores_unknown: np.ndarray) -> float:
    """Open-Set Classification Rate (area under OSCR curve).

    Parameters
    ----------
    correct_known_mask : (N_known,) bool — True for samples whose closed-set
                        top-1 prediction was correct.
    scores_known       : (N_known,) — OOD score for known samples.
    scores_unknown     : (N_unknown,) — OOD score for unknown samples.

    The OSCR curve is (CCR vs FPR) at various thresholds of the OOD score.
    CCR(threshold) = fraction of known samples that are BOTH correctly
    classified AND below the OOD threshold.
    FPR(threshold) = fraction of unknown samples that are below the threshold
    (falsely accepted as known).
    The curve is integrated over FPR in [0, 1]; the area is the OSCR.

    Higher is better; random baseline = closed-set accuracy * (1 - prior_ood).
    """
    correct = np.asarray(correct_known_mask, dtype=bool).ravel()
    s_known = np.asarray(scores_known, dtype=np.float64).ravel()
    s_unknown = np.asarray(scores_unknown, dtype=np.float64).ravel()
    if len(correct) != len(s_known) or len(s_unknown) == 0:
        return float('nan')
    if len(correct) == 0:
        return float('nan')

    n_unknown = len(s_unknown)
    n_known = len(s_known)

    # Convention: higher score = more OOD.  At threshold t, a sample is
    # *flagged as OOD* iff score > t.  The OSCR curve sweeps over t.
    # Candidate thresholds: every distinct score plus +/- inf endpoints.
    thresholds = np.concatenate([
        [-np.inf],
        np.unique(np.concatenate([s_known, s_unknown])),
        [np.inf],
    ])

    fprs = []
    ccrs = []
    for t in thresholds:
        # FPR (for OSCR): fraction of UNKNOWN samples with score <= t
        # (i.e., NOT flagged as OOD = falsely accepted as known).
        fpr = float((s_unknown <= t).sum() / n_unknown)
        # CCR: fraction of KNOWN samples that are BOTH correctly classified
        # AND NOT flagged as OOD (i.e., score <= t).
        kept_mask = s_known <= t
        ccr = float((correct & kept_mask).sum() / n_known)
        fprs.append(fpr)
        ccrs.append(ccr)

    # Sort by FPR ascending for the trapezoidal integration
    order = np.argsort(fprs)
    fprs = np.array(fprs)[order]
    ccrs = np.array(ccrs)[order]

    # The curve runs from (FPR=0, CCR=closed_acc) to (FPR=1, CCR=0).
    closed_acc = float(correct.mean())
    fprs = np.concatenate(([0.0], fprs, [1.0]))
    ccrs = np.concatenate(([closed_acc], ccrs, [0.0]))
    return float(_trapz(ccrs, fprs))


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    print("Testing open_set_metrics ...")
    rng = np.random.RandomState(42)

    # Known and unknown scores: known scores have lower mean (in-dist),
    # unknown scores have higher mean (OOD).  Both have some overlap.
    scores_known = rng.normal(loc=0.0, scale=1.0, size=500)
    scores_unknown = rng.normal(loc=2.0, scale=1.0, size=200)

    print(f"  AUROC         = {auroc(scores_known, scores_unknown):.3f}  (want close to 1.0)")
    print(f"  AUPR (in+)    = {aupr_in(scores_known, scores_unknown):.3f}")
    print(f"  FPR @ 95% TPR = {fpr_at_95_tpr(scores_known, scores_unknown):.3f}  (want low)")

    correct_known = rng.rand(500) > 0.2      # ~80% closed-set accuracy
    print(f"  OSCR          = {oscr(correct_known, scores_known, scores_unknown):.3f}")

    # Edge case: identical distributions
    s2 = rng.normal(loc=0.0, scale=1.0, size=200)
    print(f"  AUROC (random) = {auroc(s2, s2):.3f}  (want close to 0.5)")
    print(f"  OSCR (random)  = {oscr(correct_known, s2, s2):.3f}")

    print("open_set_metrics smoke test passed!")