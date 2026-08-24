"""
Paper 3 — Open-Set SC-BSS: Loss functions.

Provides:
  - pit_multi_task_loss: permutation-invariant joint SI-SDR + CE loss with
    per-sample best-permutation alignment. Used as the training objective.
  - si_sdr_loss: convenience wrapper returning -SI-SDR as a scalar.

The PIT loss aligns both the source assignment AND the modulation-label
assignment to the same permutation, so the per-source classification
head is trained against the modulation of whichever source ended up in
that slot after permutation.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# SI-SDR (per-sample tensor, complex-safe)
# ----------------------------------------------------------------------------
def _si_sdr_per_sample(estimate: torch.Tensor,
                       reference: torch.Tensor,
                       eps: float = 1e-8) -> torch.Tensor:
    """Scale-Invariant SDR, per-sample. Returns [B] tensor (higher is better).

    Handles complex inputs by flattening I/Q into the last dim, matching the
    convention used by paper1_cnn_se/utils.py:si_sdr.
    """
    if torch.is_complex(estimate):
        estimate = torch.view_as_real(estimate).flatten(-2)
        reference = torch.view_as_real(reference).flatten(-2)

    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)

    alpha = (reference * estimate).sum(dim=-1, keepdim=True) / (
        (reference ** 2).sum(dim=-1, keepdim=True) + eps
    )
    target = alpha * reference
    noise = estimate - target

    si_sdr_val = 10 * torch.log10(
        (target ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps) + eps
    )
    return si_sdr_val.reshape(si_sdr_val.shape[0], -1).mean(dim=-1)


# ----------------------------------------------------------------------------
# Multi-task PIT loss
# ----------------------------------------------------------------------------
def pit_multi_task_loss(s1_hat: torch.Tensor,
                        s2_hat: torch.Tensor,
                        logits1: torch.Tensor,
                        logits2: torch.Tensor,
                        src1: torch.Tensor,
                        src2: torch.Tensor,
                        mod1: torch.Tensor,
                        mod2: torch.Tensor,
                        alpha: float = 1.0,
                        eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
    """Per-sample permutation-invariant joint SI-SDR + CE loss.

    Returns
    -------
    loss          : scalar tensor — backward-able
    use_swap      : [B] bool tensor — True if permutation 2 was used
    permuted      : (s1_perm, s2_perm, logits1_perm, logits2_perm)
                    PIT-aligned predictions for downstream metrics
    """
    # ---- Separation loss: -SI-SDR (per-sample) for both permutations ----
    sdr_11 = _si_sdr_per_sample(s1_hat, src1, eps)   # [B]
    sdr_22 = _si_sdr_per_sample(s2_hat, src2, eps)
    sdr_12 = _si_sdr_per_sample(s1_hat, src2, eps)
    sdr_21 = _si_sdr_per_sample(s2_hat, src1, eps)
    sep_loss_1 = -(sdr_11 + sdr_22) / 2.0
    sep_loss_2 = -(sdr_12 + sdr_21) / 2.0

    # ---- Classification loss: CE for both permutations ----
    ce_1_1 = F.cross_entropy(logits1, mod1, reduction='none')
    ce_2_2 = F.cross_entropy(logits2, mod2, reduction='none')
    ce_1_2 = F.cross_entropy(logits1, mod2, reduction='none')
    ce_2_1 = F.cross_entropy(logits2, mod1, reduction='none')
    cls_loss_1 = (ce_1_1 + ce_2_2) / 2.0
    cls_loss_2 = (ce_1_2 + ce_2_1) / 2.0

    # ---- Per-sample joint loss ----
    loss_1 = sep_loss_1 + alpha * cls_loss_1     # [B]
    loss_2 = sep_loss_2 + alpha * cls_loss_2     # [B]

    use_swap = loss_2 < loss_1                   # [B] bool

    # ---- Permute predictions to match the chosen permutation ----
    swap_b1 = use_swap.view(-1, 1, 1)            # broadcast over (1, T)
    swap_b2 = use_swap.view(-1, 1)               # broadcast over K
    s1_perm = torch.where(swap_b1, s2_hat, s1_hat)
    s2_perm = torch.where(swap_b1, s1_hat, s2_hat)
    logits1_perm = torch.where(swap_b2, logits2, logits1)
    logits2_perm = torch.where(swap_b2, logits1, logits2)

    # Final scalar loss = mean over batch after per-sample PIT
    per_sample = torch.where(use_swap, loss_2, loss_1)
    loss = per_sample.mean()

    return loss, use_swap, (s1_perm, s2_perm, logits1_perm, logits2_perm)


# ----------------------------------------------------------------------------
# Convenience wrappers
# ----------------------------------------------------------------------------
def si_sdr_loss(s_hat: torch.Tensor, s_ref: torch.Tensor) -> torch.Tensor:
    """Scalar -SI-SDR (averaged over batch)."""
    return -_si_sdr_per_sample(s_hat, s_ref).mean()


# ----------------------------------------------------------------------------
# Center Loss (Wen et al. 2016)
# ----------------------------------------------------------------------------
class CenterLoss(nn.Module):
    """Center loss: penalize per-sample L2 distance from a learnable
    per-class centre vector.

    L_c = 0.5 * lambda_c * mean_i || x_i - c_{y_i} ||^2

    Centres are parameters in the optimizer (same as Wen et al.) so the
    pull towards them is the only force competing with the classification
    CE loss.  Initialized to small random values; after a few warm-up
    batches they stabilize.
    """

    def __init__(self, num_classes: int, feat_dim: int, lambda_c: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c
        # Small random init so the first batch has non-trivial loss
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.1)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """embeddings: [B, D], labels: [B] long."""
        centers_batch = self.centers[labels]                  # [B, D]
        dist = ((embeddings - centers_batch) ** 2).sum(dim=-1)  # [B]
        return 0.5 * self.lambda_c * dist.mean()


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    print("Testing pit_multi_task_loss ...")
    torch.manual_seed(0)

    B, T = 4, 4096
    K = 4
    # Random complex signals
    s1_hat = torch.randn(B, 1, T, dtype=torch.complex64)
    s2_hat = torch.randn(B, 1, T, dtype=torch.complex64)
    src1   = torch.randn(B, 1, T, dtype=torch.complex64)
    src2   = torch.randn(B, 1, T, dtype=torch.complex64)
    logits1 = torch.randn(B, K)
    logits2 = torch.randn(B, K)
    mod1 = torch.randint(0, K, (B,))
    mod2 = torch.randint(0, K, (B,))

    loss, use_swap, perm = pit_multi_task_loss(
        s1_hat, s2_hat, logits1, logits2,
        src1, src2, mod1, mod2, alpha=1.0,
    )
    print(f"  loss = {loss.item():.4f}  (requires_grad={loss.requires_grad})")
    print(f"  use_swap = {use_swap.tolist()}  ({use_swap.float().mean().item():.0%} swapped)")
    s1p, s2p, l1p, l2p = perm
    print(f"  permuted shapes: s1={tuple(s1p.shape)}  logits1={tuple(l1p.shape)}")

    # Backward sanity
    loss.backward()
    print("  backward() succeeded")

    print("pit_multi_task_loss smoke test passed!")