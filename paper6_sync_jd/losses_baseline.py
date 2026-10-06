"""Paper 6 — vendored baseline training losses (self-contained).

Copied from paper5_task_oriented/losses.py on 2026-09-19:
_si_sdr_per_sample, assignment_table, variable_k_pit_loss.  The
paper-5-only branches (lambda_ser soft-SER, lambda_llr joint head) are NOT
vendored — they raise NotImplementedError (paper 6's baseline recipe is
lambda_ser=0, lambda_llr=0).
"""
from __future__ import annotations

import itertools
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


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
# Assignment tables: injective maps of K sources into k_slots slots
# ----------------------------------------------------------------------------
_ASSIGN_TABLE_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}


def assignment_table(k: int, k_slots: int) -> torch.Tensor:
    """All injective assignments of k sources into k_slots slots.

    Returns a LongTensor [P(k_slots, k), k] where row a, column j is the
    slot index that source j maps to.  Sizes at k_slots=4: k=1 -> 4,
    k=2 -> 12, k=3 -> 24.
    """
    key = (k, k_slots)
    if key not in _ASSIGN_TABLE_CACHE:
        _ASSIGN_TABLE_CACHE[key] = torch.tensor(
            list(itertools.permutations(range(k_slots), k)),
            dtype=torch.long,
        )
    return _ASSIGN_TABLE_CACHE[key]


# ----------------------------------------------------------------------------
# Variable-count PIT loss (slot model) — baseline recipe
# ----------------------------------------------------------------------------
def variable_k_pit_loss(slots: torch.Tensor,
                        occ_logits: torch.Tensor,
                        count_logits: torch.Tensor | None,
                        sources: torch.Tensor,
                        k: torch.Tensor,
                        lambda_occ: float = 1.0,
                        lambda_cnt: float = 0.1,
                        lambda_mse: float = 0.0,
                        lambda_ser: float = 0.0,
                        lambda_sep: float = 1.0,
                        lambda_llr: float = 0.0,
                        eps: float = 1e-8):
    """Variable-count PIT loss for SlotSepNet (paper-6 baseline subset).

    Supports exactly the paper-6 baseline recipe: SI-SDR PIT separation
    term (lambda_sep) + MSE anchor (lambda_mse) + occupancy BCE
    (lambda_occ) + count-head CE (lambda_cnt).  lambda_ser / lambda_llr
    (paper-5 task-oriented terms) are NOT vendored and raise if enabled.
    """
    if lambda_ser > 0 or lambda_llr > 0:
        raise NotImplementedError(
            "paper-5 task-oriented terms are not vendored into paper6")

    B, S, T = slots.shape
    device = slots.device
    src = sources.squeeze(2)                              # [B, S, T]

    per_sample_sep = torch.zeros(B, device=device)
    per_sample_occ = torch.zeros(B, device=device)
    per_sample_cnt = torch.zeros(B, device=device)
    per_sample_mse = torch.zeros(B, device=device)
    assigned = torch.full((B, S), -1, dtype=torch.long, device=device)

    # Group the batch by true K so each group uses its own assignment table.
    for k_val in torch.unique(k):
        kv = int(k_val.item())
        idx = (k == k_val).nonzero(as_tuple=True)[0]
        b = idx.numel()
        slots_g = slots[idx]                              # [b, S, T]
        src_g = src[idx, :kv]                             # [b, kv, T]

        # Pair matrix: SI-SDR between every slot and every true source.
        est = slots_g.unsqueeze(2).expand(b, S, kv, T).reshape(b * S * kv, T)
        ref = src_g.unsqueeze(1).expand(b, S, kv, T).reshape(b * S * kv, T)
        pair = _si_sdr_per_sample(est, ref, eps).view(b, S, kv)

        # Score every injective assignment and pick the best per sample.
        table = assignment_table(kv, S).to(device)        # [A, kv]
        src_idx = torch.arange(kv, device=device).expand(table.shape[0], kv)
        scores = pair[:, table, src_idx].sum(dim=-1)      # [b, A]
        best = scores.argmax(dim=1)                       # [b]
        chosen = table[best]                              # [b, kv]
        best_score = scores.gather(1, best.unsqueeze(1)).squeeze(1)  # [b]
        per_sample_sep[idx] = -best_score / kv
        assigned[idx, :kv] = chosen

        # MSE anchor on the assigned pairs only (same assignment as PIT).
        if lambda_mse > 0:
            b_idx = torch.arange(b, device=device).unsqueeze(1)
            est_pairs = slots_g[b_idx, chosen]            # [b, kv, T]
            per_sample_mse[idx] = (est_pairs - src_g).abs().pow(2).mean(
                dim=(-2, -1))

        # Occupancy BCE against the assignment indicator.
        occ_target = torch.zeros(b, S, device=device)
        occ_target.scatter_(1, chosen, 1.0)
        per_sample_occ[idx] = F.binary_cross_entropy_with_logits(
            occ_logits[idx], occ_target, reduction='none').mean(dim=-1)

        # Optional explicit count CE (classes are K-1 in {0,1,2}).
        if count_logits is not None and lambda_cnt > 0:
            cnt_target = torch.full((b,), kv - 1, dtype=torch.long,
                                    device=device)
            per_sample_cnt[idx] = F.cross_entropy(
                count_logits[idx], cnt_target, reduction='none')

    per_sample = (lambda_sep * per_sample_sep
                  + lambda_occ * per_sample_occ
                  + lambda_cnt * per_sample_cnt
                  + lambda_mse * per_sample_mse)
    loss = per_sample.mean()

    aux = {
        'per_sample_sep': per_sample_sep.detach(),
        'per_sample_occ': per_sample_occ.detach(),
        'per_sample_cnt': per_sample_cnt.detach(),
        'assigned': assigned.detach(),
        'si_sdr': -per_sample_sep.detach().mean(),
        'mse': per_sample_mse.detach().mean(),
        # kept for paper5-train.py logging compatibility
        'ser_soft': torch.zeros((), device=device),
        'llr': torch.zeros((), device=device),
    }
    return loss, aux


# ============================================================================
# Smoke test
# ============================================================================
if __name__ == '__main__':
    print("Testing paper6 losses_baseline ...")
    torch.manual_seed(0)
    B, S, T = 4, 4, 512
    slots = (torch.randn(B, S, T) + 1j * torch.randn(B, S, T)).to(torch.complex64)
    slots.requires_grad_(True)
    occ_logits = torch.randn(B, S)
    count_logits = torch.randn(B, 3)
    sources = (torch.randn(B, S, 1, T)
               + 1j * torch.randn(B, S, 1, T)).to(torch.complex64)
    k = torch.tensor([1, 2, 3, 2])
    loss, aux = variable_k_pit_loss(slots, occ_logits, count_logits,
                                    sources, k, lambda_mse=1.0)
    loss.backward()
    assert torch.isfinite(loss)
    print(f"  loss={float(loss):.4f}  si_sdr={float(aux['si_sdr']):.3f}  "
          f"backward OK")
    print("\nlosses_baseline smoke test passed!")
