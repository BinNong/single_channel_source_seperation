"""
Paper 5 — Task-Oriented SC-BSS: Loss functions.

Paper4 base (variable-count PIT + occupancy BCE + count CE + MSE anchor)
PLUS the paper-5 addition: `variable_k_pit_loss` accepts `lambda_ser`
(default 0.0), a differentiable soft-demodulation term (soft_ser_pairs
from soft_demod.py) computed on the PIT-ASSIGNED pairs inside the same
K-grouped loop — reusing `chosen`, so no second assignment enumeration
(plan v2 O1).  0.0 disables it and is byte-identical to paper4 behavior;
`aux['ser_soft']` always reports the term (0.0 when disabled).

        loss = mean_k[ -SI-SDR(best assignment) ]        # separation
             + lambda_occ * BCE(occ_logits, assignment indicator)
             + lambda_cnt * CE(count_logits, K - 1)      # optional
             + lambda_mse * MSE(assigned pairs)          # optional anchor
             + lambda_ser * soft-SER(assigned pairs)     # paper 5

Provides:
  - variable_k_pit_loss : slot model (the only arch wired for lambda_ser).
  - recursive_pit_loss  : OneAndRestNet (unchanged from paper4).
  - specialist_loss     : SpecialistBankNet (unchanged from paper4).

`_si_sdr_per_sample` is copied verbatim from paper3_open_set/losses.py
(complex-safe, per-sample) so paper5 stays self-contained.
"""

from __future__ import annotations

import itertools
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from soft_demod import soft_ser_pairs, ref_hard_labels, CONST_MASK


# ----------------------------------------------------------------------------
# SI-SDR (per-sample tensor, complex-safe) — copied from paper3 losses.py
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
# Assignment tables: injective maps of K sources into k_slots slots
# ----------------------------------------------------------------------------
_ASSIGN_TABLE_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}


def assignment_table(k: int, k_slots: int) -> torch.Tensor:
    """All injective assignments of k sources into k_slots slots.

    Returns a LongTensor [P(k_slots, k), k] where row a, column j is the
    slot index that source j maps to.  Sizes at k_slots=4: k=1 -> 4,
    k=2 -> 12, k=3 -> 24 (spec §6.1).
    """
    key = (k, k_slots)
    if key not in _ASSIGN_TABLE_CACHE:
        _ASSIGN_TABLE_CACHE[key] = torch.tensor(
            list(itertools.permutations(range(k_slots), k)),
            dtype=torch.long,
        )
    return _ASSIGN_TABLE_CACHE[key]


# ----------------------------------------------------------------------------
# Variable-count PIT loss (slot model)
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
                        sym_logits: torch.Tensor | None = None,
                        ser_sigma2: float = 0.1,
                        carriers: torch.Tensor | None = None,
                        mods: torch.Tensor | None = None,
                        sample_rate: int = 16000,
                        n_symbols: int = 256,
                        roll_off: float = 0.35,
                        num_taps: int = 64,
                        eps: float = 1e-8):
    """Variable-count PIT loss for SlotSepNet.

    Args:
        slots        : [B, S, T] complex — RAW (ungated) slot outputs
        occ_logits   : [B, S]
        count_logits : [B, 3] (classes K=1,2,3) or None
        sources      : [B, S, 1, T] complex, zero-padded beyond the first K
        k            : [B] long — true source count per sample
        lambda_occ   : weight on the occupancy BCE term
        lambda_cnt   : weight on the count CE term (ignored when no head)
        lambda_mse   : weight on the MSE anchor term (per ASSIGNED pair,
                       same assignment the SI-SDR PIT chose).  The pure
                       -SI-SDR term is scale-invariant, so without this
                       anchor the output magnitude drifts and plain
                       SDR/SIR read depressed (paper1's combined loss was
                       0.5*MSE + 0.5*(-SI-SDR)).  0.0 disables it and is
                       byte-identical to the pre-anchor behavior.
        lambda_ser   : weight on the soft-demodulation term (paper 5) —
                       differentiable soft-SER (soft_ser_pairs) per
                       ASSIGNED pair, reusing the PIT assignment (`chosen`)
                       already computed above, so no second enumeration.
                       Requires `carriers` and `mods`.  0.0 disables it
                       (byte-identical to paper4 behavior).
        lambda_sep   : weight on the -SI-SDR separation term itself.
                       1.0 = paper4 behavior.  0.0 gives the PURE
                       task-oriented configuration (e): the PIT assignment
                       still uses the SI-SDR pair matrix, but the waveform
                       term contributes no gradient.
        lambda_llr   : weight on the JOINT soft-information term (paper 5,
                       S3): cross-entropy of the JointLLRHead's per-symbol
                       logits (masked to the true constellation) against
                       the reference hard labels (ref_hard_labels), on the
                       PIT-ASSIGNED slots only.  Requires `sym_logits`,
                       `carriers`, `mods`.  0.0 disables it.
        sym_logits   : [B, S, N_sym, 16] — JointLLRHead outputs
        ser_sigma2   : Gaussian softness of the soft log-likelihood on the
                       unit-power grid (TaskConfig.ser_sigma2)
        carriers     : [B, S] float — TRUE per-source carriers incl. jitter
                       (NaN padded), from the dataset's return_carriers
        mods         : [B, S] long — per-source modulation index, -1 padded

    Returns
    -------
    loss : scalar tensor (backward-able)
    aux  : dict with detached tensors for logging —
           'per_sample_sep' [B]  (separation term, = -mean SI-SDR)
           'per_sample_occ' [B], 'per_sample_cnt' [B],
           'assigned'       [B, S] long, slot indices per source, -1 padded
           'si_sdr'         scalar (mean SI-SDR over assigned pairs; kept
                            PURE — the MSE anchor never enters it)
           'mse'            scalar (mean MSE over assigned pairs)
           'ser_soft'       scalar (mean soft-SER CE over assigned pairs)
    """
    B, S, T = slots.shape
    device = slots.device
    src = sources.squeeze(2)                              # [B, S, T]

    if lambda_ser > 0:
        assert carriers is not None and mods is not None, \
            'lambda_ser requires dataset return_carriers=True (mods+carriers)'
    if lambda_llr > 0:
        assert sym_logits is not None, \
            'lambda_llr requires a model with use_joint_head=True'
        assert carriers is not None and mods is not None, \
            'lambda_llr requires dataset return_carriers=True (mods+carriers)'

    per_sample_sep = torch.zeros(B, device=device)
    per_sample_occ = torch.zeros(B, device=device)
    per_sample_cnt = torch.zeros(B, device=device)
    per_sample_mse = torch.zeros(B, device=device)
    per_sample_ser = torch.zeros(B, device=device)
    per_sample_llr = torch.zeros(B, device=device)
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

        # Soft-demodulation term (paper 5): differentiable soft-SER on the
        # assigned pairs, reusing THIS PIT assignment (no second pass).
        if lambda_ser > 0:
            b_idx = torch.arange(b, device=device).unsqueeze(1)
            est_pairs = slots_g[b_idx, chosen]            # [b, kv, T]
            car_g = carriers[idx, :kv].to(device).float().reshape(-1)
            mod_g = mods[idx, :kv].to(device).reshape(-1)
            ce = soft_ser_pairs(
                est_pairs.reshape(b * kv, T),
                src_g.reshape(b * kv, T),
                car_g, mod_g,
                sigma2=ser_sigma2, sample_rate=sample_rate,
                n_symbols=n_symbols, roll_off=roll_off, num_taps=num_taps)
            per_sample_ser[idx] = ce.view(b, kv).mean(dim=-1)

        # Joint soft-information term (paper 5, S3): CE of the JointLLRHead
        # symbol logits (masked to the true constellation) against the
        # reference hard labels, on the assigned slots only.
        if lambda_llr > 0:
            b_idx = torch.arange(b, device=device).unsqueeze(1)
            lg_pairs = sym_logits[idx][b_idx, chosen]     # [b, kv, N, 16]
            n_sym = lg_pairs.shape[2]
            labels = ref_hard_labels(
                src_g.reshape(b * kv, T),
                carriers[idx, :kv].to(device).float().reshape(-1),
                mods[idx, :kv].to(device).reshape(-1),
                sample_rate=sample_rate, n_symbols=n_symbols,
                roll_off=roll_off, num_taps=num_taps)     # [b*kv, N]
            vmask = CONST_MASK.to(device)[
                mods[idx, :kv].to(device).reshape(-1)]    # [b*kv, 16]
            lg_flat = lg_pairs.reshape(b * kv * n_sym, 16).masked_fill(
                ~vmask.unsqueeze(1).expand(b * kv, n_sym, 16)
                 .reshape(b * kv * n_sym, 16), -1e9)
            ce_llr = F.cross_entropy(lg_flat, labels.reshape(-1),
                                     reduction='none')
            per_sample_llr[idx] = ce_llr.view(b, kv, n_sym).mean(dim=(-2, -1))

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
                  + lambda_mse * per_sample_mse
                  + lambda_ser * per_sample_ser
                  + lambda_llr * per_sample_llr)
    loss = per_sample.mean()

    aux = {
        'per_sample_sep': per_sample_sep.detach(),
        'per_sample_occ': per_sample_occ.detach(),
        'per_sample_cnt': per_sample_cnt.detach(),
        'assigned': assigned.detach(),
        'si_sdr': -per_sample_sep.detach().mean(),
        'mse': per_sample_mse.detach().mean(),
        'ser_soft': per_sample_ser.detach().mean(),
        'llr': per_sample_llr.detach().mean(),
    }
    return loss, aux


# ----------------------------------------------------------------------------
# Specialist-bank loss (ablation A6, EDSNet-style)
# ----------------------------------------------------------------------------
def specialist_loss(slots: torch.Tensor,
                    count_logits: torch.Tensor,
                    sources: torch.Tensor,
                    occ_mask: torch.Tensor,
                    k: torch.Tensor,
                    lambda_cnt: float = 0.1,
                    lambda_mse: float = 0.0,
                    eps: float = 1e-8):
    """Joint count-CE + uPIT loss for SpecialistBankNet.

    Standard PIT over the k! permutations WITHIN the routed specialist's
    k slots: slots[b, :k_b] are matched against the k_b true sources using
    the square assignment table assignment_table(kv, kv) (1 / 2 / 6
    permutations for K = 1 / 2 / 3).  Unlike the slot model there is no
    occupancy term — the count head is the ONLY counting mechanism, which
    is exactly what ablation A6 isolates.

    Args:
        slots        : [B, S, T] complex — teacher-routed, zero-padded
                       beyond each sample's routed K
        count_logits : [B, 3] (classes K=1,2,3)
        sources      : [B, S, 1, T] complex, zero-padded beyond the first K
        occ_mask     : [B, S] — accepted for call-site uniformity with the
                       slot loss; `k` is authoritative and this is unused
        k            : [B] long — true source count per sample
        lambda_cnt   : weight on the count CE term
        lambda_mse   : weight on the MSE anchor term (per ASSIGNED pair);
                       0.0 disables it (byte-identical to pre-anchor
                       behavior)

    Returns
    -------
    loss : scalar tensor (backward-able)
    aux  : dict — 'per_sample_sep' [B], 'per_sample_cnt' [B],
           'assigned' [B, S] long (-1 padded), 'si_sdr' scalar (PURE),
           'mse' scalar
    """
    B, S, T = slots.shape
    device = slots.device
    src = sources.squeeze(2)                              # [B, S, T]

    per_sample_sep = torch.zeros(B, device=device)
    per_sample_cnt = torch.zeros(B, device=device)
    per_sample_mse = torch.zeros(B, device=device)
    assigned = torch.full((B, S), -1, dtype=torch.long, device=device)

    # Group the batch by true K (the routing was by true K in training).
    for k_val in torch.unique(k):
        kv = int(k_val.item())
        idx = (k == k_val).nonzero(as_tuple=True)[0]
        b = idx.numel()
        slots_g = slots[idx, :kv]                         # [b, kv, T]
        src_g = src[idx, :kv]                             # [b, kv, T]

        # Square pair matrix over the specialist's kv slots.
        est = slots_g.unsqueeze(2).expand(b, kv, kv, T).reshape(b * kv * kv, T)
        ref = src_g.unsqueeze(1).expand(b, kv, kv, T).reshape(b * kv * kv, T)
        pair = _si_sdr_per_sample(est, ref, eps).view(b, kv, kv)

        # k! permutations within the specialist's slots.
        table = assignment_table(kv, kv).to(device)       # [kv!, kv]
        src_idx = torch.arange(kv, device=device).expand(table.shape[0], kv)
        scores = pair[:, table, src_idx].sum(dim=-1)      # [b, kv!]
        best = scores.argmax(dim=1)                       # [b]
        chosen = table[best]                              # [b, kv]
        best_score = scores.gather(1, best.unsqueeze(1)).squeeze(1)
        per_sample_sep[idx] = -best_score / kv
        assigned[idx, :kv] = chosen

        # MSE anchor on the assigned pairs only (same assignment as PIT).
        if lambda_mse > 0:
            b_idx = torch.arange(b, device=device).unsqueeze(1)
            est_pairs = slots_g[b_idx, chosen]            # [b, kv, T]
            per_sample_mse[idx] = (est_pairs - src_g).abs().pow(2).mean(
                dim=(-2, -1))

        if lambda_cnt > 0:
            cnt_target = torch.full((b,), kv - 1, dtype=torch.long,
                                    device=device)
            per_sample_cnt[idx] = F.cross_entropy(
                count_logits[idx], cnt_target, reduction='none')

    per_sample = (per_sample_sep
                  + lambda_cnt * per_sample_cnt
                  + lambda_mse * per_sample_mse)
    loss = per_sample.mean()

    aux = {
        'per_sample_sep': per_sample_sep.detach(),
        'per_sample_cnt': per_sample_cnt.detach(),
        'assigned': assigned.detach(),
        'si_sdr': -per_sample_sep.detach().mean(),
        'mse': per_sample_mse.detach().mean(),
    }
    return loss, aux


# ----------------------------------------------------------------------------
# Recursive one-and-rest loss
# ----------------------------------------------------------------------------
def recursive_pit_loss(model,
                       mixture: torch.Tensor,
                       sources: torch.Tensor,
                       k: torch.Tensor,
                       lambda_stop: float = 1.0,
                       lambda_mse: float = 0.0,
                       eps: float = 1e-8):
    """Teacher-forced recursive loss for OneAndRestNet.

    Unrolls exactly K_max (= S) steps.  At step i:

      - input: the TRUE remainder (teacher forcing) = mixture minus the
        sources assigned at earlier steps (noise stays in);
      - "one" output: PIT-matched to the best REMAINING ground-truth source
        (SI-SDR over the not-yet-assigned set, argmax) — skipped when no
        source remains (i >= K);
      - "rest" target: remainder minus the source assigned at THIS step,
        i.e. the remaining sources plus the noise (skipped when i >= K);
      - stop head: BCE against "more sources remain after this step",
        target = 1 iff (i + 1) < K — this is the ONLY term for steps
        beyond K.

    All steps are weighted equally.  `model` must expose
    `forward_step(remainder) -> (one, rest, stop_logit)`.

    Args:
        mixture    : [B, 1, T] complex
        sources    : [B, S, 1, T] complex, zero-padded beyond the first K
        k          : [B] long
        lambda_stop: weight on the stop-head BCE
        lambda_mse : weight on the MSE anchor — on "one" vs its assigned
                     source AND on "rest" vs its rest-target (active
                     samples only); 0.0 disables it (byte-identical to
                     pre-anchor behavior)

    Returns
    -------
    loss : scalar tensor (backward-able)
    aux  : dict — 'per_sample' [B], 'si_sdr' scalar (mean SI-SDR of "one"
           vs assigned source over all extraction steps; kept PURE),
           'step_si_sdr' list of per-step means (None where no sample was
           active), 'stop_acc' scalar, 'mse' scalar (mean over active
           extraction steps)
    """
    B, S, _, T = sources.shape
    device = mixture.device
    src = sources.squeeze(2)                              # [B, S, T]

    # Remaining-source mask: True for real, not-yet-assigned sources.
    remaining = (torch.arange(S, device=device).unsqueeze(0)
                 < k.unsqueeze(1))                        # [B, S] bool
    remainder = mixture

    per_sample = torch.zeros(B, device=device)
    step_si_sum = torch.zeros(S, device=device)
    step_si_cnt = torch.zeros(S, device=device)
    stop_correct = 0
    mse_sum = 0.0
    mse_cnt = 0

    for i in range(S):
        one, rest, stop_logit = model.forward_step(remainder)
        # one/rest: [B, 1, T]; stop_logit: [B]

        stop_target = (i + 1 < k).float()                 # [B]
        stop_bce = F.binary_cross_entropy_with_logits(
            stop_logit, stop_target, reduction='none')    # [B]
        stop_correct += int(((stop_logit > 0).float() == stop_target)
                            .sum().item())
        step_loss = lambda_stop * stop_bce

        active = (i < k)                                  # [B] bool
        if active.any():
            a_idx = active.nonzero(as_tuple=True)[0]
            b = a_idx.numel()
            one_a = one[a_idx].squeeze(1)                 # [b, T]
            src_a = src[a_idx]                            # [b, S, T]

            # PIT of "one" over the remaining ground-truth sources.
            est = one_a.unsqueeze(1).expand(b, S, T).reshape(b * S, T)
            ref = src_a.reshape(b * S, T)
            si = _si_sdr_per_sample(est, ref, eps).view(b, S)
            si = si.masked_fill(~remaining[a_idx], -1e9)
            best_src = si.argmax(dim=1)                   # [b]
            best_si = si.gather(1, best_src.unsqueeze(1)).squeeze(1)

            # "rest" target: remainder minus this step's source (noise kept).
            best_wave = src_a.gather(
                1, best_src.view(b, 1, 1).expand(b, 1, T))     # [b, 1, T]
            rest_target = remainder[a_idx] - best_wave
            rest_si = _si_sdr_per_sample(rest[a_idx], rest_target, eps)

            sep_rest = torch.zeros(B, device=device)
            sep_rest[a_idx] = -best_si - rest_si
            step_loss = step_loss + sep_rest

            # MSE anchor: "one" vs its assigned source AND "rest" vs its
            # rest-target (active samples only).
            if lambda_mse > 0:
                mse_full = torch.zeros(B, device=device)
                mse_full[a_idx] = (
                    (one[a_idx] - best_wave).abs().pow(2).mean(dim=(-2, -1))
                    + (rest[a_idx] - rest_target).abs().pow(2).mean(dim=(-2, -1))
                )
                step_loss = step_loss + lambda_mse * mse_full
                mse_sum += float(mse_full.detach().sum())
                mse_cnt += b

            step_si_sum[i] += best_si.detach().sum()
            step_si_cnt[i] += b

            # Teacher forcing: mark assigned, feed the TRUE remainder next.
            rem_a = remaining[a_idx]
            rem_a[torch.arange(b, device=device), best_src] = False
            remaining[a_idx] = rem_a
            remainder = remainder.clone()
            remainder[a_idx] = rest_target

        per_sample = per_sample + step_loss

    per_sample = per_sample / S                           # equal weight per step
    loss = per_sample.mean()

    step_si_sdr = [
        (step_si_sum[i] / step_si_cnt[i]).item() if step_si_cnt[i] > 0 else None
        for i in range(S)
    ]
    n_extr = step_si_cnt.sum()
    aux = {
        'per_sample': per_sample.detach(),
        'si_sdr': (step_si_sum.sum() / n_extr) if n_extr > 0
                  else torch.tensor(0.0),
        'step_si_sdr': step_si_sdr,
        'stop_acc': stop_correct / (B * S),
        'mse': mse_sum / max(mse_cnt, 1),
    }
    return loss, aux


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    print("Testing paper4 losses ...")
    torch.manual_seed(0)

    B, S, T = 4, 4, 512
    slots = torch.randn(B, S, T, dtype=torch.complex64).requires_grad_(True)
    occ_logits = torch.randn(B, S, requires_grad=True)
    count_logits = torch.randn(B, 3, requires_grad=True)
    sources = torch.randn(B, S, 1, T, dtype=torch.complex64)
    k = torch.tensor([1, 2, 3, 2])
    # Zero-pad sources beyond each sample's K
    for b in range(B):
        sources[b, k[b]:] = 0

    print("\n-- variable_k_pit_loss (random tensors) --")
    loss, aux = variable_k_pit_loss(slots, occ_logits, count_logits,
                                    sources, k)
    print(f"  loss={loss.item():.4f}  si_sdr={aux['si_sdr'].item():.3f} dB")
    print(f"  assigned=\n{aux['assigned'].numpy()}")
    for b in range(B):
        kv = int(k[b])
        row = aux['assigned'][b, :kv].tolist()
        assert len(set(row)) == kv, "assignment must be injective"
        assert all(0 <= s < S for s in row)
        assert (aux['assigned'][b, kv:] == -1).all()
    loss.backward()
    print("  backward() succeeded")

    print("\n-- assignment-table sizes (spec: 4 / 12 / 24 at K_max=4) --")
    for kv, n in ((1, 4), (2, 12), (3, 24)):
        tab = assignment_table(kv, S)
        print(f"  K={kv}: {tuple(tab.shape)}")
        assert tab.shape == (n, kv)

    print("\n-- perfect-assignment sanity --")
    # slots = the true sources permuted + an unused noise slot
    perm = torch.tensor([2, 0, 3, 1])
    slots_perfect = torch.zeros(1, S, T, dtype=torch.complex64)
    src1 = torch.randn(1, 3, T, dtype=torch.complex64)
    for j in range(3):
        slots_perfect[0, perm[j]] = src1[0, j]
    sources1 = torch.zeros(1, S, 1, T, dtype=torch.complex64)
    sources1[0, :3, 0] = src1
    loss_p, aux_p = variable_k_pit_loss(
        slots_perfect, torch.zeros(1, S), None, sources1,
        torch.tensor([3]), lambda_occ=0.0)
    print(f"  si_sdr={aux_p['si_sdr'].item():.1f} dB  "
          f"assigned={aux_p['assigned'][0].tolist()} (expect {perm.tolist()} + -1)")
    assert aux_p['si_sdr'].item() > 40.0
    assert aux_p['assigned'][0, :3].tolist() == perm[:3].tolist()

    print("\n-- recursive_pit_loss (random tensors, toy model) --")
    from models import OneAndRestNet
    torch.manual_seed(1)
    oar = OneAndRestNet(hidden_channels=8, n_layers=1, k_max=S)
    mixture = torch.randn(B, 1, T, dtype=torch.complex64)
    loss_r, aux_r = recursive_pit_loss(oar, mixture, sources, k)
    print(f"  loss={loss_r.item():.4f}  si_sdr={aux_r['si_sdr'].item():.3f} dB  "
          f"stop_acc={aux_r['stop_acc']:.2f}")
    print(f"  step_si_sdr={[None if v is None else round(v, 2) for v in aux_r['step_si_sdr']]}")
    loss_r.backward()
    print("  backward() succeeded")

    # Stop-target check: with K=2 and S=4, targets over steps are 1,0,0,0.
    k2 = torch.full((B,), 2)
    _, aux2 = recursive_pit_loss(oar, mixture, sources, k2)
    print(f"  K=2 stop_acc on random head: {aux2['stop_acc']:.2f} "
          f"(~0.5 expected)")

    print("\n-- specialist_loss (random tensors + perfect assignment) --")
    torch.manual_seed(2)
    sp_slots = torch.randn(B, S, T, dtype=torch.complex64).requires_grad_(True)
    sp_cnt = torch.randn(B, 3, requires_grad=True)
    occ = torch.zeros(B, S)
    for b in range(B):
        occ[b, :k[b]] = 1.0
        sp_slots.data[b, k[b]:] = 0     # routed specialist: zero padding
    loss_s, aux_s = specialist_loss(sp_slots, sp_cnt, sources, occ, k)
    print(f"  loss={loss_s.item():.4f}  si_sdr={aux_s['si_sdr'].item():.3f} dB")
    for b in range(B):
        kv = int(k[b])
        row = aux_s['assigned'][b, :kv].tolist()
        assert sorted(row) == list(range(kv)), \
            "specialist PIT must stay within the routed k slots"
        assert (aux_s['assigned'][b, kv:] == -1).all()
    loss_s.backward()
    print("  assigned rows are permutations of range(k): OK")
    print("  backward() succeeded")

    # Perfect assignment inside the routed specialist's k slots
    src3 = torch.randn(1, 3, T, dtype=torch.complex64)
    sp_perfect = torch.zeros(1, S, T, dtype=torch.complex64)
    p3 = [2, 0, 1]                       # source j sits in slot p3[j]
    for j in range(3):
        sp_perfect[0, p3[j]] = src3[0, j]
    sources3 = torch.zeros(1, S, 1, T, dtype=torch.complex64)
    sources3[0, :3, 0] = src3
    _, aux_sp = specialist_loss(sp_perfect, torch.zeros(1, 3), sources3,
                                torch.ones(1, S), torch.tensor([3]),
                                lambda_cnt=0.0)
    print(f"  perfect: si_sdr={aux_sp['si_sdr'].item():.1f} dB  "
          f"assigned={aux_sp['assigned'][0].tolist()} (expect {p3} + -1)")
    assert aux_sp['si_sdr'].item() > 40.0
    assert aux_sp['assigned'][0, :3].tolist() == p3

    print("\n-- lambda_mse=1.0 anchor on all three losses --")
    import math
    torch.manual_seed(3)
    # Default must stay byte-identical: MSE term absent at lambda_mse=0.
    _, aux0 = variable_k_pit_loss(slots.detach(), occ_logits.detach(),
                                  count_logits.detach(), sources, k)
    assert aux0['mse'].item() == 0.0
    print("  lambda_mse=0.0 default: aux['mse'] == 0 (anchor off)  OK")

    sl = torch.randn(B, S, T, dtype=torch.complex64, requires_grad=True)
    oc = torch.randn(B, S, requires_grad=True)
    cl = torch.randn(B, 3, requires_grad=True)
    l1, a1 = variable_k_pit_loss(sl, oc, cl, sources, k, lambda_mse=1.0)
    l1.backward()
    assert math.isfinite(l1.item()) and sl.grad is not None
    print(f"  slot:       loss={l1.item():.4f}  mse={a1['mse'].item():.4f}  "
          f"si_sdr={a1['si_sdr'].item():.3f} dB (pure)  backward OK")

    sl2 = torch.randn(B, S, T, dtype=torch.complex64, requires_grad=True)
    cl2 = torch.randn(B, 3, requires_grad=True)
    for b in range(B):
        sl2.data[b, k[b]:] = 0
    l2, a2 = specialist_loss(sl2, cl2, sources, occ, k, lambda_mse=1.0)
    l2.backward()
    assert math.isfinite(l2.item()) and sl2.grad is not None
    print(f"  specialist: loss={l2.item():.4f}  mse={a2['mse'].item():.4f}  "
          f"backward OK")

    oar2 = OneAndRestNet(hidden_channels=8, n_layers=1, k_max=S)
    l3, a3 = recursive_pit_loss(oar2, mixture, sources, k, lambda_mse=1.0)
    l3.backward()
    assert math.isfinite(l3.item())
    print(f"  recursive:  loss={l3.item():.4f}  mse={a3['mse']:.4f}  "
          f"backward OK")

    print("\n-- lambda_ser soft-demod term (paper 5) --")
    # Default must stay byte-identical: SER term absent at lambda_ser=0.
    assert aux0['ser_soft'].item() == 0.0
    print("  lambda_ser=0.0 default: aux['ser_soft'] == 0 (term off)  OK")

    sl4 = torch.randn(B, S, T, dtype=torch.complex64, requires_grad=True)
    oc4 = torch.randn(B, S, requires_grad=True)
    cl4 = torch.randn(B, 3, requires_grad=True)
    car4 = 2000.0 + 5.0 * torch.rand(B, S)        # fake carriers (Hz)
    mod4 = torch.randint(0, 4, (B, S))            # fake modulation indices
    l4, a4 = variable_k_pit_loss(sl4, oc4, cl4, sources, k,
                                 lambda_ser=1.0, ser_sigma2=0.1,
                                 carriers=car4, mods=mod4,
                                 n_symbols=128)   # sps=4 at T=512
    l4.backward()
    assert math.isfinite(l4.item()) and sl4.grad is not None
    assert a4['ser_soft'].item() > 0
    print(f"  slot+ser: loss={l4.item():.4f}  ser_soft={a4['ser_soft'].item():.4f}  "
          f"si_sdr={a4['si_sdr'].item():.3f} dB (pure)  backward OK")

    # Pure task-oriented config (e): no -SI-SDR term in the loss.
    sl5 = torch.randn(B, S, T, dtype=torch.complex64, requires_grad=True)
    l5, a5 = variable_k_pit_loss(sl5, torch.randn(B, S), None, sources, k,
                                 lambda_sep=0.0, lambda_occ=0.0,
                                 lambda_ser=1.0, carriers=car4, mods=mod4,
                                 n_symbols=128)
    l5.backward()
    assert math.isfinite(l5.item()) and sl5.grad is not None
    # with sep/occ/cnt/mse all off, loss == lambda_ser * ser_soft mean
    assert abs(l5.item() - a5['ser_soft'].item()) < 1e-5
    print(f"  config (e) pure-ser: loss={l5.item():.4f} == ser_soft "
          f"{a5['ser_soft'].item():.4f}  backward OK")

    print("\n-- lambda_llr joint soft-information term (S3) --")
    sl6 = torch.randn(B, S, T, dtype=torch.complex64, requires_grad=True)
    sym6 = torch.randn(B, S, 128, 16, requires_grad=True)  # n_sym=128 at T=512
    l6, a6 = variable_k_pit_loss(sl6, torch.randn(B, S), None, sources, k,
                                 lambda_llr=1.0, sym_logits=sym6,
                                 carriers=car4, mods=mod4, n_symbols=128)
    l6.backward()
    assert math.isfinite(l6.item()) and sym6.grad is not None
    assert a6['llr'].item() > 0
    print(f"  slot+llr: loss={l6.item():.4f}  llr={a6['llr'].item():.4f}  "
          f"backward OK (sym_logits grad norm "
          f"{sym6.grad.abs().sum().item():.2f})")

    # Assertion guard: lambda_ser without carriers/mods must fail loudly.
    try:
        variable_k_pit_loss(sl4.detach(), oc4.detach(), cl4.detach(),
                            sources, k, lambda_ser=1.0)
        raise SystemExit("missing-carriers guard failed")
    except AssertionError:
        print("  missing carriers/mods guard: AssertionError raised  OK")

    print("\nlosses smoke test passed!")
