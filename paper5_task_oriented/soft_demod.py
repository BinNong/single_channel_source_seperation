"""Paper 5 — differentiable soft demodulation (the task-oriented core).

`soft_ser_pairs` is a DIFFERENTIABLE proxy of the compensated receiver in
ser_comp.py, mirroring compute_ser_compensated line by line:

    oracle down-conversion at the TRUE per-source carrier
    -> RRC matched filter (exact np.convolve(mode='same') semantics)
    -> symbol grid sampling at 0::sps (zero net delay, as in ser_comp)
    -> unit-power normalisation of each stream
    -> phase-only alignment of est to ref (coefficient fitted on DETACHED
       tensors and applied as a fixed rotation — the alignment fit itself
       must not become a gradient exploit, plan v2 O4)
    -> soft log-likelihoods -|sym - c_m|^2 / sigma2 over the constellation
    -> cross-entropy against the reference's hard decision (argmin, detached)

Because the same front-end operations are used, the hard version of this
pipeline (argmax of the soft logits) reproduces ser_comp's numpy decisions
symbol-for-symbol on clean sources — enforced by the __main__ smoke test.
That parity is the guard against another zero-result paper: the training
proxy and the evaluation truth are the same receiver.

BER note: the generator has no bit stream, so BER is DEFINED via the Gray
labelling in ser_comp.GRAY_BITS (post-hoc mapping of decided symbol
indices); this module only needs symbol-level cross-entropy for training.

Usage (inside losses.variable_k_pit_loss, on PIT-assigned pairs only):
    ce = soft_ser_pairs(est_pairs, ref_pairs, carriers_hz, mod_idx,
                        sigma2=0.1)   # -> [P] per-pair cross-entropy
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Pull paper1's data generator for the RRC taps (same sys.path trick as
# data_generator_vark / ser_comp).
_PAPER1_DIR = Path(__file__).resolve().parent.parent / "paper1_cnn_se"
if str(_PAPER1_DIR) not in sys.path:
    sys.path.append(str(_PAPER1_DIR))

from data_generator import rrc_filter  # noqa: E402


# ---------------------------------------------------------------------------
# Constellation tables — MUST match paper1_cnn_se/utils.py:CONSTELLATIONS
# (closed set) and data_generator_vark.MOD_TYPES ordering.
# ---------------------------------------------------------------------------
MOD_TYPES = ['BPSK', 'QPSK', '8PSK', '16QAM']
_M_MAX = 16

_CONST_LISTS = {
    'BPSK': [-1 + 0j, 1 + 0j],
    'QPSK': [(-1 - 1j) / np.sqrt(2), (-1 + 1j) / np.sqrt(2),
             (1 - 1j) / np.sqrt(2), (1 + 1j) / np.sqrt(2)],
    '8PSK': list(np.exp(1j * (2 * np.pi * np.arange(8) / 8 + np.pi / 8))),
    '16QAM': [(i + 1j * q) / np.sqrt(10)
              for i in (-3, -1, 1, 3) for q in (-3, -1, 1, 3)],
}


def _build_tables():
    mat = np.zeros((len(MOD_TYPES), _M_MAX), dtype=np.complex64)
    mask = np.zeros((len(MOD_TYPES), _M_MAX), dtype=bool)
    for m, name in enumerate(MOD_TYPES):
        pts = _CONST_LISTS[name]
        mat[m, :len(pts)] = pts
        mask[m, :len(pts)] = True
    return torch.from_numpy(mat), torch.from_numpy(mask)


CONST_MAT, CONST_MASK = _build_tables()      # [4, 16] complex64 / bool


# ---------------------------------------------------------------------------
# RRC matched filter with exact np.convolve(x, h, mode='same') semantics
# ---------------------------------------------------------------------------
_RRC_CACHE: dict[tuple[int, float, int], torch.Tensor] = {}


def _rrc_torch(num_taps: int, roll_off: float, sps: int) -> torch.Tensor:
    key = (num_taps, roll_off, sps)
    if key not in _RRC_CACHE:
        # flip: F.conv1d cross-correlates, np.convolve convolves
        taps = rrc_filter(num_taps, roll_off, sps)[::-1].copy()
        _RRC_CACHE[key] = torch.from_numpy(taps.astype(np.float32))
    return _RRC_CACHE[key]


def _matched_filter_same(x: torch.Tensor, rrc_t: torch.Tensor) -> torch.Tensor:
    """x: [P, T] complex64 -> [P, T] complex64, np.convolve 'same' crop.

    Full convolution via padding=L-1 on both sides, then crop the central
    window starting at (L-1)//2 — numpy's 'same' convention for M >= L.
    The kernel is real, so real/imag channels are filtered independently.
    """
    L = rrc_t.numel()
    T = x.shape[-1]
    xr = torch.view_as_real(x).permute(0, 2, 1)              # [P, 2, T]
    kern = rrc_t.to(x.device).view(1, 1, L).expand(2, 1, L)  # [2, 1, L]
    y = F.conv1d(xr, kern, padding=L - 1, groups=2)          # [P, 2, T+L-1]
    start = (L - 1) // 2
    y = y[:, :, start:start + T]
    return torch.complex(y[:, 0], y[:, 1])


# ---------------------------------------------------------------------------
# Soft-SER loss on PIT-assigned pairs
# ---------------------------------------------------------------------------
def _soft_frontend(est: torch.Tensor,
                   ref: torch.Tensor,
                   carriers_hz: torch.Tensor,
                   mod_idx: torch.Tensor,
                   sigma2: float = 0.1,
                   sample_rate: int = 16000,
                   n_symbols: int = 256,
                   roll_off: float = 0.35,
                   num_taps: int = 64,
                   eps: float = 1e-10):
    """Shared receiver front-end (mirrors ser_comp._demod_pair_labels).

    Returns (logits [P, N, 16], labels [P, N]): soft log-likelihoods of the
    ALIGNED estimate over the per-sample constellation, and the reference's
    hard decisions (detached).  argmax of logits = the torch pipeline's own
    hard decision (invariant to sigma2).
    """
    P, T = est.shape
    device = est.device
    assert T % n_symbols == 0, f"T={T} not divisible by n_symbols={n_symbols}"
    sps = T // n_symbols

    # 1. Oracle down-conversion at each pair's TRUE carrier.
    t = torch.arange(T, device=device, dtype=torch.float32) / sample_rate
    ph = (-2.0 * np.pi) * carriers_hz.to(device=device,
                                         dtype=torch.float32).view(P, 1) * t
    osc = torch.polar(torch.ones_like(ph), ph)               # e^{j*ph}, [P, T]
    est_bb = est * osc
    ref_bb = ref * osc

    # 2. RRC matched filter (np.convolve 'same' semantics).
    rrc_t = _rrc_torch(num_taps, roll_off, sps)
    est_mf = _matched_filter_same(est_bb, rrc_t)
    ref_mf = _matched_filter_same(ref_bb, rrc_t)

    # 3. Symbol grid: offset 0, stride sps (zero net delay — see ser_comp).
    n_use = min(n_symbols, T // sps)
    est_syms = est_mf[:, 0::sps][:, :n_use]                  # [P, N]
    ref_syms = ref_mf[:, 0::sps][:, :n_use]

    # 4. Unit-power normalisation (gradient flows through est).
    est_u = est_syms / (est_syms.abs().pow(2).mean(dim=-1, keepdim=True)
                        .sqrt() + eps)
    ref_u = ref_syms / (ref_syms.abs().pow(2).mean(dim=-1, keepdim=True)
                        .sqrt() + eps)

    # 5. Phase-only alignment of est to ref: the coefficient is fitted on
    #    DETACHED tensors and applied as a fixed rotation (plan v2 O4) —
    #    same semantics as ser_comp's exp(1j*angle(z)).
    est_d = est_u.detach()
    z = (ref_u.detach() * est_d.conj()).sum(dim=-1) / (
        est_d.abs().pow(2).sum(dim=-1) + eps)                # [P] complex
    rot = z / (z.abs() + eps)
    est_a = est_u * rot.unsqueeze(1)                         # gradient via est_u

    # 6. Soft log-likelihoods on the per-sample constellation.
    const = CONST_MAT.to(device)                             # [4, 16]
    valid = CONST_MASK.to(device)                            # [4, 16]
    c = const[mod_idx]                                       # [P, 16]
    v = valid[mod_idx].unsqueeze(1)                          # [P, 1, 16]

    d2_est = (est_a.unsqueeze(-1) - c.unsqueeze(1)).abs().pow(2)  # [P, N, 16]
    logits = -d2_est / sigma2
    logits = logits.masked_fill(~v, -1e9)

    # 7. Reference hard decision (detached) as the CE label.
    d2_ref = (ref_u.unsqueeze(-1) - c.unsqueeze(1)).abs().pow(2)
    labels = d2_ref.masked_fill(~v, float('inf')).argmin(dim=-1).detach()
    return logits, labels


def soft_ser_pairs(est: torch.Tensor,
                   ref: torch.Tensor,
                   carriers_hz: torch.Tensor,
                   mod_idx: torch.Tensor,
                   sigma2: float = 0.1,
                   sample_rate: int = 16000,
                   n_symbols: int = 256,
                   roll_off: float = 0.35,
                   num_taps: int = 64,
                   eps: float = 1e-10) -> torch.Tensor:
    """Differentiable soft-SER proxy for a batch of est/ref waveform PAIRS.

    Args:
        est, ref    : [P, T] complex64 — separated estimate / true source
        carriers_hz : [P] float — TRUE per-source carriers (with jitter)
        mod_idx     : [P] long — index into MOD_TYPES
        sigma2      : Gaussian softness of the soft log-likelihood on the
                      unit-power grid (config TaskConfig.ser_sigma2)

    Returns
    -------
    ce : [P] float32 — per-pair cross-entropy (mean over symbols), the
         differentiable SER proxy.  Lower = decisions land closer to the
         reference's hard decisions, ON the constellation grid.
    """
    logits, labels = _soft_frontend(est, ref, carriers_hz, mod_idx, sigma2,
                                    sample_rate, n_symbols, roll_off,
                                    num_taps, eps)
    P, N = labels.shape
    ce = F.cross_entropy(logits.reshape(P * N, _M_MAX),
                         labels.reshape(P * N), reduction='none')
    return ce.view(P, N).mean(dim=-1)


def ref_hard_labels(ref: torch.Tensor,
                    carriers_hz: torch.Tensor,
                    mod_idx: torch.Tensor,
                    sample_rate: int = 16000,
                    n_symbols: int = 256,
                    roll_off: float = 0.35,
                    num_taps: int = 64,
                    eps: float = 1e-10) -> torch.Tensor:
    """Reference hard decisions for a batch of waveforms, batched/torch.

    Mirrors the reference side of _soft_frontend exactly (oracle
    down-conversion at the TRUE carrier -> RRC matched filter -> 0::sps
    grid -> unit-power normalisation -> masked argmin over the
    constellation).  Used to label the JointLLRHead's symbol logits (S3).
    Returns [P, N] long (detached).
    """
    P, T = ref.shape
    device = ref.device
    assert T % n_symbols == 0, f"T={T} not divisible by n_symbols={n_symbols}"
    sps = T // n_symbols
    t = torch.arange(T, device=device, dtype=torch.float32) / sample_rate
    ph = (-2.0 * np.pi) * carriers_hz.to(device=device,
                                         dtype=torch.float32).view(P, 1) * t
    ref_bb = ref * torch.polar(torch.ones_like(ph), ph)
    rrc_t = _rrc_torch(num_taps, roll_off, sps)
    ref_mf = _matched_filter_same(ref_bb, rrc_t)
    n_use = min(n_symbols, T // sps)
    ref_syms = ref_mf[:, 0::sps][:, :n_use]
    ref_u = ref_syms / (ref_syms.abs().pow(2).mean(dim=-1, keepdim=True)
                        .sqrt() + eps)
    c = CONST_MAT.to(device)[mod_idx]                        # [P, 16]
    v = CONST_MASK.to(device)[mod_idx].unsqueeze(1)          # [P, 1, 16]
    d2 = (ref_u.unsqueeze(-1) - c.unsqueeze(1)).abs().pow(2)  # [P, N, 16]
    return d2.masked_fill(~v, float('inf')).argmin(dim=-1).detach()


# ---------------------------------------------------------------------------
# Smoke-test helpers: hard decisions of both pipelines for the parity check
# ---------------------------------------------------------------------------
def _numpy_labels(est_w, ref_w, mod_id, carrier_hz):
    """ser_comp's numpy receiver decisions, for the parity check."""
    from data_generator_vark import IDX_TO_MOD
    from ser_comp import _demod_pair_labels
    return _demod_pair_labels(
        est_w, ref_w, mod_type=IDX_TO_MOD[int(mod_id)],
        carrier_freq=float(carrier_hz))


def _torch_labels(est_w, ref_w, mod_id, carrier_hz):
    """This module's hard decisions: argmax of the soft logits."""
    with torch.no_grad():
        logits, labels = _soft_frontend(
            est_w.unsqueeze(0), ref_w.unsqueeze(0),
            torch.as_tensor([float(carrier_hz)]),
            torch.as_tensor([int(mod_id)], dtype=torch.long))
    est_labels = logits[0].argmax(dim=-1).cpu().numpy()
    return est_labels, labels[0].cpu().numpy()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Testing paper5 soft_demod ...")
    torch.manual_seed(0)

    # -- 0. constellation tables must match utils.CONSTELLATIONS ----------
    from utils import CONSTELLATIONS, _get_constellation  # symlink to paper1
    for m, name in enumerate(MOD_TYPES):
        ref_const = _get_constellation(name)
        ours = CONST_MAT[m, :len(ref_const)].numpy()
        assert np.allclose(ours, ref_const, atol=1e-6), name
    print("  constellation parity with utils.CONSTELLATIONS: OK")

    # -- 1. matched filter must equal np.convolve(..., 'same') ------------
    T = 4096
    x = (torch.randn(2, T) + 1j * torch.randn(2, T)).to(torch.complex64)
    rrc_np = rrc_filter(64, 0.35, 16)
    y_t = _matched_filter_same(x, _rrc_torch(64, 0.35, 16)).numpy()
    y_n = np.stack([np.convolve(xi.numpy(), rrc_np, mode='same')
                    for xi in x])
    err = np.max(np.abs(y_t - y_n)) / np.max(np.abs(y_n))
    print(f"  matched filter vs np.convolve: rel max err = {err:.2e}")
    assert err < 1e-4

    # -- 2. Gray property: nearest neighbour differs in exactly 1 bit -----
    from ser_comp import GRAY_BITS
    for name in MOD_TYPES:
        const = _get_constellation(name)
        bits = GRAY_BITS[name]
        d = np.abs(const[:, None] - const[None, :])
        np.fill_diagonal(d, np.inf)
        nn = d.argmin(axis=1)
        hamming = (bits[:, None, :] != bits[None, :, :]).sum(-1)
        assert (hamming[np.arange(len(const)), nn] == 1).all(), name
    print("  Gray labelling: every nearest neighbour differs in 1 bit: OK")

    # -- 3. torch pipeline vs numpy receiver on CLEAN sources -------------
    #    Hard decisions must match ser_comp symbol-for-symbol; identity CE
    #    must be ~0.
    from data_generator_vark import CommBSSVarKTestDataset, MOD_TYPES as DMOD
    assert DMOD == MOD_TYPES, "MOD_TYPES ordering must match the generator"
    ds = CommBSSVarKTestDataset(n_per_cell=1, snr_points=[20], seed=99999,
                                return_carriers=True)
    n_checked = 0
    for mix, sources, _occ, k, _snr, mods, carriers in ds:
        kt = int(k)
        if kt > 3:
            continue
        for j in range(kt):
            ref_w = sources[j, 0]
            est_w = ref_w.clone()                      # identity estimate
            # NOTE: identity CE is NOT ~0 by design — the proxy penalises
            # the off-grid distance of the (ISI-carrying) clean symbols;
            # that residual IS the gradient signal we want.  Measured on
            # this grid (2026-09-01): identity CE <= 0.45 (16QAM worst),
            # cross-reference CE in 11..18, ratio <= 0.04.  Acceptance:
            # loose absolute bound + a wide discriminative margin.
            ce = soft_ser_pairs(est_w.unsqueeze(0), ref_w.unsqueeze(0),
                                carriers[j].view(1).float(),
                                mods[j].view(1), sigma2=0.1)
            assert float(ce) < 0.5, f"identity CE too large: {float(ce)}"
            if kt >= 2:
                j2 = (j + 1) % kt
                ce_cross = soft_ser_pairs(
                    est_w.unsqueeze(0), sources[j2, 0].unsqueeze(0),
                    carriers[j2].view(1).float(), mods[j2].view(1),
                    sigma2=0.1)
                assert float(ce) < 0.25 * float(ce_cross), \
                    f"no discriminative margin: id={float(ce):.3f} " \
                    f"cross={float(ce_cross):.3f}"

            # hard decisions of the torch pipeline vs numpy receiver
            est_l_np, ref_l_np = _numpy_labels(est_w, ref_w,
                                               mods[j], carriers[j])
            est_l_t, ref_l_t = _torch_labels(est_w, ref_w,
                                             mods[j], carriers[j])
            assert np.array_equal(est_l_np, est_l_t)
            assert np.array_equal(ref_l_np, ref_l_t)
            n_checked += 1
    print(f"  identity CE < 0.5 with >=4x cross-reference margin and "
          f"torch==numpy decisions on {n_checked} clean sources: OK")

    # -- 4. gradient flows -------------------------------------------------
    est = (torch.randn(1, T) + 1j * torch.randn(1, T)).to(torch.complex64)
    est.requires_grad_(True)
    ref = (torch.randn(1, T) + 1j * torch.randn(1, T)).to(torch.complex64)
    loss = soft_ser_pairs(est, ref, torch.tensor([2003.0]),
                          torch.tensor([1]), sigma2=0.1)
    loss.backward()
    assert est.grad is not None and torch.isfinite(est.grad).all()
    assert est.grad.abs().sum() > 0
    print(f"  backward OK: loss={float(loss):.4f}, |grad| sum="
          f"{float(est.grad.abs().sum()):.4f}")

    print("\nsoft_demod smoke test passed!")
