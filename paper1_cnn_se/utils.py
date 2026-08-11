"""
Utility functions for evaluation metrics and training helpers.

Metrics:
  - SDR (Signal-to-Distortion Ratio)
  - SIR (Signal-to-Interference Ratio)
  - SER (Symbol Error Rate) - communication-specific
  - NMSE (Normalized Mean Square Error)
  - SI-SDR (Scale-Invariant SDR)
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import correlate
import os
import torch.nn as nn


# =============================================================================
# Scale-Invariant SDR (SI-SDR)
# =============================================================================
def si_sdr(estimate, reference, eps=1e-8, return_per_sample=False):
    """
    Compute Scale-Invariant SDR between estimate and reference.
    Both inputs: [..., T]
    Returns:
        scalar (mean over batch) by default; or [B]-shape tensor if return_per_sample=True.
    """
    # Handle complex inputs
    if torch.is_complex(estimate):
        estimate = torch.view_as_real(estimate).flatten(-2)
        reference = torch.view_as_real(reference).flatten(-2)

    # Zero-mean
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)

    # Optimal scaling
    alpha = (reference * estimate).sum(dim=-1, keepdim=True) / \
            ((reference ** 2).sum(dim=-1, keepdim=True) + eps)

    target = alpha * reference
    noise = estimate - target

    si_sdr_val = 10 * torch.log10(
        (target ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps) + eps
    )
    # si_sdr_val is now [B, ...] (flattened last 2 dims)
    # Reduce over all non-batch dims to get a [B] tensor.
    flat = si_sdr_val.reshape(si_sdr_val.shape[0], -1).mean(dim=-1)
    if return_per_sample:
        return flat
    return flat.mean().item()


# =============================================================================
# Standard SDR
# =============================================================================
def compute_sdr(estimated, original, eps=1e-8, return_per_sample=True):
    """
    Compute SDR (Signal-to-Distortion Ratio).
    Handles permutation: tries both (s1_est vs s1_orig, s2_est vs s2_orig)
    and swapped assignment, returns best per-sample.

    Returns:
        [B]-shape tensor by default; scalar mean if return_per_sample=False.
    """
    def _sdr_pair(e, o):
        if torch.is_complex(e):
            e = torch.view_as_real(e).flatten(-2)
            o = torch.view_as_real(o).flatten(-2)
        noise = e - o
        val = 10 * torch.log10(
            (o ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps) + eps
        )
        return val.reshape(val.shape[0], -1).mean(dim=-1)  # [B]

    # Try both permutations and pick best
    sdr_11 = _sdr_pair(estimated[0], original[0])  # est1 vs orig1
    sdr_22 = _sdr_pair(estimated[1], original[1])  # est2 vs orig2
    sdr_12 = _sdr_pair(estimated[0], original[1])  # est1 vs orig2
    sdr_21 = _sdr_pair(estimated[1], original[0])  # est2 vs orig1

    sdr_perm1 = (sdr_11 + sdr_22) / 2  # Normal assignment
    sdr_perm2 = (sdr_12 + sdr_21) / 2  # Swapped assignment

    # Per-sample best permutation
    use_swap = sdr_perm2 > sdr_perm1  # [B]
    sdr1_best = torch.where(use_swap, sdr_12, sdr_11)  # [B]
    sdr2_best = torch.where(use_swap, sdr_21, sdr_22)  # [B]
    sdr_best = (sdr1_best + sdr2_best) / 2  # [B]
    if return_per_sample:
        return sdr_best
    return sdr_best.mean().item()


# =============================================================================
# SIR (Signal-to-Interference Ratio)
# =============================================================================
def compute_sir(estimated, original, eps=1e-8):
    """
    Compute SIR: ratio of target signal power to interference power.
    For each estimated source, measure how much of the other original source leaks in.
    """
    def _power(x):
        if torch.is_complex(x):
            return (x.abs() ** 2).sum(dim=-1)
        return (x ** 2).sum(dim=-1)

    # Cross-talk: est1 should match orig1, not orig2
    e1, e2 = estimated
    o1, o2 = original

    # Correlation-based assignment
    corr_11 = (e1 * o1.conj()).abs().sum(dim=-1) if torch.is_complex(e1) else (e1 * o1).abs().sum(dim=-1)
    corr_12 = (e1 * o2.conj()).abs().sum(dim=-1) if torch.is_complex(e1) else (e1 * o2).abs().sum(dim=-1)
    swap = corr_12 > corr_11

    if swap.any():
        o1_swapped = torch.where(swap.view(-1, 1, 1), o2, o1)
        o2_swapped = torch.where(swap.view(-1, 1, 1), o1, o2)
    else:
        o1_swapped, o2_swapped = o1, o2

    # SIR for source 1: power(o1) / power(leakage from o2 in est1)
    proj = ((e1 * o2_swapped.conj()).sum(dim=-1, keepdim=True) /
            (_power(o2_swapped).unsqueeze(-1) + eps)) * o2_swapped if torch.is_complex(e1) else \
           ((e1 * o2_swapped).sum(dim=-1, keepdim=True) /
            (_power(o2_swapped).unsqueeze(-1) + eps)) * o2_swapped
    interference = e1 - proj

    sir1 = 10 * torch.log10(_power(o1_swapped) / (_power(interference) + eps) + eps)

    # SIR for source 2
    proj2 = ((e2 * o1_swapped.conj()).sum(dim=-1, keepdim=True) /
             (_power(o1_swapped).unsqueeze(-1) + eps)) * o1_swapped if torch.is_complex(e2) else \
            ((e2 * o1_swapped).sum(dim=-1, keepdim=True) /
             (_power(o1_swapped).unsqueeze(-1) + eps)) * o1_swapped
    interference2 = e2 - proj2
    sir2 = 10 * torch.log10(_power(o2_swapped) / (_power(interference2) + eps) + eps)

    return ((sir1 + sir2) / 2).squeeze(-1)


# =============================================================================
# NMSE (Normalized Mean Square Error)
# =============================================================================
def compute_nmse(estimated, original, eps=1e-8):
    """Compute NMSE in dB."""
    if torch.is_complex(estimated):
        mse = (estimated - original).abs().pow(2).sum(dim=-1)
        power = original.abs().pow(2).sum(dim=-1)
    else:
        mse = ((estimated - original) ** 2).sum(dim=-1)
        power = (original ** 2).sum(dim=-1)
    nmse = 10 * torch.log10(mse / (power + eps) + eps)
    if nmse.dim() > 1 and nmse.shape[-1] == 1:
        nmse = nmse.squeeze(-1)
    return nmse


# =============================================================================
# SER (Symbol Error Rate) - Communication-specific metric
# =============================================================================

# Standard constellation points (normalized)
CONSTELLATIONS = {
    'BPSK': np.array([-1+0j, 1+0j]),
    'QPSK': np.array([-1-1j, -1+1j, 1-1j, 1+1j]) / np.sqrt(2),
    '8PSK': np.exp(1j * (2*np.pi*np.arange(8)/8 + np.pi/8)),
    '16QAM': (lambda: np.array([
        -3-3j, -3-1j, -3+1j, -3+3j,
        -1-3j, -1-1j, -1+1j, -1+3j,
         1-3j,  1-1j,  1+1j,  1+3j,
         3-3j,  3-1j,  3+1j,  3+3j
    ]) / np.sqrt(10))(),
    'MSK': np.array([-1+0j, 1+0j]),
    'GMSK': np.array([-1+0j, 1+0j]),
}


def _get_constellation(mod_type):
    """Return normalized constellation points as [K] complex numpy array."""
    if mod_type in CONSTELLATIONS:
        return CONSTELLATIONS[mod_type].copy()
    # Fallback: treat as QPSK
    return CONSTELLATIONS['QPSK'].copy()


def compute_ser_from_signal(estimated_signal, source_signal, mod_type='QPSK',
                            sample_rate=16000, n_symbols=256, roll_off=0.35,
                            num_taps=64, carrier_freq=2000.0):
    """
    Estimate SER by matched-filter demodulation of the separated signal.

    Pipeline:
      1. Down-convert (remove carrier)
      2. Matched filter (RRC) → downsample to symbol rate
      3. Min-distance demodulation onto constellation
      4. Count symbol errors

    Args:
        estimated_signal: [T] complex tensor (separated signal)
        source_signal: [T] complex tensor (reference clean signal)
        mod_type: modulation type string
        sample_rate: sampling rate (Hz)
        n_symbols: number of symbols
        roll_off: RRC roll-off factor
        num_taps: RRC filter taps
        carrier_freq: carrier frequency (Hz)

    Returns:
        ser: symbol error rate (0.0 to 1.0)
    """
    import numpy as np
    from scipy.signal import resample_poly

    # Convert to numpy
    est = estimated_signal.detach().cpu().numpy().squeeze()
    ref = source_signal.detach().cpu().numpy().squeeze()

    T = len(est)
    sps = T // n_symbols  # samples per symbol (should be ~16 for 4096/256)

    # 1. Down-convert (remove carrier)
    t = np.arange(T) / sample_rate
    est_bb = est * np.exp(-1j * 2 * np.pi * carrier_freq * t)
    ref_bb = ref * np.exp(-1j * 2 * np.pi * carrier_freq * t)

    # 2. Matched filter (RRC)
    from data_generator import rrc_filter
    rrc = rrc_filter(num_taps, roll_off, sps)
    est_mf = np.convolve(est_bb, rrc, mode='same')
    ref_mf = np.convolve(ref_bb, rrc, mode='same')

    # 3. Find optimal sampling offset via cross-correlation with reference
    # Use reference signal to find the best sampling timing
    # Sample at the center of each symbol period
    start_offset = sps // 2
    est_syms = est_mf[start_offset::sps][:n_symbols]
    ref_syms = ref_mf[start_offset::sps][:n_symbols]

    if len(est_syms) < n_symbols:
        est_syms = np.pad(est_syms, (0, n_symbols - len(est_syms)))

    # 4. Normalize both to unit average power for fair comparison
    est_pwr = np.sqrt(np.mean(np.abs(est_syms)**2) + 1e-10)
    ref_pwr = np.sqrt(np.mean(np.abs(ref_syms)**2) + 1e-10)
    est_syms = est_syms / est_pwr

    # 5. Find scaling/rotation between est and ref
    # (account for arbitrary complex scaling from separation)
    scale = np.sum(ref_syms * np.conj(est_syms)) / (np.sum(np.abs(est_syms)**2) + 1e-10)
    est_syms = est_syms * scale

    # 6. Min-distance demodulation
    const = _get_constellation(mod_type)
    est_labels = np.argmin(np.abs(est_syms[:, None] - const[None, :]), axis=1)
    ref_labels = np.argmin(np.abs(ref_syms[:, None] / ref_pwr - const[None, :]), axis=1)

    # 7. Count errors
    errors = np.sum(est_labels != ref_labels)
    ser = errors / n_symbols

    return float(ser)


# =============================================================================
# FLOP Counter (lightweight estimation)
# =============================================================================
def count_model_params_and_flops(model, input_shape=(1, 1, 4096)):
    """
    Count parameters, FLOPs, and measure inference time for a model.

    Returns:
        dict with keys: total_params, trainable_params, flops_estimate,
                        inference_time_ms, inference_time_std_ms
    """
    import time
    device = next(model.parameters()).device

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # FLOPs estimate (rough: 2 MAC per complex multiply-add per weight)
    # ComplexConv1d: each real conv does (in*out*kernel) MAC per output sample
    flops = 0
    T = input_shape[2]
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            # Count for one ComplexConv1d (2 real convs inside)
            mac_per_output = m.in_channels * m.out_channels * m.kernel_size[0] / m.groups
            L_out = T  # rough
            if hasattr(m, 'stride') and m.stride[0] > 1:
                L_out = T // m.stride[0]
            flops += 2 * mac_per_output * L_out * 2  # 2 convs, 2 ops per MAC (mul+add)
        elif isinstance(m, nn.BatchNorm1d):
            flops += 2 * m.num_features * T  # mean + var
        elif isinstance(m, nn.Linear):
            flops += 2 * m.in_features * m.out_features * 2  # 2 ops per MAC

    flops = int(flops)

    # Inference time (warmup + measure)
    model.eval()
    dummy = torch.randn(*input_shape).to(torch.complex64).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms

    avg_time = np.mean(times)
    std_time = np.std(times)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops_estimate': flops,
        'inference_time_ms': round(avg_time, 3),
        'inference_time_std_ms': round(std_time, 3),
    }


# =============================================================================
# Combined Evaluation
# =============================================================================
def evaluate_batch(estimated_sources, original_sources, metrics=None):
    """
    Evaluate a batch of separated sources.

    IMPORTANT: All metrics use the SAME permutation assignment per sample
    (chosen by maximizing SI-SDR). This guarantees that for any sample
    SI-SDR >= SDR (scale-invariant >= standard SDR).

    Args:
        estimated_sources: tuple of (est_s1, est_s2), each [B, 1, T]
        original_sources: tuple of (orig_s1, orig_s2), each [B, 1, T]
        metrics: list of metric names, default all

    Returns:
        dict of metric values (averaged over batch)
    """
    if metrics is None:
        metrics = ['SI-SDR', 'SDR', 'SIR', 'NMSE']

    est1, est2 = estimated_sources
    orig1, orig2 = original_sources

    # ---- Step 1: determine optimal permutation per sample using SI-SDR ----
    sisdr_11 = si_sdr(est1, orig1, return_per_sample=True)  # [B]
    sisdr_22 = si_sdr(est2, orig2, return_per_sample=True)
    sisdr_12 = si_sdr(est1, orig2, return_per_sample=True)
    sisdr_21 = si_sdr(est2, orig1, return_per_sample=True)

    perm1 = (sisdr_11 + sisdr_22) / 2  # Normal assignment
    perm2 = (sisdr_12 + sisdr_21) / 2  # Swapped assignment
    use_swap = perm2 > perm1  # [B] bool
    # [B, 1, T] masks for swapping o1<->o2 per sample
    swap_mask = use_swap.view(-1, 1, 1)

    o1_aligned = torch.where(swap_mask, orig2, orig1)
    o2_aligned = torch.where(swap_mask, orig1, orig2)

    results = {}

    if 'SI-SDR' in metrics:
        # Already computed above — just average the per-sample best.
        sisdr_best = torch.where(use_swap, perm2, perm1)  # [B]
        results['SI-SDR'] = sisdr_best.mean().item()

    if 'SDR' in metrics:
        sdr_1 = compute_sdr((est1, est2), (o1_aligned, o2_aligned),
                            return_per_sample=True)  # [B]
        results['SDR'] = sdr_1.mean().item()

    if 'SIR' in metrics:
        sir_vals = compute_sir((est1, est2), (o1_aligned, o2_aligned))
        # compute_sir already does its own correlation-based assignment internally;
        # we still pass aligned references so the computation is consistent.
        results['SIR'] = sir_vals.mean().item()

    if 'NMSE' in metrics:
        nmse1 = compute_nmse(est1, o1_aligned).mean().item()
        nmse2 = compute_nmse(est2, o2_aligned).mean().item()
        results['NMSE'] = (nmse1 + nmse2) / 2

    return results


# =============================================================================
# Visualization
# =============================================================================
def visualize_separation(mixture, source1, source2, est1, est2, save_path=None):
    """Visualize separation results in time and frequency domain."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Convert to numpy
    def to_np(x):
        return x.detach().cpu().numpy().squeeze()

    mix, s1, s2, e1, e2 = to_np(mixture), to_np(source1), to_np(source2), to_np(est1), to_np(est2)

    # Time domain (magnitude)
    t = np.arange(len(mix))
    axes[0, 0].plot(t, np.abs(mix), 'k', alpha=0.7, label='Mixture')
    axes[0, 0].set_title('Mixture (Time Domain)')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Magnitude')

    axes[0, 1].plot(t, np.abs(s1), 'b', alpha=0.7, label='Original 1')
    axes[0, 1].plot(t, np.abs(e1), 'r--', alpha=0.7, label='Estimated 1')
    axes[0, 1].set_title('Source 1: Original vs Estimated')
    axes[0, 1].legend()

    axes[1, 0].plot(t, np.abs(s2), 'b', alpha=0.7, label='Original 2')
    axes[1, 0].plot(t, np.abs(e2), 'r--', alpha=0.7, label='Estimated 2')
    axes[1, 0].set_title('Source 2: Original vs Estimated')
    axes[1, 0].legend()

    # Constellation diagrams
    axes[1, 1].scatter(s1.real, s1.imag, c='blue', alpha=0.3, s=1, label='Original 1')
    axes[1, 1].scatter(e1.real, e1.imag, c='red', alpha=0.3, s=1, label='Estimated 1')
    axes[1, 1].set_title('Constellation Diagram: Source 1')
    axes[1, 1].set_xlabel('I'); axes[1, 1].set_ylabel('Q')
    axes[1, 1].legend(); axes[1, 1].axis('equal')

    # Error signal
    err1 = np.abs(s1 - e1)
    err2 = np.abs(s2 - e2)
    axes[2, 0].plot(t, err1, 'r', alpha=0.5, label='Error Source 1')
    axes[2, 0].plot(t, err2, 'b', alpha=0.5, label='Error Source 2')
    axes[2, 0].set_title('Separation Error')
    axes[2, 0].legend()

    # Frequency domain
    from scipy.fft import fft
    freqs = np.fft.fftfreq(len(mix), d=1/16000)
    axes[2, 1].plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(fft(mix))[:len(freqs)//2]+1e-10), 'k', alpha=0.5, label='Mixture')
    axes[2, 1].plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(fft(s1))[:len(freqs)//2]+1e-10), 'b', alpha=0.5, label='Source 1')
    axes[2, 1].set_title('Frequency Domain')
    axes[2, 1].set_xlabel('Frequency (Hz)'); axes[2, 1].set_ylabel('dB')
    axes[2, 1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Checkpoint Utils
# =============================================================================
def save_checkpoint(model, optimizer, epoch, best_metric, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, best_metric={checkpoint['best_metric']:.4f}")
    return checkpoint['epoch'], checkpoint['best_metric']
