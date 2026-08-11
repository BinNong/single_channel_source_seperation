"""
Neural Network Models for Communication Signal Blind Source Separation

Includes:
  1. ComplexConv1d - Complex-valued 1D convolution
  2. ComplexConvTranspose1d - Complex-valued transposed 1D convolution
  3. ComplexBatchNorm1d - Complex batch normalization
  4. ComplexReLU / ComplexPReLU - Complex activations
  5. ComplexGlobalLayerNorm - Global layer normalization for complex tensors
  6. ComplexSEBlock - Complex Squeeze-and-Excitation (core innovation)
  7. ComplexResidualBlock - Residual block with optional SE
  8. ComplexLightweightSepNet - Proposed model (total params < 1M)
  9. RealValuedBaseline - Real-valued CNN baseline (similar to Hou & Gao 2022)
 10. SimpleComplexCNN - Simple complex CNN (ablation)
 11. ComplexConvTasNet - Complex-domain Conv-TasNet (Luo & Mesgarani 2019) baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# Complex Conv1d
# =============================================================================
class ComplexConv1d(nn.Module):
    """
    Complex-valued 1D convolution.
    For input x = a + jb, weight w = u + jv:
        x * w = (a*u - b*v) + j(a*v + b*u)
    Implemented using two real-valued convolutions.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, dilation=1, groups=1, bias=True):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) * dilation // 2

        self.real_conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias=bias)
        self.imag_conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias=bias)

    def forward(self, x):
        # x: [B, C, T] complex64
        return torch.complex(
            self.real_conv(x.real) - self.imag_conv(x.imag),
            self.real_conv(x.imag) + self.imag_conv(x.real)
        )


# =============================================================================
# Complex ConvTranspose1d
# =============================================================================
class ComplexConvTranspose1d(nn.Module):
    """
    Complex-valued transposed 1D convolution.
    For input x = a + jb, weight w = u + jv:
        x * w = (a*u - b*v) + j(a*v + b*u)
    Implemented using two real-valued transposed convolutions.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, output_padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.real_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                            stride, padding, output_padding,
                                            groups, bias, dilation)
        self.imag_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                            stride, padding, output_padding,
                                            groups, bias, dilation)

    def forward(self, x):
        return torch.complex(
            self.real_conv(x.real) - self.imag_conv(x.imag),
            self.real_conv(x.imag) + self.imag_conv(x.real)
        )


# =============================================================================
# Complex Batch Normalization
# =============================================================================
class ComplexBatchNorm1d(nn.Module):
    """
    Complex batch normalization.
    Normalizes both real and imaginary parts independently.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn_real = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        self.bn_imag = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x):
        return torch.complex(self.bn_real(x.real), self.bn_imag(x.imag))


# =============================================================================
# Complex ReLU (CReLU) - applies ReLU to both real and imag parts
# =============================================================================
class ComplexReLU(nn.Module):
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))


# =============================================================================
# Complex PReLU (CPReLU) - applies PReLU to both real and imag parts
# =============================================================================
class ComplexPReLU(nn.Module):
    """Complex PReLU – learnable negative slope applied to real & imag separately."""
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.prelu_real = nn.PReLU(num_parameters, init)
        self.prelu_imag = nn.PReLU(num_parameters, init)

    def forward(self, x):
        return torch.complex(self.prelu_real(x.real), self.prelu_imag(x.imag))


# =============================================================================
# Complex Global Layer Normalization (for Conv-TasNet style normalization)
# =============================================================================
class ComplexGlobalLayerNorm(nn.Module):
    """
    Global layer normalization for complex tensors.
    Normalizes over all feature dimensions (C, T) except batch.
    Used in Conv-TasNet architecture instead of BatchNorm for causal / streaming.
    """
    def __init__(self, channels, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, T] complex
        r, i = x.real, x.imag
        r_mean = r.mean(dim=(1, 2), keepdim=True)
        i_mean = i.mean(dim=(1, 2), keepdim=True)
        r_var = ((r - r_mean) ** 2).mean(dim=(1, 2), keepdim=True)
        i_var = ((i - i_mean) ** 2).mean(dim=(1, 2), keepdim=True)
        r_norm = self.gamma * (r - r_mean) / torch.sqrt(r_var + self.eps) + self.beta
        i_norm = self.gamma * (i - i_mean) / torch.sqrt(i_var + self.eps) + self.beta
        return torch.complex(r_norm, i_norm)


# =============================================================================
# Complex Squeeze-and-Excitation Block (CORE INNOVATION)
# =============================================================================
class ComplexSEBlock(nn.Module):
    """
    Complex Squeeze-and-Excitation Block.

    Squeeze: Global pooling over time + concatenation of per-channel statistics.
    Excitation: FC layer -> ReLU -> FC layer -> Sigmoid to produce channel weights.
    Scale: Multiply complex features by learned channel weights.

    Args:
        channels: number of complex channels (each channel = 1 complex value).
        reduction: SE bottleneck reduction ratio.
        scale_mode: 'real' (default — same real weight on real and imag,
                     preserves phase, adjusts magnitude),
                    'complex_mean' (averages two sigmoid outputs),
                    'separate' (different weights for real and imag parts).
        pooling_mode: 'mean' (default — average real and imag separately, concat to 2C),
                      'power' (mean of squared values, real and imag separately),
                      'magnitude' (mean of absolute values, real and imag separately),
                      'mean+power' (concat mean and power, total 4C).
    """

    def __init__(self, channels, reduction=4, scale_mode='real',
                 pooling_mode='mean'):
        super().__init__()
        assert scale_mode in ('real', 'complex_mean', 'separate')
        assert pooling_mode in ('mean', 'power', 'magnitude', 'mean+power')
        self.scale_mode = scale_mode
        self.pooling_mode = pooling_mode
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # Input dim to FC depends on pooling mode: 2C, 2C, 2C, 4C respectively.
        if pooling_mode == 'mean+power':
            in_dim = channels * 4
        else:
            in_dim = channels * 2
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // reduction, in_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, T] complex
        b, c, t = x.shape

        # Squeeze: global pooling of real and imag separately, with mode-specific stats
        z_real = self.avg_pool(x.real).view(b, c)  # [B, C]
        z_imag = self.avg_pool(x.imag).view(b, c)  # [B, C]

        if self.pooling_mode == 'mean':
            z = torch.cat([z_real, z_imag], dim=1)                       # [B, 2C]
        elif self.pooling_mode == 'power':
            # E[Re^2], E[Im^2] -- energy-like statistics per channel
            p_real = (x.real ** 2).mean(dim=-1)                          # [B, C]
            p_imag = (x.imag ** 2).mean(dim=-1)                          # [B, C]
            z = torch.cat([p_real, p_imag], dim=1)                       # [B, 2C]
        elif self.pooling_mode == 'magnitude':
            # E[|Re|], E[|Im|] -- magnitude-like statistics per channel
            m_real = x.real.abs().mean(dim=-1)                           # [B, C]
            m_imag = x.imag.abs().mean(dim=-1)                           # [B, C]
            z = torch.cat([m_real, m_imag], dim=1)                       # [B, 2C]
        elif self.pooling_mode == 'mean+power':
            p_real = (x.real ** 2).mean(dim=-1)
            p_imag = (x.imag ** 2).mean(dim=-1)
            z = torch.cat([z_real, z_imag, p_real, p_imag], dim=1)       # [B, 4C]

        # Excitation
        scale = self.fc(z)                                               # [B, in_dim]
        # Map back to per-channel real and imag weights (only the first 2C entries are used
        # for scale; the extra 2C from 'mean+power' is absorbed into the same 2C weights)
        scale_real = scale[:, :c].view(b, c, 1)                          # [B, C, 1]
        scale_imag = scale[:, c:2*c].view(b, c, 1)                       # [B, C, 1]

        if self.scale_mode == 'real':
            # Same real weight for both real and imag parts of each channel.
            # (scale_real + scale_imag) / 2 acts as a learned magnitude scaler
            # that preserves phase.
            weight = (scale_real + scale_imag) / 2       # [B, C, 1] real
            return x * weight
        elif self.scale_mode == 'complex_mean':
            # Build a complex weight: w = (scale_real + j*scale_imag), then
            # output = x * w  (complex multiplication).
            w = torch.complex(scale_real, scale_imag)
            return x * w
        elif self.scale_mode == 'separate':
            # Apply different real weights to real and imag parts (no phase
            # preservation guarantee).
            return torch.complex(x.real * scale_real, x.imag * scale_imag)


# =============================================================================
# Complex Residual Block (Conv + BN + ReLU + SE + Residual)
# =============================================================================
class ComplexResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, use_se=True, se_reduction=4,
                 se_scale_mode='real', se_pooling_mode='mean'):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = ComplexConv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = ComplexBatchNorm1d(channels)
        self.relu = ComplexReLU()
        self.conv2 = ComplexConv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = ComplexBatchNorm1d(channels)
        self.se = ComplexSEBlock(channels, se_reduction,
                                 scale_mode=se_scale_mode,
                                 pooling_mode=se_pooling_mode) if use_se else None

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        return self.relu(out + residual)


# =============================================================================
# Proposed Model: Complex Lightweight Separation Network
# =============================================================================
class ComplexLightweightSepNet(nn.Module):
    """
    Proposed Complex-Valued Lightweight CNN for Communication Signal BSS.

    Architecture:
        Complex Conv Encoder -> [Residual Block x N] -> Complex Conv Decoder
        with Complex Squeeze-and-Excitation attention.

    Output strategy (mask-based + learnable scale):
        The decoder predicts two complex masks, applied to the mixture to
        produce the estimated sources. A learnable complex scalar `output_scale`
        is multiplied with the masks so that the initial output is approximately
        `0.5 * mixture` for both sources (a sensible non-degenerate starting
        point). This avoids the trivial-solution collapse observed when the
        decoder directly predicts sources from a Kaiming-initialized head.

    Target: < 1M parameters, trainable on 8GB GPU.
    """

    def __init__(self, in_channels=1, hidden_channels=32, n_layers=4,
                 kernel_size_enc=7, kernel_size_hidden=3, kernel_size_dec=7,
                 use_se=True, se_reduction=4, se_scale_mode='real',
                 se_pooling_mode='mean'):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            ComplexConv1d(in_channels, hidden_channels, kernel_size_enc, padding=kernel_size_enc//2),
            ComplexBatchNorm1d(hidden_channels),
            ComplexReLU()
        )

        # Residual blocks with SE
        self.res_blocks = nn.ModuleList([
            ComplexResidualBlock(hidden_channels, kernel_size_hidden, use_se,
                                 se_reduction, se_scale_mode, se_pooling_mode)
            for _ in range(n_layers)
        ])

        # Decoder: output 2 complex masks
        self.decoder = ComplexConv1d(hidden_channels, 2, kernel_size_dec, padding=kernel_size_dec//2)

        # Learnable complex scaling so initial output ≈ 0.5 * mixture
        # (decoder weights start near 0 -> mask ≈ 0; scale=0.5 -> output ≈ 0.5*mix)
        self.output_scale = nn.Parameter(torch.tensor(0.5 + 0j, dtype=torch.complex64))

        self._count_params()

    def forward(self, mixture):
        # mixture: [B, 1, T] complex
        x = self.encoder(mixture)                       # [B, H, T]
        for block in self.res_blocks:
            x = block(x)                                 # [B, H, T]
        masks = self.decoder(x)                          # [B, 2, T] complex
        # Apply learnable complex scale, then apply to mixture
        scaled_masks = masks * self.output_scale         # [B, 2, T] complex
        s1 = scaled_masks[:, 0:1, :] * mixture          # [B, 1, T]
        s2 = scaled_masks[:, 1:2, :] * mixture          # [B, 1, T]
        return s1, s2

    def _count_params(self):
        n = sum(p.numel() for p in self.parameters())
        print(f"[ComplexLightweightSepNet] Total parameters: {n:,} ({n/1e6:.3f}M)")


# =============================================================================
# Baseline 1: Real-Valued CNN (similar to Hou & Gao 2022 CNSE)
# =============================================================================
class RealValuedBaseline(nn.Module):
    """
    Real-valued CNN baseline.
    Treats complex signal as 2-channel real input (I and Q).
    Similar architecture to Hou & Gao 2022 (Digital Signal Processing).
    """

    def __init__(self, in_channels=2, hidden=64, n_layers=6, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )

        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Conv1d(hidden, hidden, kernel_size, padding=padding),
                nn.BatchNorm1d(hidden),
                nn.ReLU()
            ])
        self.middle = nn.Sequential(*layers)

        # Output 4 channels: 2 sources x 2 (mask_real, mask_imag)
        self.decoder = nn.Conv1d(hidden, 4, kernel_size=7, padding=3)

        # Learnable scale (real-valued) to make initial output ≈ 0.5 * mixture
        self.output_scale = nn.Parameter(torch.tensor(0.5))

        self._count_params()

    def forward(self, mixture):
        # mixture: [B, 1, T] complex -> convert to [B, 2, T] real
        x_real = mixture.real  # [B, 1, T]
        x_imag = mixture.imag  # [B, 1, T]
        x = torch.cat([x_real, x_imag], dim=1)  # [B, 2, T]

        x = self.encoder(x)
        x = self.middle(x)
        m = self.decoder(x)  # [B, 4, T]
        m = m * self.output_scale  # scale

        # Apply complex mask to mixture: s = (m_re + j m_im) * (x_re + j x_im)
        m1_re = m[:, 0:1, :]; m1_im = m[:, 1:2, :]
        m2_re = m[:, 2:3, :]; m2_im = m[:, 3:4, :]
        s1 = torch.complex(m1_re * x_real - m1_im * x_imag,
                           m1_re * x_imag + m1_im * x_real)
        s2 = torch.complex(m2_re * x_real - m2_im * x_imag,
                           m2_re * x_imag + m2_im * x_real)
        return s1, s2

    def _count_params(self):
        n = sum(p.numel() for p in self.parameters())
        print(f"[RealValuedBaseline] Total parameters: {n:,} ({n/1e6:.3f}M)")


# =============================================================================
# Baseline 2: Simple Complex CNN (no SE, no residual)
# =============================================================================
class SimpleComplexCNN(nn.Module):
    """Simple complex CNN without attention or residual connections."""

    def __init__(self, in_channels=1, hidden=48, n_layers=6):
        super().__init__()
        layers = [
            ComplexConv1d(in_channels, hidden, 7, padding=3),
            ComplexBatchNorm1d(hidden),
            ComplexReLU()
        ]
        for _ in range(n_layers):
            layers.extend([
                ComplexConv1d(hidden, hidden, 3, padding=1),
                ComplexBatchNorm1d(hidden),
                ComplexReLU()
            ])
        layers.append(ComplexConv1d(hidden, 2, 7, padding=3))
        self.net = nn.Sequential(*layers)
        # Learnable complex scaling so initial output ≈ 0.5 * mixture
        self.output_scale = nn.Parameter(torch.tensor(0.5 + 0j, dtype=torch.complex64))
        self._count_params()

    def forward(self, mixture):
        masks = self.net(mixture)                                # [B, 2, T] complex
        scaled_masks = masks * self.output_scale                 # [B, 2, T]
        s1 = scaled_masks[:, 0:1, :] * mixture
        s2 = scaled_masks[:, 1:2, :] * mixture
        return s1, s2

    def _count_params(self):
        n = sum(p.numel() for p in self.parameters())
        print(f"[SimpleComplexCNN] Total parameters: {n:,} ({n/1e6:.3f}M)")


# =============================================================================
# Conv-TasNet TCN Block (Complex-valued, depthwise separable convolution)
# =============================================================================
class ConvTasNetTCNBlock(nn.Module):
    """
    1-D dilated convolutional block for Conv-TasNet, adapted to complex domain.
    Uses depthwise separable convolution (D-conv) as in Luo & Mesgarani 2019.

    Input → 1×1-conv(B→H) → PReLU → BN → D-conv(H→H, P, dil) → PReLU → BN
          ├→ 1×1-conv(H→B) + BN → residual (+ input)
          └→ 1×1-conv(H→Sc) → skip (summed across all blocks)
    """
    def __init__(self, B, H, Sc, P, dilation):
        super().__init__()
        # First pointwise: B → H
        self.conv1x1_1 = ComplexConv1d(B, H, 1)
        self.prelu1 = ComplexPReLU(H)
        self.norm1 = ComplexBatchNorm1d(H)

        # Depthwise dilated: H → H
        self.dconv = ComplexConv1d(H, H, P, dilation=dilation, groups=H)
        self.prelu2 = ComplexPReLU(H)
        self.norm2 = ComplexBatchNorm1d(H)

        # Residual path: H → B
        self.res_conv = ComplexConv1d(H, B, 1)
        self.norm_res = ComplexBatchNorm1d(B)

        # Skip-connection path: H → Sc
        self.skip_conv = ComplexConv1d(H, Sc, 1)

    def forward(self, x):
        # x: [B, B, T']
        out = self.conv1x1_1(x)
        out = self.norm1(out)
        out = self.prelu1(out)

        out = self.dconv(out)
        out = self.norm2(out)
        out = self.prelu2(out)

        residual = self.res_conv(out)
        residual = self.norm_res(residual)

        skip = self.skip_conv(out)

        return residual, skip


# =============================================================================
# Complex Conv-TasNet (Luo & Mesgarani 2019, adapted to complex domain)
# =============================================================================
class ComplexConvTasNet(nn.Module):
    """
    Complex-valued Conv-TasNet for communication signal BSS.

    Adapts the fully-convolutional time-domain audio separation network
    (Luo & Mesgarani, IEEE/ACM TASLP 2019) to the complex domain using
    ComplexConv1d, ComplexPReLU, ComplexBatchNorm1d and
    ComplexConvTranspose1d throughout.

    Architecture (scaled for 8 GB GPU, ~0.7M real params):
        Encoder (L=16, N=64) → BN → Bottleneck (N→B) →
        X×R TCN blocks (1,2,4,... dilation) with skip-sum →
        PReLU → 1×1-mask-conv (Sc→2N) → apply masks →
        Decoder (transposed conv) → s1, s2

    Uses the same mask-based + learnable complex scale strategy as other
    models in this repo for fair comparison.

    Reference hyperparameters from original paper (5.1M params, SI-SNRi 15.3dB):
        N=512, L=16, B=128, Sc=128, H=512, P=3, X=8, R=3, gLN
    Our scaled-down version:
        N=64,  L=16, B=64,  Sc=64,  H=128, P=3, X=5, R=3
    """

    def __init__(self, N=64, B=64, Sc=64, H=128, P=3, X=5, R=3, L=16, C=2):
        super().__init__()
        self.N = N
        self.L = L

        # ---- Encoder ----
        # Conv1d(1, N, L, stride=L//2) — 50% overlap between frames
        self.encoder = ComplexConv1d(1, N, L, stride=L // 2, padding=0)
        self.encoder_norm = ComplexGlobalLayerNorm(N)

        # ---- Bottleneck (1×1 conv) ----
        self.bottleneck = ComplexConv1d(N, B, 1)

        # ---- TCN blocks ----
        self.tcn_blocks = nn.ModuleList()
        for _ in range(X):
            for r in range(R):
                dilation = 2 ** r
                self.tcn_blocks.append(ConvTasNetTCNBlock(B, H, Sc, P, dilation))

        # ---- Mask estimation ----
        self.mask_prelu = ComplexPReLU(Sc)
        self.mask_conv = ComplexConv1d(Sc, C * N, 1)

        # ---- Decoder (transposed conv) ----
        # ConvTranspose1d(N, 1, L, stride=L//2)
        self.decoder = ComplexConvTranspose1d(N, 1, L, stride=L // 2, padding=0)

        # ---- Learnable complex scale (shared with other models) ----
        self.output_scale = nn.Parameter(torch.tensor(0.5 + 0j, dtype=torch.complex64))

        self._count_params()

    def forward(self, mixture):
        # mixture: [B, 1, T] complex
        Bsz = mixture.shape[0]

        # 1. Encode to latent representation
        w = self.encoder(mixture)          # [B, N, T']
        w_norm = self.encoder_norm(w)      # [B, N, T']

        # 2. Bottleneck
        x = self.bottleneck(w_norm)        # [B, B, T']

        # 3. TCN separation with skip connections
        skip_sum = 0.0
        for block in self.tcn_blocks:
            residual, skip = block(x)
            x = x + residual               # residual path → next block
            skip_sum = skip_sum + skip     # skip-connection sum

        # 4. Estimate masks from skip-connection sum
        masks = self.mask_prelu(skip_sum)           # [B, Sc, T']
        masks = self.mask_conv(masks)               # [B, 2*N, T']
        masks = masks * self.output_scale           # [B, 2*N, T']
        masks = masks.view(Bsz, 2, self.N, -1)      # [B, 2, N, T']

        # 5. Apply masks to (normalized) encoder output
        s1_w = w * masks[:, 0, :, :]     # [B, N, T']
        s2_w = w * masks[:, 1, :, :]     # [B, N, T']

        # 6. Decode each masked representation back to waveform
        s1 = self.decoder(s1_w)           # [B, 1, T]
        s2 = self.decoder(s2_w)           # [B, 1, T]

        return s1, s2

    def _count_params(self):
        n = sum(p.numel() for p in self.parameters())
        print(f"[ComplexConvTasNet] Total parameters: {n:,} ({n/1e6:.3f}M)")


# =============================================================================
# CNSE: Convolutional time-domain Network with Squeeze-and-Excitation blocks
# Hou & Gao, "Single-channel blind separation of co-frequency signals based on
# convolutional network," Digital Signal Processing, vol. 129, p. 103654, 2022.
#
# Architecture (scaled-down for 8GB GPU; original uses hidden=512, ~18M params):
#   Input:  [B, 2, T]  (Re and Im of mixture as 2 channels)
#   Encoder: 3 x Conv1D(in=2, hidden, k=16, stride=2) + PReLU
#   Bottleneck: 1x1 Conv1D(hidden, hidden)
#   Separator: 3 x StackedBlock
#     StackedBlock = 3 x SepBlock + 1 x SEBlock (with skip from input)
#       SepBlock = 1x1 Conv + GN + PReLU + DWConv(dil) + GN + PReLU + 1x1 Conv + residual
#       SEBlock  = AvgPool + 1x1 Conv(hidden, hidden/4) + PReLU + 1x1 Conv(hidden/4, hidden)
#   Output: Conv1d(hidden, 4) -> 2 complex sources (Re, Im, Re, Im)
# =============================================================================


class CNSEEncoder(nn.Module):
    """Three Conv1D layers with stride=2 (8x total downsampling) interleaved with PReLU."""
    def __init__(self, in_channels=2, hidden=256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=16, stride=2, padding=7)
        self.prelu1 = nn.PReLU(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=16, stride=2, padding=7)
        self.prelu2 = nn.PReLU(hidden)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=16, stride=2, padding=7)
        self.prelu3 = nn.PReLU(hidden)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        return x  # [B, hidden, T/8]


class CNSESepBlock(nn.Module):
    """SepBlock: 1x1 Conv + GN + PReLU + DWConv(dil) + GN + PReLU + 1x1 Conv + residual."""
    def __init__(self, channels, kernel_size=3, dilation=1, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.prelu1 = nn.PReLU(channels)
        # Use kernel=3 with padding=dilation for length preservation (paper uses k=2
        # with asymmetric padding; k=3 with dilations 1/2/4 gives equivalent receptive
        # field of 3/5/9 vs the paper's 2/3/5).
        self.dwconv = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                                 padding=dilation, dilation=dilation,
                                 groups=channels)
        self.gn2 = nn.GroupNorm(num_groups, channels)
        self.prelu2 = nn.PReLU(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.prelu1(self.gn1(self.conv1(x)))
        out = self.prelu2(self.gn2(self.dwconv(out)))
        out = self.conv2(out)
        return out + residual


class CNSESEBlock(nn.Module):
    """Real-valued SE block with reduction ratio and PReLU activation."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.prelu = nn.PReLU(channels // reduction)
        self.conv2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        z = self.pool(x)            # [B, C, 1]
        z = self.prelu(self.conv1(z))
        z = self.conv2(z)           # [B, C, 1]
        return x * z


class CNSEStackedBlock(nn.Module):
    """StackedBlock = 3 x SepBlock (dilations 1, 2, 4) + 1 x SEBlock, with skip from input."""
    def __init__(self, channels, dilations=(1, 2, 4)):
        super().__init__()
        self.sep1 = CNSESepBlock(channels, dilation=dilations[0])
        self.sep2 = CNSESepBlock(channels, dilation=dilations[1])
        self.sep3 = CNSESepBlock(channels, dilation=dilations[2])
        self.se = CNSESEBlock(channels)

    def forward(self, x):
        residual = x
        out = self.sep1(x)
        out = self.sep2(out)
        out = self.sep3(out)
        out = self.se(out)
        return out + residual


class CNSEDecoder(nn.Module):
    """Three ConvTranspose1D layers with stride=2 (8x total upsampling) + final 1x1 Conv."""
    def __init__(self, hidden=256, out_channels=4):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(hidden, hidden, kernel_size=16, stride=2, padding=7, output_padding=0)
        self.prelu1 = nn.PReLU(hidden)
        self.deconv2 = nn.ConvTranspose1d(hidden, hidden, kernel_size=16, stride=2, padding=7, output_padding=0)
        self.prelu2 = nn.PReLU(hidden)
        self.deconv3 = nn.ConvTranspose1d(hidden, hidden, kernel_size=16, stride=2, padding=7, output_padding=0)
        self.prelu3 = nn.PReLU(hidden)
        self.final = nn.Conv1d(hidden, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.prelu1(self.deconv1(x))
        x = self.prelu2(self.deconv2(x))
        x = self.prelu3(self.deconv3(x))
        return self.final(x)  # [B, 4, T]


class CNSE(nn.Module):
    """Real-valued CNSE-style separation network (Hou & Gao 2022).

    Accepts a complex mixture [B, 1, T] and outputs two complex sources [B, 1, T].
    Internally treats I/Q as 2 real channels and predicts 4 real channels
    (Re(s1), Im(s1), Re(s2), Im(s2)).
    """

    def __init__(self, in_channels=2, hidden=256, n_stacks=3, kernel_size=2,
                 num_groups=8):
        super().__init__()
        self.encoder = CNSEEncoder(in_channels, hidden)
        self.bottleneck_in = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.stacks = nn.ModuleList([
            CNSEStackedBlock(hidden) for _ in range(n_stacks)
        ])
        self.bottleneck_out = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.decoder = CNSEDecoder(hidden, out_channels=4)
        # Learnable real-valued scale (initial output ≈ 0.5 * mixture, for stability).
        self.output_scale = nn.Parameter(torch.tensor(0.5))
        self._count_params()

    def forward(self, mixture):
        # mixture: [B, 1, T] complex
        # Pack Re and Im as 2 real channels
        x = torch.cat([mixture.real, mixture.imag], dim=1)        # [B, 2, T]
        x = self.encoder(x)                                       # [B, hidden, T/8]
        x = self.bottleneck_in(x)
        for stack in self.stacks:
            x = stack(x)
        x = self.bottleneck_out(x)
        out = self.decoder(x) * self.output_scale                 # [B, 4, T]
        # Split into two complex sources
        s1 = torch.complex(out[:, 0], out[:, 1]).unsqueeze(1)     # [B, 1, T]
        s2 = torch.complex(out[:, 2], out[:, 3]).unsqueeze(1)
        return s1, s2

    def _count_params(self):
        n = sum(p.numel() for p in self.parameters())
        print(f"[CNSE] Total parameters: {n:,} ({n/1e6:.3f}M)")


# =============================================================================
# S4-UNET: Structured State Space U-Net for Communication Signal BSS
# Gao, Guo, Shi, Peng, "S4-UNET: A Long-Sequence Modeling Blind Source Separation
# Method for Single-Channel Co-Channel Overlapped Communication Signals,"
# J. Electron. Inf. Technol., 2026, DOI: 10.11999/JEIT251144.
#
# Simplified implementation: a U-Net whose odd-numbered encoder stages embed a
# bidirectional state-space block (FFT convolution + diagonal A parameterization)
# capturing long-range temporal dependencies, and even-numbered stages are
# convolutional. TSEM (Temporal State Enhancement Module) acts as the basic
# encoder/decoder block: LayerNorm + Conv1d + LeakyReLU + Conv1d + LeakyReLU
# + InstanceNorm + Conv1d with a residual sum. Scaled-down for 8GB GPU
# (4 encoder + 3 decoder stages instead of the paper's 5 + 4).
# =============================================================================


class TSEM(nn.Module):
    """Temporal State Enhancement Module (S4-UNET 2026, Fig. 2 left).

    Two residual Conv stacks: each Conv1d is followed by LeakyReLU and the
    second stack has InstanceNorm. A final 1x1 Conv maps back to the channel
    dimension. With two stacked sub-blocks for depth, matching the paper's
    2x residual pattern shown in Fig. 2.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.ln = nn.GroupNorm(1, channels)   # LayerNorm across channel
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.in1 = nn.GroupNorm(channels, channels)  # InstanceNorm (per-channel)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.in2 = nn.GroupNorm(channels, channels)
        self.conv4 = nn.Conv1d(channels, channels, kernel_size, padding=pad)

    def forward(self, x):
        residual = x
        out = self.ln(x)
        out = self.lrelu1(self.conv1(out))
        out = self.lrelu2(self.conv2(out))
        out = out + residual
        # Second sub-block with InstanceNorm
        residual2 = out
        out = self.in1(out)
        out = self.lrelu3(self.conv3(out))
        out = self.in2(out)
        out = self.conv4(out)
        return out + residual2


class S4DSimpleBlock(nn.Module):
    """Simplified S4D-style block: diagonal A, FFT-based fast convolution.

    Approximates the original S4 with diagonal A + bilinear discretization.
    Bi-directional via separate forward/backward passes whose outputs are summed.

    Args:
        channels: feature dimension (C).
        state_dim: SSM state dimension N (paper uses N=64, here N=32 for GPU memory).
        kernel_size: depthwise conv kernel inside the SSM feed-forward.
    """
    def __init__(self, channels, state_dim=32, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.state_dim = state_dim
        # Log-diagonal A parameter: A = -exp(log_A_real) + 1j*pi*log_A_imag
        # Initialised so eigenvalues are inside unit circle (stable).
        log_real = torch.log(torch.arange(1, state_dim + 1).float())
        self.log_A_real = nn.Parameter(log_real.view(1, state_dim, 1).expand(channels, -1, 1).clone())
        self.log_A_imag = nn.Parameter(torch.zeros(channels, state_dim, 1))
        # B, C parameters: shape [C, N]
        self.B = nn.Parameter(torch.randn(channels, state_dim, 1) * 0.5)
        self.C = nn.Parameter(torch.randn(channels, state_dim, 1) * 0.5)
        # Skip connection D (per-channel)
        self.D = nn.Parameter(torch.zeros(1, channels, 1))
        # Depthwise conv before SSM (input mixer)
        self.dwconv = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=channels)
        # Output linear
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)
        # Dropout
        self.dropout = nn.Dropout(0.1)
        # GeLU activation (paper uses GeLU for the S4 stage)
        self.act = nn.GELU()

    def _ssm_step(self, x):
        """x: [B, C, T] -> y: [B, C, T] via FFT convolution with discretised SSM kernel."""
        B, C, T = x.shape
        N = self.state_dim
        # Discretise A: A_bar = exp(Delta * A), with fixed Delta=1.0 here.
        A_real = -torch.exp(self.log_A_real)  # [C, N, 1]
        A_imag = torch.pi * self.log_A_imag
        # Magnitude decays (stable); oscillate with imag part.
        # Build discretised kernel B_bar in time domain by IFFT of (B / (1 - z * A_bar)).
        # For diagonal SSM the kernel is sum_n C_n * A_bar_n^k for k=0..T-1.
        # We compute it via a closed-form recurrence of length T.
        # A_bar magnitude decays, so kernel length can be truncated to T (no pad).
        # Initialise discretised B via first-order hold: B_bar = (exp(Delta*A) - I)/(A * Delta) * B
        # For diagonal A, closed form per channel n:
        A_bar_real = torch.exp(A_real.squeeze(-1)) * torch.cos(A_imag.squeeze(-1))  # [C, N]
        A_bar_imag = torch.exp(A_real.squeeze(-1)) * torch.sin(A_imag.squeeze(-1))  # [C, N]
        B_real = self.B.squeeze(-1)  # [C, N]
        B_imag = torch.zeros_like(B_real)
        # C: [C, N]
        C_real = self.C.squeeze(-1)
        C_imag = torch.zeros_like(C_real)
        # Compute kernel[k] = sum_n C_n * A_bar_n^k * B_n   for k=0..T-1
        # A_bar_n^k = exp(k*log(A_bar_n))
        log_abs = torch.log(A_bar_real.clamp(min=1e-8))  # [C, N]
        theta = torch.atan2(A_bar_imag, A_bar_real)      # [C, N]
        # kernel[c, k] = sum_n |A_bar_n|^k * cos(k*theta_n + phi_n) * |C_n B_n|
        # where phi_n = arg(C_n) + arg(B_n) (both zero here)
        # vectorize over (C, N, T):
        k_range = torch.arange(T, device=x.device).view(1, 1, T)  # [1, 1, T]
        log_kernel = log_abs.unsqueeze(-1) * k_range             # [C, N, T]
        angle = theta.unsqueeze(-1) * k_range                    # [C, N, T]
        # Magnitude: |A_bar_n|^k * |C_n * B_n|; phase: k*theta_n + arg(C_n*B_n)
        cb_mag = (C_real * B_real).abs()  # [C, N]
        cb_ang = torch.atan2(C_imag * B_real + C_real * B_imag,
                             C_real * B_real - C_imag * B_imag + 1e-8)  # [C, N]
        # kernel_real[c, k] = sum_n |A_bar_n|^k * |C_n B_n| * cos(k*theta_n + phi_n)
        # kernel_imag[c, k] = sum_n |A_bar_n|^k * |C_n B_n| * sin(k*theta_n + phi_n)
        # log_kernel is in log-space; use exp().
        magnitude = cb_mag.unsqueeze(-1) * torch.exp(log_kernel)  # [C, N, T]
        kernel_real = (magnitude * torch.cos(angle + cb_ang.unsqueeze(-1))).sum(dim=1)  # [C, T]
        kernel_imag = (magnitude * torch.sin(angle + cb_ang.unsqueeze(-1))).sum(dim=1)  # [C, T]
        kernel = torch.complex(kernel_real, kernel_imag)         # [C, T]

        # FFT convolution
        fft_size = 2 * T
        K_fft = torch.fft.fft(kernel, n=fft_size, dim=-1)        # [C, fft_size]
        X_fft = torch.fft.fft(x, n=fft_size, dim=-1)             # [B, C, fft_size]
        Y_fft = X_fft * K_fft.unsqueeze(0)                       # [B, C, fft_size]
        y_complex = torch.fft.ifft(Y_fft, n=fft_size, dim=-1)[..., :T]  # [B, C, T]

        # Skip connection via D (shape [1, channels, 1] broadcasts with x [B, C, T])
        y = y_complex + self.D * x
        return y

    def forward(self, x):
        # x: [B, C, T] real
        # Bidirectional SSM: forward pass + backward pass on flipped input.
        x_dw = self.dwconv(x)
        y_fwd = self._ssm_step(x_dw)
        y_bwd = self._ssm_step(x_dw.flip(-1)).flip(-1)
        y = y_fwd + y_bwd
        # Take magnitude to convert complex -> real (phase-free), then activation.
        # The complex-valued SSM is treated as a long-range temporal feature mixer;
        # the activation and projection operate on the magnitude which is invariant
        # to per-sample phase rotation.
        y_real = y.abs()
        y_real = self.act(y_real)
        y_real = self.dropout(y_real)
        return self.out_proj(y_real)


class S4UNETStage(nn.Module):
    """Encoder/decoder stage: TSEM + (optional) S4 block, with downsampling/upsampling.

    For encoder: TSEM/S4 -> merge to out_channels -> skip (pre-downsample) -> downsample.
    For decoder: TSEM/S4 on concat -> merge -> upsample.
    """
    def __init__(self, in_channels, out_channels, use_s4=False, down=True,
                 kernel_size=3, state_dim=32):
        super().__init__()
        self.tsem = TSEM(in_channels, kernel_size=kernel_size)
        self.use_s4 = use_s4
        self.down = down
        if use_s4:
            self.s4 = S4DSimpleBlock(in_channels, state_dim=state_dim,
                                     kernel_size=kernel_size)
            self.merge_pre = nn.Conv1d(in_channels * 2, out_channels, kernel_size=1)
        else:
            self.merge_pre = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        if down:
            self.pool = nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            # Decoder upsamples after the merge.
            self.up = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, return_skip=False):
        tsem_out = self.tsem(x)
        if self.use_s4:
            s4_out = self.s4(tsem_out)
            merged = self.merge_pre(torch.cat([tsem_out, s4_out], dim=1))
        else:
            merged = self.merge_pre(tsem_out)
        if self.down:
            skip = merged
            out = self.pool(merged)
        else:
            out = self.up(merged)
            skip = None
        return (out, skip) if return_skip else out


class S4UNETDecoderStage(nn.Module):
    """Decoder stage: takes deeper features + encoder skip, processes, then upsamples.

    Per the paper's U-NET pattern: concat skip with deeper features, process through
    TSEM/S4 blocks, then upsample before passing to the next decoder stage.
    """
    def __init__(self, in_channels, skip_channels, out_channels, use_s4=False,
                 kernel_size=3, state_dim=32):
        super().__init__()
        # in_channels: channels from the deeper decoder layer
        # skip_channels: channels from the corresponding encoder skip
        # Combined channels after concat: in_channels + skip_channels
        combined = in_channels + skip_channels
        self.tsem = TSEM(combined, kernel_size=kernel_size)
        self.use_s4 = use_s4
        if use_s4:
            self.s4 = S4DSimpleBlock(combined, state_dim=state_dim,
                                     kernel_size=kernel_size)
            self.merge = nn.Conv1d(combined * 2, out_channels, kernel_size=1)
        else:
            self.merge = nn.Conv1d(combined, out_channels, kernel_size=1)
        # Upsample after merge (so output temporal dim doubles for next decoder stage)
        self.up = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        tsem_out = self.tsem(x)
        if self.use_s4:
            s4_out = self.s4(tsem_out)
            out = self.merge(torch.cat([tsem_out, s4_out], dim=1))
        else:
            out = self.merge(tsem_out)
        out = self.up(out)
        return out


class S4UNET(nn.Module):
    """S4-UNET style architecture (simplified for 8GB GPU).

    Real-valued 2-channel (I/Q) input. 4 encoder stages + 3 decoder stages
    (instead of paper's 5+4). Odd-numbered encoder stages embed S4.

    Args:
        base_channels: width of the first stage (paper uses 32; default 16 here for memory).
        state_dim: SSM state dimension (paper N=64; here default 16).
    """
    def __init__(self, base_channels=16, state_dim=16):
        super().__init__()
        ch = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        # Encoder stages (4 total, odd-numbered with S4)
        self.enc1 = S4UNETStage(2, ch[0], use_s4=True,  down=True,  state_dim=state_dim)
        self.enc2 = S4UNETStage(ch[0], ch[1], use_s4=False, down=True)
        self.enc3 = S4UNETStage(ch[1], ch[2], use_s4=True,  down=True,  state_dim=state_dim)
        self.enc4 = S4UNETStage(ch[2], ch[3], use_s4=False, down=True)
        # Bottleneck: TSEM + S4 on the deepest features (no downsample)
        self.bottleneck = S4UNETStage(ch[3], ch[3], use_s4=True, down=False, state_dim=state_dim)
        # Decoder stages (3): each upsamples its input, concats with skip, processes.
        self.dec3 = S4UNETDecoderStage(in_channels=ch[3], skip_channels=ch[3],
                                         out_channels=ch[2])
        self.dec2 = S4UNETDecoderStage(in_channels=ch[2], skip_channels=ch[2],
                                         out_channels=ch[1])
        self.dec1 = S4UNETDecoderStage(in_channels=ch[1], skip_channels=ch[1],
                                         out_channels=ch[0])
        # Final projection to 4 channels (Re s1, Im s1, Re s2, Im s2)
        self.final = nn.Conv1d(ch[0], 4, kernel_size=1)
        # Learnable scale
        self.output_scale = nn.Parameter(torch.tensor(0.5))
        self._count_params()

    def forward(self, mixture):
        # mixture: [B, 1, T] complex -> [B, 2, T] real (I/Q as 2 channels)
        x = torch.cat([mixture.real, mixture.imag], dim=1)
        # Encoder: collect skips (pre-downsample features)
        x1, skip1 = self.enc1(x, return_skip=True)
        x2, skip2 = self.enc2(x1, return_skip=True)
        x3, skip3 = self.enc3(x2, return_skip=True)
        x4, skip4 = self.enc4(x3, return_skip=True)
        # Bottleneck (no downsample, but project channels)
        bn = self.bottleneck(x4)
        # bn is at T/16 (ch[3]=128). Upsample to T/8 to match skip4, then concat + process.
        bn_up = nn.functional.interpolate(bn, size=skip4.shape[-1], mode='linear', align_corners=False)
        # Decoder stages: each upsample doubles temporal resolution, then concat skip.
        d3 = self.dec3(bn_up, skip4)   # output: T/4, ch[2]
        d2 = self.dec2(d3, skip3)       # output: T/2, ch[1]
        d1 = self.dec1(d2, skip2)       # output: T, ch[0]
        out = self.final(d1) * self.output_scale                  # [B, 4, T]
        s1 = torch.complex(out[:, 0], out[:, 1]).unsqueeze(1)
        s2 = torch.complex(out[:, 2], out[:, 3]).unsqueeze(1)
        return s1, s2

    def _count_params(self):
        n = sum(p.numel() for p in self.parameters())
        print(f"[S4UNET] Total parameters: {n:,} ({n/1e6:.3f}M)")

# =============================================================================
# Utility: Model Summary
# =============================================================================
def model_summary(model, input_shape=(1, 1, 4096)):
    """Print model summary including parameter count and memory estimate."""
    device = next(model.parameters()).device
    dummy = torch.randn(input_shape).to(torch.complex64).to(device)

    # Forward pass to check
    with torch.no_grad():
        s1, s2 = model(dummy)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory (params + forward activation)
    param_mem = n_params * 8 / 1e6  # complex64 = 8 bytes per param
    act_mem = dummy.numel() * 8 / 1e6 * 4  # rough activation estimate

    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {n_params:,} ({n_params/1e6:.3f}M)")
    print(f"Trainable:  {n_trainable:,}")
    print(f"Output shape: {s1.shape}, {s2.shape}")
    print(f"Estimated param memory: {param_mem:.2f} MB")
    print(f"Estimated total memory: {param_mem + act_mem:.2f} MB")
    print("=" * 60)
    return n_params


# =============================================================================
# Quick Test
# =============================================================================
if __name__ == '__main__':
    print("Testing models...\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dummy = torch.randn(2, 1, 4096).to(torch.complex64).to(device)

    # Test proposed model
    print("--- Proposed Model ---")
    model = ComplexLightweightSepNet(hidden_channels=32, n_layers=4, use_se=True).to(device)
    model_summary(model)
    s1, s2 = model(dummy)
    print(f"Output: s1={s1.shape}, s2={s2.shape}\n")

    # Test without SE (ablation)
    print("--- Without SE (ablation) ---")
    model_no_se = ComplexLightweightSepNet(hidden_channels=32, n_layers=4, use_se=False).to(device)
    model_summary(model_no_se)

    # Test real-valued baseline
    print("--- Real-Valued Baseline ---")
    baseline = RealValuedBaseline(hidden=64, n_layers=6).to(device)
    model_summary(baseline)
    s1b, s2b = baseline(dummy)
    print(f"Output: s1={s1b.shape}, s2={s2b.shape}\n")

    # Test simple complex CNN
    print("--- Simple Complex CNN ---")
    simple = SimpleComplexCNN(hidden=48, n_layers=6).to(device)
    model_summary(simple)

    # Test complex Conv-TasNet
    print("--- Complex Conv-TasNet ---")
    conv_tasnet = ComplexConvTasNet(N=64, B=64, Sc=64, H=128, P=3, X=5, R=3, L=16).to(device)
    model_summary(conv_tasnet)
    s1ct, s2ct = conv_tasnet(dummy)
    print(f"Output: s1={s1ct.shape}, s2={s2ct.shape}\n")

    print("All model tests passed!")
