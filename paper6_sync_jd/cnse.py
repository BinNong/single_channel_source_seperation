"""Paper 6 — vendored CNSE separator (Hou & Gao 2022 strong baseline).

Copied VERBATIM from paper1_cnn_se/models.py (lines ~596-735) on 2026-10-05,
per the repo convention "copy, not share, for models".  Real-valued network:
complex mixture [B, 1, T] in (I/Q packed as 2 real channels) -> two complex
sources [B, 1, T] out.  Standard torch only.

Reference: Hou & Gao, "Single-channel blind separation of co-frequency
signals based on convolutional network," Digital Signal Processing,
vol. 129, p. 103654, 2022.

Usage:
    python cnse.py          # smoke test (forward pass + param count)
"""
from __future__ import annotations

import torch
import torch.nn as nn


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


# ============================================================================
# Smoke test
# ============================================================================
if __name__ == '__main__':
    print("Testing paper6 vendored CNSE ...")
    torch.manual_seed(0)
    B, T = 2, 4096
    mix = (torch.randn(B, 1, T)
           + 1j * torch.randn(B, 1, T)).to(torch.complex64)
    model = CNSE(hidden=256, n_stacks=3)
    s1, s2 = model(mix)
    assert s1.shape == (B, 1, T) and s2.shape == (B, 1, T), \
        f"bad shapes: {s1.shape} {s2.shape}"
    assert s1.dtype == torch.complex64 and s2.dtype == torch.complex64
    loss = s1.abs().pow(2).mean() + s2.abs().pow(2).mean()
    loss.backward()
    print(f"  in {tuple(mix.shape)} {mix.dtype} -> "
          f"s1 {tuple(s1.shape)} {s1.dtype}, s2 {tuple(s2.shape)} {s2.dtype}")
    print("\ncnse.py smoke test passed!")
