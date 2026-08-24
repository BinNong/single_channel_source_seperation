"""
Paper 3 — Open-Set SC-BSS: Models.

Self-contained:
  - All complex-valued base layers copied from paper1_cnn_se/models.py so
    paper3 does not import paper1 at runtime (paper3 may run side-by-side
    with paper1 experiments without import side effects).
  - Adds ModulationHead (per-source embedding + classifier) and OpenSetCSE
    (C-SE backbone + per-source head).

Design choice for OpenSetCSE:
  - The two source heads share weights (a single ModulationHead module applied
    to both feat1 and feat2).  Sharing keeps the head small (~8.6K params) and
    matches the permutation-invariant training regime: the model must learn
    one classifier that works for "the first source" and "the second source"
    after PIT alignment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Complex-valued base layers (copied verbatim from paper1_cnn_se/models.py
# so paper3 stays self-contained)
# ============================================================================
class ComplexConv1d(nn.Module):
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
        return torch.complex(
            self.real_conv(x.real) - self.imag_conv(x.imag),
            self.real_conv(x.imag) + self.imag_conv(x.real),
        )


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn_real = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        self.bn_imag = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x):
        return torch.complex(self.bn_real(x.real), self.bn_imag(x.imag))


class ComplexReLU(nn.Module):
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))


class ComplexSEBlock(nn.Module):
    """Complex SE block (paper1's exact implementation)."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(2 * channels, 2 * channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2 * channels // reduction, 2 * channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _t = x.shape
        z = torch.cat([self.avg_pool(x.real).view(b, c),
                        self.avg_pool(x.imag).view(b, c)], dim=1)  # [B, 2C]
        scale = self.fc(z)
        scale_real = scale[:, :c].view(b, c, 1)
        scale_imag = scale[:, c:2 * c].view(b, c, 1)
        weight = (scale_real + scale_imag) / 2       # real weight, preserves phase
        return x * weight


class ComplexResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, use_se=True, se_reduction=4):
        super().__init__()
        p = kernel_size // 2
        self.conv1 = ComplexConv1d(channels, channels, kernel_size, padding=p)
        self.bn1   = ComplexBatchNorm1d(channels)
        self.relu  = ComplexReLU()
        self.conv2 = ComplexConv1d(channels, channels, kernel_size, padding=p)
        self.bn2   = ComplexBatchNorm1d(channels)
        self.se    = ComplexSEBlock(channels, se_reduction) if use_se else None

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        return self.relu(out + residual)


# ============================================================================
# ModulationHead (per-source embedding + K-way classifier)
# ============================================================================
class ModulationHead(nn.Module):
    """Embed per-source masked bottleneck features, classify modulation.

    Inputs come from the C-SE bottleneck features (post-residual-blocks,
    pre-decoder) after element-wise multiplication with the per-source
    complex mask.  Because the mask is what the backbone has learned to
    produce for each source, the masked features are already a
    source-specific representation — pooling them gives a compact
    embedding that:
      - For known modulations: clusters by modulation class (so
        Prototype and VOS scoring work).
      - For unknown modulations: lands far from any known prototype
        (so OOD detection works).
    """

    def __init__(self, in_channels: int, embed_dim: int, num_known: int):
        super().__init__()
        pool_dim = 2 * in_channels                       # real+imag concat
        self.embed = nn.Sequential(
            nn.Linear(pool_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_known)
        self.embed_dim = embed_dim
        self.num_known = num_known

    def forward(self, masked_features):
        """masked_features: [B, C, T] complex.
        Returns
        -------
        embedding : [B, embed_dim]
        logits    : [B, num_known]
        """
        # Pool real and imag separately, then concat (same recipe as the
        # ComplexSEBlock squeeze, applied channel-wise here).
        b, c, _t = masked_features.shape
        pooled = torch.cat(
            [masked_features.real.mean(dim=-1),
             masked_features.imag.mean(dim=-1)],
            dim=1,
        )                                                 # [B, 2C]
        emb = self.embed(pooled)                          # [B, embed_dim]
        logits = self.classifier(emb)                     # [B, num_known]
        return emb, logits


# ============================================================================
# OpenSetCSE — C-SE backbone + per-source modulation head
# ============================================================================
class OpenSetCSE(nn.Module):
    """C-SE separation backbone + shared ModulationHead for per-source OOD.

    Forward returns
    ---------------
    s1, s2    : [B, 1, T] complex — separated sources
    emb1, emb2: [B, embed_dim]  — per-source embeddings (for OOD scoring)
    logits1, logits2: [B, num_known] — per-source modulation logits
    """

    def __init__(self,
                 hidden_channels: int = 32,
                 n_layers: int = 4,
                 kernel_size_enc: int = 7,
                 kernel_size_hidden: int = 3,
                 kernel_size_dec: int = 7,
                 use_se: bool = True,
                 se_reduction: int = 4,
                 embed_dim: int = 64,
                 num_known_classes: int = 4):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            ComplexConv1d(1, hidden_channels, kernel_size_enc,
                          padding=kernel_size_enc // 2),
            ComplexBatchNorm1d(hidden_channels),
            ComplexReLU(),
        )
        # Residual stack
        self.res_blocks = nn.ModuleList([
            ComplexResidualBlock(hidden_channels, kernel_size_hidden,
                                  use_se, se_reduction)
            for _ in range(n_layers)
        ])
        # Decoder -> 2 complex masks
        self.decoder = ComplexConv1d(hidden_channels, 2,
                                       kernel_size_dec,
                                       padding=kernel_size_dec // 2)
        # Learnable complex scale; paper1 trick so initial output ~ 0.5*mix
        self.output_scale = nn.Parameter(torch.tensor(0.5 + 0j,
                                                       dtype=torch.complex64))

        # Per-source modulation head (shared weights; PIT-aligned training)
        self.head = ModulationHead(hidden_channels, embed_dim, num_known_classes)
        self.hidden_channels = hidden_channels
        self.embed_dim = embed_dim
        self.num_known_classes = num_known_classes

        self._count_params()

    def forward(self, mixture):
        # mixture: [B, 1, T] complex
        x = self.encoder(mixture)                                # [B, H, T]
        for block in self.res_blocks:
            x = block(x)                                          # [B, H, T]

        masks = self.decoder(x)                                   # [B, 2, T]
        scaled_masks = masks * self.output_scale                  # [B, 2, T]
        s1 = scaled_masks[:, 0:1, :] * mixture
        s2 = scaled_masks[:, 1:2, :] * mixture

        # Per-source masked features for the head.
        feat1 = scaled_masks[:, 0:1, :] * x                       # [B, H, T]
        feat2 = scaled_masks[:, 1:2, :] * x                       # [B, H, T]
        emb1, logits1 = self.head(feat1)
        emb2, logits2 = self.head(feat2)

        return s1, s2, emb1, emb2, logits1, logits2

    def _count_params(self):
        sections = {
            'encoder':     self.encoder,
            'res_blocks':  self.res_blocks,
            'decoder':     self.decoder,
            'head':        self.head,
        }
        per_section = {n: sum(p.numel() for p in m.parameters())
                       for n, m in sections.items()}
        per_section['output_scale'] = self.output_scale.numel()
        total = sum(per_section.values())
        breakdown = ', '.join(f'{n}={v:,}' for n, v in per_section.items())
        print(f"[OpenSetCSE] Total parameters: {total:,} ({total / 1e3:.1f}K) "
              f"  [{breakdown}]")


# ============================================================================
# Smoke test
# ============================================================================
if __name__ == '__main__':
    print("Testing OpenSetCSE ...")
    dummy = torch.randn(2, 1, 4096).to(torch.complex64)
    model = OpenSetCSE(hidden_channels=32, n_layers=4,
                       embed_dim=64, num_known_classes=4)
    s1, s2, emb1, emb2, logits1, logits2 = model(dummy)
    print(f"  s1={tuple(s1.shape)}  s2={tuple(s2.shape)}")
    print(f"  emb1={tuple(emb1.shape)}  emb2={tuple(emb2.shape)}")
    print(f"  logits1={tuple(logits1.shape)}  logits2={tuple(logits2.shape)}")
    print("OpenSetCSE smoke test passed!")