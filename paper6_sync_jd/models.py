"""Paper 6 — vendored SlotSepNet baseline model (self-contained).

Copied from paper5_task_oriented/models.py on 2026-09-19: the complex-valued
base layers (which paper5 itself copied verbatim from paper3_open_set/
models.py, which copied them from paper1_cnn_se), CSEBackbone, the
OccupancyHead/CountHead, and SlotSepNet.  Dropped: StopHead, JointLLRHead,
OneAndRestNet, SpecialistBankNet — not used by paper 6.  SlotSepNet's
use_joint_head parameter is kept for checkpoint-construction compatibility
but must be False (the head class is not vendored).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Complex-valued base layers (copied via paper5 from paper3/paper1)
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
# Shared C-SE backbone (paper3 OpenSetCSE topology, decoder factored out)
# ============================================================================
class CSEBackbone(nn.Module):
    """encoder -> n_layers x ComplexResidualBlock. Returns bottleneck x."""

    def __init__(self, hidden_channels=64, n_layers=4,
                 kernel_size_enc=7, kernel_size_hidden=3,
                 use_se=True, se_reduction=4):
        super().__init__()
        self.encoder = nn.Sequential(
            ComplexConv1d(1, hidden_channels, kernel_size_enc,
                          padding=kernel_size_enc // 2),
            ComplexBatchNorm1d(hidden_channels),
            ComplexReLU(),
        )
        self.res_blocks = nn.ModuleList([
            ComplexResidualBlock(hidden_channels, kernel_size_hidden,
                                  use_se, se_reduction)
            for _ in range(n_layers)
        ])

    def forward(self, mixture):
        # mixture: [B, 1, T] complex -> x: [B, H, T] complex
        x = self.encoder(mixture)
        for block in self.res_blocks:
            x = block(x)
        return x


def _pool_complex(x):
    """Pool complex features [B, C, T] -> real vector [B, 2C]
    (real/imag pooled separately, then concat — same recipe as the
    ComplexSEBlock squeeze and paper3's ModulationHead)."""
    return torch.cat([x.real.mean(dim=-1), x.imag.mean(dim=-1)], dim=1)


# ============================================================================
# Heads
# ============================================================================
class OccupancyHead(nn.Module):
    """Shared per-slot occupancy classifier: masked features (mask_m * x)
    -> pool -> MLP -> 1 logit.

    The same module is applied to every slot, so it must learn a single
    "does this mask isolate a real source?" decision that works for any
    slot after PIT alignment.  sigma(logit) doubles as the output gate and
    the emergent counter (K_hat = # slots above threshold).
    """

    def __init__(self, in_channels: int, embed_dim: int = 64):
        super().__init__()
        pool_dim = 2 * in_channels
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, masked_features):
        """masked_features: [B, C, T] complex -> logit [B]."""
        return self.mlp(_pool_complex(masked_features)).squeeze(-1)


class CountHead(nn.Module):
    """Explicit K-classifier on the pooled bottleneck embedding.

    Outputs 3 logits for K in {1, 2, 3}.  By construction it CANNOT
    extrapolate to K=4.
    """

    def __init__(self, in_channels: int, embed_dim: int = 64,
                 n_count_classes: int = 3):
        super().__init__()
        pool_dim = 2 * in_channels
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, n_count_classes),
        )

    def forward(self, bottleneck):
        """bottleneck: [B, C, T] complex -> count_logits [B, n_classes]."""
        return self.mlp(_pool_complex(bottleneck))


# ============================================================================
# SlotSepNet — headline slot-occupancy model
# ============================================================================
class SlotSepNet(nn.Module):
    """C-SE backbone + K_max complex masks + shared occupancy head.

    Forward returns
    ---------------
    slots        : [B, K_max, T] complex — RAW (ungated) slot outputs;
                   gating by sigma(occ_logits) is applied at inference only
    occ_logits   : [B, K_max]
    count_logits : [B, 3] for K in {1,2,3}, or None if use_count_head=False
    """

    def __init__(self,
                 hidden_channels: int = 64,
                 n_layers: int = 4,
                 k_max: int = 4,
                 kernel_size_enc: int = 7,
                 kernel_size_hidden: int = 3,
                 kernel_size_dec: int = 7,
                 use_se: bool = True,
                 se_reduction: int = 4,
                 head_embed_dim: int = 64,
                 use_count_head: bool = True,
                 use_joint_head: bool = False,
                 sps: int = 16):
        super().__init__()
        if use_joint_head:
            raise NotImplementedError(
                "JointLLRHead is not vendored into paper6 (not used)")
        self.k_max = k_max
        self.hidden_channels = hidden_channels

        self.backbone = CSEBackbone(hidden_channels, n_layers,
                                    kernel_size_enc, kernel_size_hidden,
                                    use_se, se_reduction)
        # Decoder -> K_max complex masks
        self.decoder = ComplexConv1d(hidden_channels, k_max,
                                       kernel_size_dec,
                                       padding=kernel_size_dec // 2)
        # Learnable complex scale; paper1 trick so initial output ~ 0.5*mix
        self.output_scale = nn.Parameter(torch.tensor(0.5 + 0j,
                                                       dtype=torch.complex64))

        self.occ_head = OccupancyHead(hidden_channels, head_embed_dim)
        self.count_head = (CountHead(hidden_channels, head_embed_dim,
                                     n_count_classes=3)
                           if use_count_head else None)

        self._count_params()

    def forward(self, mixture):
        # mixture: [B, 1, T] complex
        x = self.backbone(mixture)                                # [B, H, T]

        masks = self.decoder(x)                                   # [B, K, T]
        scaled_masks = masks * self.output_scale                  # [B, K, T]
        slots = scaled_masks * mixture                            # [B, K, T]

        # Shared occupancy head over per-slot masked features.
        feats = scaled_masks.unsqueeze(2) * x.unsqueeze(1)        # [B, K, H, T]
        b, km, h, t = feats.shape
        occ_logits = self.occ_head(
            feats.reshape(b * km, h, t)).view(b, km)              # [B, K]

        count_logits = self.count_head(x) if self.count_head is not None else None
        return slots, occ_logits, count_logits

    def _count_params(self):
        sections = {
            'backbone.encoder':    self.backbone.encoder,
            'backbone.res_blocks': self.backbone.res_blocks,
            'decoder':             self.decoder,
            'occ_head':            self.occ_head,
        }
        if self.count_head is not None:
            sections['count_head'] = self.count_head
        per_section = {n: sum(p.numel() for p in m.parameters())
                       for n, m in sections.items()}
        per_section['output_scale'] = self.output_scale.numel()
        total = sum(per_section.values())
        breakdown = ', '.join(f'{n}={v:,}' for n, v in per_section.items())
        print(f"[SlotSepNet] Total parameters: {total:,} ({total / 1e3:.1f}K) "
              f"  [{breakdown}]")


# ============================================================================
# Smoke test
# ============================================================================
if __name__ == '__main__':
    print("Testing paper6 models (vendored SlotSepNet) ...")
    torch.manual_seed(0)
    model = SlotSepNet()
    x = (torch.randn(2, 1, 4096) + 1j * torch.randn(2, 1, 4096)) \
        .to(torch.complex64)
    slots, occ_logits, count_logits = model(x)
    assert slots.shape == (2, 4, 4096) and slots.is_complex()
    assert occ_logits.shape == (2, 4) and count_logits.shape == (2, 3)
    # expected baseline parameter count (paper5 config (b) recipe)
    total = sum(p.numel() for p in model.parameters())
    assert total == 254093, total
    print("  forward shapes + param count (254,093): OK")
    print("\nmodels smoke test passed!")
