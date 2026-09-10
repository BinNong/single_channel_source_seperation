"""
Paper 5 — Task-Oriented SC-BSS: Models (paper4 base + S3 JointLLRHead).

Self-contained:
  - All complex-valued base layers copied verbatim from
    paper3_open_set/models.py (which copied them from paper1_cnn_se) —
    repo convention is copy-not-share for model code.
  - SlotSepNet   : headline slot-occupancy model (spec §5.1).
                   C-SE backbone -> K_max complex masks; a shared
                   OccupancyHead per slot doubles as (i) output gate and
                   (ii) emergent counter K_hat = sum(sigma(o_m) > tau);
                   an optional CountHead (K in {1,2,3} classifier) is the
                   ablation reference that CANNOT extrapolate to K=4.
  - OneAndRestNet: recursive one-and-rest comparison model (spec §5.2).
                   Same backbone, 2 masks ("one"/"rest") + stop head;
                   weights shared across steps.
  - SpecialistBankNet: EDSNet-style ablation A6 (closest prior art,
                   Modern Radar 2026): CountHead (K in {1,2,3}) + a bank of
                   per-K specialist decoders, routed by true K in training
                   (teacher routing) and by predicted K at inference.  Its
                   architectural inability to represent K=4 (count head
                   capped at 3 classes, no K=4 specialist) is the intended
                   failure demonstration vs. our unified variable-K archs.

Forward returns RAW (ungated) slot outputs plus occ/count logits —
gating is an inference-time decision, the training loss sees raw outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Complex-valued base layers (copied verbatim from paper3_open_set/models.py
# so paper4 stays self-contained)
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
    """Shared per-slot occupancy classifier (paper3 shared-ModulationHead
    pattern): masked features (mask_m * x) -> pool -> MLP -> 1 logit.

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
    extrapolate to K=4 — this is the ablation reference that highlights
    the occupancy mechanism's zero-shot counting ability.
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


class StopHead(nn.Module):
    """OneAndRestNet stop classifier: rest features -> 1 logit.

    Target: "more sources remain after this step" (1) vs "done" (0).
    """

    def __init__(self, in_channels: int, embed_dim: int = 64):
        super().__init__()
        pool_dim = 2 * in_channels
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, rest_features):
        """rest_features: [B, C, T] complex -> logit [B]."""
        return self.mlp(_pool_complex(rest_features)).squeeze(-1)


# ============================================================================
# JointLLRHead — paper 5, S3: joint soft-information demodulation
# ============================================================================
class JointLLRHead(nn.Module):
    """Per-slot symbol-posterior head (shared across slots).

    Taps the per-slot masked features, the un-gated backbone bottleneck
    AND the slot waveform itself, so the head demodulates each slot while
    still seeing the joint (un-separated) representation — the neural
    analogue of joint multiuser detection (plan v2, S3).

    Input  : [P, 2H+1, T] complex  (P = B * K_slots)
    Output : [P, N_sym, 16] real logits — per-symbol posteriors over a
             shared label space of capacity M_max = 16 (label j = j-th
             point of the sample's own constellation); per-modulation
             masking is applied by the loss / evaluation, not here.

    Frame-rate -> symbol-rate reduction is a LEARNED strided complex conv
    (k=2*sps, stride=sps).  NOTE: the v1 head used a fixed window MEAN,
    which provably destroys the signal: the carrier (2000 Hz at fs 16000)
    completes exactly 2 cycles per 16-sample window, so its boxcar mean is
    exactly zero and the head was starved of carrier-coherent information
    (v1 result: joint SER 0.73 ≈ chance, 2026-09-03).  The learned strided
    conv can implement down-conversion + matched filtering on its own.
    """

    def __init__(self, hidden_channels: int = 64, sps: int = 16,
                 n_out: int = 16):
        super().__init__()
        self.sps = sps
        H2 = 2 * hidden_channels + 1
        self.in_proj = nn.Sequential(
            ComplexConv1d(H2, 32, kernel_size=1),
            ComplexBatchNorm1d(32),
            ComplexReLU(),
        )
        # Learned down-sampler to symbol rate (replaces the v1 window mean).
        # k=2*sps, stride=sps, padding=sps//2: output length = T//sps and
        # position i covers samples [i*sps - sps/2, i*sps + 3*sps/2).
        self.to_symrate = nn.Sequential(
            ComplexConv1d(32, 32, kernel_size=2 * sps, stride=sps,
                          padding=sps // 2),
            ComplexBatchNorm1d(32),
            ComplexReLU(),
        )
        self.sym_blocks = nn.Sequential(
            ComplexConv1d(32, 32, kernel_size=5, padding=2),
            ComplexBatchNorm1d(32),
            ComplexReLU(),
            ComplexConv1d(32, 32, kernel_size=5, padding=2),
            ComplexBatchNorm1d(32),
            ComplexReLU(),
        )
        self.to_logits = nn.Conv1d(64, n_out, kernel_size=1)  # real+imag -> logits

    def forward(self, x):
        # x: [P, 2H+1, T] complex -> [P, N_sym, 16] real logits
        assert x.shape[-1] % self.sps == 0, (
            f"T={x.shape[-1]} not divisible by sps={self.sps}; "
            "N_sym would not match the symbol grid")
        h = self.in_proj(x)                                   # [P, 32, T]
        h = self.to_symrate(h)                                # [P, 32, N_sym]
        h = self.sym_blocks(h)                                # [P, 32, N_sym]
        h = torch.cat([h.real, h.imag], dim=1)                # [P, 64, N_sym]
        return self.to_logits(h).transpose(1, 2)              # [P, N_sym, 16]


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
    sym_logits   : [B, K_max, N_sym, 16] — ONLY when use_joint_head=True
                   (paper 5 S3; returned as a 4th element)
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
        self.joint_head = (JointLLRHead(hidden_channels, sps=sps)
                           if use_joint_head else None)

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
        if self.joint_head is not None:
            # Joint head input: per-slot masked features + the UN-GATED
            # bottleneck (the joint view) + the slot waveform itself
            # (carries the carrier-coherent signal the head demodulates).
            joint_in = torch.cat(
                [feats, x.unsqueeze(1).expand_as(feats),
                 slots.unsqueeze(2)], dim=2)                      # [B,K,2H+1,T]
            sym_logits = self.joint_head(
                joint_in.reshape(b * km, 2 * h + 1, t))           # [B*K, N, 16]
            sym_logits = sym_logits.view(b, km, *sym_logits.shape[1:])
            return slots, occ_logits, count_logits, sym_logits
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
        if self.joint_head is not None:
            sections['joint_head'] = self.joint_head
        per_section = {n: sum(p.numel() for p in m.parameters())
                       for n, m in sections.items()}
        per_section['output_scale'] = self.output_scale.numel()
        total = sum(per_section.values())
        breakdown = ', '.join(f'{n}={v:,}' for n, v in per_section.items())
        print(f"[SlotSepNet] Total parameters: {total:,} ({total / 1e3:.1f}K) "
              f"  [{breakdown}]")


# ============================================================================
# OneAndRestNet — recursive one-and-rest comparison model
# ============================================================================
class OneAndRestNet(nn.Module):
    """Same C-SE backbone, decoder -> 2 masks ("one" / "rest") + stop head.

    Weights are shared across steps: at inference the model is unrolled
    mixture -> (s1, rest1) -> (s2, rest2) -> ... until the stop head fires
    or max_steps is reached.

    forward_step(remainder)
    -----------------------
    remainder : [B, 1, T] complex — current residual mixture
    returns one [B, 1, T] complex, rest [B, 1, T] complex,
            stop_logit [B]

    forward(mixture, max_steps)
    ---------------------------
    Inference unroll (predicted rest feeds the next step).
    Returns ones [B, max_steps, T] complex, stop_logits [B, max_steps].
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
                 head_embed_dim: int = 64):
        super().__init__()
        self.k_max = k_max
        self.hidden_channels = hidden_channels

        self.backbone = CSEBackbone(hidden_channels, n_layers,
                                    kernel_size_enc, kernel_size_hidden,
                                    use_se, se_reduction)
        # Decoder -> 2 complex masks: "one" (extracted source) + "rest"
        self.decoder = ComplexConv1d(hidden_channels, 2,
                                       kernel_size_dec,
                                       padding=kernel_size_dec // 2)
        self.output_scale = nn.Parameter(torch.tensor(0.5 + 0j,
                                                       dtype=torch.complex64))
        self.stop_head = StopHead(hidden_channels, head_embed_dim)

        self._count_params()

    def forward_step(self, remainder):
        # remainder: [B, 1, T] complex
        x = self.backbone(remainder)                              # [B, H, T]
        masks = self.decoder(x) * self.output_scale               # [B, 2, T]
        one = masks[:, 0:1, :] * remainder                        # [B, 1, T]
        rest = masks[:, 1:2, :] * remainder                       # [B, 1, T]
        # Stop head reads the rest-masked bottleneck features.
        rest_feat = masks[:, 1:2, :] * x                          # [B, H, T]
        stop_logit = self.stop_head(rest_feat)                    # [B]
        return one, rest, stop_logit

    def forward(self, mixture, max_steps: int | None = None):
        if max_steps is None:
            max_steps = self.k_max
        ones, stop_logits = [], []
        remainder = mixture
        for _ in range(max_steps):
            one, rest, stop_logit = self.forward_step(remainder)
            ones.append(one.squeeze(1))                           # [B, T]
            stop_logits.append(stop_logit)                        # [B]
            remainder = rest
        return torch.stack(ones, dim=1), torch.stack(stop_logits, dim=1)

    def _count_params(self):
        sections = {
            'backbone.encoder':    self.backbone.encoder,
            'backbone.res_blocks': self.backbone.res_blocks,
            'decoder':             self.decoder,
            'stop_head':           self.stop_head,
        }
        per_section = {n: sum(p.numel() for p in m.parameters())
                       for n, m in sections.items()}
        per_section['output_scale'] = self.output_scale.numel()
        total = sum(per_section.values())
        breakdown = ', '.join(f'{n}={v:,}' for n, v in per_section.items())
        print(f"[OneAndRestNet] Total parameters: {total:,} ({total / 1e3:.1f}K) "
              f"  [{breakdown}]")


# ============================================================================
# SpecialistBankNet — EDSNet-style ablation A6 (closest prior art)
# ============================================================================
class SpecialistBankNet(nn.Module):
    """Count-classification head + bank of per-K specialist mask decoders
    (EDSNet, Modern Radar 2026), trained with joint count-CE + uPIT.

    Specialist k (k in {1,2,3}) is a ComplexConv1d(hidden, k, 7) producing
    k complex masks; its outputs live in slots[:, :k].  Routing by K is
    done per sample by grouping the batch on `k_route`:

      - training  : pass the TRUE K per sample (teacher routing);
      - inference : pass nothing — routes by its own count-head argmax.

    Because the count head has 3 classes and there is no K=4 specialist,
    this architecture CANNOT represent or count K=4 — the architectural-
    failure demonstration that motivates the unified slot design (A6).

    forward(mixture, k_route=None)
    ------------------------------
    mixture      : [B, 1, T] complex
    k_route      : [B] long in {1,2,3}, or None for self-routing
    returns slots [B, k_max, T] complex (zero-padded beyond each sample's
    routed K) and count_logits [B, 3].
    """

    N_SPECIALISTS: int = 3    # K in {1,2,3} — the training range

    def __init__(self,
                 hidden_channels: int = 64,
                 n_layers: int = 4,
                 k_max: int = 4,
                 kernel_size_enc: int = 7,
                 kernel_size_hidden: int = 3,
                 kernel_size_dec: int = 7,
                 use_se: bool = True,
                 se_reduction: int = 4,
                 head_embed_dim: int = 64):
        super().__init__()
        self.k_max = k_max
        self.hidden_channels = hidden_channels

        self.backbone = CSEBackbone(hidden_channels, n_layers,
                                    kernel_size_enc, kernel_size_hidden,
                                    use_se, se_reduction)
        self.count_head = CountHead(hidden_channels, head_embed_dim,
                                    n_count_classes=self.N_SPECIALISTS)
        self.specialists = nn.ModuleList([
            ComplexConv1d(hidden_channels, k, kernel_size_dec,
                          padding=kernel_size_dec // 2)
            for k in range(1, self.N_SPECIALISTS + 1)
        ])
        # One shared learnable complex scale across specialists (paper1
        # trick so initial output ~ 0.5*mix).
        self.output_scale = nn.Parameter(torch.tensor(0.5 + 0j,
                                                       dtype=torch.complex64))

        self._count_params()

    def forward(self, mixture, k_route: torch.Tensor | None = None):
        # mixture: [B, 1, T] complex
        x = self.backbone(mixture)                                # [B, H, T]
        count_logits = self.count_head(x)                         # [B, 3]
        if k_route is None:
            k_route = count_logits.argmax(dim=1) + 1              # [B]

        B, _H, T = x.shape
        slots = mixture.new_zeros(B, self.k_max, T)
        for k_val in range(1, self.N_SPECIALISTS + 1):
            idx = (k_route == k_val).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            masks = (self.specialists[k_val - 1](x[idx])
                     * self.output_scale)                         # [b, k, T]
            slots[idx, :k_val] = masks * mixture[idx]             # broadcast
        return slots, count_logits

    def _count_params(self):
        sections = {
            'backbone.encoder':    self.backbone.encoder,
            'backbone.res_blocks': self.backbone.res_blocks,
            'count_head':          self.count_head,
            'specialists':         self.specialists,
        }
        per_section = {n: sum(p.numel() for p in m.parameters())
                       for n, m in sections.items()}
        per_section['output_scale'] = self.output_scale.numel()
        total = sum(per_section.values())
        breakdown = ', '.join(f'{n}={v:,}' for n, v in per_section.items())
        print(f"[SpecialistBankNet] Total parameters: {total:,} "
              f"({total / 1e3:.1f}K)  [{breakdown}]")


# ============================================================================
# Smoke test
# ============================================================================
if __name__ == '__main__':
    print("Testing paper4 models ...")
    dummy = torch.randn(2, 1, 4096).to(torch.complex64)

    print("\n-- SlotSepNet (defaults: h64, l4, k_max=4, count head on) --")
    slot = SlotSepNet()
    slots, occ_logits, count_logits = slot(dummy)
    print(f"  slots={tuple(slots.shape)}  occ_logits={tuple(occ_logits.shape)}  "
          f"count_logits={tuple(count_logits.shape)}")
    assert slots.shape == (2, 4, 4096) and slots.is_complex()
    assert occ_logits.shape == (2, 4) and count_logits.shape == (2, 3)
    n_slot = sum(p.numel() for p in slot.parameters())

    print("\n-- SlotSepNet (no count head) --")
    slot_nc = SlotSepNet(use_count_head=False)
    _, _, cnt_none = slot_nc(dummy)
    assert cnt_none is None
    print("  count_logits is None as expected")

    print("\n-- SlotSepNet + JointLLRHead (paper 5 S3) --")
    slot_j = SlotSepNet(use_joint_head=True)
    slots_j, occ_j, cnt_j, sym_j = slot_j(dummy)
    print(f"  sym_logits={tuple(sym_j.shape)} (expect (2, 4, 256, 16))")
    assert sym_j.shape == (2, 4, 256, 16) and not sym_j.is_complex()
    n_joint = sum(p.numel() for p in slot_j.parameters())
    n_base = sum(p.numel() for p in slot.parameters())
    print(f"  joint head adds {n_joint - n_base:,} params "
          f"({(n_joint - n_base) / 1e3:.1f}K)")
    sym_j.mean().backward()
    print("  backward() succeeded through the joint head")

    print("\n-- OneAndRestNet (defaults: h64, l4, k_max=4) --")
    oar = OneAndRestNet()
    one, rest, stop = oar.forward_step(dummy)
    print(f"  forward_step: one={tuple(one.shape)}  rest={tuple(rest.shape)}  "
          f"stop={tuple(stop.shape)}")
    assert one.shape == (2, 1, 4096) and rest.shape == (2, 1, 4096)
    assert one.is_complex() and stop.shape == (2,)
    ones, stops = oar(dummy, max_steps=4)
    print(f"  forward: ones={tuple(ones.shape)}  stop_logits={tuple(stops.shape)}")
    assert ones.shape == (2, 4, 4096) and stops.shape == (2, 4)
    n_oar = sum(p.numel() for p in oar.parameters())

    # Weight sharing across steps: one parameter set, reused in the unroll.
    assert n_oar < n_slot, "recursive model should be smaller (2 masks, no occ head)"

    # Backward sanity on both
    (slots.abs().mean() + occ_logits.mean() + count_logits.mean()).backward()
    (ones.abs().mean() + stops.mean()).backward()
    print("\n  backward() succeeded on both models")

    print("\n-- SpecialistBankNet (defaults: h64, l4, k_max=4; A6 baseline) --")
    spec = SpecialistBankNet()
    # Teacher routing with a mixed-K batch: samples route through
    # different specialists; slots zero-padded beyond each routed K.
    k_route = torch.tensor([1, 3])
    s_slots, s_cnt = spec(dummy, k_route=k_route)
    print(f"  teacher-routed: slots={tuple(s_slots.shape)}  "
          f"count_logits={tuple(s_cnt.shape)}")
    assert s_slots.shape == (2, 4, 4096) and s_slots.is_complex()
    assert s_cnt.shape == (2, 3)
    # K=1 sample: slots 1..3 must be exactly zero; K=3 sample: slot 3 zero
    assert s_slots[0, 1:].abs().sum().item() == 0.0
    assert s_slots[1, 3].abs().sum().item() == 0.0
    assert s_slots[0, 0].abs().sum().item() > 0.0
    assert s_slots[1, :3].abs().sum().item() > 0.0
    print("  zero-padding outside the routed specialist: OK")
    # Self-routing (inference path): count-head argmax drives the routing
    s_slots2, s_cnt2 = spec(dummy)
    assert s_slots2.shape == (2, 4, 4096) and s_cnt2.shape == (2, 3)
    pred_k = s_cnt2.argmax(dim=1) + 1
    for b in range(2):
        assert s_slots2[b, int(pred_k[b]):].abs().sum().item() == 0.0
    print(f"  self-routed: pred_k={pred_k.tolist()}, padding consistent")
    (s_slots.abs().mean() + s_cnt.mean()).backward()
    print("  backward() succeeded")

    print("\nmodels smoke test passed!")
