r"""
Draw the OpenSetCSE + SNR-routed OOD detection block diagram (Fig. 1).

Layout (left to right):
  mixture y(t)
    -> Complex Encoder -> 4x Complex Residual Blocks (C-SE) -> Complex Decoder
    -> (s1_hat, s2_hat) separated waveforms            [separation path]
  per-source mask-weighted bottleneck features
    -> shared Modulation Head -> (z_i, f_i)
    -> two scorer families (logit: Energy/MSP/ODIN; geometry: Proto/VOS/Maha)
    -> SNR router (E if SNR <= 0 dB else P)
    -> per-source label in {known classes, unknown}

Output: figures/fig_architecture.{pdf,png}
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

for _f in glob.glob('/usr/local/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf'):
    font_manager.fontManager.addfont(_f)
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Latin Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
})

fig, ax = plt.subplots(figsize=(8.9, 4.8))
ax.set_xlim(-0.1, 15.9)
ax.set_ylim(-2.7, 7.2)
ax.axis('off')

C_BACK_F, C_BACK_E = '#eaf1fb', '#2f5d8f'   # blue: separation backbone
C_HEAD_F, C_HEAD_E = '#f3e9f7', '#7d2e8f'   # purple: modulation head
C_LOGI_F, C_LOGI_E = '#fdf3e3', '#b07d2b'   # amber: logit family
C_GEOM_F, C_GEOM_E = '#e9f7ec', '#2e7d46'   # green: geometry family
C_ROUT_F, C_ROUT_E = '#fde3e3', '#b02b2b'   # red: router
C_OUT_F,  C_OUT_E  = '#f2f2f2', '#555555'   # grey: io


def box(x, y, w, h, text, fc, ec, fs=8.5):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle='round,pad=0.05,rounding_size=0.12',
                       fc=fc, ec=ec, lw=1.2, zorder=2)
    ax.add_patch(p)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, zorder=3)


def arrow(x1, y1, x2, y2, rad=0.0, lw=1.0, ls='-', color='#333333'):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='-|>', mutation_scale=9,
                        connectionstyle=f'arc3,rad={rad}',
                        lw=lw, ls=ls, color=color, zorder=1)
    ax.add_patch(a)


def elbow_arrow(verts, lw=1.0, color='#333333'):
    """Right-angled polyline connector; arrowhead follows the last segment."""
    path = Path(verts, [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1))
    a = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=9,
                        lw=lw, color=color, zorder=1)
    ax.add_patch(a)


# ---------------- top row: separation backbone ----------------
ytop = 5.7
box(1.0, ytop, 1.6, 1.1, 'mixture\n$y(t)$\n$\\mathbb{C}^{4096}$',
    C_OUT_F, C_OUT_E)
box(3.4, ytop, 2.2, 1.1, 'Complex\nEncoder\nConv k=7',
    C_BACK_F, C_BACK_E)
box(6.4, ytop, 2.4, 1.1, '$4\\times$ Complex\nResidual Block\n(C-SE)',
    C_BACK_F, C_BACK_E)
box(9.4, ytop, 2.2, 1.1, 'Complex\nDecoder\nConv k=7',
    C_BACK_F, C_BACK_E)
box(12.3, ytop, 2.2, 1.1, '$\\hat{s}_1, \\hat{s}_2$\nseparated\nsources',
    C_OUT_F, C_OUT_E)

arrow(1.8, ytop, 2.3, ytop)
arrow(4.5, ytop, 5.2, ytop)
arrow(7.6, ytop, 8.3, ytop)
arrow(10.5, ytop, 11.2, ytop)

# PIT training loss annotation
ax.text(6.4, 6.85, 'training: $\\mathcal{L}_{\\mathrm{SI\\text{-}SDR}}'
        ' + \\alpha\\,\\mathcal{L}_{\\mathrm{CE}}$ (PIT, known classes only)',
        ha='center', va='center', fontsize=7.5, color='#555555', style='italic')

# ---------------- middle: modulation head ----------------
ymid = 3.7
box(4.4, ymid, 3.2, 1.15,
    'mask-weighted\nbottleneck features\n(per source)',
    C_HEAD_F, C_HEAD_E)
box(8.0, ymid, 2.6, 1.15,
    'Modulation\nHead (shared)\n$\\sim$8.6K params',
    C_HEAD_F, C_HEAD_E)
box(11.2, ymid, 2.4, 1.15,
    '$z_i \\in \\mathbb{R}^{64}$\n$f_i \\in \\mathbb{R}^{4}$',
    C_OUT_F, C_OUT_E)

# taps from residual stack down to features, and features -> head -> (z, f)
arrow(6.4, ytop - 0.55, 5.3, ymid + 0.58)
arrow(6.0, ymid, 6.7, ymid)
arrow(9.3, ymid, 10.0, ymid)

# ---------------- bottom: scorer families + router ----------------
ybot = 1.6
box(5.4, ybot, 3.4, 1.15,
    'logit family\nEnergy / MSP / ODIN',
    C_LOGI_F, C_LOGI_E)
box(9.8, ybot, 3.8, 1.15,
    'geometry family\nPrototype / VOS / Mahalanobis',
    C_GEOM_F, C_GEOM_E)

arrow(10.6, ymid - 0.58, 6.3, ybot + 0.58, rad=0.18)  # f_i -> logit family
arrow(11.5, ymid - 0.58, 10.4, ybot + 0.58, rad=0.0)  # z_i -> geometry family

yrout = -0.15
box(7.6, yrout, 3.2, 1.5,
    'SNR router\n$\\leq 0$ dB: $E$\n$> 0$ dB: $P$',
    C_ROUT_F, C_ROUT_E, fs=8.5)
arrow(5.6, ybot - 0.58, 6.7, yrout + 0.75, rad=-0.12)  # logit -> router
arrow(9.6, ybot - 0.58, 8.5, yrout + 0.75, rad=0.12)  # geom -> router

# SNR estimate input (below the router)
box(7.6, -1.95, 2.6, 0.9, 'SNR\nestimate', C_OUT_F, C_OUT_E, fs=7.5)
arrow(7.6, -1.5, 7.6, yrout - 0.75)

# final decision output (right column, fed by an elbow connector)
box(14.0, ymid, 2.9, 1.05,
    'per-source label\n$\\hat{m}_i \\in \\mathcal{K} \\cup'
    ' \\{\\mathrm{unknown}\\}$', C_OUT_F, C_OUT_E, fs=7)
elbow_arrow([(9.2, yrout), (14.0, yrout), (14.0, ymid - 0.53)])

fig.tight_layout()
os.makedirs('figures', exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(f'figures/fig_architecture.{ext}', bbox_inches='tight')
print('figures/fig_architecture.{pdf,png} written')
