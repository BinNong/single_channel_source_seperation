"""
Paper 3 — PIT label-anchoring schematic (review-3 follow-up).

Draws fig_pit_label_anchor.{pdf,png} into ../paper3/figures/: how the
dump stage paired PIT-aligned output CONTENT with slot-anchored LABELS
(the second misalignment), vs the corrected truth-anchored pairing, for
one ku mixture whose PIT permutation swaps the slots.

Usage:  python make_fig_pit_labels.py
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

for _f in glob.glob('/usr/local/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf'):
    font_manager.fontManager.addfont(_f)
for _f in glob.glob('/Library/TeX/Root/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf'):
    font_manager.fontManager.addfont(_f)
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Latin Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
})

OUT = os.path.join('..', 'paper3', 'figures')
GREEN = '#d7e8d4'   # known source s1 (QPSK)
PINK = '#f4d9d0'    # unknown source s2 (64QAM)
GRAY = '#efefef'


def box(ax, x, y, w, h, lines, fc, fs=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                                facecolor=fc, edgecolor='black',
                                linewidth=0.7))
    ax.text(x + w / 2, y + h / 2, lines, ha='center', va='center',
            fontsize=fs, linespacing=1.3)


def main():
    fig, ax = plt.subplots(figsize=(3.5, 1.62))
    ax.axis('off')
    H = 0.78
    y1, y2 = 1.18, 0.22          # slot-1 row, slot-2 row

    # left: PIT-aligned outputs (pi swaps the slots of this ku mixture)
    box(ax, 0.0, y1, 2.35, H, 'slot 1 output\n= content of $s_2$', PINK)
    box(ax, 0.0, y2, 2.35, H, 'slot 2 output\n= content of $s_1$', GREEN)
    ax.text(1.18, 2.22, 'PIT outputs\n($\\pi$ swaps slots)', ha='center',
            fontsize=8.0, style='italic', linespacing=1.3)

    # middle: (a) labels stay with the slots (the bug)
    box(ax, 3.30, y1, 1.95, H, 'QPSK\nknown', GRAY)
    box(ax, 3.30, y2, 1.95, H, '64QAM\nunknown', GRAY)
    ax.text(4.28, 2.22, '(a) labels stay\nwith slots (bug)', ha='center',
            fontsize=8.0, style='italic', color='red', linespacing=1.3)
    for y in (y1, y2):
        ax.annotate('', xy=(3.24, y + H / 2), xytext=(2.42, y + H / 2),
                    arrowprops=dict(arrowstyle='-|>', lw=1.1, color='red'))
        ax.text(5.38, y + H / 2, '$\\times$', ha='left', va='center',
                fontsize=10, color='red', fontweight='bold')

    # right: (b) labels follow the content (the fix)
    box(ax, 6.55, y1, 1.95, H, '64QAM\nunknown', PINK)
    box(ax, 6.55, y2, 1.95, H, 'QPSK\nknown', GREEN)
    ax.text(7.52, 2.22, '(b) labels follow\ncontent (fix)', ha='center',
            fontsize=8.0, style='italic', color='tab:green', linespacing=1.3)
    ax.annotate('', xy=(6.49, y1 + H / 2), xytext=(5.85, y2 + H / 2),
                arrowprops=dict(arrowstyle='-|>', lw=1.1, color='tab:green'))
    ax.annotate('', xy=(6.49, y2 + H / 2), xytext=(5.85, y1 + H / 2),
                arrowprops=dict(arrowstyle='-|>', lw=1.1, color='tab:green'))
    for y in (y1, y2):
        ax.text(8.63, y + H / 2, '$\\checkmark$', ha='left', va='center',
                fontsize=9, color='tab:green', fontweight='bold')

    ax.set_xlim(-0.1, 9.05)
    ax.set_ylim(0.08, 2.62)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_pit_label_anchor.{ext}'),
                    bbox_inches='tight', pad_inches=0.04, dpi=300)
    plt.close(fig)
    print('fig_pit_label_anchor done')


if __name__ == '__main__':
    main()
