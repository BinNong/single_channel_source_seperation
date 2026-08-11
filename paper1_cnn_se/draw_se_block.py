"""
Draw Complex Squeeze-and-Excitation (C-SE) block architecture diagram.
Output: paper/figures/se_block.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# High-quality settings
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'text.usetex': False,
})

fig, ax = plt.subplots(1, 1, figsize=(16, 5))
ax.set_xlim(0, 20)
ax.set_ylim(0, 4)
ax.axis('off')

# Color palette
COLORS = {
    'input': '#FFFACD',      # light yellow
    'squeeze': '#BFE4FF',    # light blue
    'concat': '#C1F0C1',     # light green
    'excitation': '#FFCCCC',  # light red/pink
    'scale': '#E8D1F0',      # light purple
    'combine': '#FFE0B2',    # light orange
    'output': '#FFFACD',     # light yellow
    'border': '#333333',
    'arrow': '#555555',
}

def draw_block(ax, x, y, w, h, text, color, fontsize=9, bold=False):
    """Draw a rounded rectangle block with text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.15",
                         facecolor=COLORS[color], edgecolor=COLORS['border'],
                         linewidth=1.2, alpha=0.9)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, color='#222222')

def draw_arrow(ax, x1, y1, x2, y2, color=COLORS['arrow']):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                               lw=1.5, connectionstyle='arc3,rad=0'))

def draw_label(ax, x, y, text, fontsize=10, bold=True):
    """Draw a section label."""
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal', color='#444444')

# ==================== BLOCKS ====================

# Input
draw_block(ax, 1.2, 2.0, 2.2, 1.0,
           '$\mathbf{X} \in \mathbb{C}^{C \\times T}$\n(Complex Feature)',
           'input', fontsize=9, bold=True)

# ----- SQUEEZE phase -----
draw_block(ax, 3.7, 3.0, 2.4, 0.8,
           'GlobalAvgPool\n$\mathrm{Re}(\mathbf{X})$',
           'squeeze', fontsize=9)
draw_block(ax, 3.7, 1.0, 2.4, 0.8,
           'GlobalAvgPool\n$\mathrm{Im}(\mathbf{X})$',
           'squeeze', fontsize=9)

# Arrows from input to squeeze
draw_arrow(ax, 2.3, 2.0, 2.5, 3.0)
draw_arrow(ax, 2.3, 2.0, 2.5, 1.0)

# ----- CONCAT -----
draw_block(ax, 6.2, 2.0, 2.2, 1.0,
           'Concat\n$\mathbf{z} \in \mathbb{R}^{2C}$',
           'concat', fontsize=9)

# Arrows from squeeze to concat
draw_arrow(ax, 4.9, 3.0, 5.1, 2.3)
draw_arrow(ax, 4.9, 1.0, 5.1, 1.7)

# ----- EXCITATION phase -----
draw_block(ax, 8.8, 2.0, 2.2, 0.8,
           'FC: $2C \\to 2C/r$\n$\mathbf{W}_1$',
           'excitation', fontsize=9)
draw_arrow(ax, 7.3, 2.0, 7.7, 2.0)

draw_block(ax, 11.4, 2.0, 1.4, 0.8, 'ReLU', 'excitation', fontsize=9)
draw_arrow(ax, 9.9, 2.0, 10.7, 2.0)

draw_block(ax, 14.0, 2.0, 2.2, 0.8,
           'FC: $2C/r \\to 2C$\n$\mathbf{W}_2$',
           'excitation', fontsize=9)
draw_arrow(ax, 12.1, 2.0, 12.9, 2.0)

draw_block(ax, 16.2, 2.0, 1.4, 0.8, '$\sigma$', 'excitation', fontsize=9)
draw_arrow(ax, 15.1, 2.0, 15.5, 2.0)

# ----- SCALE phase -----
draw_block(ax, 18.4, 3.0, 2.4, 0.8,
           'Combine\n$\mathbf{w} = \\frac{\mathbf{s}_r + \mathbf{s}_i}{2}$',
           'combine', fontsize=8)

# Arrow from sigmoid to combine (split implied)
draw_arrow(ax, 17.3, 2.0, 17.4, 2.8)

# ----- OUTPUT -----
draw_block(ax, 18.4, 1.0, 2.4, 0.8,
           'Channel Scale\n$\mathbf{w} \odot \mathbf{X}$',
           'output', fontsize=9)

# Skip connection
ax.annotate('', xy=(17.4, 1.2), xytext=(1.2, 2.5),
            arrowprops=dict(arrowstyle='->', color='#888888',
                           lw=1.2, linestyle='dashed',
                           connectionstyle='arc3,rad=0.35'))
ax.text(7.0, 3.7, 'Skip Connection (Identity)', ha='center', fontsize=8,
        color='#888888', style='italic')
draw_arrow(ax, 17.3, 2.0, 17.3, 1.4)

# ==================== SECTION LABELS ====================
# Background bands for sections
for (x_start, x_end, color, label) in [
    (2.5, 5.0, 'blue', 'SQUEEZE'),
    (5.5, 7.5, 'green', ''),
    (7.5, 17.3, 'red', 'EXCITATION'),
    (17.3, 19.8, 'orange', 'SCALE'),
]:
    rect = mpatches.Rectangle((x_start, 3.55), x_end - x_start, 0.35,
                               facecolor=color, alpha=0.12, edgecolor=color,
                               linewidth=0.8, linestyle='--', zorder=0)
    ax.add_patch(rect)
    if label:
        ax.text((x_start + x_end)/2, 3.72, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color='#333333')

# ==================== BRACES / ANNOTATIONS ====================
# Real/Imag labels
ax.text(2.4, 2.6, 'real', ha='center', fontsize=7, color='#1f77b4', style='italic')
ax.text(2.4, 2.4, 'imag', ha='center', fontsize=7, color='#d62728', style='italic')

# Reduction ratio note
ax.text(9.7, 1.2, '$r=4$', ha='center', fontsize=8, color='#666666')

# Real-valued weight note
ax.annotate('Real-valued\nweight (preserves phase)', xy=(19.0, 0.1),
            ha='center', fontsize=8, color='#444444',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', alpha=0.6))

plt.tight_layout(pad=0.5)
plt.savefig('../paper/figures/se_block.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('SE block diagram saved to paper/figures/se_block.png')
