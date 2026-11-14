"""
Figure 3.

Panel A - fatigue against radial stiffness, every population, each with its own front
Panel B - material against radial stiffness, with the screen's measured f_metal floor

Both panels share the stiffness axis, which is logarithmic.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
from screen import pareto

OUT = config.FIG_DEV_DIR / 's10_2_pareto.png'

OURS = '#2a78d6'
PARAM = '#eb6834'
SIMP = '#7a3fbf'
HAND = '#8a8a86'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'
SURFACE = '#fcfcfb'

SERIES = (
    ('parametric', 'parametric family', PARAM, 'o', 9, 0.30),
    ('generated', 'generated (ours)', OURS, 'o', 9, 0.30),
    ('handmade', 'handmade', HAND, 'D', 26, 0.95),
    ('simp', 'SIMP, screened', SIMP, 's', 70, 1.0),
    ('simp_unscreened', 'SIMP, unscreened', SIMP, 's', 70, 1.0),
)

def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)

def scatter(ax, report, key):
    """Every population on (K_radial, key)."""
    for name, _, color, marker, size, alpha in SERIES:
        rows = report['rows'][name]
        if not rows:
            continue
        x = [r['K_radial'] for r in rows]
        y = [r[key] for r in rows]
        open_face = name == 'simp_unscreened'
        ax.scatter(x, y, s=size, marker=marker, alpha=alpha, zorder=3,
                   facecolors=SURFACE if open_face else color, edgecolors=color,
                   linewidths=1.4 if size > 20 else 0.0)
    # The conditional frontier: best value from each stiffness band.
    for name, color in (('generated', OURS), ('parametric', PARAM)):
        band = [b for b in report['band_best'][name][key] if key in b]
        if len(band) < 2:
            continue
        ax.plot([b['K_radial'] for b in band], [b[key] for b in band],
                color=color, lw=2.0, zorder=4, solid_capstyle='round',
                marker='o', markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax.set_xscale('log')
    style(ax)

def main():
    report = pareto.load()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.13, wspace=0.22)

    scatter(axes[0], report, 'A_over_lim')
    axes[0].set_ylabel('$A_{over}/A$  -  material over the fatigue limit', fontsize=9, color=TEXT_PRIMARY)
    axes[0].set_title('A  fatigue against radial stiffness', fontsize=11, color=TEXT_PRIMARY, loc='left')

    scatter(axes[1], report, 'f_metal')
    axes[1].set_ylabel('$f_{metal}$  -  area fraction of metal', fontsize=9, color=TEXT_PRIMARY)
    axes[1].set_title('B  material against radial stiffness', fontsize=11, color=TEXT_PRIMARY, loc='left')

    for ax in axes:
        ax.set_xlabel('$K_{radial}$  (N/mm$^3$)', fontsize=9, color=TEXT_PRIMARY)

    handles = [plt.Line2D([], [], marker=m, color='none',
                          markerfacecolor=SURFACE if n == 'simp_unscreened' else c,
                          markeredgecolor=c, markeredgewidth=1.4,
                          markersize=5 if s < 20 else 7, label=label)
               for n, label, c, m, s, _ in SERIES]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.07, 0.955),
               ncol=5, frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
               handletextpad=0.4, columnspacing=1.4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, facecolor=SURFACE)
    print(f'wrote {OUT}')

if __name__ == '__main__':
    main()
