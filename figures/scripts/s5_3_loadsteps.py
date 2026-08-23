"""
The four-load-step history.

Three stacked panels sharing one time axis, with the load-step boundaries marked. 
Stacked: displacement (mm), force (N) and strain (%).

Top    - Imposed displacements: radial on the outer surface, axial at the far end
Middle - Reaction forces, which is where a run that applied no load is obvious at a glance
Bottom - The strain excursion the fatigue amplitude is formed from

Usage: python figures/scripts/s5_3_loadsteps.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
from sim3d.loadsteps import load_history, summarize

HIST = config.PROJECT_ROOT / 'sim3d' / 'results' / 's5_3_loadsteps_hist.txt'
STATUS = config.PROJECT_ROOT / 'sim3d' / 'results' / 's5_3_loadsteps.txt'
OUT = config.FIG_DEV_DIR / 's5_3_loadsteps.png'

SERIES_1 = '#2a78d6'
SERIES_2 = '#eb6834'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'
LIMIT = '#e34948'

STEPS = [(0, 1, 'LS1\nradial in'), (1, 2, 'LS2\nrelease'),
         (2, 3, 'LS3\naxial −9%'), (3, 4, 'LS4\nrelease')]

def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for boundary in (1, 2, 3):
        ax.axvline(boundary, color=TEXT_SECONDARY, lw=0.9, ls=':', zorder=1)

def main():
    t, ux, uz, fx, fz = load_history(HIST)
    status = summarize(STATUS)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True)

    # DISPLACEMENT #
    ax = axes[0]
    ax.plot(t, ux, color=SERIES_1, lw=2, marker='o', ms=4, label='radial, outer surface')
    ax.plot(t, uz, color=SERIES_2, lw=2, marker='s', ms=4, label='axial, far end')
    style(ax)
    ax.set_ylabel('displacement (mm)', color=TEXT_SECONDARY)
    ax.set_title('Load history - sector, cyclic symmetry, no crimp',
                 color=TEXT_PRIMARY, fontsize=12, loc='left')
    leg = ax.legend(frameon=False, loc='center left', fontsize=9)
    for text in leg.get_texts():
        text.set_color(TEXT_SECONDARY)

    # REACTIONS #
    ax = axes[1]
    ax.plot(t, fx, color=SERIES_1, lw=2, marker='o', ms=4, label='radial reaction')
    ax.plot(t, fz, color=SERIES_2, lw=2, marker='s', ms=4, label='axial reaction')
    style(ax)
    ax.set_ylabel('reaction force (N)', color=TEXT_SECONDARY)
    ax.set_ylim(min(fz.min(), fx.min()) * 1.45, max(fx.max(), fz.max()) * 1.35)
    leg = ax.legend(frameon=False, loc='lower right', fontsize=9)
    for text in leg.get_texts():
        text.set_color(TEXT_SECONDARY)
    ax.annotate(f'radial reaction at t=1: {status["FRADIAL"]:.3f} N'
                f'   ->   K_radial_3D = {status["K_radial_3D"]:.1f} N/mm³',
                xy=(0.02, 0.06), xycoords='axes fraction',
                color=TEXT_PRIMARY, fontsize=9)

    # STRAIN EXCURSION #
    ax = axes[2]
    eps3, eps4, amp = status['EPS3'], status['EPS4'], status['EPSAMP']
    ax.plot([2, 3, 4], [eps4 * 100, eps3 * 100, eps4 * 100], color=SERIES_1, lw=2,
            marker='o', ms=8, markeredgecolor='white', markeredgewidth=1.4, zorder=3)
    ax.axhline(config.EPS_A_LIM * 100, color=LIMIT, lw=1.4, ls='--', zorder=2)
    ax.annotate(f'nitinol amplitude limit {config.EPS_A_LIM * 100:.1f}%',
                xy=(0.05, config.EPS_A_LIM * 100), xytext=(0, 7),
                textcoords='offset points', color=LIMIT, fontsize=9)
    ax.annotate('peak {:.2f}%{}amplitude {:.2f}%'.format(eps3 * 100, chr(10), amp * 100),
                xy=(3, eps3 * 100), xytext=(-96, -12), textcoords='offset points',
                color=TEXT_PRIMARY, fontsize=9)
    style(ax)
    ax.set_ylabel('max principal strain (%)', color=TEXT_SECONDARY)
    ax.set_xlabel('load step time', color=TEXT_SECONDARY)
    ax.set_xlim(0, 4.15)
    ax.set_ylim(-0.5, eps3 * 100 * 1.28)
    ax.set_xticks([0, 1, 2, 3, 4])
    for lo, hi, label in STEPS:
        ax.annotate(label.replace(chr(10), ' '), xy=((lo + hi) / 2, 0.97),
                    xycoords=('data', 'axes fraction'), ha='center', va='top',
                    color=TEXT_SECONDARY, fontsize=9)

    fig.tight_layout(pad=1.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight', facecolor='white')

    print(f'{len(t)} time points across 4 load steps')
    print(f'  radial reaction at t=1   {status["FRADIAL"]:.4f} N')
    print(f'  K_radial_3D              {status["K_radial_3D"]:.3f} N/mm^3')
    print(f'  peak strain (LS3)        {eps3 * 100:.3f} %')
    print(f'  residual strain (LS4)    {eps4 * 100:.5f} %')
    print(f'  amplitude                {amp * 100:.3f} %   limit {config.EPS_A_LIM * 100:.1f} %')
    print()
    print('Wrote', OUT)

if __name__ == "__main__":
    main()
