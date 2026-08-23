"""
This exists to measure how long ~5000 cells will take, and whether 
the dataset generation plan is possible. Also confirms the per-cell time scales 
with element count rather than jumping around.

Usage: python figures/scripts/s4_4_throughput.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
from geom import reference
from geom.parametric import sweep
from sim2d.label import describe, label

OUT = config.FIG_DEV_DIR / 's4_4_throughput.png'
S7_TARGET = 5000

SERIES_1 = '#2a78d6'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'

def main():
    print('Reference cell')
    print(describe(label(reference.build())))
    print()

    valid, _ = sweep()
    rows = []
    for _, cell in valid:
        result = label(cell)
        rows.append((result.metrics['n_elements'], result.seconds))
    elements = np.array([r[0] for r in rows])
    seconds = np.array([r[1] for r in rows])

    total = seconds.sum()
    per_cell = seconds.mean()
    print(f'labeled          {len(rows)} cells in {total:.1f} s')
    print(f'per cell          mean {per_cell * 1000:.0f} ms, '
          f'median {np.median(seconds) * 1000:.0f} ms, max {seconds.max() * 1000:.0f} ms')
    print(f'budget (S4.4)     5 s per cell -> {5.0 / per_cell:.0f}x headroom')
    print()
    print(f'extrapolated to {S7_TARGET} cells (S7):')
    print(f'  single core     {S7_TARGET * per_cell / 60:.1f} min')
    for workers in (4, 8, 16):
        print(f'  {workers:2d} cores        {S7_TARGET * per_cell / 60 / workers:.1f} min')

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(elements, seconds * 1000, 'o', color=SERIES_1, ms=8, alpha=0.85,
            markeredgecolor='white', markeredgewidth=1.2, zorder=3)
    fit = np.polyfit(elements, seconds * 1000, 1)
    xs = np.linspace(elements.min(), elements.max(), 50)
    ax.plot(xs, np.polyval(fit, xs), color=TEXT_SECONDARY, lw=1.2, ls='--', zorder=2)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.set_xlabel('elements in the cell', color=TEXT_SECONDARY)
    ax.set_ylabel('label time (ms)', color=TEXT_SECONDARY)
    ax.set_title(f'Labeling cost scales with mesh size - mean {per_cell * 1000:.0f} ms/cell',
                 color=TEXT_PRIMARY, fontsize=12, loc='left')
    ax.annotate(f'{len(rows)} cells\n{S7_TARGET} cells ≈ '
                f'{S7_TARGET * per_cell / 60:.0f} min single-core',
                xy=(0.04, 0.86), xycoords='axes fraction',
                color=TEXT_PRIMARY, fontsize=10)

    fig.tight_layout(pad=1.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight', facecolor='white')
    print(f'\nWrote {OUT}')

if __name__ == '__main__':
    main()
