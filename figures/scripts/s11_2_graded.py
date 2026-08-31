"""
The graded stent: cells that differ along the axis, joined into one device.

Three things:
  Top    - The cell sequence, each labeled with what it was asked for and what it achieved
  Left   - The assembled stack, with every interface marked. A break would show here
  Right  - Achieved against asked. A chain that assembles but doesn't vary is a uniform stent with extra steps

Usage: python figures/scripts/s11_2_graded.py
"""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
from diffusion import graded

OUT = config.FIG_DEV_DIR / 's11_2_graded.png'
REPORT = config.PROJECT_ROOT / 'diffusion' / 'results' / 's11_2_graded_report.json'

ACCENT = '#0E7C86'
ASKED = '#52514e'
GOT = '#2a78d6'
SEAM = '#eb6834'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'

def main():
    chain, uniform, sigma, meta = graded.load_chain()
    rep = json.loads(REPORT.read_text(encoding='utf-8'))
    n = len(chain)
    key = rep['key']

    fig = plt.figure(figsize=(14.0, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.72, 1.5], hspace=0.22)

    # CELL SEQUENCE #
    top = gs[0].subgridspec(1, n, wspace=0.08)
    for i in range(n):
        ax = fig.add_subplot(top[0, i])
        ax.imshow(chain[i], cmap='binary', interpolation='nearest')
        got = rep['achieved'][i]
        ax.set_title(f"asked {rep['asked'][i]:.0f}\n"
                     f"got {'n/a' if got is None else f'{got:.0f}'}",
                     fontsize=8.5, color=TEXT_SECONDARY, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(ACCENT); s.set_linewidth(1.4)
        if i == 0:
            ax.set_ylabel('cells,\nin order', fontsize=9, color=TEXT_PRIMARY,
                          rotation=0, ha='right', va='center', labelpad=26)

    bottom = gs[1].subgridspec(1, 2, width_ratios=[1.0, 2.2], wspace=0.22)

    # ASSEMBLED STACK #
    ax = fig.add_subplot(bottom[0, 0])
    stacked = graded.interface.stack(list(chain))
    ax.imshow(stacked, cmap='binary', interpolation='nearest', aspect='auto')
    for i in range(1, n):
        ax.axhline(i * config.GRID_N - 0.5, color=SEAM, lw=1.2, alpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"assembled  ({rep['interfaces_joined']}/{n} interfaces joined)",
                 fontsize=10.5, color=TEXT_PRIMARY, loc='left')
    ax.set_xlabel('orange lines are the interfaces', fontsize=8.5, color=TEXT_SECONDARY)

    # ACHIEVED VS ASKED #
    ax = fig.add_subplot(bottom[0, 1])
    idx = np.arange(n)
    ax.plot(idx, rep['asked'], color=ASKED, lw=1.8, ls='--', marker='o', ms=5,
            label='asked', zorder=3)
    got = [np.nan if g is None else g for g in rep['achieved']]
    ax.plot(idx, got, color=GOT, lw=2.4, marker='s', ms=6, label='achieved', zorder=4)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.set_xticks(idx)
    ax.set_xlabel('position along the stent', color=TEXT_SECONDARY)
    ax.set_ylabel(key, color=TEXT_SECONDARY)
    ax.legend(fontsize=9, frameon=False, loc='upper left')
    ax.set_title(f"rank correlation rho = {rep['gradient_rho']:.3f}   "
                 f"achieved span {rep['achieved_span']:.0f}",
                 fontsize=10.5, color=TEXT_PRIMARY, loc='left')

    length = n * config.cell_extent_mm()[1]
    fig.suptitle(f"A stent that varies along its length   |   {n} cells, "
                 f"{length:.1f} mm, {key} {meta['lo']:.0f} to {meta['hi']:.0f}   |   "
                 f"{rep['n_valid_cells']}/{n} cells valid, stack connected",
                 color=TEXT_PRIMARY, fontsize=12.5, x=0.012, ha='left')
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.07)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=155, facecolor='white')
    print(f"{n} cells, {length:.2f} mm")
    print(f"  interfaces joined {rep['interfaces_joined']}/{n}, "
          f"cells valid {rep['n_valid_cells']}/{n}, "
          f"connected {rep['stack_connected']}")
    print(f"  gradient rho {rep['gradient_rho']:.3f}, span {rep['achieved_span']:.1f}")
    print('Wrote', OUT)

if __name__ == "__main__":
    main()
