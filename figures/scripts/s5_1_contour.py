"""
Left  - Voxel edges (what geom/tube.py meshes) against the smoothed contour, zoomed on a
        crown where the stair steps are worst. The steps are the reason a voxel mesh
        can't be trusted for eps_a_max: each one is an artificial stress raiser.
Right - How the smoothed geometry's narrowest feature and metal fraction move with the
        smoothing sigma, against the manufacturing floor. Width is the 1st percentile of
        medial-ridge thickness.

Usage: python figures/scripts/s5_1_contour.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
from geom import reference
from sim3d import contour

OUT = config.FIG_DEV_DIR / 's5_1_contour.png'
SIGMA = 0.8

SERIES_1 = '#2a78d6'
SERIES_2 = '#eb6834'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'
LIMIT = '#e34948'

def main():
    cell = reference.build()
    arr = cell.to_array()
    mm = config.mm_per_px()[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # BOUNDARY COMPARISON #
    ax1.imshow(arr, origin='lower', cmap='Greys', alpha=0.30, interpolation='nearest',
               extent=(0, arr.shape[1], 0, arr.shape[0]))
    for poly in contour.contours(cell, sigma_px=SIGMA, n_circ=1, n_axial=1):
        ax1.plot(poly[:, 1], poly[:, 0], color=SERIES_1, lw=2.0, zorder=3)
    for poly in contour.contours(cell, sigma_px=0.0001, n_circ=1, n_axial=1):
        ax1.plot(poly[:, 1], poly[:, 0], color=SERIES_2, lw=1.0, ls='--', zorder=2)

    ax1.set_xlim(20, 44)
    ax1.set_ylim(0, 24)
    ax1.set_aspect('equal')
    ax1.set_xlabel('circumferential (px)', color=TEXT_SECONDARY)
    ax1.set_ylabel('axial (px)', color=TEXT_SECONDARY)
    ax1.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax1.set_title('Smoothed boundary replaces the voxel staircase',
                  color=TEXT_PRIMARY, fontsize=12, loc='left')
    ax1.plot([], [], color=SERIES_1, lw=2.0, label=f'contour, σ={SIGMA} px')
    ax1.plot([], [], color=SERIES_2, lw=1.0, ls='--', label='voxel edge (σ→0)')
    leg = ax1.legend(frameon=False, loc='upper right')
    for t in leg.get_texts():
        t.set_color(TEXT_SECONDARY)

    # SIGMA SWEEP #
    sigmas = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5])
    width, frac = [], []
    for s in sigmas:
        r = contour.summarize(cell, sigma_px=float(s))
        width.append(r['min_width_mm'])
        frac.append(100 * (r['f_metal_smoothed'] - r['f_metal_voxel']) / r['f_metal_voxel'])
    width, frac = np.array(width), np.array(frac)

    ax2.plot(sigmas, width, color=SERIES_1, lw=2, marker='o', ms=8,
             markeredgecolor='white', markeredgewidth=1.4, zorder=3)
    ax2.axhline(config.MIN_FEATURE_MM, color=LIMIT, lw=1.4, ls='--', zorder=2)
    ax2.annotate(f'manufacturing floor {config.MIN_FEATURE_MM} mm',
                 xy=(sigmas[0], config.MIN_FEATURE_MM), xytext=(2, 5),
                 textcoords='offset points', color=LIMIT, fontsize=9)
    ax2.axvline(SIGMA, color=TEXT_SECONDARY, lw=1.0, ls=':', zorder=1)
    ax2.annotate(f'chosen σ={SIGMA} px\n({SIGMA * mm * 1000:.0f} µm)',
                 xy=(SIGMA, width.max()), xytext=(6, -6), textcoords='offset points',
                 color=TEXT_PRIMARY, fontsize=9)
    for s, w, d in zip(sigmas, width, frac):
        ax2.annotate(f'{d:+.1f}%', xy=(s, w), xytext=(0, -16),
                     textcoords='offset points', color=TEXT_SECONDARY, fontsize=8,
                     ha='center')

    ax2.set_ylim(0.08, max(width.max() * 1.12, config.MIN_FEATURE_MM * 1.35))
    ax2.grid(True, color=GRID, lw=0.7, zorder=0)
    ax2.set_axisbelow(True)
    for side in ('top', 'right'):
        ax2.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax2.spines[side].set_color(GRID)
    ax2.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax2.set_xlabel('smoothing σ (px)', color=TEXT_SECONDARY)
    ax2.set_ylabel('narrowest feature, p1 (mm)', color=TEXT_SECONDARY)
    ax2.set_title('Smoothing does not thin the struts (labels: Δ metal fraction)',
                  color=TEXT_PRIMARY, fontsize=12, loc='left')

    fig.tight_layout(pad=1.7)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight', facecolor='white')

    r = contour.summarize(cell, sigma_px=SIGMA)
    print(f'chosen sigma      {SIGMA} px = {SIGMA * mm * 1000:.1f} um')
    print(f'polygons          {r["n_polygons"]} ({r["n_points"]} points) over {config.N_CIRC} cells')
    print(f'narrowest (p1)    {r["min_width_mm"]:.4f} mm   floor {r["min_feature_mm"]:.2f} mm')
    print(f'metal fraction    {r["f_metal_smoothed"]:.4f} vs voxel {r["f_metal_voxel"]:.4f}'
          f'  ({100 * (r["f_metal_smoothed"] - r["f_metal_voxel"]) / r["f_metal_voxel"]:+.1f}%)')
    print(f'\nWrote {OUT}')

if __name__ == "__main__":
    main()
