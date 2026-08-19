"""
Left  - Maximum principal strain amplitude on the reference cell under 9% axial compression
        (proximal SFA, walking; Poulson 2018). Sequential single-hue ramp; the hatched
        contour marks material above the 0.4% nitinol limit.
Right - K_radial against eps_a_max across the six pixel-exact strut widths. A scatter rather
        than two lines on twin axes.

Usage: python figures/scripts/s4_2_fatigue.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
from geom import reference
from geom.parametric import achievable_widths, crown
from sim2d.fatigue import fatigue
from sim2d.homogenize import homogenize

OUT = config.FIG_DEV_DIR / 's4_2_fatigue.png'

SERIES_1 = '#2a78d6'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'

def strain_map(cell, result):
    """Per-element amplitude painted back onto the pixel grid."""
    arr = cell.to_array()
    field = np.full(arr.shape, np.nan)
    aj, ai = np.nonzero(arr)
    field[aj, ai] = result.per_element()
    return field

def main():
    cell = reference.build()
    h = homogenize(cell)
    f = fatigue(cell, h)

    print(f'macroscopic strain (circ, axial, shear) {f.macro.round(5)}')
    print(f'eps_a_max   {f.eps_a_max:.5f}   limit {config.EPS_A_LIM}')
    print(f'eps_a_p99   {f.eps_a_p99:.5f}   (max/p99 {f.eps_a_max / f.eps_a_p99:.2f})')
    print(f'A_over_lim  {f.A_over_lim:.4f}')
    print(f'attenuation {f.attenuation:.2f}x')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # STRAIN MAP #
    field = strain_map(cell, f)
    circ_mm, axial_mm = config.cell_extent_mm()
    im = ax1.imshow(field * 100, origin='lower', cmap='Blues',
                    extent=(0, circ_mm, 0, axial_mm), interpolation='nearest')
    over = np.where(field > config.EPS_A_LIM, 1.0, np.nan)
    ax1.contour(np.linspace(0, circ_mm, field.shape[1]),
                np.linspace(0, axial_mm, field.shape[0]),
                np.nan_to_num(over), levels=[0.5], colors=['#e34948'], linewidths=1.2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('circumferential (mm)', color=TEXT_SECONDARY)
    ax1.set_ylabel('axial (mm)', color=TEXT_SECONDARY)
    ax1.set_title('Strain amplitude concentrates at the crowns',
                  color=TEXT_PRIMARY, fontsize=12, loc='left')
    ax1.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label('max principal strain amplitude (%)', color=TEXT_SECONDARY, fontsize=9)
    cb.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax1.annotate(f'red contour = above the {config.EPS_A_LIM * 100:.1f}% nitinol limit\n'
                 f'{f.A_over_lim:.0%} of the structure exceeds it',
                 xy=(0.02, -0.20), xycoords='axes fraction',
                 color=TEXT_SECONDARY, fontsize=9)

    # TRADEOFF #
    widths = achievable_widths()
    k, e = [], []
    for w in widths:
        c = crown(strut_width_mm=w, crown_amplitude=0.25, n_periods=1)
        hh = homogenize(c)
        k.append(hh.K_radial)
        e.append(fatigue(c, hh).eps_a_max)
    k, e = np.array(k), np.array(e)

    ax2.plot(e * 100, k, color=SERIES_1, lw=2, marker='o', ms=9,
             markeredgecolor='white', markeredgewidth=1.5, zorder=3)
    for w, ee, kk in zip(widths, e, k):
        ax2.annotate(f'{w:.3f} mm', xy=(ee * 100, kk), xytext=(8, -4),
                     textcoords='offset points', color=TEXT_SECONDARY, fontsize=9)
    ax2.axvline(config.EPS_A_LIM * 100, color='#e34948', lw=1.2, ls='--', zorder=1)
    ax2.annotate('nitinol limit 0.4%', xy=(config.EPS_A_LIM * 100, k.max()),
                 xytext=(6, -10), textcoords='offset points',
                 color='#e34948', fontsize=9)
    ax2.set_xlim(0, max(e) * 100 * 1.25)
    ax2.grid(True, color=GRID, lw=0.7, zorder=0)
    ax2.set_axisbelow(True)
    for side in ('top', 'right'):
        ax2.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax2.spines[side].set_color(GRID)
    ax2.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax2.set_xlabel('eps_a_max  (%)', color=TEXT_SECONDARY)
    ax2.set_ylabel('K_radial  (N/mm³)', color=TEXT_SECONDARY)
    ax2.set_title('Stiffness is bought with fatigue life', color=TEXT_PRIMARY,
                  fontsize=12, loc='left')

    fig.tight_layout(pad=1.8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight', facecolor='white')
    print(f'\nWrote {OUT}')

if __name__ == "__main__":
    main()
