"""
Left  - K_radial across the family's six pixel-exact strut widths, log-log, with the fitted
        power-law exponent labeled. Bending-dominated cellular solids scale near w^3, 
        stretch-dominated near w^1; where this lands says which regime the cell is in.
Right - Relative error against closed-form answers the homogenisation: a solid cell 
        (no holes means no homogenisation) and aligned bars (rule of mixtures). 
        Log axis, because the claim is "machine precision".

Usage: python figures/scripts/s4_1_stiffness.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

import config
from geom.cell import UnitCell
from geom.parametric import achievable_widths, crown
from sim2d.homogenize import homogenize

OUT = config.FIG_DEV_DIR / 's4_1_stiffness.png'

# dataviz reference palette, light mode: categorical slot 1 + ink tokens.
SERIES_1 = '#2a78d6'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'

N = config.GRID_N
E_BULK = config.NITINOL['E_austenite_MPa']
NU = config.NITINOL['poisson_austenite']

def stiffness_sweep():
    widths, k = [], []
    for w in achievable_widths():
        cell = crown(strut_width_mm=w, crown_amplitude=0.25, n_periods=1)
        widths.append(w)
        k.append(homogenize(cell).K_radial)
    return np.array(widths), np.array(k)

def validation_cases():
    exact_C = E_BULK / (1 - NU ** 2) * np.array([[1, NU, 0], [NU, 1, 0],
                                                 [0, 0, (1 - NU) / 2]])
    cases = []

    C = homogenize(UnitCell(np.ones((N, N), dtype=bool))).C_eff
    cases.append(('solid cell\nvs plane stress',
                  np.max(np.abs(C - exact_C)) / exact_C.max()))

    for fill in (0.25, 0.5, 0.75):
        a = np.zeros((N, N), dtype=bool)
        a[:, :int(fill * N)] = True
        got = homogenize(UnitCell(a)).E_axial
        cases.append((f'bars, fill {fill:g}\nvs rule of mixtures',
                      abs(got - E_BULK * fill) / (E_BULK * fill)))
    return cases

def main():
    widths, k = stiffness_sweep()
    exponent, intercept = np.polyfit(np.log(widths), np.log(k), 1)
    cases = validation_cases()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax in (ax1, ax2):
        ax.set_facecolor('white')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)

    # POWER LAW #
    fit = np.exp(intercept) * widths ** exponent
    ax1.plot(widths, fit, color=TEXT_SECONDARY, lw=1.2, ls='--', zorder=1)
    ax1.plot(widths, k, color=SERIES_1, lw=2, marker='o', ms=8,
             markeredgecolor='white', markeredgewidth=1.5, zorder=3)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    # Log minor-tick labels collide at this range, label only the six real widths.
    ax1.xaxis.set_major_locator(FixedLocator(widths))
    ax1.xaxis.set_minor_locator(FixedLocator([]))
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.xaxis.set_minor_formatter(NullFormatter())
    ax1.ticklabel_format(axis='x', style='plain')
    ax1.yaxis.set_major_locator(FixedLocator([20, 40, 60, 80, 100, 130]))
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.grid(True, which='major', color=GRID, lw=0.7, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_xlabel('strut width (mm)', color=TEXT_SECONDARY)
    ax1.set_ylabel('K_radial  (N/mm³)', color=TEXT_SECONDARY)
    ax1.set_title('Radial stiffness rises with strut width',
                  color=TEXT_PRIMARY, fontsize=12, loc='left')
    ax1.annotate(f'K ∝ w^{exponent:.2f}\n(bending-dominated ≈3, stretch ≈1)',
                 xy=(widths[-2], k[-2]), xytext=(0.06, 0.80),
                 textcoords='axes fraction', color=TEXT_PRIMARY, fontsize=10)
    for x, y in ((widths[0], k[0]), (widths[-1], k[-1])):
        ax1.annotate(f'{y:.1f}', xy=(x, y), xytext=(4, -12),
                     textcoords='offset points', color=TEXT_SECONDARY, fontsize=9)

    # VALIDATION #
    labels = [c[0] for c in cases]
    errors = np.array([max(c[1], 1e-17) for c in cases])
    y = np.arange(len(cases))[::-1]
    ax2.plot(errors, y, 'o', color=SERIES_1, ms=9,
             markeredgecolor='white', markeredgewidth=1.5, zorder=3)
    ax2.axvline(1e-9, color=TEXT_SECONDARY, lw=1.2, ls='--', zorder=1)
    ax2.annotate('1e-9 tolerance', xy=(1e-9, y.max() + 0.42),
                 xytext=(6, 0), textcoords='offset points',
                 color=TEXT_SECONDARY, fontsize=9)
    ax2.set_xscale('log')
    ax2.set_xlim(1e-18, 1e-6)
    ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=9, color=TEXT_PRIMARY)
    ax2.set_ylim(-0.6, len(cases) - 0.15)
    ax2.grid(True, axis='x', color=GRID, lw=0.7, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_xlabel('relative error vs closed form', color=TEXT_SECONDARY)
    ax2.set_title('Exact solutions are reproduced to machine precision',
                  color=TEXT_PRIMARY, fontsize=12, loc='left')
    for e, yy in zip(errors, y):
        ax2.annotate(f'{e:.0e}', xy=(e, yy), xytext=(9, -3),
                     textcoords='offset points', color=TEXT_SECONDARY, fontsize=9)

    fig.tight_layout(pad=1.8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight', facecolor='white')

    print(f'{"w mm":>8} {"K_radial":>10}')
    for w, kk in zip(widths, k):
        print(f'{w:8.4f} {kk:10.4f}')
    print(f'\npower-law exponent  {exponent:.3f}')
    print('validation (relative error vs closed form):')
    for label, err in cases:
        print(f'  {label.replace(chr(10), " "):38} {err:.2e}')
    print(f'\nWrote {OUT}')

if __name__ == "__main__":
    main()
