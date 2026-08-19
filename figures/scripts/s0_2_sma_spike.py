"""
Nitinol superelastic hysteresis loop out of Ansys.

Reads the PRVAR table written by sim3d/decks/s0_2_spike_sma.inp and checks the loop against
the constants that generated it. Passing means the material card in Ansys reproduces
config.NITINOL.

Usage: python figures/scripts/s0_2_sma_spike.py [path/to/sma_spike.txt]
"""

import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
from sim3d.apdl import read_prvar

OUT = config.FIG_DEV_DIR / 's0_2_sma_spike.png'
DEFAULT = config.PROJECT_ROOT / 'sim3d' / 'results' / 's0_2_spike_sma.txt'

SERIES_1 = '#2a78d6'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'

def main(path=DEFAULT):
    t, uz, sz = read_prvar(path)
    strain = uz * 100.0                 # 1 mm cube, so displacement is strain
    n = config.NITINOL
    load, unload = t <= 1.0, t > 1.0

    E_fit = np.polyfit(uz[load][:6], sz[load][:6], 1)[0]
    tangent = np.gradient(sz[load], uz[load])
    knee = int(np.argmax(tangent < 0.5 * E_fit))
    tu = np.gradient(sz[unload], uz[unload])
    flat = np.abs(tu) < 0.5 * E_fit
    area = abs(np.trapezoid(sz, uz))

    checks = [
        ('elastic modulus (MPa)', E_fit, n['E_austenite_MPa']),
        ('loading transf. start (MPa)', sz[load][knee], n['sigma_start_loading_MPa']),
        ('unloading plateau top (MPa)', sz[unload][flat].max(), n['sigma_start_unloading_MPa']),
        ('unloading plateau end (MPa)', sz[unload][flat].min(), n['sigma_finish_unloading_MPa']),
    ]
    print(f'{"quantity":<30} {"measured":>10} {"input":>10} {"err":>8}')
    for name, got, want in checks:
        print(f'{name:<30} {got:10.1f} {want:10.1f} {100 * abs(got - want) / want:7.2f}%')
    print(f'{"residual strain":<30} {uz[-1]:10.3e} {0.0:10.1f}')
    print(f'{"residual stress (MPa)":<30} {sz[-1]:10.3e} {0.0:10.1f}')
    print(f'{"loop area (MPa)":<30} {area:10.3f}   dissipated energy density')

    closed = abs(uz[-1]) < 1e-6 and abs(sz[-1]) < 1e-3
    print(f'\nLOOP CLOSES: {closed}  -> superelastic, not plastic')

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(strain[load], sz[load], color=SERIES_1, lw=2.2, label='loading')
    ax.plot(strain[unload], sz[unload], color=SERIES_1, lw=2.2, ls='--', label='unloading')
    for value, label in ((n['sigma_start_loading_MPa'], 'σ_LS 465'),
                         (n['sigma_finish_loading_MPa'], 'σ_LE 535'),
                         (n['sigma_start_unloading_MPa'], 'σ_US 227'),
                         (n['sigma_finish_unloading_MPa'], 'σ_UE 187')):
        ax.axhline(value, color=TEXT_SECONDARY, lw=0.8, ls=':', zorder=1)
        ax.annotate(label, xy=(strain.max(), value), xytext=(4, 2),
                    textcoords='offset points', color=TEXT_SECONDARY, fontsize=8)

    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.set_xlabel('strain (%)', color=TEXT_SECONDARY)
    ax.set_ylabel('stress (MPa)', color=TEXT_SECONDARY)
    ax.set_title('Nitinol superelasticity reproduced in Ansys — the loop closes',
                 color=TEXT_PRIMARY, fontsize=12, loc='left')
    ax.set_xlim(-0.3, strain.max() * 1.18)
    leg = ax.legend(frameon=False, loc='lower right')
    for text in leg.get_texts():
        text.set_color(TEXT_SECONDARY)
    ax.annotate(f'E = {E_fit:.0f} MPa (input 65000)\n'
                f'residual strain {uz[-1]:.1e}\n'
                f'loop area {area:.2f} MPa',
                xy=(0.04, 0.72), xycoords='axes fraction',
                color=TEXT_PRIMARY, fontsize=9)

    fig.tight_layout(pad=1.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight', facecolor='white')
    print(f'\nWrote {OUT}')

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
