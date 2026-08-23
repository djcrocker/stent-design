"""
Does GRID_N = 128 rank better than 64?

Left  - Rank correlation against the same 3D truth at each resolution, with bootstrap CIs.
        If 128 helped, its rho would clear 64's upper bound. It doesn't.
Right - Why not. Refining the grid moves the 2D values a lot while barely reordering 
        anything (rank agreement between 64 and 128 is rho ~0.99). So the screen's 
        disagreement with 3D isn't a discretization error that a finer grid could fix.

Usage: python figures/scripts/s6_4_resolution.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import config
from sim3d import resolution

OUT = config.FIG_DEV_DIR / 's6_4_resolution.png'

SERIES = {'K_radial': '#2a78d6', 'eps_a_max': '#eb6834', 'A_over_lim': '#7a3fbf'}
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'
LIMIT = '#e34948'

def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)

def main():
    result = resolution.compare(n_boot=3000)
    rows64 = {r['name']: r for r in resolution.relabel(64)}
    rows128 = {r['name']: r for r in resolution.relabel(128)}
    names = sorted(rows64)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # RHO vs RESOLUTION #
    ax = axes[0]
    xs = [0, 1]
    for metric, color in SERIES.items():
        if metric not in result['metrics']:
            continue
        d = result['metrics'][metric]
        ys = [d['64']['rho'], d['128']['rho']]
        lo = [y - d[k]['ci95'][0] for y, k in zip(ys, ('64', '128'))]
        hi = [d[k]['ci95'][1] - y for y, k in zip(ys, ('64', '128'))]
        ax.errorbar(xs, ys, yerr=[lo, hi], color=color, lw=2, marker='o', ms=7,
                    capsize=5, zorder=3, label=metric)
    ax.axhline(0.7, color=LIMIT, lw=1.2, ls='--', zorder=2, label='gate (0.7)')
    style(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(['GRID_N = 64', 'GRID_N = 128'])
    ax.set_ylabel('Spearman rho vs the same 3D truth', color=TEXT_SECONDARY)
    ax.set_ylim(0.6, 1.0)
    ax.set_title('refining the grid does not improve ranking', color=TEXT_PRIMARY,
                 fontsize=12, loc='left')
    ax.legend(fontsize=9, frameon=False, loc='lower left')

    # WHY NOT #
    ax = axes[1]
    for metric, color in SERIES.items():
        key = f'{metric}_2D'
        x = np.array([rows64[n][key] for n in names])
        y = np.array([rows128[n][key] for n in names])
        rel = 100 * np.median(np.abs(y - x) / np.abs(x))
        rho = stats.spearmanr(x, y).statistic
        ax.scatter(x, y, s=34, color=color, alpha=0.75, edgecolor='white', lw=0.5,
                   zorder=3, label=f'{metric}: values move {rel:.0f}%, rank rho {rho:.3f}')
    lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, color=TEXT_SECONDARY, lw=1, ls='--', zorder=2, label='unchanged')
    style(ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('2D label at GRID_N = 64', color=TEXT_SECONDARY)
    ax.set_ylabel('2D label at GRID_N = 128', color=TEXT_SECONDARY)
    ax.set_title('values shift, order does not', color=TEXT_PRIMARY, fontsize=12,
                 loc='left')
    ax.legend(fontsize=8, frameon=False, loc='upper left')

    fig.suptitle(f"Resolution test - same {len(names)} cells, same 3D truth, "
                 f"no new Ansys jobs   |   recommended GRID_N = "
                 f"{result['recommended_grid_n']}",
                 color=TEXT_PRIMARY, fontsize=13, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, facecolor='white')

    for metric, d in result['metrics'].items():
        print(f"  {metric:12} rho 64={d['64']['rho']:.4f}  128={d['128']['rho']:.4f}  "
              f"delta {d['delta_rho']:+.4f}  materially better: {d['materially_better']}")
    print(f"  recommended GRID_N = {result['recommended_grid_n']}")
    print('Wrote', OUT)

if __name__ == "__main__":
    main()
