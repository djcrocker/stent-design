"""
Left  - Rank disagreement against solver difficulty. If the cells the screen gets wrong are
        also the ones that needed heavy bisection, then neither number is trustworthy there
        and the gate is resting partly on noise.
Right - The 3D/2D ratio against f_metal. The screen is miscalibrated and that's fine
        by design, but the spread matters: a constant offset would be harmless, a 
        but a systematic drift with density wouldn't.

Usage: python figures/scripts/s6_2_residuals.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import config
from sim3d import correlate, spikeb

OUT = config.FIG_DEV_DIR / 's6_2_residuals.png'

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
    rows, _ = correlate.load_labels()
    retries = np.array([r.get('retries') or 0 for r in rows], float)
    f_metal = np.array([r['f_metal'] for r in rows], float)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # DISAGREEMENT vs DIFFICULTY #
    ax = axes[0]
    for metric, color in SERIES.items():
        x2 = np.array([r[f'{metric}_2D'] for r in rows], float)
        y3 = np.array([r[f'{metric}_3D'] for r in rows], float)
        disagree = np.abs(stats.rankdata(x2) - stats.rankdata(y3))
        jitter = np.random.default_rng(1).normal(0, 0.08, len(retries))
        ax.scatter(retries + jitter, disagree, s=34, color=color, alpha=0.75,
                   edgecolor='white', lw=0.5, zorder=3, label=metric)
        if len(np.unique(retries)) > 1:
            rho, p = stats.spearmanr(retries, disagree)
            print(f'  {metric:12} rank-disagreement vs retries: rho={rho:+.3f} p={p:.3f}')
    style(ax)
    ax.set_xlabel('bisection retries during the solve', color=TEXT_SECONDARY)
    ax.set_ylabel('|rank(2D) - rank(3D)|', color=TEXT_SECONDARY)
    ax.set_title('does the screen err most where the solver struggled?',
                 color=TEXT_PRIMARY, fontsize=12, loc='left')
    ax.legend(fontsize=9, frameon=False)

    # CALIBRATION SPREAD #
    ax = axes[1]
    for metric, color in SERIES.items():
        x2 = np.array([r[f'{metric}_2D'] for r in rows], float)
        y3 = np.array([r[f'{metric}_3D'] for r in rows], float)
        ratio = np.divide(y3, x2, out=np.full_like(y3, np.nan), where=x2 != 0)
        ax.scatter(f_metal, ratio, s=34, color=color, alpha=0.75, edgecolor='white',
                   lw=0.5, zorder=3, label=metric)
        good = np.isfinite(ratio)
        print(f'  {metric:12} 3D/2D ratio: median {np.nanmedian(ratio):.3f}  '
              f'spread {np.nanmin(ratio):.3f}-{np.nanmax(ratio):.3f}')
    ax.axhline(1.0, color=LIMIT, lw=1.2, ls='--', zorder=2, label='perfect agreement')
    style(ax)
    ax.set_xlabel('f_metal', color=TEXT_SECONDARY)
    ax.set_ylabel('3D / 2D', color=TEXT_SECONDARY)
    ax.set_title('calibration spread',
                 color=TEXT_PRIMARY, fontsize=12, loc='left')
    ax.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, facecolor='white')
    print('Wrote', OUT)

if __name__ == "__main__":
    main()
