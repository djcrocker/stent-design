"""
Rank-rank correlation: Spearman measures ordering, and a raw scatter can look linear while 
the ranking is poor. Perfect ranking is the diagonal.

Top row    - rank(2D) vs rank(3D) per metric, with rho and its bootstrap CI
Bottom row - top-K retention: rank by the surrogate, keep the best K, and hope the winners 
             are inside. rho can look healthy while errors concentrate at the top.

Usage: python figures/scripts/s6_2_correlation.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import config
from sim3d import correlate

OUT = config.FIG_DEV_DIR / 's6_2_correlation.png'

CROWN = '#2a78d6'
HAND = '#eb6834'
REF = '#7a3fbf'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'
LIMIT = '#e34948'

COLOR = {'crown': CROWN, 'handmade': HAND, 'reference': REF}
METRICS = ('K_radial', 'eps_a_max', 'A_over_lim')

def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)

def main():
    rows, all_rows = correlate.load_labels()
    result = correlate.analyze(n_boot=4000)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))

    for col, metric in enumerate(METRICS):
        d = result['overall'][metric]
        x = np.array([r[f'{metric}_2D'] for r in rows])
        y = np.array([r[f'{metric}_3D'] for r in rows])
        rx, ry = stats.rankdata(x), stats.rankdata(y)

        ax = axes[0, col]
        for fam in ('crown', 'handmade', 'reference'):
            m = np.array([r['family'] == fam for r in rows])
            if m.any():
                ax.scatter(rx[m], ry[m], s=42, color=COLOR[fam], alpha=0.85,
                           edgecolor='white', lw=0.6, zorder=3,
                           label=f'{fam} (n={int(m.sum())})')
        lim = [0, len(rows) + 1]
        ax.plot(lim, lim, color=TEXT_SECONDARY, lw=1, ls='--', zorder=2,
                label='perfect ranking')
        style(ax)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel('rank, 2D screen', color=TEXT_SECONDARY)
        if col == 0:
            ax.set_ylabel('rank, 3D FEA', color=TEXT_SECONDARY)
        gate = ' (GATE)' if d['gates'] else ''
        ax.set_title(f'{metric}{gate}', color=TEXT_PRIMARY, fontsize=12, loc='left')
        # "reported only" read as a failure.
        if d['gates']:
            verdict = 'passes gate' if d['passes'] else 'FAILS gate'
        else:
            side = 'above' if d['rho'] >= correlate.GATE_RHO else 'below'
            verdict = f'not gated ({side} {correlate.GATE_RHO})'
        ax.text(0.03, 0.97,
                f"rho = {d['rho']:.3f}\n95% CI [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]\n{verdict}",
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                color=TEXT_PRIMARY,
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=GRID))
        if col == 0:
            ax.legend(loc='lower right', fontsize=8, frameon=False)

        # RETENTION #
        ax = axes[1, col]
        curve = correlate.retention_curve(x, y, d['higher_is_better'])
        ks = [k for k, _ in curve]
        vs = [v for _, v in curve]
        ax.plot(ks, vs, color=COLOR['crown'], lw=2, zorder=3)
        ax.axhspan(0, 0.7, color=LIMIT, alpha=0.07, zorder=1)
        ax.axvspan(10, 30, color=TEXT_SECONDARY, alpha=0.10, zorder=1)
        ax.text(20, 0.06, 'Stage 9\nkeeps ~10-30', ha='center', fontsize=8,
                color=TEXT_SECONDARY)
        style(ax)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(1, len(rows))
        ax.set_xlabel('K (cells kept by the 2D screen)', color=TEXT_SECONDARY)
        if col == 0:
            ax.set_ylabel('fraction of the true top-K kept', color=TEXT_SECONDARY)
        direction = 'higher is better' if d['higher_is_better'] else 'lower is better'
        ax.set_title(f'top-K retention  ({direction})', color=TEXT_PRIMARY,
                     fontsize=11, loc='left')

    fig.suptitle(
        f"2D screen vs 3D FEA - {result['n_converged']} of {result['n_total']} cells"
        f"   |   gate: rho >= {correlate.GATE_RHO} on K_radial and eps_a_max",
        color=TEXT_PRIMARY, fontsize=13, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, facecolor='white')

    print(f"n = {result['n_converged']}/{result['n_total']} cells")
    for metric in METRICS:
        d = result['overall'][metric]
        tag = 'GATE' if d['gates'] else '    '
        print(f"  {tag} {metric:12} rho={d['rho']:+.3f}  "
              f"CI95=[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
    print()
    print(f"  gate passed          {result['gate_passed']}")
    print(f"  CI clears threshold  {result['gate_ci_clears']}")
    print()
    print('Wrote', OUT)

if __name__ == "__main__":
    main()
