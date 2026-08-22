"""
Left  - Parameter grid: all 82 valid sweep cells as a background, the 54 chosen for
        Spike B on top, marked by outcome. Farthest-point selection is meant to spread
        across the grid rather than cluster, and this is where that either shows or does not.
Right - Convergence against the 2D-predicted strain: if the cells the 3D tier can't solve 
        are the high-strain ones, the correlation i smeasured on an easier population.

Usage: python figures/scripts/s6_2_coverage.py
"""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
from geom import parametric
from sim3d import spikeb

OUT = config.FIG_DEV_DIR / 's6_2_coverage.png'

OK = '#2e9e5b'
RETRIED = '#e0a83a'
FAILED = '#e34948'
BACKDROP = '#c9c9c4'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'

MARKER = {1: 'o', 2: 's', 3: '^'}

def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)

def outcome(name):
    c = spikeb.read_convergence(config.PROJECT_ROOT / 'ansys_output' / f'{name}.mntr')
    if not c['converged']:
        return FAILED
    return RETRIED if c['retries'] > 2 else OK

def main():
    valid, _ = parametric.sweep()
    specs = [s for s in spikeb.select_cells() if s['family'] == 'crown']
    # 2D labels come from the manifest written at build time.
    manifest = json.loads((config.PROJECT_ROOT / 'sim3d' / 'results'
                           / 's6_1_manifest.json').read_text(encoding='utf-8'))
    labels = {c['name']: c['labels_2d'] for c in manifest['cells']}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # PARAMETER GRID #
    ax = axes[0]
    jitter = {1: -0.006, 2: 0.0, 3: 0.006}
    ax.scatter([p['strut_width_mm'] for p, _ in valid],
               [p['crown_amplitude'] for p, _ in valid],
               s=110, color=BACKDROP, alpha=0.35, zorder=2,
               label=f'valid sweep, not chosen ({len(valid)})')
    for spec in specs:
        p = spec['params']
        ax.scatter(p['strut_width_mm'] + jitter[p['n_periods']], p['crown_amplitude'],
                   s=58, marker=MARKER[p['n_periods']], color=outcome(spec['name']),
                   edgecolor='white', lw=0.6, zorder=3)
    style(ax)
    ax.set_xlabel('strut width (mm)', color=TEXT_SECONDARY)
    ax.set_ylabel('crown amplitude', color=TEXT_SECONDARY)
    ax.set_title('parameter coverage  (marker = n_periods, offset for legibility)',
                 color=TEXT_PRIMARY, fontsize=12, loc='left')
    handles = [plt.Line2D([], [], marker=MARKER[k], ls='', color=TEXT_SECONDARY,
                          label=f'n_periods = {k}') for k in (1, 2, 3)]
    handles += [plt.Line2D([], [], marker='o', ls='', color=c, label=t) for c, t in
                ((OK, 'converged'), (RETRIED, 'after retry'), (FAILED, 'failed'))]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc='upper right', ncol=2)

    # BIAS #
    ax = axes[1]
    groups = {'converged': [], 'after retry': [], 'failed': []}
    for spec in specs:
        c = outcome(spec['name'])
        key = 'failed' if c == FAILED else ('after retry' if c == RETRIED else 'converged')
        groups[key].append(labels[spec['name']]['eps_a_max_2D'] * 100)
    labels, colors = list(groups), [OK, RETRIED, FAILED]
    data = [groups[k] for k in labels]
    parts = ax.boxplot(data, tick_labels=[f'{k}\n(n={len(groups[k])})' for k in labels],
                       patch_artist=True, widths=0.55, zorder=3)
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    for key in ('whiskers', 'caps', 'medians'):
        for line in parts[key]:
            line.set_color(TEXT_SECONDARY)
    for i, (k, color) in enumerate(zip(labels, colors), start=1):
        y = groups[k]
        ax.scatter(np.random.default_rng(0).normal(i, 0.055, len(y)), y, s=22,
                   color=color, alpha=0.85, edgecolor='white', lw=0.5, zorder=4)
    style(ax)
    ax.set_ylabel('eps_a_max predicted by the 2D screen (%)', color=TEXT_SECONDARY)
    ax.set_title('convergence bias: harder designs are the ones that drop out',
                 color=TEXT_PRIMARY, fontsize=12, loc='left')

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, facecolor='white')

    for k in labels:
        v = groups[k]
        if v:
            print(f'  {k:12} n={len(v):2d}  median eps_a_max_2D {np.median(v):.3f} %')
    print('Wrote', OUT)

if __name__ == "__main__":
    main()
