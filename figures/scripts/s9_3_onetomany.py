"""
One-to-many gallery.

This is the core argument for using a generative method rather than an optimizer. An
optimizer returns one answer; the claim is that the inverse problem admits many, and that
the model surfaces them.

Rows   - Exemplars at one target each, chosen farthest-first in a translation-invariant
         descriptor so the row spans the variety present rather than showing near-copies.
         Each is annotated with what it achieved
Bottom - A design against its own torus shifts scores zero, which is what allows calling 
         anything above it a real difference; the within-target and cross-target distributions
         then show how much of the model's total variety is available at a single target.

Usage: python figures/scripts/s9_3_onetomany.py
"""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
from screen import onetomany

REPORT = config.PROJECT_ROOT / 'screen' / 'results' / 's9_3_onetomany.json'
OUT = config.FIG_DEV_DIR / 's9_3_onetomany.png'

ACCENT = '#0E7C86'
WITHIN = '#2a78d6'
CROSS = '#eb6834'
FLOOR = '#2e9e5b'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'

# Three targets spanning easy to hard, so the reader sees variety shrink as the ask tightens.
SHOW = ('control_mid', 'K200_A10', 'K400_A10')
N_SHOW = 6

def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)

def main():
    from diffusion import generate

    report = json.loads(REPORT.read_text(encoding='utf-8'))
    fields, which, meta = generate.load_samples(report['source'])
    rows, _ = generate.cached_labels(fields, report['source'])
    by_name = {t['name']: t for t in report['per_target'] if 'note' not in t}

    shown = [n for n in SHOW if n in by_name]
    fig = plt.figure(figsize=(14.5, 4.0 + 2.6 * len(shown)))
    gs = fig.add_gridspec(len(shown) + 1, N_SHOW, height_ratios=[1] * len(shown) + [1.15], hspace=0.5, wspace=0.1)

    for r, name in enumerate(shown):
        t = by_name[name]
        idx = t['exemplar_pool_indices'][:N_SHOW]
        for c in range(N_SHOW):
            ax = fig.add_subplot(gs[r, c])
            if c < len(idx):
                i = idx[c]
                ax.imshow(fields[i], cmap='binary', interpolation='nearest')
                lab = rows[i]
                ax.set_title(f"K {lab['K_radial']:.0f}   A {lab['A_over_lim']:.3f}",
                             fontsize=8, color=TEXT_SECONDARY, pad=3)
                for side in ax.spines.values():
                    side.set_color(ACCENT)
                    side.set_linewidth(1.6)
            else:
                ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                k10, k90 = t['y_spread']['K_radial']
                a10, a90 = t['y_spread']['A_over_lim']
                ax.set_ylabel(f"{name}\n{t['n_valid']} valid\n"
                              f"K {k10:.0f}–{k90:.0f}\nA {a10:.3f}–{a90:.3f}",
                              fontsize=8.5, color=TEXT_PRIMARY, rotation=0,
                              ha='right', va='center', labelpad=52)

    # DISTANCE DISTRIBUTIONS #
    bottom = gs[len(shown), :].subgridspec(1, 2, wspace=0.24)
    ax = fig.add_subplot(bottom[0, 0])
    names = [t['name'] for t in report['per_target'] if 'note' not in t]
    means = [t['within_mean'] for t in report['per_target'] if 'note' not in t]
    order = np.argsort(means)
    y = np.arange(len(order))
    ax.barh(y, [means[i] for i in order], color=WITHIN, zorder=3, height=0.68)
    ax.axvline(report['cross_target_mean'], color=CROSS, lw=1.6, ls='--', zorder=4,
               label=f"cross-target {report['cross_target_mean']:.3f}")
    ax.axvline(max(report['shift_floor_max'], 0.004), color=FLOOR, lw=1.6, zorder=4,
               label=f"shift floor {report['shift_floor_max']:.4f}")
    style(ax)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set_xlabel('mean structural distance within a target', color=TEXT_SECONDARY)
    ax.set_title('same target, different designs', color=TEXT_PRIMARY, fontsize=11,
                 loc='left')
    ax.legend(fontsize=8, frameon=False, loc='lower right')

    # HOW MUCH OF THE TOTAL VARIETY IS AVAILABLE AT ONE TARGET #
    ax = fig.add_subplot(bottom[0, 1])
    ratios = [t['ratio_to_cross_target'] for t in report['per_target'] if 'note' not in t]
    ns = [t['n_valid'] for t in report['per_target'] if 'note' not in t]
    ax.scatter(ns, ratios, s=46, color=ACCENT, edgecolor='white', lw=0.7, zorder=3)
    ax.axhline(1.0, color=CROSS, lw=1.4, ls='--', zorder=2,
               label='as varied as different targets')
    style(ax)
    ax.set_ylim(0, 1.12)
    ax.margins(x=0.06)
    ax.set_xlabel('valid designs at the target', color=TEXT_SECONDARY)
    ax.set_ylabel('within / cross-target', color=TEXT_SECONDARY)
    ax.set_title('variety shrinks as the ask tightens', color=TEXT_PRIMARY, fontsize=11,
                 loc='left')
    ax.legend(fontsize=8, frameon=False, loc='lower right')

    above = min(t['fraction_above_floor'] for t in report['per_target'] if 'note' not in t)
    fig.suptitle('One-to-many: several distinct topologies meet the same target   |   '
                 f'{100 * above:.0f}% of pairs differ by more than a torus shift, '
                 f'at every target',
                 color=TEXT_PRIMARY, fontsize=12.5, x=0.01, ha='left')
    fig.subplots_adjust(left=0.10, right=0.985, top=0.93, bottom=0.07)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=155, facecolor='white')

    print(f"shift floor {report['shift_floor_max']:.4f}   "
          f"cross-target {report['cross_target_mean']:.4f}")
    for t in report['per_target']:
        if 'note' not in t:
            print(f"  {t['name']:12} n={t['n_valid']:3d}  within {t['within_mean']:.4f}  "
                  f"ratio {t['ratio_to_cross_target']:.2f}  "
                  f"above floor {100 * t['fraction_above_floor']:.0f}%")
    print('Wrote', OUT)

if __name__ == "__main__":
    main()
