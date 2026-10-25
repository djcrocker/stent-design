"""
Figure 5 (motifs) - what the winning topologies do that the conventional families cannot.

Panel A - the pool in its own two biggest shape dimensions, by motif cluster, winners ringed
Panel B - the exclusion plane: the one descriptor pair where the two populations do not meet
Panel C - cluster exemplars beside a conventional cell, tiled 2x2 so the motif is visible
Panel D - density-matched effect sizes, so the eye-catching panels have their statistics

Usage: python figures/scripts/s10_3_motifs.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
from screen import motifs

OUT = config.FIG_DEV_DIR / 's10_3_motifs.png'

CLUSTER_COLORS = ['#2a78d6', '#eb6834', '#2f9e64', '#7a3fbf']
WINNER = '#0b0b0b'
CONV = '#e34948'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'

def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)

def panel_embedding(ax, report, pool_rows):
    cl = report['clusters']
    coords = np.array(cl['coords'], float)
    labels = np.array(cl['labels'], int)
    kept = np.array(cl['kept'], int)
    layer = np.array([pool_rows[i]['pareto_layer'] for i in kept], int)
    win = layer <= report['winner_layers']

    style(ax)
    for c in sorted(set(labels.tolist())):
        m = labels == c
        d = cl['clusters'][str(c)]
        ax.scatter(coords[m, 0], coords[m, 1], s=16, alpha=0.55,
                   color=CLUSTER_COLORS[(c - 1) % len(CLUSTER_COLORS)],
                   label=f"cluster {c} (n={d['n']}, {d['winner_frac']:.0%} win)",
                   edgecolors='none', zorder=2)
    ax.scatter(coords[win, 0], coords[win, 1], s=46, facecolors='none',
               edgecolors=WINNER, linewidths=1.0, zorder=3, label='Pareto front')
    v = cl['pc_variance_frac']
    ax.set_xlabel(f'PC1 ({v[0]:.0%} of shape variance)', color=TEXT_SECONDARY)
    ax.set_ylabel(f'PC2 ({v[1]:.0%})', color=TEXT_SECONDARY)
    ax.set_title('A  shape space of the generated pool', color=TEXT_PRIMARY,
                 fontsize=12, loc='left')
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc='lower right')

def panel_exclusion(ax, report, pool_rows, conv_rows, x='straight_run_circ', y='thickness_cv'):
    style(ax)
    layer = np.array([r['pareto_layer'] for r in pool_rows], int)
    win = layer <= report['winner_layers']
    px = np.array([r[x] for r in pool_rows], float)
    py = np.array([r[y] for r in pool_rows], float)
    cx = np.array([r[x] for r in conv_rows], float)
    cy = np.array([r[y] for r in conv_rows], float)

    # The box the conventional families live in.
    ax.add_patch(plt.Rectangle((cx.min(), cy.min()), np.ptp(cx), np.ptp(cy),
                               facecolor=CONV, alpha=0.10, edgecolor=CONV, lw=1.0,
                               ls='--', zorder=1, label='conventional envelope'))
    ax.scatter(px[~win], py[~win], s=14, color='#b9c2cc', alpha=0.7, edgecolors='none', zorder=2, label='pool, dominated')
    ax.scatter(px[win], py[win], s=30, color=WINNER, alpha=0.85, edgecolors='none', zorder=4, label='Pareto front')
    ax.scatter(cx, cy, s=34, marker='^', color=CONV, edgecolors='none', zorder=3, label='parametric + handmade')
    ex = (report['exclusion_density_matched'] or report['exclusion'])[y]
    ax.set_xlabel('straight_run_circ   (1.0 = a complete circumferential band)', color=TEXT_SECONDARY)
    ax.set_ylabel('thickness_cv   (strut width variation along the member)', color=TEXT_SECONDARY)
    ax.set_title(f"B  the exclusion plane - {ex['frac_outside']:.0%} of winners lie outside", color=TEXT_PRIMARY, fontsize=12, loc='left')
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc='upper left')

def panel_exemplars(axes, report, fields, conv_rows):
    """Cluster exemplars, tiled 2x2 so the motif reads as a pattern."""
    from geom import handmade, parametric

    cl = report['clusters']
    items = []
    for c in sorted(cl['clusters']):
        d = cl['clusters'][c]
        items.append((f"cluster {c}  ({d['winner_frac']:.0%} win)",
                      fields[d['exemplar_row']],
                      CLUSTER_COLORS[(int(c) - 1) % len(CLUSTER_COLORS)]))
    valid, _ = parametric.sweep()
    ref = max(valid, key=lambda pc: pc[1].f_metal)[1].to_array()
    items.append(('parametric crown', ref, CONV))

    for ax, (title, arr, color) in zip(axes, items):
        ax.imshow(np.tile(arr, (2, 2)), cmap='binary', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(color)
            side.set_linewidth(2.0)
        ax.set_title(title, color=color, fontsize=9, loc='left')

def panel_effects(ax, report, top=10):
    per = report['winners_vs_conventional_density_matched']['per_metric']
    rows = sorted((kv for kv in per.items() if 'cliffs_delta' in kv[1]),
                  key=lambda kv: -abs(kv[1]['cliffs_delta']))[:top][::-1]
    names = [k for k, _ in rows]
    vals = [d['cliffs_delta'] for _, d in rows]
    sig = [d['p'] < 0.05 for _, d in rows]

    style(ax)
    colors = [WINNER if s else '#b9c2cc' for s in sig]
    ax.barh(range(len(vals)), vals, color=colors, height=0.68, zorder=2)
    ax.axvline(0, color=TEXT_SECONDARY, lw=0.8, zorder=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5, color=TEXT_SECONDARY)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Cliff's delta (winners vs conventional, density-matched)", color=TEXT_SECONDARY)
    m = report['winners_vs_conventional_density_matched']['matching']
    lo, hi = m['overlap']
    ax.set_title(f"D  effect sizes   f_metal {lo:.2f}-{hi:.2f}, "
                 f"n={m['n_a']} vs {m['n_b']}  (grey: p>=0.05)",
                 color=TEXT_PRIMARY, fontsize=12, loc='left')

def main():
    report, table, fields = motifs.load()
    pool_rows, conv_rows = table['pool'], table['conventional']

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.25, 1.25, 0.75], hspace=0.32, wspace=0.34)
    panel_embedding(fig.add_subplot(gs[0, :3]), report, pool_rows)
    panel_effects(fig.add_subplot(gs[0, 3:]), report)
    panel_exclusion(fig.add_subplot(gs[1, :3]), report, pool_rows, conv_rows)

    ax_note = fig.add_subplot(gs[1, 3:])
    ax_note.axis('off')
    ex = report['exclusion_density_matched'] or report['exclusion']
    lines = ['Descriptor values the conventional', 'families never reach:', '']
    ranked = sorted((kv for kv in ex.items() if 'frac_outside' in kv[1]),
                    key=lambda kv: -kv[1]['frac_outside'])[:6]
    for key, d in ranked:
        lo, hi = d['conventional_range']
        wlo, whi = d['winner_range']
        lines.append(f"{key}")
        lines.append(f"   outside {d['frac_outside']:.0%}   conv [{lo:.2f}, {hi:.2f}]"
                     f"   win [{wlo:.2f}, {whi:.2f}]")
    lines += ['', 'PENDING: the SIMP / inverse-homogenization',
              'baseline (S10.1) is not in this comparison,',
              'so "conventional" is the parametric family',
              'plus the handmade cells only.']
    ax_note.text(0, 1, chr(10).join(lines), va='top', ha='left', fontsize=8.6,
                 color=TEXT_SECONDARY, family='monospace', linespacing=1.55)

    panel_exemplars([fig.add_subplot(gs[2, i]) for i in range(5)], report, fields, conv_rows)

    pops = report['populations']
    fig.suptitle(f"Motif analysis - {pops['winners']['n']} Pareto-front designs vs "
                 f"{pops['conventional']['n']} conventional cells, "
                 f"from a pool of {pops['pool']['n']}",
                 color=TEXT_PRIMARY, fontsize=13, x=0.01, ha='left')
    fig.subplots_adjust(top=0.94, bottom=0.05, left=0.06, right=0.98)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, facecolor='white')

    print(f"winners {pops['winners']['n']}  pool {pops['pool']['n']}  "
          f"conventional {pops['conventional']['n']}")
    for key, d in ranked[:4]:
        print(f"  {key:22s} outside {d['frac_outside']:5.1%}  "
              f"conv [{d['conventional_range'][0]:.3f}, {d['conventional_range'][1]:.3f}]")
    print('Wrote', OUT)

if __name__ == '__main__':
    main()
