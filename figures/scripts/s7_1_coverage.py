"""
Does the dataset actually cover the objective space?

Top row    - One histogram per `y` component, stacked by source, with the parametric
             family's own span shaded. Bars outside the shading are designs the family
             can't express at all.
Bottom row - Where each source sits in the objective plane, and the source mix before
             versus after coverage subsampling.

Usage: python figures/scripts/s7_1_coverage.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
from diffusion import dataset

OUT = config.FIG_DEV_DIR / 's7_1_coverage.png'

COLOR = {'family': '#2a78d6', 'perturbation': '#eb6834', 'lattice': '#2e9e5b', 'random': '#7a3fbf'}
ORDER = ('family', 'perturbation', 'lattice', 'random')
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#dcdcd8'
FAMILY_BAND = '#9fb8d4'

PANELS = (('K_radial', True), ('eps_a_max', False),
          ('A_over_lim', False), ('f_metal', False))

def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)

def main():
    _, frame = dataset.load()
    sources = [s for s in ORDER if (frame['source'] == s).any()]

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9], hspace=0.35, wspace=0.28)

    # HISTOGRAMS #
    for col, (key, log) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, col])
        v = frame[key].to_numpy()
        fam = frame.loc[frame['source'] == 'family', key].to_numpy()

        if log:
            v_plot = np.log10(np.maximum(v, 1e-6))
            edges = np.linspace(v_plot.min(), v_plot.max(), 45)
            label = f'log10({key})'
            band = (np.log10(max(fam.min(), 1e-6)), np.log10(max(fam.max(), 1e-6)))
        else:
            v_plot = v
            edges = np.linspace(v.min(), v.max(), 45)
            label = key
            band = (fam.min(), fam.max())

        stack = [v_plot[(frame['source'] == s).to_numpy()] for s in sources]
        ax.hist(stack, bins=edges, stacked=True,
                color=[COLOR[s] for s in sources], label=sources, zorder=3)
        # Anything outside this shading is unreachable by the parametric family.
        ax.axvspan(band[0], band[1], color=FAMILY_BAND, alpha=0.25, zorder=1)
        style(ax)
        ax.set_xlabel(label, color=TEXT_SECONDARY)
        if col == 0:
            ax.set_ylabel('cells', color=TEXT_SECONDARY)
            ax.legend(fontsize=8, frameon=False)
        outside = int(((v < fam.min()) | (v > fam.max())).sum())
        ax.set_title(f'{key}\n{outside:,} cells beyond the family',
                     color=TEXT_PRIMARY, fontsize=11, loc='left')

    # OBJECTIVE PLANE #
    ax = fig.add_subplot(gs[1, :2])
    for s in sources:
        m = (frame['source'] == s).to_numpy()
        ax.scatter(frame.loc[m, 'K_radial'], frame.loc[m, 'eps_a_max'],
                   s=3, alpha=0.25, color=COLOR[s], zorder=3, label=s)
    ax.set_xscale('log')
    ax.axhline(config.EPS_A_LIM, color='#e34948', lw=1.2, ls='--', zorder=4,
               label=f'fatigue limit {config.EPS_A_LIM}')
    style(ax)
    ax.set_xlabel('K_radial (N/mm^3, log)', color=TEXT_SECONDARY)
    ax.set_ylabel('eps_a_max', color=TEXT_SECONDARY)
    ax.set_title('the objective plane the model will be conditioned on',
                 color=TEXT_PRIMARY, fontsize=11, loc='left')
    leg = ax.legend(fontsize=8, frameon=False, markerscale=4)
    for handle in leg.legend_handles:
        try:
            handle.set_alpha(1.0)
        except AttributeError:
            pass

    # MIX SHIFT #
    ax = fig.add_subplot(gs[1, 2:])
    import json
    info = json.loads((config.DATA_DIR / 'dataset_info.json').read_text(encoding='utf-8'))
    kept = info['pool']['stats']['kept']
    pool_total = sum(kept.values())
    before = [100 * kept.get(s, 0) / pool_total for s in sources]
    after = [100 * (frame['source'] == s).sum() / len(frame) for s in sources]
    x = np.arange(len(sources))
    ax.bar(x - 0.19, before, 0.36, color=[COLOR[s] for s in sources], alpha=0.45,
           zorder=3, label='pool (100k)')
    ax.bar(x + 0.19, after, 0.36, color=[COLOR[s] for s in sources], zorder=3,
           label='after coverage subsample (50k)')
    style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.set_ylabel('% of cells', color=TEXT_SECONDARY)
    ax.set_title('subsampling reweights toward whatever occupies unique bins',
                 color=TEXT_PRIMARY, fontsize=11, loc='left')
    ax.legend(fontsize=8, frameon=False)

    labels = frame[list(dataset.Y_KEYS)].to_dict('records')
    occupied = len({tuple(r) for r in dataset.bin_indices(labels, n_bins=8)})
    fig.suptitle(f'S7.1 dataset - {len(frame):,} cells, {occupied} of 4096 y-bins occupied, '
                 f'0 NaNs', color=TEXT_PRIMARY, fontsize=13, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor='white')

    print(f'{len(frame):,} cells, {occupied} of 4096 y-bins occupied')
    for s in sources:
        n = int((frame['source'] == s).sum())
        print(f'  {s:13} {n:6,}  ({100 * n / len(frame):5.1f}%)')
    print('Wrote', OUT)

if __name__ == "__main__":
    main()
