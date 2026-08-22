"""
A contact sheet of all 60 Spike B cells as thumbnails, bordered by what the 3D tier made of them.

Border: green converged first try, amber converged only after the retry at finer
substepping, red never reached TIME 4.0.

Usage: python figures/scripts/s6_2_contact_sheet.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import config
from sim3d import spikeb

OUT = config.FIG_DEV_DIR / 's6_2_contact_sheet.png'

OK = '#2e9e5b'
RETRIED = '#e0a83a'
FAILED = '#e34948'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'

COLS = 10

def main():
    specs = spikeb.select_cells()
    rows = int(np.ceil(len(specs) / COLS))
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 1.45, rows * 1.65))
    axes = np.atleast_2d(axes)

    n_ok = n_retried = n_failed = 0
    for i, spec in enumerate(specs):
        ax = axes[i // COLS, i % COLS]
        conv = spikeb.read_convergence(
            config.PROJECT_ROOT / 'ansys_output' / f"{spec['name']}.mntr")
        if not conv['converged']:
            color, n_failed = FAILED, n_failed + 1
        elif conv['retries'] > 2:
            color, n_retried = RETRIED, n_retried + 1
        else:
            color, n_ok = OK, n_ok + 1

        ax.imshow(spec['cell'].to_array(), cmap='binary', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(color)
            side.set_linewidth(2.2)
        label = spec['name'].replace('s6_', '')
        ax.set_title(f"{label}\nf={spec['cell'].f_metal:.2f}", fontsize=7, color=TEXT_SECONDARY, pad=3)

    for j in range(len(specs), rows * COLS):
        axes[j // COLS, j % COLS].axis('off')

    handles = [plt.Line2D([], [], color=c, lw=3, label=t) for c, t in
               ((OK, f'converged ({n_ok})'),
                (RETRIED, f'converged after retry ({n_retried})'),
                (FAILED, f'never reached TIME 4.0 ({n_failed})'))]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f'Spike B cell set - {len(specs)} cells, {n_ok + n_retried} labeled '
                 f'by both tiers', color=TEXT_PRIMARY, fontsize=13, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor='white')

    print(f'{len(specs)} cells: {n_ok} clean, {n_retried} needed the retry, '
          f'{n_failed} failed')
    print('Wrote', OUT)

if __name__ == "__main__":
    main()
