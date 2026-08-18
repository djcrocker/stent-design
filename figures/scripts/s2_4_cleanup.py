"""
Left column is the input, right is the result of cleanup with material kept in grey, removed in red, added in blue.

Usage: python figures/scripts/s2_4_cleanup.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from geom import cleanup
from geom import handmade as H
from geom.render import plot_cell, plot_change

OUT = config.FIG_DEV_DIR / 's2_4_cleanup.png'

def main():
    cases = [('diamond (valid)', H.diamond)] + [
        (name, build) for name, (build, _) in H.BROKEN_CELLS.items()
    ]

    fig, axes = plt.subplots(len(cases), 2, figsize=(8, 4 * len(cases)))
    results = []

    for (name, build), row in zip(cases, axes):
        cell = build()
        result = cleanup.clean(cell)
        results.append(result)

        plot_cell(cell, ax=row[0], title=f'{name}  (input)')
        row[0].set_xlabel(''); row[0].set_ylabel('')

        if result.cell is not None:
            label = ('UNCHANGED - nothing to repair' if result.change_fraction == 0
                     else f'REPAIRED - changed {result.change_fraction:.1%}')
            plot_change(cell, result.cell, ax=row[1], title=label)
            row[1].title.set_color('tab:green')
        else:
            plot_cell(cell, ax=row[1],
                      title='UNFIXABLE - ' + ', '.join(result.validity.reasons))
            row[1].title.set_color('tab:red')
            row[1].set_alpha(0.3)
        row[1].set_xlabel(''); row[1].set_ylabel('')

        verdict = ('UNCHANGED' if result.fixed and result.change_fraction == 0
                   else 'REPAIRED' if result.fixed else 'UNFIXABLE')
        print(f'{name:20} {verdict:10} change={result.change_fraction:7.4f}  '
              f'{result.actions if result.fixed else result.validity.reasons}')

    print()
    print('Batch report:')
    for k, v in cleanup.summarize(results).items():
        print(f'  {k:16} {v}')

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f'\nWrote {OUT}')

if __name__ == "__main__":
    main()
