"""
Cells come from geom.handmade's VALID_CELLS / BROKEN_CELLS registries, the same ones
tests/test_validity.py asserts against.

Usage: python figures/scripts/s2_3_validity.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from geom import handmade as H
from geom import validity as V
from geom.render import plot_cell

OUT = config.FIG_DEV_DIR / 's2_3_validity.png'

def main():
    cells = ([(name, build()) for name, build in H.VALID_CELLS.items()]
             + [(name, build()) for name, (build, _) in H.BROKEN_CELLS.items()])

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ax, (name, cell) in zip(axes.ravel(), cells):
        result = V.check(cell)
        verdict = 'VALID' if result.ok else ', '.join(result.reasons)
        plot_cell(cell, ax=ax, title=f'{name}\n{verdict}')
        ax.title.set_color('tab:green' if result.ok else 'tab:red')
        ax.set_xlabel('')
        ax.set_ylabel('')
        print(f"{name:20} {'VALID' if result.ok else 'INVALID':8} "
              f"f={result.metrics['f_metal']:.3f} {result.reasons}")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"\nWrote {OUT}")

if __name__ == "__main__":
    main()
