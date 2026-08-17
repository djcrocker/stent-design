"""
Top row is the rectangle-based scipy op, bottom row is ours.
scipy's erosion messes with the struts wherever they cross a cell boundary, 
so the tiling shows a cross of damage.

Usage: python figures/scripts/s2_2_periodicity.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
import config
from geom import periodic
from geom.cell import UnitCell
from geom.handmade import diamond
from geom.render import plot_cell, plot_tiling

OUT = config.FIG_DEV_DIR / 's2_2_periodicity.png'

def main():
    cell = diamond()
    arr = cell.to_array()

    rect = UnitCell(ndimage.binary_erosion(arr, periodic.CONN4, iterations=2))
    ours = UnitCell(periodic.erosion(arr, iterations=2))

    print(f"original      f_metal={cell.f_metal:.4f}")
    print(f"scipy erosion f_metal={rect.f_metal:.4f}  (loses struts at the seam)")
    print(f"torus erosion f_metal={ours.f_metal:.4f}")
    print(f"shift-equivariant?  scipy={periodic.is_shift_equivariant(ndimage.binary_erosion, arr)}"
          f"  ours={periodic.is_shift_equivariant(periodic.erosion, arr)}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plot_cell(rect, ax=axes[0][0], title="scipy erosion (rectangle)")
    plot_tiling(rect, 2, 2, ax=axes[0][1], title="tiled: seam damage")
    plot_cell(ours, ax=axes[1][0], title="periodic erosion (torus)")
    plot_tiling(ours, 2, 2, ax=axes[1][1], title="tiled: no seam")
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"\nWrote {OUT}")

if __name__ == "__main__":
    main()