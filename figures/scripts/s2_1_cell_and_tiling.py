"""
Writes a PNG of a hand-drawn diamond cell and a 3x3 tiling of that cell.

Usage: python figures/scripts/s2_1_cell_and_tiling.py
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from geom.handmade import diamond
from geom.render import plot_cell, plot_tiling

OUT = config.FIG_DEV_DIR / 's2_1_cell_and_tiling.png'

def main():
    cell = diamond()
    print(cell)
    print(f"  grid            {cell.n} x {cell.n}")
    print(f"  extent (mm)     {cell.extent_mm[0]:.4f} circ x {cell.extent_mm[1]:.4f} axial")
    print(f"  mm per pixel    {cell.mm_per_px[0]:.5f}")
    print(f"  f_metal         {cell.f_metal:.4f}")
    print(f"  strut width     {config.STRUT_WIDTH_MM} mm = {config.min_feature_px():.1f} px")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_cell(cell, ax=axes[0], title="Hand-drawn diamond cell")
    plot_tiling(cell, 3, 3, ax=axes[1], title="3 x 3 tiling (cell boundaries in red)")
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"\nWrote {OUT}")

if __name__ == "__main__":
    main()
