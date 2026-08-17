"""
Hand-drawn reference cells.
"""

import numpy as np
import config
from geom.cell import UnitCell

def diamond(n=None, width_mm=None):
    """
    A periodic diamond/crown lattice cell.
    """
    n = config.GRID_N if n is None else n
    width_mm = config.STRUT_WIDTH_MM if width_mm is None else width_mm
    circ_mm, _ = config.cell_extent_mm()

    # Pixel centers in normalized cell coordinates, [0, 1).
    coords = (np.arange(n) + 0.5) / n
    v, u = np.meshgrid(coords, coords, indexing="ij")  # v axial (axis 0), u circ (axis 1)

    # Level sets of f are diamonds; f = 0.5 is the one touching all four edge midpoints.
    f = np.abs(u - 0.5) + np.abs(v - 0.5)

    # |grad f| = sqrt(2), so a band of half-height h has perpendicular width h*sqrt(2).
    # Solve for the band that gives the requested physical strut width.
    width_norm = width_mm / circ_mm
    h = width_norm / np.sqrt(2)

    return UnitCell(np.abs(f - 0.5) <= h)
