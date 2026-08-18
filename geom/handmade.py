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

def grid(n=None, width_mm=None):
    """
    An orthogonal grid: one circumferential bar and one axial bar.
    """
    n = config.GRID_N if n is None else n
    width_mm = config.STRUT_WIDTH_MM if width_mm is None else width_mm
    circ_mm, _ = config.cell_extent_mm()
    w_px = int(np.ceil(width_mm / (circ_mm / n)))

    a = np.zeros((n, n), dtype=bool)
    a[:w_px, :] = True
    a[:, :w_px] = True
    return UnitCell(a)

def _bar_px(width_mm=None):
    """Bar width in whole pixels, rounded UP (a minimum, never a target)."""
    width_mm = config.STRUT_WIDTH_MM if width_mm is None else width_mm
    return int(np.ceil(width_mm / config.mm_per_px()[0]))

def _disc(arr, cy, cx, r):
    """Paint a filled circle. Circles, not squares: sharp convex corners shed area under
    the opening-based min-feature test, which would trip thin_feature spuriously."""
    y, x = np.ogrid[:arr.shape[0], :arr.shape[1]]
    arr[(y - cy) ** 2 + (x - cx) ** 2 <= r * r] = True
    return arr


# BROKEN CELLS #
# All invalid, one per failure mode.
# Shared by tests/test_validity.py and figures/scripts/s2_3_validity.py.

def empty(n=None):
    """Nothing at all -> EMPTY."""
    n = config.GRID_N if n is None else n
    return UnitCell(np.zeros((n, n), dtype=bool))

def tiny_blob(n=None, r=5):
    """A speck: too sparse, wraps neither way, thinner than w."""
    n = config.GRID_N if n is None else n
    return UnitCell(_disc(np.zeros((n, n), dtype=bool), n // 4, n // 4, r))

def bands_circ_only(n=None, width_mm=None):
    """
    Rings around the circumference, nothing joining them -> NO_WRAP_AXIAL.
    Kept above F_METAL_MIN on purpose so it isolates the wrap failure.
    """
    n = config.GRID_N if n is None else n
    a = np.zeros((n, n), dtype=bool)
    a[:_bar_px(width_mm), :] = True
    return UnitCell(a)

def bars_axial_only(n=None, width_mm=None):
    """Bars along the artery with no hoop -> NO_WRAP_CIRC, i.e. no radial support."""
    n = config.GRID_N if n is None else n
    a = np.zeros((n, n), dtype=bool)
    a[:, :_bar_px(width_mm)] = True
    return UnitCell(a)

def thin_grid(n=None, w_px=2):
    """A grid of one-pixel-ish necks -> THIN_FEATURE."""
    n = config.GRID_N if n is None else n
    a = np.zeros((n, n), dtype=bool)
    a[:w_px, :] = True
    a[:, :w_px] = True
    return UnitCell(a)

def island(n=None, r=7):
    """A good diamond plus material floating free of it -> DISCONNECTED."""
    n = config.GRID_N if n is None else n
    return UnitCell(_disc(diamond(n).to_array(), n // 2, n // 2, r))

def nearly_solid(n=None):
    """Foil with a pinhole -> TOO_DENSE."""
    n = config.GRID_N if n is None else n
    a = np.ones((n, n), dtype=bool)
    a[n // 2 - 2:n // 2 + 2, n // 2 - 2:n // 2 + 2] = False
    return UnitCell(a)

# Registries, so callers iterate one list rather than re-deriving it.
VALID_CELLS = {
    'diamond w=0.20': lambda: diamond(),
    'diamond w=0.25': lambda: diamond(width_mm=0.25),
    'diamond w=0.30': lambda: diamond(width_mm=0.30),
    'grid w=0.20': lambda: grid(),
    'grid w=0.25': lambda: grid(width_mm=0.25),
}

# name -> (builder, the reason code it must trip)
BROKEN_CELLS = {
    'empty': (empty, 'empty'),
    'tiny blob': (tiny_blob, 'too_sparse'),
    'circ bands only': (bands_circ_only, 'no_wrap_axial'),
    'axial bars only': (bars_axial_only, 'no_wrap_circ'),
    'thin grid (2 px)': (thin_grid, 'thin_feature'),
    'island': (island, 'disconnected'),
    'nearly solid': (nearly_solid, 'too_dense'),
}
