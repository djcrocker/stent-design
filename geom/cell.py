"""
Unit-cell field object.

Axis convention: `array[axial, circumferential]`

axis 0 runs along the artery, axis 1 runs around the circumference. 
Displayed with `imshow` this reads as the unwrapped stent surface (x around the tube, y along it) 
and `tile(n_circ, ...)` therefore repeats along axis 1.

A cell always represents the same physical patch (`config.cell_extent_mm()`) regardless of how many pixels sample it.
"""

import numpy as np
import config

class UnitCell:
    """
    A binary material/void field on a square grid.
    True = material (metal), False = void.
    """

    def __init__(self, array):
        arr = np.asarray(array)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape {arr.shape}")
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(f"Expected a square grid, got shape {arr.shape}")
        self._array = arr.astype(bool)

    # CONSTRUCTION #
    @classmethod
    def from_array(cls, array):
        """Build a cell from any 2D array-like; nonzero is treated as material."""
        return cls(array)

    @classmethod
    def zeros(cls, n=None):
        """All-void cell, GRID_N x GRID_N unless n is given."""
        n = config.GRID_N if n is None else n
        return cls(np.zeros((n, n), dtype=bool))

    # ACCESS #
    def to_array(self):
        """A copy of the underlying boolean grid (callers cannot mutate the cell)."""
        return self._array.copy()

    @property
    def n(self):
        """Grid resolution (one number because cells are square)."""
        return self._array.shape[0]

    @property
    def shape(self):
        return self._array.shape

    # PHYSICAL SCALE #
    @property
    def extent_mm(self):
        """Physical size of the cell as (circumferential, axial), in mm."""
        return config.cell_extent_mm()

    @property
    def mm_per_px(self):
        """Physical size of one pixel as (circumferential, axial), in mm."""
        circ_mm, axial_mm = self.extent_mm
        return (circ_mm / self.n, axial_mm / self.n)

    @property
    def f_metal(self):
        """Metal area fraction."""
        return float(self._array.mean())

    # OPERATIONS #
    def tile(self, n_circ=3, n_axial=3):
        """
        Repeat the cell n_axial times along the artery and n_circ times around it.
        Returns a plain array, as a tiling is a patch of stent surface.
        """
        if n_circ < 1 or n_axial < 1:
            raise ValueError("Tile counts must be >= 1")
        return np.tile(self._array, (n_axial, n_circ))

    def __repr__(self):
        circ_mm, axial_mm = self.extent_mm
        return (
            f"UnitCell(n={self.n}, f_metal={self.f_metal:.3f}, "
            f"extent={circ_mm:.3f}x{axial_mm:.3f} mm)"
        )
