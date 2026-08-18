"""
Rendering helpers for unit cells.
"""

import matplotlib.pyplot as plt
import config

def _draw(ax, arr, extent_mm):
    circ_mm, axial_mm = extent_mm
    ax.imshow(
        arr,
        origin="lower",
        cmap="Greys",
        vmin=0,
        vmax=1,
        extent=(0.0, circ_mm, 0.0, axial_mm),
        interpolation="nearest",
    )
    ax.set_xlabel("circumferential (mm)")
    ax.set_ylabel("axial (mm)")
    ax.set_aspect("equal")

def plot_cell(cell, ax=None, title=None):
    """Render a single unit cell at physical scale."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    _draw(ax, cell.to_array(), cell.extent_mm)
    ax.set_title(title or f"unit cell  {cell.n}x{cell.n}  f_metal={cell.f_metal:.3f}")
    return ax

def plot_tiling(cell, n_circ=3, n_axial=3, ax=None, title=None, boundaries=True):
    """Render an n_circ x n_axial tiling, with cell boundaries marked."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    circ_mm, axial_mm = cell.extent_mm
    _draw(ax, cell.tile(n_circ, n_axial), (circ_mm * n_circ, axial_mm * n_axial))

    if boundaries:
        for i in range(1, n_circ):
            ax.axvline(i * circ_mm, color="tab:red", lw=0.8, alpha=0.7)
        for j in range(1, n_axial):
            ax.axhline(j * axial_mm, color="tab:red", lw=0.8, alpha=0.7)

    ax.set_title(title or f"{n_circ} x {n_axial} tiling")
    return ax

def plot_change(before, after, ax=None, title=None):
    """Show a repair: material kept in grey, removed in red, added in blue."""
    import numpy as np
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    b = before.to_array() if hasattr(before, 'to_array') else np.asarray(before, dtype=bool)
    a = after.to_array() if hasattr(after, 'to_array') else np.asarray(after, dtype=bool)

    rgb = np.ones(b.shape + (3,))
    rgb[b & a] = (0.25, 0.25, 0.25)      # Kept
    rgb[b & ~a] = (0.85, 0.20, 0.20)     # Removed
    rgb[~b & a] = (0.20, 0.40, 0.85)     # Added

    circ_mm, axial_mm = config.cell_extent_mm()
    ax.imshow(rgb, origin='lower', extent=(0.0, circ_mm, 0.0, axial_mm),
              interpolation='nearest')
    ax.set_aspect('equal')
    changed = float((a != b).mean())
    ax.set_title(title or f'changed {changed:.1%}')
    return ax
