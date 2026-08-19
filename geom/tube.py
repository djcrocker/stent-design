"""
Wrap a unit cell onto a tube.

Tiles the cell N_CIRC times around the circumference (which closes the ring exactly, since
N_CIRC is what the cell's width was derived from) and n_axial times along the artery, wraps
the result onto a cylinder of radius D/2, and extrudes it radially by the strut thickness.

Geometry conventions:

    Inner surface at D/2:     The deployed diameter is the lumen, so metal sits outside it
    `array[axial, circ]`:     As everywhere else (see geom/cell.py)
    Theta closes exactly:     The last column's neighbor is the first, by index arithmetic
                              rather than by duplicating vertices, so the tube is watertight.

Mesh type: one hexahedron per material pixel.

    This is not an FEA mesh: voxel extrusion leaves stair-stepped strut edges, which create
    artificial stress concentrations at crowns and necks. Since eps_a_max is a component of 
    the objective vector y, meshing this way for FEA would bias the quantity the project optimizes.
"""

import numpy as np

import config

VTK_HEXAHEDRON = 12

def tube_mesh(cell, n_axial=6, n_circ=None, radius_mm=None, thickness_mm=None):
    """
    Wrap a cell onto a tube. Returns a pyvista.UnstructuredGrid.

    n_circ defaults to config.N_CIRC, which is the count the cell's width was derived from,
    so the ring closes with no gap and no overlap.
    """
    import pyvista as pv

    n_circ = config.N_CIRC if n_circ is None else n_circ
    radius_mm = config.D_DEPLOYED_MM / 2.0 if radius_mm is None else radius_mm
    thickness_mm = config.STRUT_THICKNESS_MM if thickness_mm is None else thickness_mm

    field = cell.tile(n_circ, n_axial)          # (axial, circumferential)
    n_z, n_theta = field.shape

    # Vertex lattice. theta has no duplicate seam column.
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    dz = config.cell_extent_mm()[1] / cell.n
    z = np.arange(n_z + 1) * dz
    radii = np.array([radius_mm, radius_mm + thickness_mm])

    tt, zz, rr = np.meshgrid(theta, z, radii, indexing='ij')
    points = np.column_stack([
        (rr * np.cos(tt)).ravel(),
        (rr * np.sin(tt)).ravel(),
        zz.ravel(),
    ])

    def vid(i, j, k):
        """Vertex id at (theta index, z index, radius index); theta wraps."""
        return ((i % n_theta) * (n_z + 1) + j) * 2 + k

    zi, ti = np.nonzero(field)      # Material pixels
    i0, i1 = ti, ti + 1
    j0, j1 = zi, zi + 1

    cells = np.column_stack([
        np.full(ti.size, 8),
        vid(i0, j0, 0), vid(i1, j0, 0), vid(i1, j1, 0), vid(i0, j1, 0),
        vid(i0, j0, 1), vid(i1, j0, 1), vid(i1, j1, 1), vid(i0, j1, 1),
    ]).ravel()

    grid = pv.UnstructuredGrid(
        cells, np.full(ti.size, VTK_HEXAHEDRON, dtype=np.uint8), points
    )
    return grid.extract_cells(np.arange(grid.n_cells))

def expected_volume_mm3(cell, n_axial=6, n_circ=None, radius_mm=None, thickness_mm=None):
    """
    Analytic metal volume, for checking the mesh against arithmetic.

    Uses the mid-surface radius, since extruding a curved surface outwards adds volume that
    the flat unwrapped area under-counts.
    """
    n_circ = config.N_CIRC if n_circ is None else n_circ
    radius_mm = config.D_DEPLOYED_MM / 2.0 if radius_mm is None else radius_mm
    thickness_mm = config.STRUT_THICKNESS_MM if thickness_mm is None else thickness_mm

    mid_r = radius_mm + thickness_mm / 2.0
    axial_len = config.cell_extent_mm()[1] * n_axial
    return cell.f_metal * (2.0 * np.pi * mid_r) * axial_len * thickness_mm
