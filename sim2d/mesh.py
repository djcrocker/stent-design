"""
Turn a binary unit cell into a periodic quad mesh.

One quad element per material pixel, with the nodes on the right edge identified with those
on the left, and top with bottom. Without that identification a unit cell's edges behave as 
free surfaces that don't exist in the tiled stent, and the computed stiffness is wrong.

Stair-stepping is accepted here: this is the screen, and one element per pixel keeps a cell 
at ~1000 elements so thousands of cells stay affordable. The cost lands on strain concentrations 
at pixel corners, which is why sim2d reports a percentile of the strain field alongside the raw maximum.
"""

import numpy as np
from skfem import MeshQuad1, MeshQuad1DG

import config

def cell_to_mesh(cell):
    """Periodic MeshQuad1DG with one element per material pixel. Coordinates in mm."""
    arr = cell.to_array()
    n = arr.shape[0]
    circ_mm, axial_mm = config.cell_extent_mm()
    dx, dy = circ_mm / n, axial_mm / n

    # Full (n+1) x (n+1) vertex lattice; unused vertices are dropped afterwards.
    ii, jj = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
    points = np.vstack([ii.ravel() * dx, jj.ravel() * dy])   # (2, (n+1)^2), x=circ, y=axial

    def vid(i, j):
        return i * (n + 1) + j

    # arr is [axial, circumferential]; element (row=axial j, col=circ i).
    aj, ai = np.nonzero(arr)
    quads = np.vstack([
        vid(ai, aj), vid(ai + 1, aj), vid(ai + 1, aj + 1), vid(ai, aj + 1)
    ])

    used = np.unique(quads)
    remap = np.full((n + 1) ** 2, -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    mesh = MeshQuad1(points[:, used], remap[quads])

    # Identify the wrap. Fold each node's coordinates into [0, extent) and group nodes
    # that land on the same canonical position; keep one per group, eliminate the rest.
    # Doing it by folding rather than edge-by-edge matters at the corners: the node at
    # (circ, axial) must collapse twice (right onto left & top onto bottom) and a
    # single-pass edge mapping leaves it pointing at a node that is itself eliminated,
    # producing a singular system. That only shows up when material reaches the corners.
    p = mesh.p
    tol = min(dx, dy) * 1e-6
    xc = np.where(np.abs(p[0] - circ_mm) < tol, 0.0, p[0])
    yc = np.where(np.abs(p[1] - axial_mm) < tol, 0.0, p[1])
    key = np.round(np.vstack([xc, yc]) / tol).astype(np.int64)

    _, first, inverse = np.unique(key, axis=1, return_index=True, return_inverse=True)
    representative = first[inverse]
    eliminate = np.nonzero(representative != np.arange(p.shape[1]))[0]

    if eliminate.size == 0:
        return mesh
    return MeshQuad1DG.periodic(mesh, eliminate, representative[eliminate])
