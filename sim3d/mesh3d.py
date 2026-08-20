"""
Tube mesh for Ansys: pixel quads, boundary projected onto the smooth level set, extruded.

Marching squares on the padded periodic field returns curves clipped by the padding, not 
closed loops. Feeding gmsh a plane surface with holes means clipping and re-closing every 
open curve along the domain edge. Building from the pixel grid instead makes the topology 
and the circumferential periodicity correct by construction.

Voxel meshing skips contouring: element faces land on pixel boundaries, giving re-entrant 90
degree corners that act as artificial stress raisers where eps_a_max is read. The fix is to 
move the boundary nodes onto the smooth 0.5 level set, which removes those corners while 
leaving connectivity untouched.

Result: quads -> radial extrusion -> hexahedra (SOLID185/186 in Ansys). Hexes rather than
tets because these struts are thin and bending-dominated, and linear tets are over-stiff in bending.
"""

import numpy as np
from scipy import ndimage

import config
from sim3d import contour

def quad_mesh_2d(cell, n_circ=None, n_axial=1):
    """
    Pixel quads of the tiled field. Returns (nodes_px, quads, boundary_mask).

    nodes_px are (axial, circumferential) in pixels of the tiled field. Circumferential
    periodicity is applied by folding the last column onto the first.
    """
    n_circ = config.N_CIRC if n_circ is None else n_circ
    field = cell.tile(n_circ, n_axial)
    n_z, n_t = field.shape

    # Fold the seam only for a full ring. A sector spans a fraction of the circumference
    # and has two free cyclic-symmetry faces; folding it wraps a wedge into a closed ring,
    # which inverts elements and makes the total volume negative.
    full_ring = (n_circ == config.N_CIRC)
    n_cols = n_t if full_ring else n_t + 1

    def vid(j, i):
        """Node id at (axial j, circumferential i); wraps only on a full ring."""
        return j * n_cols + (i % n_t if full_ring else i)

    zi, ti = np.nonzero(field)
    quads = np.column_stack([
        vid(zi, ti), vid(zi, ti + 1), vid(zi + 1, ti + 1), vid(zi + 1, ti)
    ])

    jj, ii = np.meshgrid(np.arange(n_z + 1), np.arange(n_cols), indexing='ij')
    nodes = np.column_stack([jj.ravel().astype(float), ii.ravel().astype(float)])

    used = np.unique(quads)
    remap = np.full(nodes.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    quads = remap[quads]
    nodes = nodes[used]

    # A boundary node touches an edge used by one quad.
    edges = np.vstack([quads[:, [0, 1]], quads[:, [1, 2]],
                       quads[:, [2, 3]], quads[:, [3, 0]]])
    edges = np.sort(edges, axis=1)
    _, inv, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    boundary_edges = edges[counts[inv] == 1]
    boundary = np.zeros(nodes.shape[0], dtype=bool)
    boundary[np.unique(boundary_edges)] = True
    return nodes, quads, boundary, full_ring

def project_to_level_set(nodes_px, boundary, cell, sigma_px=0.8, n_circ=None, n_axial=1, iterations=3):
    """
    Move boundary nodes onto the smooth 0.5 isocontour, by Newton steps along the gradient.

    Interior nodes are left alone. Axial-end nodes are pinned in the axial direction: the
    tube's ends are a cut through the geometry, not a material boundary, and letting the
    level set pull them would round off the ends into something the stent does not have.
    """
    n_circ = config.N_CIRC if n_circ is None else n_circ
    field, pad = contour.smoothed_field(cell, sigma_px, n_circ, n_axial)
    n_z = cell.n * n_axial

    at_end = (np.abs(nodes_px[:, 0]) < 1e-9) | (np.abs(nodes_px[:, 0] - n_z) < 1e-9)
    movable = boundary & ~at_end

    gz, gt = np.gradient(field)
    out = nodes_px.copy()
    for _ in range(iterations):
        # Node (j, i) in tiled coords sits at (j + pad, i + pad) in the padded field.
        coords = np.vstack([out[movable, 0] + pad, out[movable, 1] + pad])
        phi = ndimage.map_coordinates(field, coords, order=1, mode='grid-wrap')
        dz = ndimage.map_coordinates(gz, coords, order=1, mode='grid-wrap')
        dt = ndimage.map_coordinates(gt, coords, order=1, mode='grid-wrap')
        denom = dz ** 2 + dt ** 2
        denom[denom < 1e-12] = 1e-12
        step = (phi - contour.LEVEL) / denom
        out[movable, 0] -= step * dz
        out[movable, 1] -= step * dt

    # Clamp inside the axial extent. A boundary node one row in from the end can otherwise
    # be pulled past the cut plane by the level set, poking the mesh out beyond z = 0 or L.
    out[:, 0] = np.clip(out[:, 0], 0.0, float(n_z))

    # A projection step must never move a node further than the smoothing length.
    moved = np.linalg.norm(out - nodes_px, axis=1)
    limit = 0.30
    clamp = moved > limit
    if clamp.any():
        scale = limit / moved[clamp]
        out[clamp] = nodes_px[clamp] + (out[clamp] - nodes_px[clamp]) * scale[:, None]
    return out

def relax_interior(nodes_px, quads, boundary, n_t, iterations=30, factor=0.3, periodic=True, free=None):
    """
    Laplacian smoothing of interior nodes, boundary held fixed, periodic in circumference.

    Projecting boundary nodes alone isn't enough: a node moved ~1 px in a 1 px grid folds
    the element behind it. Relaxing spreads that displacement over several element layers.

    The circumferential coordinate has to be averaged periodically. A seam node is stored at
    coordinate 0 while its neighbor across the wrap is stored at n_t - 1; averaging those
    directly flings the node to the far side of the tube. Neighbor offsets are wrapped into 
    [-n_t/2, n_t/2) before averaging.
    """
    n = nodes_px.shape[0]
    edges = np.vstack([quads[:, [0, 1]], quads[:, [1, 2]],
                       quads[:, [2, 3]], quads[:, [3, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    order = np.argsort(edges[:, 0], kind='stable')
    a, b = edges[order, 0], edges[order, 1]

    counts = np.bincount(a, minlength=n).astype(float)
    counts[counts == 0] = 1.0

    out = nodes_px.copy()
    interior = ~boundary if free is None else free
    for _ in range(iterations):
        d_axial = out[b, 0] - out[a, 0]
        d_circ = out[b, 1] - out[a, 1]
        if periodic:
            d_circ = (d_circ + n_t / 2.0) % n_t - n_t / 2.0   # Nearest periodic image

        mean_axial = np.bincount(a, weights=d_axial, minlength=n) / counts
        mean_circ = np.bincount(a, weights=d_circ, minlength=n) / counts

        out[interior, 0] += factor * mean_axial[interior]
        out[interior, 1] += factor * mean_circ[interior]
        if periodic:
            out[interior, 1] %= n_t
    return out

def tube_hex_mesh(cell, n_circ=None, n_axial=1, layers=4, sigma_px=0.8,
                  project=True, relax=40, passes=8):
    """
    Hexahedral tube mesh. Returns (points_mm, hexes, info).

    points_mm are 3D in mm; hexes index them with VTK/Ansys node ordering (bottom face
    counter-clockwise, then top face).
    """
    n_circ = config.N_CIRC if n_circ is None else n_circ
    nodes_px, quads, boundary, full_ring = quad_mesh_2d(cell, n_circ, n_axial)
    if project:
        # Project the boundary onto the level set, then relax the interior only.
        nodes_px = project_to_level_set(nodes_px, boundary, cell, sigma_px, n_circ, n_axial)
        nodes_px = relax_interior(nodes_px, quads, boundary, n_t=cell.n * n_circ, iterations=relax, periodic=full_ring)

    mm = config.mm_per_px()[0]
    radius = config.D_DEPLOYED_MM / 2.0
    thickness = config.STRUT_THICKNESS_MM

    arc = nodes_px[:, 1] * mm                       # Arc length around the tube
    z = nodes_px[:, 0] * config.mm_per_px()[1]
    theta = arc / radius

    n2d = nodes_px.shape[0]
    points = []
    for k in range(layers + 1):
        r = radius + thickness * k / layers
        points.append(np.column_stack([r * np.cos(theta), r * np.sin(theta), z]))
    points = np.vstack(points)

    hexes = []
    for k in range(layers):
        lo, hi = k * n2d, (k + 1) * n2d
        hexes.append(np.column_stack([quads + lo, quads + hi]))
    hexes = np.vstack(hexes)

    info = {
        'n_points': points.shape[0],
        'n_hexes': hexes.shape[0],
        'n_quads_2d': quads.shape[0],
        'layers': layers,
        'boundary_nodes': int(boundary.sum()),
        'projected': project,
        'sigma_px': sigma_px,
        'relax_iterations': relax if project else 0,
        'passes': passes if project else 0,
    }
    return points, hexes, info

def max_face_angle(points, hexes):
    """Largest interior face angle per element, in degrees.

    This is the criterion Ansys's own shape checker warns on (>155 deg for bricks), so it is
    what the smoothing should be tuned against: a scaled Jacobian can look acceptable while
    one face is nearly folded flat.
    """
    faces = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    worst = np.zeros(hexes.shape[0])
    for f in faces:
        quad = points[hexes[:, f]]
        for k in range(4):
            a = quad[:, (k - 1) % 4] - quad[:, k]
            b = quad[:, (k + 1) % 4] - quad[:, k]
            denom = np.clip(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30, None)
            cos = np.einsum('ij,ij->i', a, b) / denom
            worst = np.maximum(worst, np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return worst

def quality(points, hexes):
    """
    Element quality, by pyvista's scaled Jacobian.

    Scaled Jacobian is 1 for a perfect cube and <= 0 for an inverted or degenerate element.
    Ansys refuses to solve on non-positive elements, so `min` is the number that decides
    whether the mesh is usable at all; the low percentile says whether it will solve well.
    """
    import pyvista as pv

    cells = np.column_stack([np.full(hexes.shape[0], 8), hexes]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(hexes.shape[0], 12, dtype=np.uint8), points)
    q = grid.cell_quality('scaled_jacobian')['scaled_jacobian']
    vol = grid.compute_cell_sizes()['Volume']
    return {
        'scaled_jacobian_min': float(q.min()),
        'scaled_jacobian_p1': float(np.percentile(q, 1)),
        'scaled_jacobian_mean': float(q.mean()),
        'n_nonpositive': int((q <= 0).sum()),
        'volume_min_mm3': float(vol.min()),
        'volume_total_mm3': float(vol.sum()),
        'max_face_angle_deg': float(max_face_angle(points, hexes).max()),
        'n_over_155deg': int((max_face_angle(points, hexes) > 155.0).sum()),
    }

def orphan_nodes(points, hexes):
    """Nodes not referenced by any element."""
    return int(points.shape[0] - np.unique(hexes).size)

def write_apdl_mesh(points, hexes, path, etype=185, mat=1):
    """Write the mesh as an APDL deck of N and EN commands, read with /INPUT."""
    lines = [
        '! Mesh generated by sim3d/mesh3d.py.',
        f'! {config.UNITS_BANNER}',
        f'! {points.shape[0]} nodes, {hexes.shape[0]} hexahedra',
        '/PREP7',
        f'ET,1,SOLID{etype}',
        # Enhanced strain formulation. The default B-bar formulation shear-locks in
        # bending. Locking would inflate stiffness and distort the strain field, hitting
        # eps_a_max.
        'KEYOPT,1,2,2',
        f'TYPE,1 $ MAT,{mat}',
    ]
    lines += [f'N,{i + 1},{x:.6f},{y:.6f},{z:.6f}'
              for i, (x, y, z) in enumerate(points)]
    lines += ['EN,{},{}'.format(k + 1, ','.join(str(n + 1) for n in row))
              for k, row in enumerate(hexes)]
    # Self-documenting import: echo off while the ~20k N/EN commands stream in.
    stem = path.stem
    lines.insert(3, '/NOPR')
    lines += [
        '/GOPR',
        '*GET,ne,ELEM,0,COUNT',
        '*GET,nn,NODE,0,COUNT',
        f'/OUTPUT,{stem}_check,txt',
        '*STATUS,ne',
        '*STATUS,nn',
        'SHPP,SUMMARY',
        '/OUTPUT',
    ]
    lines.append('FINISH')
    path.write_text(chr(10).join(lines) + chr(10), encoding='utf-8')
    return path

def build_and_write(cell, path, n_circ=None, n_axial=1, layers=4, sigma_px=0.8, project=True, relax=40, passes=8):
    """Mesh a cell and write the APDL deck. Returns (info, quality)."""
    points, hexes, info = tube_hex_mesh(cell, n_circ, n_axial, layers, sigma_px,
                                        project, relax, passes)
    info['orphan_nodes'] = orphan_nodes(points, hexes)
    q = quality(points, hexes)
    write_apdl_mesh(points, hexes, path)
    info['path'] = str(path)
    info['bytes'] = path.stat().st_size
    return info, q

# Meshes this module emits: filename -> (n_circ, n_axial, note).
MESHES = {
    'spikeA_sector.inp': (1, 2, '1/12 sector, cyclic symmetry'),
    'spikeA_fullring.inp': (None, 1, 'full ring'),
}

DECK_DIR = config.PROJECT_ROOT / 'sim3d' / 'decks'

def write_meshes(directory=None, layers=4):
    """Write every mesh in MESHES. Returns a list of (path, info, quality)."""
    from geom import reference

    directory = DECK_DIR if directory is None else directory
    directory.mkdir(parents=True, exist_ok=True)
    cell = reference.build()
    written = []
    for name, (n_circ, n_axial, _) in MESHES.items():
        info, q = build_and_write(cell, directory / name, n_circ=n_circ,
                                  n_axial=n_axial, layers=layers)
        written.append((directory / name, info, q))
    return written

if __name__ == "__main__":
    from geom import reference

    cell = reference.build()
    print(f'Generating from the reference cell  (f_metal={cell.f_metal:.4f}, '
          f'GRID_N={config.GRID_N}, N_CIRC={config.N_CIRC})')
    for path, info, q in write_meshes():
        note = MESHES[path.name][2]
        print(f'\n  {path.relative_to(config.PROJECT_ROOT)}  ({info["bytes"] / 1e6:.1f} MB)')
        print(f'    {note}')
        print(f'    {info["n_hexes"]:,} hexes / {info["n_points"]:,} nodes, '
              f'orphans {info["orphan_nodes"]}')
        print(f'    scaled Jacobian  min {q["scaled_jacobian_min"]:.3f}  '
              f'mean {q["scaled_jacobian_mean"]:.3f}  non-positive {q["n_nonpositive"]}')
        print(f'    volume {q["volume_total_mm3"]:.5f} mm^3')
