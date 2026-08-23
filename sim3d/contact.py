"""
Contact surfaces for the crimp: a shrinking crimper and strut-to-strut self-contact.

Design drivers:

1. Self-contact is certain. Metal area is conserved while the surface shrinks, so coverage 
   rises from f_metal 0.248 deployed to ~0.74 at 2 mm; struts touch completely
   at D = 1.49 mm, only 26 % below the 2 mm target.

2. All side walls go into self-contact. The reduction that is defensible is geometric: the 
   inner/outer radial skins are excluded because nested cylinders can't touch each other.

The crimper is a TARGE170 cylinder with prescribed radial motion on its own nodes. It's
also not prescribed displacement on the stent's outer nodes: that would pin every outer node
to a cylinder and remove the radial freedom self-contact needs, so struts couldn't tilt or
shift when they collide and the contact forces would have nowhere to go.
"""

import numpy as np
from collections import defaultdict

import config

# Hex faces in the node ordering write_apdl_mesh() uses.
HEX_FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]

def exterior_faces(hexes):
    """Faces belonging to exactly one element. Returns (owner_elems, face_nodes)."""
    seen = defaultdict(list)
    for e, h in enumerate(hexes):
        for f in HEX_FACES:
            seen[tuple(sorted(int(h[i]) for i in f))].append((e, [int(h[i]) for i in f]))
    ext = [(e, n) for v in seen.values() if len(v) == 1 for e, n in v]
    owners = np.array([e for e, _ in ext], dtype=np.int64)
    faces = np.array([n for _, n in ext], dtype=np.int64)
    return owners, faces

def classify_faces(points, owners, faces, hexes):
    """
    Split exterior faces into 'inner', 'outer' and 'side'.

    Radial skins are recognized by radius, not by normal direction, because a face whose
    centroid sits at the inner or outer radius is a skin regardless of how it is oriented.
    """
    cen = points[faces].mean(axis=1)
    r = np.hypot(cen[:, 0], cen[:, 1])
    r_in = config.D_DEPLOYED_MM / 2.0
    r_out = r_in + config.STRUT_THICKNESS_MM
    tol = config.STRUT_THICKNESS_MM * 0.05
    inner = np.abs(r - r_in) < tol
    outer = np.abs(r - r_out) < tol
    return {'inner': np.where(inner)[0],
            'outer': np.where(outer)[0],
            'side': np.where(~(inner | outer))[0]}

def outward_faces(points, owners, faces, hexes, idx):
    """
    Face node lists reordered so the normal points away from the parent element.

    Contact elements inherit their normal from node ordering, and a contact surface facing
    into its own solid detects nothing.
    """
    out = []
    for i in idx:
        nodes = faces[i]
        p = points[nodes]
        n = np.cross(p[1] - p[0], p[3] - p[0])
        ecen = points[hexes[owners[i]]].mean(axis=0)
        if np.dot(n, p.mean(axis=0) - ecen) < 0:
            nodes = nodes[::-1]
        out.append(nodes)
    return np.array(out, dtype=np.int64)

def crimper_cylinder(points, n_theta=96, n_z=8, gap_mm=0.03):
    """
    A TARGE170 cylinder just outside the stent, as (nodes_mm, quads).

    Sits `gap_mm` clear of the outer surface so the first substep starts out of contact,
    then shrinks by prescribed radial displacement on these nodes.
    """
    r = config.D_DEPLOYED_MM / 2.0 + config.STRUT_THICKNESS_MM + gap_mm
    z0, z1 = float(points[:, 2].min()), float(points[:, 2].max())
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    z = np.linspace(z0, z1, n_z + 1)
    nodes = np.array([[r * np.cos(t), r * np.sin(t), zz] for zz in z for t in theta])

    quads = []
    for k in range(n_z):
        for j in range(n_theta):
            j2 = (j + 1) % n_theta          # Wraps, so the cylinder is closed
            a, b = k * n_theta, (k + 1) * n_theta
            quads.append([a + j, a + j2, b + j2, b + j])
    return nodes, np.array(quads, dtype=np.int64), r

def crimp_radius_mm(diameter_mm):
    """Outer radius the crimper has to reach to bring the stent to `diameter_mm` outside."""
    return diameter_mm / 2.0

def budget(points, hexes, faces_by_kind, n_crimper_elems):
    """Node + element total, for checking against the Ansys Student ceiling."""
    n_self = len(faces_by_kind['side'])
    n_outer = len(faces_by_kind['outer'])
    return {
        'nodes': int(points.shape[0]),
        'hexes': int(hexes.shape[0]),
        'self_contact': 2 * n_self,     # CONTA173 + TARGE170 on the same faces
        'crimper_contact': n_outer,     # CONTA173 on the stent's outer skin
        'crimper_target': n_crimper_elems,
        'total': int(points.shape[0] + hexes.shape[0] + 2 * n_self
                     + n_outer + n_crimper_elems),
    }

def base_anchor_nodes(points, angles_deg=(0.0, 120.0, 240.0)):
    """
    Three nodes on the z = 0 edge that pin rigid-body motion without fighting the crimp.

    NOTE: the ring is perforated, so the z = 0 edge carries material only at some angles
    and the nominal 0/120/240 is generally NOT reachable - the nearest available nodes are
    taken instead. That is fine: the requirement is that no rigid mode survives, which holds
    as long as the three angles stay well spread (see tests).

    Rigid rotation is one degree of freedom, so it never needed a fully pinned edge.
    Pinning UY across the whole base inverted elements at z ~ 0.08 within 7 % of LS1:
    crimping shrinks the circumference threefold, and a tangentially pinned edge can't
    accommodate that. Three nodes 120 deg apart remove rigid rotation (UY = const) and both
    in-plane translations (UY = -d*sin(theta), d*cos(theta); neither vanishes at all three
    angles) while leaving the ring free to shrink.

    Chosen here rather than by APDL angular selection, which is fragile: theta is stored on
    0-360, so a range straddling zero silently selects nothing and D is then given node 0.
    Returns 1-based APDL node numbers.
    """
    z0 = points[:, 2].min()
    tol = 1e-6 + 1e-3 * (points[:, 2].max() - z0)
    base = np.where(np.abs(points[:, 2] - z0) < tol)[0]
    if base.size == 0:
        raise ValueError("no nodes on the z = 0 edge")
    theta = np.degrees(np.arctan2(points[base, 1], points[base, 0])) % 360.0
    picked = []
    for target in angles_deg:
        d = np.abs((theta - target + 180.0) % 360.0 - 180.0)
        picked.append(int(base[int(np.argmin(d))]) + 1)
    if len(set(picked)) != len(picked):
        raise ValueError(f"anchor nodes are not distinct: {picked}")
    return picked

def write_contact_apdl(points, hexes, path, n_theta=96, n_z=8, gap_mm=0.03, fkn_crimp=1.0, fkn_self=0.1):
    """
    Write the crimper and the self-contact surfaces as an APDL include file.

    Element types, so the deck can refer to them:
      1 SOLID185 (from the mesh file)   2 TARGE170 crimper   3 CONTA173 stent outer skin
      4 TARGE170 self-contact           5 CONTA173 self-contact
    Real sets: 2 = crimper pair, 3 = self-contact pair. MAT 2 carries MU = 0.

    Creates node component CRIMPN, which the deck displaces radially.
    """
    owners, faces = exterior_faces(hexes)
    kinds = classify_faces(points, owners, faces, hexes)
    outer = outward_faces(points, owners, faces, hexes, kinds['outer'])
    side = outward_faces(points, owners, faces, hexes, kinds['side'])
    cyl_nodes, cyl_quads, r_c = crimper_cylinder(points, n_theta, n_z, gap_mm)

    n0 = int(points.shape[0])           # Existing nodes are 1..n0
    e0 = int(hexes.shape[0])            # Existing elements are 1..e0
    L = ['! Contact surfaces generated by sim3d/contact.py.',
         f'! {config.UNITS_BANNER}',
         f'! crimper: {len(cyl_quads)} TARGE170 at r={r_c:.4f} mm ({gap_mm} mm clear)',
         f'! stent outer skin: {len(outer)} CONTA173',
         f'! self-contact: {len(side)} CONTA173 + {len(side)} TARGE170 on side walls',
         '/PREP7', '/NOPR',
         'ET,2,TARGE170', 'ET,3,CONTA173', 'ET,4,TARGE170', 'ET,5,CONTA173',
         '! Frictionless. A real crimper grips; declared as an idealization.',
         'MP,MU,2,0',
         '! KEYOPT(9)=1 on self-contact ignores initial mesh gap/penetration, which is a',
         '! meshing artefact here rather than a physical interference.',
         'KEYOPT,3,2,0', 'KEYOPT,3,5,0', 'KEYOPT,3,9,0', 'KEYOPT,3,12,0',
         'KEYOPT,5,2,0', 'KEYOPT,5,5,0', 'KEYOPT,5,9,1', 'KEYOPT,5,12,0',
         f'R,2,,,{fkn_crimp}',
         f'R,3,,,{fkn_self}']

    # Crimper target nodes, then its elements. Normals must face the axis, i.e. the stent.
    L += [f'N,{n0 + i + 1},{x:.6f},{y:.6f},{z:.6f}'
          for i, (x, y, z) in enumerate(cyl_nodes)]
    L += [f'NSEL,S,NODE,,{n0 + 1},{n0 + len(cyl_nodes)}', 'CM,CRIMPN,NODE', 'NSEL,ALL']

    eid = e0
    L += ['TYPE,2 $ REAL,2 $ MAT,2']
    for q in cyl_quads:
        p = cyl_nodes[q]
        nrm = np.cross(p[1] - p[0], p[3] - p[0])
        cen = p.mean(axis=0)
        inward = -np.array([cen[0], cen[1], 0.0])
        order = q if np.dot(nrm, inward) > 0 else q[::-1]
        eid += 1
        L.append('EN,{},{}'.format(eid, ','.join(str(n0 + int(n) + 1) for n in order)))

    L += ['TYPE,3 $ REAL,2 $ MAT,2']
    for f in outer:
        eid += 1
        L.append('EN,{},{}'.format(eid, ','.join(str(int(n) + 1) for n in f)))

    L += ['TYPE,4 $ REAL,3 $ MAT,2']
    for f in side:
        eid += 1
        L.append('EN,{},{}'.format(eid, ','.join(str(int(n) + 1) for n in f)))

    L += ['TYPE,5 $ REAL,3 $ MAT,2']
    for f in side:
        eid += 1
        L.append('EN,{},{}'.format(eid, ','.join(str(int(n) + 1) for n in f)))

    # Base anchors are resolved here, not by APDL angular selection (see the docstring).
    anchors = base_anchor_nodes(points)
    L += ['/GOPR', '*GET,NE_ALL,ELEM,0,COUNT', '*GET,NN_ALL,NODE,0,COUNT',
          f'RCRIMP0 = {r_c:.6f}',
          f'NFIX1 = {anchors[0]}', f'NFIX2 = {anchors[1]}', f'NFIX3 = {anchors[2]}',
          'FINISH']
    path.write_text(chr(10).join(L) + chr(10), encoding='utf-8')
    return path, {'crimper_target': len(cyl_quads), 'crimper_contact': len(outer),
                  'self_pairs': len(side), 'r_crimper_mm': r_c,
                  'first_contact_elem': e0 + 1, 'last_elem': eid,
                  'anchors': anchors}

def build_and_write(layers=2, name='spikeA_contact_L2', directory=None):
    """Generate the contact include file for the crimp mesh."""
    from geom import reference
    from sim3d import mesh3d

    directory = (config.PROJECT_ROOT / 'sim3d' / 'decks') if directory is None else directory
    directory.mkdir(parents=True, exist_ok=True)
    cell = reference.build()
    points, hexes, _ = mesh3d.tube_hex_mesh(cell, n_circ=config.N_CIRC,
                                            n_axial=1, layers=layers)
    path, info = write_contact_apdl(points, hexes, directory / f'{name}.inp')
    owners, faces = exterior_faces(hexes)
    info['budget'] = budget(points, hexes,
                            classify_faces(points, owners, faces, hexes),
                            info['crimper_target'])
    return path, info

if __name__ == "__main__":
    path, info = build_and_write()
    print(f'Wrote {path.relative_to(config.PROJECT_ROOT)}  '
          f'({path.stat().st_size / 1e6:.2f} MB)')
    for key in ('crimper_target', 'crimper_contact', 'self_pairs', 'r_crimper_mm'):
        print(f'  {key:18} {info[key]}')
    b = info['budget']
    print()
    for key, value in b.items():
        print(f'  {key:18} {value}')
    limit = 128_000
    head = 100 * (1 - b['total'] / limit)
    print(f"  {'vs Student ~128k':18} "
          f"{'FITS, %.0f%% headroom' % head if b['total'] < limit else 'OVER'}")
