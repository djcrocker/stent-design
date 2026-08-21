"""
Parse loadsteps output and compute the derived metrics.

Ansys reports raw quantities: imposed displacement, reaction force, per-element strains.
Everything derived is computed here instead, where it can be unit-tested.
"""

import json
import pathlib
import re

import numpy as np

import config

def read_status(path):
    """Parse APDL *STATUS output into {name: value}."""
    text = open(path, encoding='utf-8', errors='ignore').read()
    out = {}
    for name, value in re.findall(r'^\s*([A-Z_][A-Z0-9_]*)\s+(-?[\d.]+E?[+-]?\d*)\s+SCALAR', text, re.MULTILINE):
        try:
            out[name] = float(value)
        except ValueError:
            pass
    return out

def sector_outer_area_mm2(n_axial=2, n_circ=1):
    """
    Outer cylindrical area of the modeled sector, voids included.

    Voids included: the 2D tier's K_radial is a property of the perforated shell over 
    the whole surface, so the 3D pressure must be referred to the same area or the two
    aren't the same quantity.
    """
    r_outer = config.D_DEPLOYED_MM / 2.0 + config.STRUT_THICKNESS_MM
    arc = 2.0 * np.pi * r_outer * (n_circ / config.N_CIRC)
    return arc * config.cell_extent_mm()[1] * n_axial

def k_radial_3d(radial_force_N, radial_disp_mm, n_axial=2, n_circ=1):
    """
    K_radial in N/mm^3.

    p = |F| / A over the outer surface, and K = p / |dR|, from p = E_circ*t*dR/R^2.
    """
    pressure = abs(radial_force_N) / sector_outer_area_mm2(n_axial, n_circ)
    return pressure / abs(radial_disp_mm)

def summarise(status_path, radial_mm=0.01, n_axial=2, n_circ=1):
    """Deck's *STATUS dump."""
    v = read_status(status_path)
    result = dict(v)
    if 'FRADIAL' in v:
        result['K_radial_3D'] = k_radial_3d(v['FRADIAL'], radial_mm, n_axial, n_circ)
        result['outer_area_mm2'] = sector_outer_area_mm2(n_axial, n_circ)
    if 'EPSAMP' in v:
        result['eps_a_max_3D'] = v['EPSAMP']
        result['over_limit'] = v['EPSAMP'] > config.EPS_A_LIM
    if 'EPS3ETAB' in v and v.get('EPS3'):
        result['eps_etable_over_nodal'] = v['EPS3ETAB'] / v['EPS3']
    return result


# Hex -> tets, for exact element volumes.
_TETS = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (3, 4, 6, 7), (1, 4, 5, 6)]

def read_node_amplitudes(path, halve=True):
    """
    Per-node strain amplitude from the deck's `*VWRITE` dump.

    The deck writes the strain range between the flexed and released states; the amplitude
    is half of it. Values are indexed by APDL node number, so index 0 is node 1.
    """
    values = []
    for line in pathlib.Path(path).read_text(errors='ignore').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue        # Header or banner text
    out = np.array(values, dtype=float)
    return 0.5 * out if halve else out

def element_volumes(points, hexes):
    """Exact hex volumes by tetrahedral decomposition."""
    P = points[hexes]
    vol = np.zeros(len(hexes))
    for a, b, c, d in _TETS:
        vol += np.abs(np.einsum('ij,ij->i', P[:, b] - P[:, a], np.cross(P[:, c] - P[:, a], P[:, d] - P[:, a]))) / 6.0
    return vol

def lumped_node_volumes(points, hexes):
    """
    Volume attributed to each node: each element donates vol/8 to each of its nodes.

    This is the discrete analogue of the 2D screen's quadrature weights, which is what
    makes A_over_lim_3D the same quantity as A_over_lim.
    """
    vol = element_volumes(points, hexes)
    lumped = np.zeros(points.shape[0])
    np.add.at(lumped, hexes.ravel(), np.repeat(vol / 8.0, hexes.shape[1]))
    return lumped

def a_over_lim_3d(amplitudes, points, hexes, limit=None):
    """
    Volume fraction of the structure whose strain amplitude exceeds the nitinol limit.

    Same definition as sim2d.fatigue.A_over_lim, with volume in place of area.
    """
    limit = config.EPS_A_LIM if limit is None else limit
    lumped = lumped_node_volumes(points, hexes)
    amp = amplitudes[:points.shape[0]]
    if amp.shape[0] != points.shape[0]:
        raise ValueError(f'{amp.shape[0]} amplitudes for {points.shape[0]} nodes')
    over = amp > limit
    return float(lumped[over].sum() / lumped.sum())

def extract_scalars(status_path=None, amp_path=None, n_axial=2, n_circ=1, layers=4, radial_mm=0.01, out_path=None):
    """The three 3D scalars for the reference cell."""
    from geom import reference
    from sim3d import mesh3d

    results = config.PROJECT_ROOT / 'sim3d' / 'results'
    status_path = results / 's5_3_loadsteps.txt' if status_path is None else status_path
    amp_path = results / 's5_3_loadsteps_amp.txt' if amp_path is None else amp_path
    out_path = results / 's5_4_scalars.json' if out_path is None else out_path

    summary = summarise(status_path, radial_mm, n_axial, n_circ)
    cell = reference.build()
    points, hexes, _ = mesh3d.tube_hex_mesh(cell, n_circ=n_circ, n_axial=n_axial, layers=layers)
    amp = read_node_amplitudes(amp_path)
    scalars = {
        'K_radial_3D': summary['K_radial_3D'],
        'eps_a_max_3D': summary['eps_a_max_3D'],
        'A_over_lim_3D': a_over_lim_3d(amp, points, hexes),
        'eps_a_lim': config.EPS_A_LIM,
        'mesh': {'nodes': int(points.shape[0]), 'hexes': int(hexes.shape[0]),
                 'n_axial': n_axial, 'n_circ': n_circ, 'layers': layers},
        'source': {'status': str(pathlib.Path(status_path).name),
                   'amplitudes': str(pathlib.Path(amp_path).name)},
    }
    pathlib.Path(out_path).write_text(json.dumps(scalars, indent=2), encoding='utf-8')
    return scalars

def read_prvar_history(path, n_columns=5, offset=1, width=14):
    """Parse a POST26 PRVAR table into an (n, n_columns) array."""
    rows = []
    for raw in open(path, encoding='utf-8', errors='ignore'):
        line = raw.rstrip(chr(10)).rstrip(chr(13))
        fields = [line[offset + i * width: offset + (i + 1) * width]
                  for i in range(n_columns)]
        if not fields[0].strip():
            continue
        try:
            rows.append([float(f) if f.strip() else 0.0 for f in fields])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f'no data rows parsed from {path}')
    return np.array(rows)

def load_history(path):
    """(time, radial_disp, axial_disp, radial_force, axial_force), sorted by time."""
    data = read_prvar_history(path)
    data = data[np.argsort(data[:, 0])]
    t, ux, uz, fx, fz = data.T

    if abs(t[-1] - 4.0) > 1e-6:
        raise ValueError(f'time should end at 4.0, got {t[-1]}')
    end_ls1 = int(np.argmin(np.abs(t - 1.0)))
    if abs(ux[end_ls1] + 0.01) > 1e-6:
        raise ValueError(
            f'radial displacement at t=1 is {ux[end_ls1]:.6g}, expected -0.01 - '
            f'the fixed-width parse is misaligned'
        )
    return t, ux, uz, fx, fz

if __name__ == "__main__":
    import sys

    path = (sys.argv[1] if len(sys.argv) > 1
            else str(config.PROJECT_ROOT / 'sim3d' / 'results' / 's5_3_loadsteps.txt'))
    result = summarise(path)
    print(f'Reading {path}\n')
    for key in ('NOUTER', 'ZMAX', 'DAXIAL', 'FRADIAL', 'EPS3', 'EPS4',
                'EPS3ETAB', 'EPSRANGE', 'EPSAMP'):
        if key in result:
            print(f'  {key:16} {result[key]:.6g}')
    print()
    if 'K_radial_3D' in result:
        print(f'  {"outer area":16} {result["outer_area_mm2"]:.5f} mm^2')
        print(f'  {"K_radial_3D":16} {result["K_radial_3D"]:.4f} N/mm^3')
    if 'eps_a_max_3D' in result:
        print(f'  {"eps_a_max_3D":16} {result["eps_a_max_3D"]:.6f}'
              f'   over the {config.EPS_A_LIM} limit: {result["over_limit"]}')
    if 'eps_etable_over_nodal' in result:
        r = result['eps_etable_over_nodal']
        print(f'  {"ETABLE/nodal":16} {r:.4f}   '
              f'(cantilever measured 0.7665 at 4 layers)')
        print(f'  {"":16} centroidal sampling was low by {100*(1-r):.1f} % on this mesh')

    from geom import reference
    from sim2d.fatigue import fatigue
    from sim2d.homogenize import homogenize

    cell = reference.build()
    h = homogenize(cell)
    f = fatigue(cell, h)
    print(f'\n  2D screen, same cell:')
    print(f'  {"K_radial_2D":16} {h.K_radial:.4f} N/mm^3'
          f'   ratio 3D/2D {result.get("K_radial_3D", float("nan")) / h.K_radial:.3f}')
    print(f'  {"eps_a_max_2D":16} {f.eps_a_max:.6f}'
          f'   ratio 3D/2D {result.get("eps_a_max_3D", float("nan")) / f.eps_a_max:.3f}')
    print('\n  Note that this is one cell, and we will need 20-30 for ranking.')
