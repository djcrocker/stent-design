"""
Parse loadsteps output and compute the derived metrics.

Ansys reports raw quantities: imposed displacement, reaction force, per-element strains.
Everything derived is computed here instead, where it can be unit-tested.
"""

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
    return result

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
    for key in ('NOUTER', 'ZMAX', 'DAXIAL', 'FRADIAL', 'EPS3', 'EPS4', 'EPSAMP'):
        if key in result:
            print(f'  {key:16} {result[key]:.6g}')
    print()
    if 'K_radial_3D' in result:
        print(f'  {"outer area":16} {result["outer_area_mm2"]:.5f} mm^2')
        print(f'  {"K_radial_3D":16} {result["K_radial_3D"]:.4f} N/mm^3')
    if 'eps_a_max_3D' in result:
        print(f'  {"eps_a_max_3D":16} {result["eps_a_max_3D"]:.6f}'
              f'   over the {config.EPS_A_LIM} limit: {result["over_limit"]}')

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
