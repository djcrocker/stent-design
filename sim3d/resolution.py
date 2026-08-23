"""
Settle GRID_N with evidence.

6.1's 3D numbers don't depend on the 2D pixel grid, so we can re-label the same cells at different
resolutions and see which GRID_N gives the best correlation with the 3D truth.

Cells are rebuilt from the parameters recorded in the S6.1 manifest. The parametric sweep 
quantizes strut width to whole pixels, so a finer grid admits different widths, produces a 
different valid set, and the farthest-point stratification would then pick different cells
"""

import json
import pathlib

import numpy as np

import config
from geom import handmade, parametric
from sim2d.fatigue import fatigue
from sim2d.homogenize import homogenize
from sim3d import correlate

RESULTS_DIR = config.PROJECT_ROOT / 'sim3d' / 'results'
MANIFEST = RESULTS_DIR / 's6_1_manifest.json'
LABELS = RESULTS_DIR / 's6_1_labels.json'

def rebuild(entry, n):
    """Rebuild one S6.1 cell at resolution n from its recorded parameters."""
    family, params = entry['family'], entry['params']
    if family in ('crown', 'reference'):
        return parametric.crown(strut_width_mm=params['strut_width_mm'],
                                crown_amplitude=params['crown_amplitude'],
                                n_periods=params['n_periods'], n=n)
    if family == 'handmade':
        return handmade.VALID_CELLS[params['label']](n)
    raise ValueError(f'unknown family {family!r}')

def relabel(n, manifest_path=None, labels_path=None):
    """
    Re-label every converged S6.1 cell at resolution n, keeping its 3D result.

    Returns rows in `correlate`'s shape, so the same rho machinery applies unchanged.
    """
    manifest = json.loads(pathlib.Path(manifest_path or MANIFEST).read_text(encoding='utf-8'))
    entries = {c['name']: c for c in manifest['cells']}
    labeled, _ = correlate.load_labels(labels_path or LABELS)

    rows = []
    for row in labeled:
        cell = rebuild(entries[row['name']], n)
        h = homogenize(cell)
        f = fatigue(cell, h)
        rows.append({
            'name': row['name'], 'family': row['family'], 'converged': True,
            'retries': row.get('retries', 0),
            'grid_n': cell.n, 'f_metal': float(cell.f_metal),
            'K_radial_2D': float(h.K_radial), 'K_radial_3D': row['K_radial_3D'],
            'eps_a_max_2D': float(f.eps_a_max), 'eps_a_max_3D': row['eps_a_max_3D'],
            'A_over_lim_2D': float(f.A_over_lim), 'A_over_lim_3D': row['A_over_lim_3D'],
        })
    return rows

def compare(resolutions=(64, 128), metrics=correlate.GATE_METRICS, n_boot=4000, seed=0):
    """rho at each resolution against the same 3D truth, plus the deltas."""
    out = {'resolutions': list(resolutions), 'metrics': {}, 'per_resolution': {}}
    per_res_rows = {n: relabel(n) for n in resolutions}
    for n, rows in per_res_rows.items():
        out['per_resolution'][str(n)] = {
            'n_cells': len(rows),
            'median_f_metal': float(np.median([r['f_metal'] for r in rows])),
        }
    for metric in metrics:
        entry = {}
        for n, rows in per_res_rows.items():
            d = correlate.analyze_metric(rows, metric, n_boot=n_boot, seed=seed)
            entry[str(n)] = {'rho': d['rho'], 'ci95': d['ci95']}
        base, fine = str(resolutions[0]), str(resolutions[-1])
        entry['delta_rho'] = entry[fine]['rho'] - entry[base]['rho']
        entry['materially_better'] = entry[fine]['rho'] > entry[base]['ci95'][1]
        out['metrics'][metric] = entry

    out['any_materially_better'] = any(v['materially_better']
                                       for v in out['metrics'].values())
    out['recommended_grid_n'] = (resolutions[-1] if out['any_materially_better']
                                 else resolutions[0])
    return out

def write(result=None, path=None):
    result = compare() if result is None else result
    path = (RESULTS_DIR / 's6_4_resolution.json') if path is None else pathlib.Path(path)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return path
