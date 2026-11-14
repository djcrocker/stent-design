"""
The 3D head-to-head, the SIMP baseline against our cells at matched 2D radial stiffness.

Only one SIMP cell passes `geom.validity` in everything Part 3 ran, at `K_radial` 320.9 and
`f_metal` 0.417.
"""

import argparse
import json
import pathlib

import numpy as np

import config
from sim3d import flex

STEM = 's10_5_simp'
LABEL = 's10_5'
MANIFEST_NAME = 's10_5_manifest.json'
STAGE_DIR = flex.STAGE_DIR / 'headtohead'

# The baseline cell.
SIMP_RUN = 's10_1_prescribed_screened'
SIMP_TARGET = 200.0

# Our side, as shortlist row indices.
MATCHED_ROWS = (6, 8)

def simp_cell(run=SIMP_RUN, target=SIMP_TARGET):
    """The baseline cell and the campaign row that produced it."""
    from baseline import prescribed

    record = json.loads(
        (prescribed.RESULTS_DIR / f'{run}.json').read_text(encoding='utf-8'))
    row = next(r for r in record['targets'] if r['k_target'] == target)
    if not row['valid']:
        raise ValueError(f'{run} k{target:g} is not screened-valid: {row["reasons"]}')
    blob = np.load(prescribed.RESULTS_DIR / f'{run}_cells.npz')
    return np.asarray(blob[f'cell_k{target:g}'], bool), row

def ours(rows=MATCHED_ROWS):
    """The matched cells from our side, with the 3D results S9.4 already measured."""
    from screen import shortlist

    _, summary = shortlist.load()
    audit = json.loads(
        (flex.RESULTS_DIR / 's9_4_loading_audit.json').read_text(encoding='utf-8'))
    by_row = {r['pool_index']: r for r in audit['rows']}
    out = []
    for row in rows:
        if row not in by_row:
            raise ValueError(f'shortlist row {row} was not audited in S9.4')
        entry = summary['shortlist'][row]
        out.append({'shortlist_row': row, 'pool_index': entry['pool_index'],
                    'K_radial': entry['K_radial'], 'f_metal': entry['f_metal'],
                    'audit': by_row[row]})
    return out

SOLVER = {'comb': {'nsubst': '200,40000,100', 'neqit': 50}}

def build(stage=True, arms=flex.ARMS, solver=None):
    """Mesh the baseline cell and write its decks."""
    field, row = simp_cell()
    manifest = flex.build(indices=[0], fields=field[None], arms=arms,
                          prefix=STEM, manifest_name=MANIFEST_NAME,
                          solver=SOLVER if solver is None else solver)
    manifest['baseline'] = {'run': SIMP_RUN, 'k_target': SIMP_TARGET,
                            'K_radial': row['K_radial'], 'f_metal': row['f_metal'],
                            'volfrac': row['volfrac']}
    manifest['matched'] = [{k: v for k, v in o.items() if k != 'audit'} for o in ours()]
    (flex.RESULTS_DIR / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2),
                                                  encoding='utf-8')
    staged = flex.stage(manifest, stage_dir=STAGE_DIR, label=LABEL,
                        script_path=flex.DECK_DIR / f'run_{LABEL}.ps1') if stage else []
    return manifest, staged

def analyze(manifest_path=None, results_dir=None, out_stem='s10_5_headtohead'):
    """The baseline's 3D strain next to our matched cells'."""
    from geom.cell import UnitCell
    from sim3d import loadsteps, mesh3d

    results_dir = (config.PROJECT_ROOT / 'ansys_output') if results_dir is None \
        else pathlib.Path(results_dir)
    manifest_path = (flex.RESULTS_DIR / MANIFEST_NAME) if manifest_path is None \
        else pathlib.Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    field, row = simp_cell()
    mesh = manifest['mesh']
    points, hexes, _ = mesh3d.tube_hex_mesh(UnitCell(field), n_circ=mesh['n_circ'],
                                            n_axial=mesh['n_axial'],
                                            layers=mesh['layers'], limit=mesh['limit'])
    base = dict(manifest['baseline'], stem=STEM + '00', source='simp', K_radial_binary=row['binary']['K_radial'])
    for arm in manifest['arms']:
        stem = f'{STEM}00_{arm}'
        status, amp = results_dir / f'{stem}.txt', results_dir / f'{stem}_amp.txt'
        if not ((results_dir / f'{stem}.ok').exists() and status.exists() and amp.exists()):
            base[f'{arm}_converged'] = False
            continue
        zloc = results_dir / f'{stem}_zloc.txt'
        s = loadsteps.flex_scalars(status, amp, points, hexes,
                                   zloc_path=zloc if zloc.exists() else None,
                                   fraction=manifest['sample_fraction'],
                                   halve=manifest.get('release', False))
        base[f'{arm}_converged'] = True
        for key, got in (('eps_a_max', 'eps_a_max_bend'), ('eps_a_p99', 'eps_a_p99_bend'),
                         ('eps_a_p95', 'eps_a_p95_bend'), ('over_eps_L', 'over_eps_L_bend'),
                         ('A_over_lim', 'A_over_lim_bend'),
                         ('grip_inflation', 'grip_inflation')):
            base[f'{arm}_{key}'] = s[got]

    rows = [base]
    for o in ours():
        row = {'source': 'generated', 'stem': o['audit']['stem'],
               'shortlist_row': o['shortlist_row'], 'pool_index': o['pool_index'],
               'K_radial': o['K_radial'], 'f_metal': o['f_metal']}
        for arm in manifest['arms']:
            for key in ('converged', 'eps_a_max', 'eps_a_p99', 'eps_a_p95', 'over_eps_L',
                        'A_over_lim', 'grip_inflation'):
                if f'{arm}_{key}' in o['audit']:
                    row[f'{arm}_{key}'] = o['audit'][f'{arm}_{key}']
        rows.append(row)

    report = {'source': str(manifest_path), 'arms': list(manifest['arms']),
              'match_quantity': 'K_radial, 2D homogenization, both populations',
              'bend_radius_mm': manifest['bend_radius_mm'],
              'axial_compression': manifest['axial_compression'], 'rows': rows}
    (flex.RESULTS_DIR / f'{out_stem}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report

def main():
    ap = argparse.ArgumentParser(description='S10.1 Part 5 head-to-head')
    ap.add_argument('phase', choices=('build', 'analyze'))
    ap.add_argument('--no-stage', action='store_true')
    args = ap.parse_args()

    if args.phase == 'analyze':
        rep = analyze()
        print(f"match on {rep['match_quantity']}")
        for r in rep['rows']:
            print(f"  {r['source']:10s} {r['stem']:14s} K_radial {r['K_radial']:6.1f}  "
                  f"f_metal {r['f_metal']:.3f}")
            for arm in rep['arms']:
                if r.get(f'{arm}_converged'):
                    print(f"      {arm:5s} eps_a_max {r[f'{arm}_eps_a_max']:.5f}  "
                          f"p99 {r[f'{arm}_eps_a_p99']:.5f}  "
                          f"A/lim {r[f'{arm}_A_over_lim']:.3f}")
                else:
                    print(f'      {arm:5s} MISSING')
        return

    manifest, staged = build(stage=not args.no_stage)
    cell = manifest['cells'][0]
    print(f"baseline K_radial {manifest['baseline']['K_radial']:.1f}  "
          f"f_metal {manifest['baseline']['f_metal']:.3f}")
    print(f"  mesh {cell['n_nodes']} nodes, {cell['n_hexes']} hexes, "
          f"scaled jacobian min {cell['scaled_jacobian_min']:.3f}, "
          f"max face angle {cell['max_face_angle_deg']:.1f} deg")
    for arm, path in cell['decks'].items():
        print(f"  {arm:5s} {path}")
    print('\nmatched against (already solved in S9.4):')
    for o in manifest['matched']:
        print(f"  row {o['shortlist_row']:2d} (pool {o['pool_index']:4d})  "
              f"K_radial {o['K_radial']:6.1f}  f_metal {o['f_metal']:.3f}")
    if staged:
        print(f'\nStaged {len(staged)} files in {STAGE_DIR}')

if __name__ == '__main__':
    main()
