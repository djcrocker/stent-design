"""
Fatigue against radial stiffness.

The populations are measured through the same path: `sim2d.label` on a binary cell.
"""

import json
import time

import numpy as np

import config
from sim2d import label as label_mod

RESULTS_DIR = config.PROJECT_ROOT / 'screen' / 'results'

# Maximize the first, minimize the second.
AXES = ('K_radial', 'A_over_lim')
KEYS = ('K_radial', 'eps_a_max', 'A_over_lim', 'f_metal')

def label_rows(cells, progress=True, group=''):
    """Objective vectors for a list of UnitCells, invalid ones included."""
    from geom import validity

    rows = []
    for i, cell in enumerate(cells, 1):
        check = validity.check(cell)
        got = label_mod.label(cell, check_validity=False)
        rows.append({**{k: float(v) for k, v in zip(config.OBJECTIVE_KEYS, got.y)},
                     'valid': bool(check.ok), 'reasons': list(check.reasons)})
        if progress and i % 100 == 0:
            print(f'    {group} {i}/{len(cells)}', flush=True)
    return rows

def populations(progress=True):
    """Every population on the plane, each row carrying its own objectives and validity."""
    from geom import handmade, parametric
    from geom.cell import UnitCell
    from screen import motifs, shortlist

    out = {}

    pool = shortlist.load_pool(progress=progress)
    out['generated'] = [{**{k: float(pool.rows[i][k]) for k in KEYS},
                         'valid': True, 'reasons': [], 'pool_index': int(i)}
                        for i in pool.unique]

    valid, _ = parametric.extended_sweep()
    if progress:
        print(f'  labeling {len(valid)} parametric cells', flush=True)
    rows = label_rows([cell for _, cell in valid], progress, 'parametric')
    for row, (params, _) in zip(rows, valid):
        row['extended'] = bool(params.get('band_width_mm') or params.get('width_taper'))
        row['params'] = params
    out['parametric'] = rows

    cells = [factory() for _, factory in sorted(handmade.VALID_CELLS.items())]
    out['handmade'] = label_rows(cells, progress, 'handmade')
    for row, (name, _) in zip(out['handmade'], sorted(handmade.VALID_CELLS.items())):
        row['name'] = name

    simp = motifs.simp_cells()
    rows = label_rows([UnitCell(c['field']) for c in simp], progress, 'simp')
    for row, meta in zip(rows, simp):
        row['name'], row['screened'] = meta['name'], meta['screened']
    out['simp'] = [r for r in rows if r['screened']]
    out['simp_unscreened'] = [r for r in rows if not r['screened']]
    return out

def front(rows, axes=AXES):
    """Indices of the non-dominated set on (maximize axes[0], minimize axes[1])."""
    from screen import shortlist

    if not rows:
        return []
    objectives = np.array([[r[axes[0]], r[axes[1]]] for r in rows], float)
    layers = shortlist.nondominated_layers(objectives)
    order = np.argsort([rows[i][axes[0]] for i in layers[0]])
    return [int(layers[0][i]) for i in order]

# Bands bracketing the K_radial conditioning ladder (100/200/300).
BANDS = (0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 600.0)

def band_best(rows, key='A_over_lim', bands=BANDS, axis=AXES[0]):
    """Best `key` attained inside each stiffness band."""
    out = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        inside = [r for r in rows if lo <= r[axis] < hi]
        if not inside:
            out.append({'lo': lo, 'hi': hi, 'n': 0})
            continue
        best = min(inside, key=lambda r: r[key])
        out.append({'lo': lo, 'hi': hi, 'n': len(inside), key: float(best[key]), axis: float(best[axis])})
    return out

def _summary(rows, key):
    v = np.array([r[key] for r in rows], float)
    if not len(v):
        return {'n': 0}
    return {'n': int(len(v)), 'min': float(v.min()), 'median': float(np.median(v)), 'max': float(v.max())}

def analyze(out_stem='s10_2_pareto', progress=True):
    """Label every population, peel each one's front, and write the figure's data."""
    started = time.perf_counter()
    pops = populations(progress=progress)
    report = {'axes': list(AXES), 'keys': list(KEYS),
              'measured_through': 'sim2d.label on the binary cell, every population',
              'bands': list(BANDS),
              'populations': {}, 'fronts': {}, 'band_best': {}}
    for name, rows in pops.items():
        report['populations'][name] = {'n': len(rows),
                                       'n_valid': sum(1 for r in rows if r['valid']),
                                       **{key: _summary(rows, key) for key in KEYS}}
        report['fronts'][name] = front(rows)
        report['band_best'][name] = {key: band_best(rows, key)
                                     for key in ('A_over_lim', 'f_metal')}
    report['screened_front'] = front([r for r in pops['generated'] if r['valid']])
    report['seconds'] = time.perf_counter() - started
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f'{out_stem}.json').write_text(json.dumps({**report, 'rows': pops}, indent=2), encoding='utf-8')
    return {**report, 'rows': pops}

def load(out_stem='s10_2_pareto'):
    return json.loads((RESULTS_DIR / f'{out_stem}.json').read_text(encoding='utf-8'))

def main():
    rep = analyze()
    print(f"\nmeasured through {rep['measured_through']} ({rep['seconds']:.0f}s)")
    print(f"{'population':18s} {'n':>5s} {'valid':>6s} {'K_radial':>18s} {'A_over_lim':>18s}")
    for name, d in rep['populations'].items():
        k, a = d['K_radial'], d['A_over_lim']
        print(f"{name:18s} {d['n']:5d} {d['n_valid']:6d} "
              f"{k['min']:7.1f}-{k['max']:<9.1f} {a['min']:7.4f}-{a['max']:<9.4f}")
    print('\nbest A_over_lim per stiffness band - the comparison that answers the design '
          'question:')
    def cell(value, width, places=4):
        return f'{value:{width}.{places}f}' if value is not None else f"{'-':>{width}}"

    print(f"  {'band':14s} {'ours':>8s} {'parametric':>11s} {'ratio':>7s} {'SIMP':>8s}")
    for i, lo in enumerate(BANDS[:-1]):
        got = {name: rep['band_best'][name]['A_over_lim'][i].get('A_over_lim')
               for name in rep['band_best']}
        ours, param = got['generated'], got['parametric']
        simp = got['simp'] if got['simp'] is not None else got['simp_unscreened']
        ratio = param / ours if ours and param else None
        print(f'  {lo:4.0f}-{BANDS[i + 1]:<9.0f} {cell(ours, 8)} {cell(param, 11)} '
              f'{cell(ratio, 7, 2)} {cell(simp, 8)}')

    print('\nfronts (K_radial, A_over_lim), best fatigue first:')
    for name, idx in rep['fronts'].items():
        rows = rep['rows'][name]
        pts = ', '.join(f"({rows[i]['K_radial']:.0f}, {rows[i]['A_over_lim']:.3f})"
                        for i in idx[:6])
        print(f'  {name:18s} {len(idx):3d} points  {pts}'
              f"{' ...' if len(idx) > 6 else ''}")

if __name__ == '__main__':
    main()
