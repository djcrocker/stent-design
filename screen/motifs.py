"""
What do the winning topologies do that the conventional families can't?

An embedding can show two clouds that miss each other, but it can't say what the gap is.
So the result here is a set of named morphological descriptors, measured identically on 
three populations - the generated pool, the S3 parametric sweep, and the handmade cells - 
and the finding is which descriptor regions the conventional families never reach.

Every descriptor here is shift-invariant, computed on the torus.

Axis convention follows `geom.validity.wraps`: axis 1 (columns) is circumferential, axis 0
(rows) is axial.
"""

import json

import numpy as np
from scipy import ndimage

import config
from geom import periodic

RESULTS_DIR = config.PROJECT_ROOT / 'screen' / 'results'

# validity.wraps returns (circumferential, axial) in this order.
AXIS = {'circ': 1, 'axial': 0}

SHAPE_KEYS = ('skeleton_length', 'junction_frac', 'endpoint_frac', 'n_junctions',
              'n_pores', 'tortuosity_axial_min', 'tortuosity_axial_mean',
              'tortuosity_circ_min', 'tortuosity_circ_mean', 'straight_run_axial',
              'straight_run_circ', 'thickness_mean', 'thickness_cv', 'orient_axial',
              'orient_circ', 'orient_diag')

def skeleton(arr):
    """
    Medial axis on the torus.

    `skimage.skeletonize` knows nothing about wrapping, so it is run on a 3x3 tiling and the
    center tile taken. One full tile of context on every side is more than the operator's
    reach, so the center comes out as if the array were periodic.
    """
    from skimage.morphology import skeletonize

    arr = np.asarray(arr, bool)
    n = arr.shape[0]
    skel = skeletonize(np.tile(arr, (3, 3)))
    return skel[n:2 * n, n:2 * n]

def _skeleton_neighbors(skel):
    """Count of 8-neighbors that are also skeleton, per skeleton pixel, wrapping."""
    k = np.ones((3, 3), dtype=np.uint8)
    k[1, 1] = 0
    counts = ndimage.convolve(skel.astype(np.uint8), k, mode='wrap')
    return counts * skel

# Offsets for the four distinct skeleton orientations.
_ORIENT = {'orient_axial': (1, 0), 'orient_circ': (0, 1),
           '_diag_a': (1, 1), '_diag_b': (1, -1)}

def orientation(skel):
    """
    How the skeleton's own length is distributed over direction.

    Counts skeleton-to-skeleton adjacencies in each of the four lattice directions and
    normalizes. A family of straight axial links loads `orient_axial`; something that
    serpentines spreads into the diagonals. Coarse, but it needs no tangent estimate 
    and inherits shift-invariance from `np.roll`.
    """
    skel = np.asarray(skel, bool)
    counts = {}
    for name, (dr, dc) in _ORIENT.items():
        counts[name] = float((skel & np.roll(skel, (dr, dc), axis=(0, 1))).sum())
    total = sum(counts.values())
    if total <= 0:
        return {'orient_axial': 0.0, 'orient_circ': 0.0, 'orient_diag': 0.0}
    return {'orient_axial': counts['orient_axial'] / total,
            'orient_circ': counts['orient_circ'] / total,
            'orient_diag': (counts['_diag_a'] + counts['_diag_b']) / total}

def longest_run(arr, axis):
    """
    Longest unbroken run of metal along one direction, over the cell width.

    1.0 means some line of the cell is solid metal end to end. Computed on a 
    doubled array so a run crossing the seam isn'dt cut in two.
    """
    arr = np.asarray(arr, bool)
    n = arr.shape[0]
    doubled = np.concatenate([arr, arr], axis=axis).astype(np.int32)
    run = np.zeros(doubled.shape[1 - axis], dtype=np.int32)
    best = 0
    for k in range(doubled.shape[axis]):
        line = doubled[k] if axis == 0 else doubled[:, k]
        run = (run + 1) * line
        best = max(best, int(run.max()))
    return float(min(best, n)) / n

def tortuosity(arr, axis):
    """
    Shortest metal path that advances exactly one period, over the straight distance.

    1.0 is a strut running straight across the cell. Higher means the only way across
    is to double back.

    One run of the geodesic per start line, because the quantity has to be shift-invariant. 

    Every cell in the pool wraps in both directions (`geom.validity` requires it), so each
    start line holds metal and a path always exists.
    """
    from skimage import graph

    arr = np.asarray(arr, bool)
    n = arr.shape[0]
    tiled = np.tile(arr, (3, 3))
    costs = np.where(tiled, 1.0, np.inf)
    mcp = graph.MCP_Geometric(costs, fully_connected=True)

    lengths = []
    for i in range(n):
        line = arr[i, :] if axis == 0 else arr[:, i]
        where = np.flatnonzero(line)
        if not len(where):
            continue
        if axis == 0:
            starts = [(n + i, n + int(c)) for c in where]
            ends = [(2 * n + i, n + int(c)) for c in where]
        else:
            starts = [(n + int(r), n + i) for r in where]
            ends = [(n + int(r), 2 * n + i) for r in where]
        reached, _ = mcp.find_costs(starts)
        d = min(reached[e] for e in ends)
        if np.isfinite(d):
            lengths.append(d / n)
    if not lengths:
        return {'min': float('nan'), 'mean': float('nan')}
    return {'min': float(np.min(lengths)), 'mean': float(np.mean(lengths))}

def describe(arr):
    """Every descriptor for one cell, as a flat dict."""
    arr = np.asarray(arr, bool)
    n = arr.shape[0]
    skel = skeleton(arr)
    nb = _skeleton_neighbors(skel)
    n_skel = float(skel.sum())

    # Inscribed radius along the centerline
    dist = periodic.distance_transform(arr)
    thick = dist[skel] if n_skel else np.array([0.0])

    _, n_pores = periodic.label(~arr)

    out = {'f_metal': float(arr.mean()),
           'skeleton_length': n_skel / n,
           'n_junctions': float((nb >= 3).sum()),
           'n_endpoints': float((nb == 1).sum()),
           'junction_frac': float((nb >= 3).sum()) / n_skel if n_skel else 0.0,
           'endpoint_frac': float((nb == 1).sum()) / n_skel if n_skel else 0.0,
           'n_pores': float(n_pores),
           'thickness_mean': float(thick.mean()),
           'thickness_cv': float(thick.std() / thick.mean()) if thick.mean() else 0.0,
           'straight_run_axial': longest_run(arr, 0),
           'straight_run_circ': longest_run(arr, 1)}
    out.update(orientation(skel))
    for name, axis in AXIS.items():
        t = tortuosity(arr, axis)
        out[f'tortuosity_{name}_min'] = t['min']
        out[f'tortuosity_{name}_mean'] = t['mean']
    return out

def describe_many(fields, progress=True, label=''):
    """Descriptors for a stack of fields."""
    rows = []
    for i, arr in enumerate(fields, 1):
        rows.append(describe(arr))
        if progress and i % 50 == 0:
            print(f'    {label} {i}/{len(fields)}', flush=True)
    return rows

# Which Pareto layers count as winners. Layer 1 is the actual non-dominated front.
WINNER_LAYERS = 1

def conventional():
    """
    The populations a conventional designer would have produced: the S3 parametric sweep
    and the handmade cells.

    PENDING, 10.1 SIMP / inverse-homogenization.
    """
    from geom import handmade, parametric

    valid, _ = parametric.sweep()
    rows = [{'name': f'param{i:02d}', 'family': 'parametric', 'field': cell.to_array()}
            for i, (_, cell) in enumerate(valid)]
    for label, factory in sorted(handmade.VALID_CELLS.items()):
        rows.append({'name': label, 'family': 'handmade', 'field': factory().to_array()})
    return rows

def _summary(values):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if not len(v):
        return {'n': 0}
    return {'n': int(len(v)), 'min': float(v.min()), 'p25': float(np.percentile(v, 25)),
            'median': float(np.median(v)), 'p75': float(np.percentile(v, 75)),
            'max': float(v.max()), 'mean': float(v.mean()), 'std': float(v.std())}

def _cliffs_delta(a, b):
    """Nonparametric effect size: P(a > b) - P(a < b), in [-1, 1]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = float((a[:, None] > b[None, :]).sum())
    lt = float((a[:, None] < b[None, :]).sum())
    return (gt - lt) / (len(a) * len(b))

def compare(winners, others, keys=SHAPE_KEYS):
    """Per-descriptor two-sample comparison, nonparametric, with an effect size."""
    from scipy import stats

    out = {}
    for key in keys:
        a = np.array([r[key] for r in winners if np.isfinite(r[key])], float)
        b = np.array([r[key] for r in others if np.isfinite(r[key])], float)
        if len(a) < 3 or len(b) < 3:
            out[key] = {'n_a': len(a), 'n_b': len(b), 'skipped': 'too few'}
            continue
        u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        out[key] = {'n_a': int(len(a)), 'n_b': int(len(b)),
                    'median_a': float(np.median(a)), 'median_b': float(np.median(b)),
                    'p': float(p), 'cliffs_delta': _cliffs_delta(a, b)}
    return out

def exclusion(winners, conv, keys=SHAPE_KEYS):
    """
    Which descriptor values the conventional families never reach.

    For each descriptor: the conventional range, and the fraction of winners outside it.
    """
    out = {}
    for key in keys:
        c = np.array([r[key] for r in conv if np.isfinite(r[key])], float)
        w = np.array([r[key] for r in winners if np.isfinite(r[key])], float)
        if len(c) < 3 or len(w) < 3:
            out[key] = {'skipped': 'too few'}
            continue
        lo, hi = float(c.min()), float(c.max())
        below, above = int((w < lo).sum()), int((w > hi).sum())
        out[key] = {'conventional_range': [lo, hi],
                    'winner_range': [float(w.min()), float(w.max())],
                    'n_winners': int(len(w)),
                    'frac_below': below / len(w), 'frac_above': above / len(w),
                    'frac_outside': (below + above) / len(w)}
    return out

def density_matched(rows_a, rows_b, key='f_metal'):
    """
    Restrict both populations to the f_metal band they share.

    Nearly every shape descriptor moves with density, and the populations don't sit at the
    same densities, so an unmatched comparison can report a difference in `f_metal` while 
    showing a difference in shape.
    """
    a = np.array([r[key] for r in rows_a], float)
    b = np.array([r[key] for r in rows_b], float)
    lo, hi = max(a.min(), b.min()), min(a.max(), b.max())
    if not (hi > lo):
        return [], [], {'overlap': None}
    ka = [r for r in rows_a if lo <= r[key] <= hi]
    kb = [r for r in rows_b if lo <= r[key] <= hi]
    return ka, kb, {'overlap': [float(lo), float(hi)], 'n_a': len(ka), 'n_b': len(kb)}

def cluster(rows, keys=SHAPE_KEYS, k=4, seed=0):
    """Group the pool by shape, to see whether the winners recur as a few motifs or spread."""

    from scipy.cluster import hierarchy

    x = np.array([[r[key] for key in keys] for r in rows], float)
    finite = np.isfinite(x).all(axis=1)
    x = x[finite]
    mu, sd = x.mean(axis=0), x.std(axis=0)
    z = (x - mu) / np.where(sd > 0, sd, 1.0)

    labels = hierarchy.fcluster(hierarchy.linkage(z, method='ward'), k, criterion='maxclust')
    u, s, vt = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
    coords = (u[:, :2] * s[:2])

    out = {'k': int(k), 'keys': list(keys), 'n': int(len(z)),
           'kept': np.flatnonzero(finite).tolist(),
           'labels': labels.tolist(),
           'coords': coords.tolist(),
           'pc_variance_frac': (s ** 2 / (s ** 2).sum())[:4].tolist(),
           'pc1_loadings': dict(zip(keys, vt[0].tolist())),
           'pc2_loadings': dict(zip(keys, vt[1].tolist())),
           'clusters': {}}
    for c in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == c)
        centroid = z[members].mean(axis=0)
        # The member nearest its own centroid: the cell to put on the figure as the motif.
        exemplar = members[int(np.argmin(np.linalg.norm(z[members] - centroid, axis=1)))]
        out['clusters'][str(c)] = {
            'n': int(len(members)),
            'exemplar_row': int(np.flatnonzero(finite)[exemplar]),
            'mean': {key: float(x[members, i].mean()) for i, key in enumerate(keys)},
            'z_mean': {key: float(centroid[i]) for i, key in enumerate(keys)}}
    return out

def analyze(source_stem='s9_1_generated', out_stem='s10_3_motifs', winner_layers=WINNER_LAYERS, k=4, progress=True):
    """
    Descriptors on three populations, then the exclusion comparison.

    Writes the descriptor table and the report, plus the exemplar fields the figure needs.
    """
    from screen import shortlist

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pool = shortlist.load_pool(source_stem, progress=progress)
    fields = pool.fields[pool.unique]
    objectives = pool.objectives()
    layers = shortlist.nondominated_layers(objectives)
    layer_of = np.zeros(len(objectives), dtype=int)
    for depth, members in enumerate(layers, 1):
        layer_of[members] = depth

    if progress:
        print(f'  descriptors on {len(fields)} pool cells', flush=True)
    pool_rows = describe_many(fields, progress, 'pool')
    for row, idx, depth in zip(pool_rows, pool.unique, layer_of):
        row['pool_index'] = int(idx)
        row['pareto_layer'] = int(depth)
        for key in shortlist.OBJECTIVES:
            row[key] = float(pool.rows[idx][key])

    conv = conventional()
    if progress:
        print(f'  descriptors on {len(conv)} conventional cells', flush=True)
    conv_rows = describe_many([c['field'] for c in conv], progress, 'conventional')
    for row, meta in zip(conv_rows, conv):
        row['name'], row['family'] = meta['name'], meta['family']

    winners = [r for r in pool_rows if r['pareto_layer'] <= winner_layers]
    rest = [r for r in pool_rows if r['pareto_layer'] > winner_layers]

    mw, mc, match = density_matched(winners, conv_rows)
    report = {
        'source': source_stem,
        'winner_layers': int(winner_layers),
        'pending': 'The SIMP / inverse-homogenization baseline is not built yet (S10.1). '
                   'Conventional here means the S3 parametric sweep plus the handmade cells.',
        'populations': {
            'pool': {'n': len(pool_rows),
                     **{key: _summary([r[key] for r in pool_rows])
                        for key in ('f_metal',) + SHAPE_KEYS}},
            'winners': {'n': len(winners),
                        **{key: _summary([r[key] for r in winners])
                           for key in ('f_metal',) + SHAPE_KEYS}},
            'conventional': {'n': len(conv_rows),
                             **{key: _summary([r[key] for r in conv_rows])
                                for key in ('f_metal',) + SHAPE_KEYS}}},
        'winners_vs_rest_of_pool': compare(winners, rest),
        'winners_vs_conventional': compare(winners, conv_rows),
        'winners_vs_conventional_density_matched': {
            'matching': match, **({} if not mw else {'per_metric': compare(mw, mc)})},
        'exclusion': exclusion(winners, conv_rows),
        'exclusion_density_matched': {} if not mw else exclusion(mw, mc),
        'clusters': cluster(pool_rows, k=k),
    }
    # Pareto layer composition per cluster
    labels = report['clusters']['labels']
    kept = report['clusters']['kept']
    for c in report['clusters']['clusters']:
        rows_in = [pool_rows[kept[i]] for i, lab in enumerate(labels) if str(lab) == c]
        n_win = sum(1 for r in rows_in if r['pareto_layer'] <= winner_layers)
        report['clusters']['clusters'][c]['n_winners'] = int(n_win)
        report['clusters']['clusters'][c]['winner_frac'] = n_win / len(rows_in)

    (RESULTS_DIR / f'{out_stem}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    table = {'pool': pool_rows, 'conventional': conv_rows}
    (RESULTS_DIR / f'{out_stem}_descriptors.json').write_text(json.dumps(table, indent=2), encoding='utf-8')
    exemplars = np.array([report['clusters']['clusters'][c]['exemplar_row']
                          for c in sorted(report['clusters']['clusters'])], dtype=int)
    np.savez_compressed(RESULTS_DIR / f'{out_stem}_cells.npz',
                        fields=np.packbits(fields, axis=None),
                        shape=np.array(fields.shape),
                        pool_index=pool.unique,
                        pareto_layer=layer_of,
                        exemplar_row=exemplars)
    return report

def load(out_stem='s10_3_motifs'):
    """The saved report, descriptor table, and pool fields."""
    report = json.loads((RESULTS_DIR / f'{out_stem}.json').read_text(encoding='utf-8'))
    table = json.loads(
        (RESULTS_DIR / f'{out_stem}_descriptors.json').read_text(encoding='utf-8'))
    blob = np.load(RESULTS_DIR / f'{out_stem}_cells.npz')
    shape = tuple(blob['shape'])
    fields = np.unpackbits(blob['fields'], count=int(np.prod(shape))).reshape(shape).astype(bool)
    return report, table, fields

def main():
    import argparse

    ap = argparse.ArgumentParser(description='S10.3 motif analysis')
    ap.add_argument('--layers', type=int, default=WINNER_LAYERS, help='Pareto layers counted as winners')
    ap.add_argument('--k', type=int, default=4, help='number of motif clusters')
    args = ap.parse_args()

    r = analyze(winner_layers=args.layers, k=args.k)
    pops = r['populations']
    print(f"\npool {pops['pool']['n']}  winners {pops['winners']['n']}  "
          f"conventional {pops['conventional']['n']}")
    print(f"  f_metal  winners {pops['winners']['f_metal']['median']:.3f}  "
          f"conventional {pops['conventional']['f_metal']['median']:.3f}")

    print('\nExclusion - descriptor values the conventional families never reach:')
    ex = r['exclusion_density_matched'] or r['exclusion']
    ranked = sorted((v for v in ex.items() if 'frac_outside' in v[1]), key=lambda kv: -kv[1]['frac_outside'])
    for key, d in ranked[:8]:
        lo, hi = d['conventional_range']
        wlo, whi = d['winner_range']
        print(f"  {key:24s} outside {d['frac_outside']:5.1%}  "
              f"(below {d['frac_below']:4.0%} above {d['frac_above']:4.0%})  "
              f"conv [{lo:.3f}, {hi:.3f}]  winners [{wlo:.3f}, {whi:.3f}]")

    print('\nWinners vs conventional, density-matched:')
    m = r['winners_vs_conventional_density_matched']
    band = m['matching'].get('overlap')
    print(f"  f_metal band {band}  n {m['matching'].get('n_a')} vs {m['matching'].get('n_b')}")
    per = m.get('per_metric', {})
    ranked = sorted((v for v in per.items() if 'cliffs_delta' in v[1]), key=lambda kv: -abs(kv[1]['cliffs_delta']))
    for key, d in ranked[:8]:
        print(f"  {key:24s} delta {d['cliffs_delta']:+.3f}  p {d['p']:.2e}  "
              f"median {d['median_a']:.3f} vs {d['median_b']:.3f}")

    print('\nMotif clusters:')
    cl = r['clusters']
    print(f"  PC variance {[round(v, 3) for v in cl['pc_variance_frac']]}")
    for c in sorted(cl['clusters']):
        d = cl['clusters'][c]
        print(f"  cluster {c}  n {d['n']:4d}  winners {d['n_winners']:3d} "
              f"({d['winner_frac']:5.1%})  "
              f"tort_ax {d['mean']['tortuosity_axial_mean']:.3f}  "
              f"run_ax {d['mean']['straight_run_axial']:.3f}  "
              f"junc {d['mean']['junction_frac']:.3f}")

if __name__ == '__main__':
    main()
