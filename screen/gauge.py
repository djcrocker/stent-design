"""
Gauge audit: does the screen's ranking survive a lighter strut cross-section?

Kamenskiy's DOE carries width x amplitude and width x n_struts interaction terms, so
cross-section is not a separable scale coefficient and a ranking measured at one gauge
can't be assumed to hold at another. In this tier, radial thickness t is out of plane, while in-plane width w is auditable.
"""

import argparse
import json
import time

import numpy as np
from scipy import stats

import config
from geom import periodic, validity
from geom.cell import UnitCell
from screen import motifs, shortlist
from sim2d.fatigue import fatigue
from sim2d.homogenize import homogenize

RESULTS_DIR = config.PROJECT_ROOT / 'screen' / 'results'

REFERENCE_MM = (config.STRUT_WIDTH_MM, config.STRUT_THICKNESS_MM)   # Current config
OFFSETS_PX = (1, 2)                                                 # Material off each side

# Ranking keys
RANK_KEYS = ('K_radial', 'eps_a_max', 'eps_a_p99', 'A_over_lim', 'f_metal')
TOP_K = (5, 10, shortlist.DEFAULT_K)

# Reasons that make a cell unsolvable.
FATAL = (validity.EMPTY, validity.DISCONNECTED, validity.NO_WRAP_CIRC, validity.NO_WRAP_AXIAL)

def gauge_mm(offset_px):
    """
    Nominal (width, thickness) in mm after taking `offset_px` off every side.

    Thickness is scaled with width so the cross-section keeps its aspect ratio.
    """
    width = config.STRUT_WIDTH_MM - 2.0 * offset_px * config.mm_per_px()[0]
    return (width, config.STRUT_THICKNESS_MM * width / config.STRUT_WIDTH_MM)

def narrow(field, offset_px):
    """Every boundary offset inward by `offset_px`."""
    return periodic.distance_transform(np.asarray(field, bool)) > offset_px

def band_depth_px(field):
    """Deepest stack of consecutive all-metal rows, on the torus."""
    rows = np.asarray(field, bool).all(axis=1)
    if rows.all():
        return int(rows.size)
    best = run = 0
    for present in np.concatenate([rows, rows]):
        run = run + 1 if present else 0
        best = max(best, run)
    return int(min(best, rows.size))

def member_width_mm(field):
    """Mean member width from the inscribed radius along the medial axis."""
    arr = np.asarray(field, bool)
    skel = motifs.skeleton(arr)
    if not skel.any():
        return 0.0
    return float(2.0 * periodic.distance_transform(arr)[skel].mean() * config.mm_per_px()[0])

def relabel(field, thickness_mm):
    """The screen's objectives for one field at a given radial thickness."""
    cell = UnitCell(field)
    h = homogenize(cell)
    f = fatigue(cell, h)
    return {'K_radial': h.E_circ * thickness_mm / (config.D_DEPLOYED_MM / 2.0) ** 2,
            'E_circ': h.E_circ, 'eps_a_max': f.eps_a_max, 'eps_a_p99': f.eps_a_p99,
            'A_over_lim': f.A_over_lim, 'f_metal': cell.f_metal}

def arm(fields, offset_px=0, progress=True, label=''):
    """One gauge over the whole pool. Returns a row per cell; dead cells carry no metrics."""
    thickness = gauge_mm(offset_px)[1]
    rows = []
    for i, field in enumerate(fields, 1):
        arr = narrow(field, offset_px) if offset_px else np.asarray(field, bool)
        check = validity.check(arr)
        row = {'alive': not any(r in FATAL for r in check.reasons),
               'valid': bool(check.ok), 'reasons': list(check.reasons),
               'straight_run_circ': motifs.longest_run(arr, 1),
               'band_depth_px': band_depth_px(arr),
               'width_mm': member_width_mm(arr)}
        if row['alive']:
            row.update(relabel(arr, thickness))
        rows.append(row)
        if progress and i % 100 == 0:
            print(f'    {label} {i}/{len(fields)}', flush=True)
    return rows

def spearman(ref, light, keys=RANK_KEYS):
    """Rank correlation per objective over the cells alive under both gauges."""
    out = {}
    for key in keys:
        a = np.array([r[key] for r in ref], float)
        b = np.array([r[key] for r in light], float)
        rho, p = stats.spearmanr(a, b)
        out[key] = {'spearman': float(rho), 'p_value': float(p), 'n': int(len(a))}
    return out

def top_k_overlap(ref, light, ks=TOP_K):
    """How much of the shortlist survives the gauge change."""
    def chosen(rows, k):
        objectives = np.array([[r[key] for key in shortlist.OBJECTIVES] for r in rows], float)
        return set(shortlist.select(objectives, k=k)[0].tolist())

    out = {}
    for k in ks:
        if k > len(ref):
            continue
        a, b = chosen(ref, k), chosen(light, k)
        out[str(k)] = {'k': int(k), 'shared': len(a & b), 'overlap': len(a & b) / float(k)}
    return out

def front_overlap(ref, light):
    """Overlap of the non-dominated front itself."""
    def front(rows):
        objectives = np.array([[r[key] for key in shortlist.OBJECTIVES] for r in rows], float)
        return set(shortlist.nondominated_layers(objectives)[0].tolist())

    a, b = front(ref), front(light)
    return {'reference': len(a), 'light': len(b), 'shared': len(a & b),
            'jaccard': len(a & b) / float(len(a | b)) if a | b else 0.0,
            'kept': len(a & b) / float(len(a)) if a else 0.0}

def compare(ref, light, pool_rows, offset_px):
    """Everything S9.5 records for one light gauge against the reference arm."""
    alive = [i for i in range(len(ref)) if ref[i]['alive'] and light[i]['alive']]
    winners = [i for i, r in enumerate(pool_rows) if r['pareto_layer'] <= motifs.WINNER_LAYERS]
    banded = [i for i, r in enumerate(ref) if r['straight_run_circ'] >= 1.0]
    a, b = [ref[i] for i in alive], [light[i] for i in alive]
    return {
        'offset_px': int(offset_px),
        'offset_mm_per_side': offset_px * config.mm_per_px()[0],
        'gauge_mm': list(gauge_mm(offset_px)),
        'width_mm_median': float(np.median([r['width_mm'] for r in light])),
        'survival': {
            'alive': sum(1 for r in light if r['alive']),
            'alive_both': len(alive),
            'valid': sum(1 for r in light if r['valid']),
            'winners': len(winners),
            'winners_alive': sum(1 for i in winners if light[i]['alive']),
            'reasons': {reason: sum(1 for r in light if reason in r['reasons'])
                        for reason in validity.CRITERIA},
        },
        'circumferential_band': {
            'reference': len(banded),
            'kept': sum(1 for i in banded if light[i]['straight_run_circ'] >= 1.0),
            'depth_px_median': float(np.median([light[i]['band_depth_px'] for i in banded]))
                               if banded else 0.0,
        },
        'spearman': spearman(a, b),
        'front_overlap': front_overlap(a, b),
        'top_k_overlap': top_k_overlap(a, b),
        'median_change': {key: float(np.median([r[key] for r in b])
                                     / np.median([r[key] for r in a]) - 1.0)
                          for key in RANK_KEYS},
    }

def audit(out_stem='s9_5_gauge', offsets=OFFSETS_PX, progress=True):
    """Relabel the pool at the reference gauge and at each lighter one."""
    started = time.perf_counter()
    _, table, fields = motifs.load()
    pool_rows = table['pool']
    if progress:
        print(f'  {len(fields)} pool cells, GRID_N {config.GRID_N} at '
              f'{config.mm_per_px()[0]:.5f} mm/px, reference {REFERENCE_MM[0]}/'
              f'{REFERENCE_MM[1]} mm', flush=True)

    ref = arm(fields, 0, progress=progress, label='reference')
    drift = max(abs(a['K_radial'] / b['K_radial'] - 1.0) for a, b in zip(ref, pool_rows))

    arms, per_cell = [], {'0': ref}
    for offset in offsets:
        if progress:
            print(f'  offset {offset} px -> nominal {gauge_mm(offset)[0]:.4f}/'
                  f'{gauge_mm(offset)[1]:.4f} mm', flush=True)
        light = arm(fields, offset, progress=progress, label=f'light {offset}px')
        per_cell[str(offset)] = light
        arms.append(compare(ref, light, pool_rows, offset))

    depths = [r['band_depth_px'] for r in ref if r['straight_run_circ'] >= 1.0]
    record = {
        'pool': int(len(fields)),
        'grid_n': config.GRID_N,
        'mm_per_px': config.mm_per_px()[0],
        'reference': {'gauge_mm': list(REFERENCE_MM),
                      'label_drift': float(drift),
                      'width_mm_median': float(np.median([r['width_mm'] for r in ref])),
                      'alive': sum(1 for r in ref if r['alive']),
                      'valid': sum(1 for r in ref if r['valid']),
                      # The S10.3 carried question: how deep is the band, really?
                      'banded': len(depths),
                      'band_depth_px': {'min': int(min(depths)) if depths else 0,
                                        'median': float(np.median(depths)) if depths else 0.0,
                                        'max': int(max(depths)) if depths else 0},
                      'band_depth_mm_median': float(np.median(depths) * config.mm_per_px()[0])
                                              if depths else 0.0},
        'arms': arms,
        'seconds': time.perf_counter() - started,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f'{out_stem}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    (RESULTS_DIR / f'{out_stem}_rows.json').write_text(json.dumps(per_cell), encoding='utf-8')
    return record

def main():
    ap = argparse.ArgumentParser(description='S9.5 gauge audit')
    ap.add_argument('--offsets', type=int, nargs='+', default=list(OFFSETS_PX), help='pixels of material off each side, one light arm per value')
    ap.add_argument('--stem', default='s9_5_gauge')
    args = ap.parse_args()

    r = audit(out_stem=args.stem, offsets=tuple(args.offsets))
    ref = r['reference']
    print(f"\npool {r['pool']}   reference {ref['gauge_mm'][0]}/{ref['gauge_mm'][1]} mm   "
          f"width median {ref['width_mm_median']:.4f} mm   "
          f"label drift {ref['label_drift']:.2e}")
    d = ref['band_depth_px']
    print(f"  complete circ band in {ref['banded']} cells, depth {d['min']}-{d['max']} px, "
          f"median {d['median']:.0f} px = {ref['band_depth_mm_median']:.4f} mm")
    for a in r['arms']:
        s, b = a['survival'], a['circumferential_band']
        print(f"\n  offset {a['offset_px']} px ({a['offset_mm_per_side']:.4f} mm/side) -> "
              f"nominal {a['gauge_mm'][0]:.4f}/{a['gauge_mm'][1]:.4f} mm, "
              f"width median {a['width_mm_median']:.4f} mm")
        print(f"    alive {s['alive']}/{r['pool']}   winners {s['winners_alive']}/"
              f"{s['winners']}   fully valid {s['valid']}   "
              f"complete circ band {b['kept']}/{b['reference']} kept")
        print(f"    ranked on {s['alive_both']} cells alive under both gauges")
        for key, d in a['spearman'].items():
            print(f"      {key:<12} spearman {d['spearman']:+.3f}  p {d['p_value']:.2e}  "
                  f"median {a['median_change'][key]:+.1%}")
        f = a['front_overlap']
        print(f"      front        {f['reference']} -> {f['light']} cells, {f['shared']} shared, "
              f"jaccard {f['jaccard']:.3f}, kept {f['kept']:.3f}")
        for d in a['top_k_overlap'].values():
            print(f"      top-{d['k']:<3} overlap {d['overlap']:.3f}  ({d['shared']}/{d['k']})")
    print(f"\n  {r['seconds']:.0f}s")

if __name__ == '__main__':
    main()
