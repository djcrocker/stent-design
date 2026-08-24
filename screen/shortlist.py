"""
Filter the generated pool to valid designs, rank them, and cut a shortlist.

Ranking is by NON-DOMINATED SORTING with crowding-distance tie-breaking, on two objectives:
`K_radial` maximized and `A_over_lim` minimized.
"""

import argparse
import json

import numpy as np

import config
from diffusion import dataset

RESULTS_DIR = config.PROJECT_ROOT / 'screen' / 'results'
DEFAULT_K = 24
OBJECTIVES = ('K_radial', 'A_over_lim')

def _as_minimization(objectives):
    """Both columns as things to minimize, with `K_radial` on a log scale."""
    k = np.log10(np.maximum(objectives[:, 0], 1e-12))
    return np.column_stack([-k, objectives[:, 1]])

def nondominated_layers(objectives):
    """Peel successive non-dominated fronts. Returns a list of index arrays, best first."""
    f = _as_minimization(np.asarray(objectives, float))
    remaining = np.arange(len(f))
    layers = []
    while len(remaining):
        sub = f[remaining]
        keep = []
        for i in range(len(remaining)):
            worse_or_equal = np.all(sub <= sub[i], axis=1)
            strictly_better = np.any(sub < sub[i], axis=1)
            if not np.any(worse_or_equal & strictly_better):
                keep.append(i)
        layers.append(remaining[keep])
        remaining = np.delete(remaining, keep)
    return layers

def crowding_distance(objectives, members):
    """
    NSGA-II crowding distance within one layer.

    Boundary designs get infinity so the extremes of the front are always kept; interior
    designs are scored by the normalized gap to their neighbors, so a shortlist spreads
    along the trade-off instead of clustering where the model happened to sample densely.
    """
    members = np.asarray(members)
    f = _as_minimization(np.asarray(objectives, float))[members]
    n = len(members)
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    for c in range(f.shape[1]):
        order = np.argsort(f[:, c], kind='mergesort')
        col = f[order, c]
        spread = col[-1] - col[0]
        dist[order[0]] = dist[order[-1]] = np.inf
        if spread <= 0:
            continue
        dist[order[1:-1]] += (col[2:] - col[:-2]) / spread
    return dist

def select(objectives, k=DEFAULT_K):
    """
    Take whole layers until one overflows, then fill from it by crowding distance.

    Returns (chosen indices, layer number per chosen design), ordered best layer first.
    """
    objectives = np.asarray(objectives, float)
    if k >= len(objectives):
        layers = nondominated_layers(objectives)
        rank = np.empty(len(objectives), int)
        for n, layer in enumerate(layers, 1):
            rank[layer] = n
        return np.arange(len(objectives)), rank

    chosen, ranks = [], []
    for n, layer in enumerate(nondominated_layers(objectives), 1):
        room = k - len(chosen)
        if len(layer) <= room:
            chosen.extend(layer.tolist())
            ranks.extend([n] * len(layer))
        else:
            d = crowding_distance(objectives, layer)
            # Descending crowding distance: sparse regions of the front first.
            pick = layer[np.argsort(-d, kind='mergesort')[:room]]
            chosen.extend(pick.tolist())
            ranks.extend([n] * len(pick))
        if len(chosen) >= k:
            break
    return np.array(chosen), np.array(ranks)

def build(k=DEFAULT_K, source_stem='s9_1_generated', out_stem='s9_2_shortlist'):
    """Load the generated pool, label it, filter to valid, rank, and save the shortlist."""
    from diffusion import generate, splits
    from diffusion.fidelity import label_fields

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fields, which, meta = generate.load_samples(source_stem)
    rows, dropped = label_fields(fields)

    valid = np.array([i for i, r in enumerate(rows) if r is not None])
    if not len(valid):
        raise RuntimeError('no valid designs in the pool')

    groups = splits.group_ids(fields[valid])
    _, first = np.unique(groups, return_index=True)
    unique = valid[np.sort(first)]

    objectives = np.array([[rows[i][key] for key in OBJECTIVES] for i in unique], float)
    chosen_local, layer_of = select(objectives, k=k)
    chosen = unique[chosen_local]

    targets = meta['targets']
    entries = []
    for slot, (idx, layer) in enumerate(zip(chosen, layer_of), 1):
        r = rows[idx]
        t = targets[int(which[idx])]
        entries.append({
            'slot': slot,
            'pool_index': int(idx),
            'layer': int(layer),
            'target_name': t['name'],
            'target_asked': t['asked'],
            **{key: float(r[key]) for key in dataset.Y_KEYS},
            'eps_a_p99': float(r['eps_a_p99']),
        })
    # Best layer first, then stiffest within a layer.
    entries.sort(key=lambda e: (e['layer'], -e['K_radial']))
    for slot, e in enumerate(entries, 1):
        e['slot'] = slot

    summary = {
        'k': int(len(entries)),
        'objectives': {'maximize': OBJECTIVES[0], 'minimize': OBJECTIVES[1]},
        'rule': 'non-dominated sorting, crowding-distance tie-break',
        'pool_sampled': int(len(fields)),
        'pool_valid': int(len(valid)),
        'pool_distinct': int(len(unique)),
        'duplicates_removed': int(len(valid) - len(unique)),
        'layers_used': int(max(e['layer'] for e in entries)),
        # We measured top-K retention at 83-90% in this K range, so roughly 10-17% of the
        # actual best designs are missed because the 2D screen ranks them wrong. That's
        # the price of the two-tier architecture, and it belongs in the record.
        'expected_top_k_retention': [0.83, 0.90],
        'shortlist': entries,
    }
    (RESULTS_DIR / f'{out_stem}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    np.savez_compressed(RESULTS_DIR / f'{out_stem}_cells.npz',
                        fields=np.packbits(fields[chosen], axis=None),
                        shape=np.array(fields[chosen].shape),
                        pool_index=chosen)
    return summary

def load(out_stem='s9_2_shortlist'):
    summary = json.loads((RESULTS_DIR / f'{out_stem}.json').read_text(encoding='utf-8'))
    blob = np.load(RESULTS_DIR / f'{out_stem}_cells.npz')
    shape = tuple(blob['shape'])
    fields = np.unpackbits(blob['fields'], count=int(np.prod(shape))).reshape(shape).astype(bool)
    return fields, summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Rank and shortlist')
    ap.add_argument('-k', type=int, default=DEFAULT_K)
    ap.add_argument('--source', default='s9_1_generated')
    ap.add_argument('--stem', default='s9_2_shortlist')
    args = ap.parse_args()
    out = build(k=args.k, source_stem=args.source, out_stem=args.stem)

    print(f"pool {out['pool_sampled']} sampled -> {out['pool_valid']} valid -> "
          f"{out['pool_distinct']} distinct")
    print(f"Shortlist {out['k']} designs across {out['layers_used']} non-dominated layers")
    lo, hi = out['expected_top_k_retention']
    print(f"Expected top-K retention {100 * lo:.0f}-{100 * hi:.0f}%, so roughly "
          f"{100 * (1 - hi):.0f}-{100 * (1 - lo):.0f}% of the true best are missed")
    print()
    print(f"{'#':>3} {'layer':>5} {'K_radial':>9} {'A_over_lim':>11} {'eps_a_max':>10} "
          f"{'f_metal':>8}  target")
    for e in out['shortlist']:
        print(f"{e['slot']:3d} {e['layer']:5d} {e['K_radial']:9.1f} {e['A_over_lim']:11.4f} "
              f"{e['eps_a_max']:10.4f} {e['f_metal']:8.3f}  {e['target_name']}")
