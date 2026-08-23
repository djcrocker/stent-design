"""
Label a large pool in parallel, then subsample it for coverage in `y`.

Generation is cheap and labeling is not, so the pool is the thing to over-produce, and 
labeling is the thing to parallelize.

Here, we subsample for coverage in `y` rather than fix a source mix. A ratio would be a guess
about both yield and about where each source lands in `y`. The sources overlap heavily in
objective space even though they look nothing alike geometrically. Binning `y` and drawing
round-robin across occupied bins targets the spread we actually need, and self-corrects
when a source turns out to pile up where another already sits.
"""

import json
import multiprocessing as mp
import pathlib
import time

import numpy as np

import config
from geom.cell import UnitCell

Y_KEYS = ('K_radial', 'eps_a_max', 'A_over_lim', 'f_metal')
LOG_KEYS = ('K_radial',)
DATA_DIR = config.DATA_DIR

def label_one(arr):
    """Label a single cell."""
    from sim2d.fatigue import fatigue
    from sim2d.homogenize import homogenize

    cell = UnitCell(np.asarray(arr, dtype=bool))
    h = homogenize(cell)
    f = fatigue(cell, h)
    return {'K_radial': float(h.K_radial), 'eps_a_max': float(f.eps_a_max),
            'eps_a_p99': float(f.eps_a_p99), 'A_over_lim': float(f.A_over_lim),
            'f_metal': float(cell.f_metal)}

def label_chunk(arrs):
    return [label_one(a) for a in arrs]

def label_parallel(cells, workers=4, chunk=250, progress=True):
    """
    Label every cell across `workers` processes.

    Cells ship as raw bool arrays rather than UnitCell objects: arrays pickle compactly and
    the worker rebuilds the object, which keeps the parent from paying object overhead on
    every one of 100,000 items.
    """
    arrs = [c.to_array() if hasattr(c, 'to_array') else np.asarray(c, bool) for c in cells]
    chunks = [arrs[i:i + chunk] for i in range(0, len(arrs), chunk)]
    out, done, t0 = [], 0, time.time()

    if workers <= 1:
        for c in chunks:
            out.extend(label_chunk(c))
        return out

    with mp.Pool(processes=workers) as pool:
        for result in pool.imap(label_chunk, chunks):
            out.extend(result)
            done += len(result)
            if progress:
                rate = done / max(time.time() - t0, 1e-9)
                left = (len(arrs) - done) / max(rate, 1e-9)
                print(f'    labeled {done}/{len(arrs)}  {rate:.0f} cells/s  '
                      f'~{left / 60:.1f} min left', flush=True)
    return out

def bin_indices(labels, n_bins=8, keys=Y_KEYS):
    """Bin each cell into a cell of the `y` grid. Returns an (N, len(keys)) int array."""
    idx = np.empty((len(labels), len(keys)), dtype=np.int32)
    for c, key in enumerate(keys):
        v = np.array([row[key] for row in labels], float)
        if key in LOG_KEYS:
            v = np.log10(np.maximum(v, 1e-12))
        lo, hi = np.nanmin(v), np.nanmax(v)
        if not np.isfinite(lo) or hi <= lo:
            idx[:, c] = 0
            continue
        edges = np.linspace(lo, hi, n_bins + 1)[1:-1]
        idx[:, c] = np.digitize(v, edges)
    return idx

def coverage_subsample(labels, n_target, n_bins=8, keys=Y_KEYS, seed=0):
    """Draw round-robin across occupied `y` bins until `n_target` cells are taken."""
    if n_target >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    idx = bin_indices(labels, n_bins=n_bins, keys=keys)

    buckets = {}
    for i, key in enumerate(map(tuple, idx)):
        buckets.setdefault(key, []).append(i)
    for members in buckets.values():
        rng.shuffle(members)

    order = list(buckets)
    rng.shuffle(order)
    chosen, exhausted = [], set()
    while len(chosen) < n_target and len(exhausted) < len(order):
        for key in order:
            if key in exhausted:
                continue
            members = buckets[key]
            if not members:
                exhausted.add(key)
                continue
            chosen.append(members.pop())
            if len(chosen) >= n_target:
                break
    return np.array(sorted(chosen))

def save(cells, labels, meta, out_dir=None, stem='dataset'):
    """Cells as packed bits (512 bytes each), labels as parquet."""
    import pandas as pd
    from diffusion import sampler

    out_dir = DATA_DIR if out_dir is None else pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    packed, shape = sampler.pack(cells)
    np.savez_compressed(out_dir / f'{stem}_cells.npz', packed=packed, shape=np.array(shape))
    frame = pd.DataFrame(labels)
    frame.insert(0, 'cell_index', np.arange(len(cells)))
    frame['source'] = [m['source'] for m in meta]
    frame['change_fraction'] = [m['change_fraction'] for m in meta]
    frame.to_parquet(out_dir / f'{stem}.parquet', index=False)
    return {'cells': str(out_dir / f'{stem}_cells.npz'),
            'labels': str(out_dir / f'{stem}.parquet'),
            'n': len(cells), 'grid_n': int(shape[1])}

def load(out_dir=None, stem='dataset'):
    import pandas as pd
    from diffusion import sampler

    out_dir = DATA_DIR if out_dir is None else pathlib.Path(out_dir)
    blob = np.load(out_dir / f'{stem}_cells.npz')
    arrs = sampler.unpack(blob['packed'], tuple(blob['shape']))
    frame = pd.read_parquet(out_dir / f'{stem}.parquet')
    return arrs, frame

def build(n_pool=100_000, n_target=50_000, workers=4, seed=0, n_bins=8, out_dir=None, stem='dataset'):
    """Generate a pool, label it in parallel, subsample for coverage."""
    from diffusion import sampler

    t0 = time.time()
    print(f'[1/4]: Generating a pool of {n_pool:,} valid cells', flush=True)
    cells, meta, stats = sampler.generate_pool(n_pool, seed=seed, progress=max(1, n_pool // 20))
    print(f'      {len(cells):,} valid from {stats["total_attempts"]:,} attempts '
          f'({time.time() - t0:.0f} s)', flush=True)
    for source in sampler.SOURCES:
        if stats['attempts'].get(source):
            print(f'        {source:13} yield {100 * stats["yield"][source]:5.1f}%  '
                  f'kept {stats["kept"][source]:,}')

    print(f'[2/4]: Labeling {len(cells):,} cells on {workers} workers', flush=True)
    t1 = time.time()
    labels = label_parallel(cells, workers=workers)
    print(f'      labeled in {(time.time() - t1) / 60:.1f} min', flush=True)

    print(f'[3/4]: Subsampling to {n_target:,} for coverage in y', flush=True)
    keep = coverage_subsample(labels, n_target, n_bins=n_bins, seed=seed)
    cells = [cells[i] for i in keep]
    labels = [labels[i] for i in keep]
    meta = [meta[i] for i in keep]
    print(f'      kept {len(keep):,}', flush=True)

    print('[4/4]: Saving', flush=True)
    info = save(cells, labels, meta, out_dir=out_dir, stem=stem)
    info['pool'] = {'kept': int(len(keep)), 'requested': n_pool, 'stats': stats}
    info['elapsed_min'] = (time.time() - t0) / 60
    out = (DATA_DIR if out_dir is None else pathlib.Path(out_dir)) / f'{stem}_info.json'
    out.write_text(json.dumps(info, indent=2, default=str), encoding='utf-8')
    print(f'      {info["n"]:,} cells -> {info["cells"]}')
    print(f'      labels           -> {info["labels"]}')
    print(f'      total {info["elapsed_min"]:.1f} min')
    return info

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description='Dataset build')
    ap.add_argument('--pool', type=int, default=100_000)
    ap.add_argument('--target', type=int, default=50_000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--bins', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--stem', default='dataset')
    args = ap.parse_args()
    build(n_pool=args.pool, n_target=args.target, workers=args.workers,
          seed=args.seed, n_bins=args.bins, stem=args.stem)
