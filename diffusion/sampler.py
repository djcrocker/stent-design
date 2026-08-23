"""
Generate unit cells that are valid and spread across the objective space.

The hard part is diversity IN `y`. A conditional diffusion model learns P(topology | y); 
if the training `y` clusters in a narrow band, the model can only generate inside that band,
and we want targets between and beyond the parametric family's reach.

Three sources, pulling against each other on purpose:
  Family       - The parametric sweep. Valid, narrow.
  Perturbation - Family members knocked off the family by pixel noise, morphological ops
                 and local deletions. Valid often enough to be useful, different.
  Random       - Smoothed random fields through S2.4 cleanup. Geometrically diverse, low
                 yield, and the only source that can reach topologies nobody designed.

The mix isn't fixed as a ratio. Ratios would be a guess about yield and about where each
source lands in `y`. Instead: over-generate, label everything (cheap), then subsample with a
cap per `y`-space bin. That targets spread directly and self-corrects when a source turns
out to be unproductive or to pile up where another already sits.
"""

import numpy as np
from scipy import ndimage

import config
from geom import cleanup, parametric

SOURCES = ('family', 'perturbation', 'lattice', 'random')

def perturb(cell, rng, flip_p=0.04, dilate_p=0.35, erode_p=0.35, blob_p=0.5, max_blobs=3):
    """
    Knock a cell off its family without destroying it.

    Three mechanisms, because each leaves a different trace: salt-and-pepper flips roughen
    strut edges, a morphological dilate/erode thickens or thins members wholesale, and blob
    deletion removes material somewhere specific, which is what actually changes the load
    path rather than just the line widths.
    """
    arr = cell.to_array().copy()
    n = arr.shape[0]

    if flip_p > 0:
        arr = np.where(rng.random(arr.shape) < flip_p, ~arr, arr)

    roll = rng.random()
    if roll < dilate_p:
        arr = ndimage.binary_dilation(arr, iterations=int(rng.integers(1, 3)))
    elif roll < dilate_p + erode_p:
        arr = ndimage.binary_erosion(arr, iterations=int(rng.integers(1, 3)))

    if rng.random() < blob_p:
        for _ in range(int(rng.integers(1, max_blobs + 1))):
            r = int(rng.integers(max(2, n // 24), max(3, n // 8)))
            ci, cj = rng.integers(0, n, 2)
            ii, jj = np.ogrid[:n, :n]
            # Wrapped distance: the cell is a torus, so a blob near an edge has to bite the
            # opposite side too, or the deletion is not periodic.
            di = np.minimum(np.abs(ii - ci), n - np.abs(ii - ci))
            dj = np.minimum(np.abs(jj - cj), n - np.abs(jj - cj))
            arr[(di ** 2 + dj ** 2) <= r ** 2] = False
    return arr

def random_field(rng, n=None, sigma_px=None, target_f_metal=None):
    """
    A smoothed random field thresholded to a target metal fraction.

    Thresholding on a quantile rather than a fixed level is what makes the yield tolerable.
    """
    n = config.GRID_N if n is None else n
    sigma = float(rng.uniform(1.5, 5.0)) if sigma_px is None else sigma_px
    if target_f_metal is None:
        target_f_metal = float(rng.uniform(config.F_METAL_MIN + 0.03,
                                           config.F_METAL_MAX - 0.05))
    field = rng.normal(size=(n, n))
    # wrap mode keeps the smoothing periodic, so the field is torus-compatible from the start
    field = ndimage.gaussian_filter(field, sigma=sigma, mode='wrap')
    return field >= np.quantile(field, 1.0 - target_f_metal)

def to_valid_cell(field_like, structure=None):
    """S2.4 cleanup then the S2.3 check. Returns (cell, change_fraction) or (None, None)."""
    result = cleanup.clean(field_like, structure)
    if not result.fixed or result.cell is None:
        return None, None
    return result.cell, result.change_fraction

def generate_pool(n_target, weights=None, n=None, seed=0, max_attempts_factor=6, progress=None):
    """Over-generate a pool of valid cells across all three sources."""
    n = config.GRID_N if n is None else n
    rng = np.random.default_rng(seed)
    # The random source is kept at a low weight to ensure a diverse pool of valid cells.
    weights = ({'family': 0.15, 'perturbation': 0.35, 'lattice': 0.45, 'random': 0.05}
               if weights is None else weights)
    names = list(weights)
    probs = np.array([weights[k] for k in names], float)
    probs = probs / probs.sum()

    sweep, _ = parametric.sweep()
    family_pool = [cell for _, cell in sweep]
    if not family_pool:
        raise RuntimeError('parametric sweep produced no valid cells')

    cells, meta = [], []
    attempts = {k: 0 for k in names}
    kept = {k: 0 for k in names}
    limit = int(n_target * max_attempts_factor)
    total = 0

    while len(cells) < n_target and total < limit:
        source = names[int(rng.choice(len(names), p=probs))]
        attempts[source] += 1
        total += 1

        if source == 'family':
            try:
                raw = parametric.crown(
                    strut_width_mm=parametric.snap_width_mm(
                        float(rng.uniform(config.MIN_FEATURE_MM, 0.25)), n=n),
                    crown_amplitude=float(rng.uniform(0.12, 0.42)),
                    n_periods=int(rng.integers(1, 4)), n=n).to_array()
            except Exception:
                continue
        elif source == 'perturbation':
            raw = perturb(family_pool[int(rng.integers(len(family_pool)))], rng)
        elif source == 'lattice':
            raw = random_lattice(rng, n=n)
        else:
            raw = random_field(rng, n=n)

        cell, change = to_valid_cell(raw)
        if cell is None:
            continue
        cells.append(cell)
        meta.append({'source': source, 'change_fraction': float(change)})
        kept[source] += 1
        if progress and len(cells) % progress == 0:
            print(f'    {len(cells)}/{n_target} valid  ({total} attempts)')

    stats = {'attempts': attempts, 'kept': kept, 'total_attempts': total,
             'yield': {k: (kept[k] / attempts[k] if attempts[k] else 0.0) for k in names}}
    return cells, meta, stats

def pack(cells):
    """Packed bits: a 64x64 bool cell is 512 bytes, so 50k cells is ~26 MB."""
    arr = np.stack([c.to_array() for c in cells]).astype(bool)
    return np.packbits(arr, axis=None), arr.shape

def unpack(packed, shape):
    total = int(np.prod(shape))
    return np.unpackbits(packed, count=total).reshape(shape).astype(bool)

def random_lattice(rng, n=None, n_seeds=None, k_neighbors=None, width_px=None):
    """
    A connected strut network built by construction, not by thresholding noise.

    `random_field` almost never survives validity, and the reason is structural: a thresholded 
    random field is a site-percolation problem, whose 2D threshold is about 0.59 metal fraction,
    while `F_METAL_MAX` is 0.50. Below threshold the field is islands, so it can't wrap.

    So, we build the skeleton first: scatter seeds on the torus, join each to its nearest
    neighbors under wrapped distance, rasterize those segments with wrap, then thicken to a
    strut width. Connectivity and wrapping come from the construction; `f_metal` comes from
    the thickening. This is how lattice metamaterials are generated, and it reaches
    topologies the parametric family can't.
    """
    n = config.GRID_N if n is None else n
    n_seeds = int(rng.integers(5, 13)) if n_seeds is None else n_seeds
    k = int(rng.integers(3, 7)) if k_neighbors is None else k_neighbors
    if width_px is None:
        width_px = int(rng.integers(max(1, int(config.min_feature_px() / 2)),
                                    max(2, int(config.min_feature_px()) + 2)))

    pts = rng.integers(0, n, size=(n_seeds, 2)).astype(int)
    arr = np.zeros((n, n), dtype=bool)

    for i in range(n_seeds):
        delta = pts - pts[i]
        # Minimal image: the shortest way to a neighbor may be across the seam.
        delta = delta - n * np.round(delta / n).astype(int)
        dist = (delta ** 2).sum(axis=1)
        for j in np.argsort(dist)[1:k + 1]:
            d = delta[j]
            steps = int(max(abs(d[0]), abs(d[1]))) * 2 + 1
            t = np.linspace(0.0, 1.0, steps)
            ii = np.round(pts[i, 0] + t * d[0]).astype(int) % n
            jj = np.round(pts[i, 1] + t * d[1]).astype(int) % n
            arr[ii, jj] = True

    from geom import periodic
    return periodic.dilation(arr, iterations=max(1, width_px))
