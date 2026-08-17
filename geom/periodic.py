"""
Torus-aware primitives.

Unit cell wraps: its left edge meets its right, its top meets its bottom. As a result, the 
field lives on a torus, and every neighbourhood operation has to know that. Here, we make 
the operations periodic, so periodicity can't be violated.

The invariant that matters here is shift equivariance, where rolling the input must roll
the output, `op(roll(x, k)) == roll(op(x), k)`, for every shift.
"""

import numpy as np
from scipy import ndimage

# 4-connectivity (edges only) and 8-connectivity (edges + corners).
CONN4 = ndimage.generate_binary_structure(2, 1)
CONN8 = ndimage.generate_binary_structure(2, 2)

def _pad_wrap(arr, pad):
    return np.pad(arr, pad, mode='wrap')

def _crop(arr, pad):
    return arr[pad:-pad, pad:-pad] if pad else arr

def erosion(arr, structure=None, iterations=1):
    """Binary erosion on the torus."""
    structure = CONN4 if structure is None else structure
    pad = max(structure.shape) * iterations
    out = ndimage.binary_erosion(_pad_wrap(arr, pad), structure, iterations=iterations)
    return _crop(out, pad)

def dilation(arr, structure=None, iterations=1):
    """Binary dilation on the torus."""
    structure = CONN4 if structure is None else structure
    pad = max(structure.shape) * iterations
    out = ndimage.binary_dilation(_pad_wrap(arr, pad), structure, iterations=iterations)
    return _crop(out, pad)

def opening(arr, structure=None, iterations=1):
    """Erosion then dilation, which removes thin necks and specks."""
    return dilation(erosion(arr, structure, iterations), structure, iterations)

def closing(arr, structure=None, iterations=1):
    """Dilation then erosion, which fills pinholes and hairline gaps."""
    return erosion(dilation(arr, structure, iterations), structure, iterations)

def distance_transform(arr, sampling=None):
    """Euclidean distance from each True pixel to the nearest False pixel on the torus."""
    n = arr.shape[0]
    # Half the grid covers any distance possible on a torus of this size.
    pad = n // 2 + 1
    out = ndimage.distance_transform_edt(_pad_wrap(arr, pad), sampling=sampling)
    return _crop(out, pad)

def label(arr, structure=None):
    """
    Connected components on the torus. Returns (labels, count).

    scipy's `label` treats the array as a rectangle, so a strut running off the right edge
    and back in on the left is counted twice.
    """
    structure = CONN4 if structure is None else structure
    lab, count = ndimage.label(arr, structure=structure)
    if count <= 1:
        return lab, count

    # Neighbor offsets implied by the structure, taken once per direction 
    # (the opposite direction is the same pair seen from the other side).
    offsets = [(0, 1), (1, 0)]
    if structure[0, 0]:  # 8-connectivity includes the diagonals
        offsets += [(1, 1), (1, -1)]

    parent = np.arange(count + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for di, dj in offsets:
        rolled = np.roll(lab, shift=(-di, -dj), axis=(0, 1))
        both = (lab > 0) & (rolled > 0)
        for a, b in set(zip(lab[both].tolist(), rolled[both].tolist())):
            union(a, b)

    # Compact the merged labels back to 1..k.
    roots = np.array([find(i) for i in range(count + 1)])
    uniq = np.unique(roots[1:])
    remap = np.zeros(count + 1, dtype=int)
    for new, old in enumerate(uniq, start=1):
        remap[roots == old] = new
    remap[0] = 0
    return remap[lab], len(uniq)

def is_shift_equivariant(op, arr, shifts=((1, 0), (0, 1), (7, 13), (-5, 3))):
    """Check if `op` commutes with rolling, the shift-equivariance property of a periodic operator."""
    base = op(arr)
    for di, dj in shifts:
        rolled = op(np.roll(arr, (di, dj), axis=(0, 1)))
        if not np.allclose(rolled, np.roll(base, (di, dj), axis=(0, 1))):
            return False
    return True
