"""
Validity checker.

Defines what counts as an acceptable stent unit cell.

Three criteria, all evaluated on the torus (see geom/periodic.py):


  Connectivity:  One connected component - no islands floating free of the structure
  Wrapping:      A load path must run all the way around the circumference and all the way
                 along the axis. On a torus "spans both axes" can't mean "touches all four
                 edges", because every column is interior; it means the component contains a
                 non-contractible loop in each direction. Circumferentially that is radial
                 support; axially it is what stops the stent falling apart into loose rings.
  Min feature:   Nothing thinner than MIN_FEATURE_MM, checked by morphological opening with
                 a disk of radius w/2, the standard length-scale test in topology
                 optimization, since a solid admits a min feature size 2r when
                 opening by a disk of radius r leaves it unchanged.

Additionally, f_metal bounds, which are degeneracy guards rather than a design constraint.

Periodicity isn't a criterion, as it's enforced by construction.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

import config
from geom import periodic

# Reason codes.
DISCONNECTED = 'disconnected'
NO_WRAP_CIRC = 'no_wrap_circ'
NO_WRAP_AXIAL = 'no_wrap_axial'
THIN_FEATURE = 'thin_feature'
TOO_SPARSE = 'too_sparse'
TOO_DENSE = 'too_dense'
EMPTY = 'empty'

# Fraction of material that opening may remove before a cell is called too thin. Non-zero
# because discretising a curved or diagonal edge always removes a few pixels at corners.
THIN_TOLERANCE = 0.02

def disk(radius_px):
    """Disk structuring element of the given radius, in pixels."""
    r = int(np.ceil(radius_px))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= radius_px ** 2

def wraps(arr, structure=None):
    """
    Does the structure wrap? Returns (circumferential, axial) as bools.

    Tiles 3x3 and labels without wrapping, then asks whether a pixel and its own copy a tile over
    share a label. If they do, a path connects them: a non-contractible loop around that direction of the torus.
    """
    structure = periodic.CONN4 if structure is None else structure
    n = arr.shape[0]
    lab, _ = ndimage.label(np.tile(arr, (3, 3)), structure=structure)
    centre = lab[n:2 * n, n:2 * n]
    right = lab[n:2 * n, 2 * n:3 * n]
    down = lab[2 * n:3 * n, n:2 * n]
    solid = centre > 0
    return (bool(np.any(solid & (centre == right))),
            bool(np.any(solid & (centre == down))))

@dataclass
class Validity:
    """Outcome of a validity check."""
    ok: bool
    reasons: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def __iter__(self):
        """Unpacks as (ok, reasons), the S2.3 signature."""
        return iter((self.ok, self.reasons))

def check(cell, structure=None):
    """Full validity check. Returns a Validity with reasons and measured metrics."""
    arr = cell.to_array() if hasattr(cell, 'to_array') else np.asarray(cell, dtype=bool)
    reasons = []
    metrics = {}

    f_metal = float(arr.mean())
    metrics['f_metal'] = f_metal

    if not arr.any():
        return Validity(False, [EMPTY], metrics)

    # Connectivity on the torus.
    _, n_components = periodic.label(arr, structure)
    metrics['n_components'] = n_components
    if n_components != 1:
        reasons.append(DISCONNECTED)

    # Wrapping in both directions.
    wrap_circ, wrap_axial = wraps(arr, structure)
    metrics['wrap_circ'] = wrap_circ
    metrics['wrap_axial'] = wrap_axial
    if not wrap_circ:
        reasons.append(NO_WRAP_CIRC)
    if not wrap_axial:
        reasons.append(NO_WRAP_AXIAL)

    # Minimum feature size.
    mm_px = config.mm_per_px()[0]
    radius_px = (config.MIN_FEATURE_MM / 2.0) / mm_px
    opened = periodic.opening(arr, structure=disk(radius_px))
    removed = float((arr & ~opened).sum()) / float(arr.sum())
    metrics['min_feature_radius_px'] = radius_px
    metrics['thin_fraction'] = removed
    if removed > THIN_TOLERANCE:
        reasons.append(THIN_FEATURE)

    # Coverage guards.
    if f_metal < config.F_METAL_MIN:
        reasons.append(TOO_SPARSE)
    if f_metal > config.F_METAL_MAX:
        reasons.append(TOO_DENSE)

    return Validity(not reasons, reasons, metrics)

def is_valid(cell, structure=None):
    """(bool, reasons)."""
    result = check(cell, structure)
    return result.ok, result.reasons