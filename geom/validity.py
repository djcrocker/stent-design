"""
Validity checker.

Defines what counts as an acceptable stent unit cell.

Three criteria, all evaluated on the torus:

  Connectivity:  One connected component, no islands floating free of the structure
  Wrapping:      A load path has to run all the way around the circumference and all the way
                 along the axis.
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
VOID_THIN_FEATURE = 'void_thin_feature'
TOO_SPARSE = 'too_sparse'
TOO_DENSE = 'too_dense'
EMPTY = 'empty'

# Every reason `check` can return, so downstream caches can fingerprint the envelope.
CRITERIA = (DISCONNECTED, NO_WRAP_CIRC, NO_WRAP_AXIAL, THIN_FEATURE, VOID_THIN_FEATURE, TOO_SPARSE, TOO_DENSE, EMPTY)

# Fraction of material that opening may remove before a cell is called too thin. Non-zero
# because discretising a curved or diagonal edge always removes a few pixels at corners.
THIN_TOLERANCE = 0.02

# Same allowance on the void side.
VOID_THIN_TOLERANCE = 0.02

def disk(radius_px):
    """Disk structuring element of the given radius, in pixels."""
    r = int(np.ceil(radius_px))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= radius_px ** 2

def min_feature_radius_px():
    """Radius of the structuring element that defines MIN_FEATURE_MM, in pixels."""
    return (config.MIN_FEATURE_MM / 2.0) / config.mm_per_px()[0]

def void_thin_fraction(arr, radius_px=None, structure=None):
    """How much of the void is narrower than the minimum feature."""
    arr = np.asarray(arr, dtype=bool)
    radius_px = min_feature_radius_px() if radius_px is None else radius_px
    closed = periodic.closing(arr, structure=disk(radius_px))
    void = int((~arr).sum())
    if not void:
        return 0.0
    return float((closed & ~arr).sum()) / float(void)

def has_thin_void(arr, tol=None, radius_px=None, structure=None):
    """True when the cell contains voids below the minimum feature size."""
    tol = VOID_THIN_TOLERANCE if tol is None else tol
    return void_thin_fraction(arr, radius_px, structure) > tol

def wraps(arr, structure=None):
    """
    Does the structure wrap? Returns (circumferential, axial) as bools.

    Tiles 3x3 and labels without wrapping, then asks whether a pixel and its own copy a tile over
    share a label. If they do, a path connects them: a non-contractible loop around that direction of the torus.
    """
    structure = periodic.CONN4 if structure is None else structure
    rows, cols = arr.shape
    lab, _ = ndimage.label(np.tile(arr, (3, 3)), structure=structure)
    center = lab[rows:2 * rows, cols:2 * cols]
    right = lab[rows:2 * rows, 2 * cols:3 * cols]
    down = lab[2 * rows:3 * rows, cols:2 * cols]
    solid = center > 0
    return (bool(np.any(solid & (center == right))),
            bool(np.any(solid & (center == down))))

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

    void_removed = void_thin_fraction(arr, radius_px, structure)
    metrics['void_thin_fraction'] = void_removed
    if void_removed > VOID_THIN_TOLERANCE:
        reasons.append(VOID_THIN_FEATURE)

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