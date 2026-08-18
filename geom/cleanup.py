"""
Cleanup / projection op.

Takes a raw field and either repairs it into a valid cell or says that it can't.

Pipeline:

    Closing:       Fill pinholes and hairline gaps; may join near-touching struts
    Opening:       Remove thin necks and specks that violate the min feature size
    Largest CC:    Sweep up whatever the opening orphaned
    Re-check:      geom.validity has the final word

Order matters: Opening can sever a neck and create new islands, so components must be swept after opening, not before.

What is repairable, and what is not:

    Repairable:    islands, thin necks, specks, pinholes, hairline gaps
    Unfixable:     empty; does not wrap after cleanup; f_metal outside the guard range

Cleanup won't dilate a non-wrapping cell until it connects. Forcing a load path creates 
structure the generator never produced, which would inflate the valid-generation rate 
with designs the model did not actually discover.

`change_fraction` is the fraction of the field cleanup altered, and it should be reported 
with the valid-generation rate: if repair is rewriting a third of every sample, a high valid 
rate is evidence about the post-processor, not about the model.
"""

from dataclasses import dataclass, field

import numpy as np

import config
from geom import periodic, validity
from geom.cell import UnitCell

@dataclass
class Cleanup:
    """Outcome of a repair attempt."""
    cell: object                                    # The repaired UnitCell, or None if unfixable
    fixed: bool                                     # Did we end up with a valid cell?
    validity: object = None                         # The final Validity verdict
    actions: list = field(default_factory=list)     # What the pipeline actually did
    change_fraction: float = 0.0                    # Fraction of pixels cleanup altered - see module docstring

    def __iter__(self):
        return iter((self.cell, self.fixed))

def _disk_radius_px():
    return (config.MIN_FEATURE_MM / 2.0) / config.mm_per_px()[0]

def keep_largest_component(arr, structure=None):
    """Drop everything not part of the biggest connected component (on the torus)."""
    lab, count = periodic.label(arr, structure)
    if count <= 1:
        return arr, 0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax(), count - 1

def clean(field_like, structure=None, radius_px=None):
    """
    Repair a raw field. Returns a Cleanup.

    `field_like` may be a UnitCell or any 2D array; anything nonzero counts as material.
    """
    arr = (field_like.to_array() if hasattr(field_like, 'to_array')
           else np.asarray(field_like, dtype=bool))
    original = arr.copy()
    actions = []

    if not arr.any():
        return Cleanup(None, False, validity.Validity(False, [validity.EMPTY]),
                       ['empty input'], 0.0)

    # Nothing to repair on a cell that is already valid.
    already = validity.check(UnitCell(arr), structure)
    if already.ok:
        return Cleanup(UnitCell(arr), True, already, [], 0.0)

    radius_px = _disk_radius_px() if radius_px is None else radius_px
    se = validity.disk(radius_px)

    closed = periodic.closing(arr, structure=se)
    if not np.array_equal(closed, arr):
        actions.append(f'Closing filled {int(closed.sum() - arr.sum())} px')
    arr = closed

    opened = periodic.opening(arr, structure=se)
    if not np.array_equal(opened, arr):
        actions.append(f'Opening removed {int(arr.sum() - opened.sum())} px')
    arr = opened

    if not arr.any():
        return Cleanup(None, False, validity.Validity(False, [validity.EMPTY]),
                       actions + ['Opening removed everything'],
                       float((original != arr).mean()))

    arr, dropped = keep_largest_component(arr, structure)
    if dropped:
        actions.append(f'Dropped {dropped} island(s)')

    change_fraction = float((original != arr).mean())
    cell = UnitCell(arr)
    verdict = validity.check(cell, structure)

    return Cleanup(cell if verdict.ok else None, verdict.ok, verdict,
                   actions, change_fraction)

def summarize(results):
    """
    Aggregate a batch of Cleanup results.

    The valid-generation rate on its own isn't interpretable. A model that emits noise can
    still score well if repair is allowed to rewrite most of each sample, so the rate must
    always be read next to how much repair it took. `mean_change` and `p90_change` are over
    the fixed results only.
    """
    results = list(results)
    n = len(results)
    fixed = [r for r in results if r.fixed]
    changes = np.array([r.change_fraction for r in fixed]) if fixed else np.array([])

    reasons = {}
    for r in results:
        if not r.fixed and r.validity is not None:
            for code in r.validity.reasons:
                reasons[code] = reasons.get(code, 0) + 1

    return {
        'n': n,
        'n_fixed': len(fixed),
        'valid_rate': len(fixed) / n if n else 0.0,
        'untouched_rate': float((changes == 0).mean()) if len(changes) else 0.0,
        'mean_change': float(changes.mean()) if len(changes) else 0.0,
        'p90_change': float(np.percentile(changes, 90)) if len(changes) else 0.0,
        'max_change': float(changes.max()) if len(changes) else 0.0,
        'failure_reasons': dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }
