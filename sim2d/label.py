"""
label(cell) -> y : the objective vector.

Returns the vector y = [K_radial, eps_a_max, A_over_lim, f_metal], ordered by
config.OBJECTIVE_KEYS rather than hard-coded, so the vector can't drift from the config
that names it.

Notes:

  - An INVALID cell returns a Label with valid=False and y=None.
  - label() doesn't repair.
  - Both eps_a_max and eps_a_p99 are recorded. y carries eps_a_max per S0.5, but the p99 is
    kept alongside because the voxel mesh inflates the raw max by a 1.75-2x.
  - One homogenisation is computed and reused for stiffness and fatigue; about half the
    work of computing each separately.
"""

import time
from dataclasses import dataclass, field

import numpy as np

import config
from geom import validity
from sim2d.fatigue import fatigue
from sim2d.homogenize import homogenize

UNITS = {
    'K_radial': 'N/mm^3',
    'eps_a_max': '-',        # Strain amplitude, dimensionless
    'A_over_lim': '-',       # Area fraction
    'f_metal': '-',          # Area fraction
}

@dataclass
class Label:
    """One cell's objective vector, plus what it took to get there."""
    valid: bool
    reasons: list = field(default_factory=list)
    y: np.ndarray = None
    metrics: dict = field(default_factory=dict)
    seconds: float = 0.0

    def __iter__(self):
        """Unpacks as (y, valid)."""
        return iter((self.y, self.valid))

    def as_dict(self):
        """
        Returns a flat dictionary representation of the label.

        On an invalid cell the three FE-derived components are None, but `f_metal` is
        still populated.
        """
        row = {'valid': self.valid,
               'reasons': ','.join(self.reasons),
               'seconds': self.seconds}
        row.update(self.metrics)
        if self.y is None:
            row.update({k: None for k in config.OBJECTIVE_KEYS})
            row['f_metal'] = self.metrics.get('f_metal')
        else:
            row.update({k: float(v) for k, v in zip(config.OBJECTIVE_KEYS, self.y)})
        return row

def label(cell, check_validity=True):
    """Objective vector for one cell. Returns a Label."""
    started = time.perf_counter()

    if check_validity:
        verdict = validity.check(cell)
        if not verdict.ok:
            return Label(False, list(verdict.reasons), None,
                         {'f_metal': cell.f_metal},
                         time.perf_counter() - started)

    h = homogenize(cell)
    f = fatigue(cell, h)

    values = {
        'K_radial': h.K_radial,
        'eps_a_max': f.eps_a_max,
        'A_over_lim': f.A_over_lim,
        'f_metal': cell.f_metal,
    }
    y = np.array([values[k] for k in config.OBJECTIVE_KEYS], dtype=float)

    metrics = {
        'eps_a_p99': f.eps_a_p99,
        'E_circ': h.E_circ,
        'E_axial': h.E_axial,
        'attenuation': f.attenuation,
        'n_elements': int(cell.to_array().sum()),
    }
    return Label(True, [], y, metrics, time.perf_counter() - started)

def describe(result):
    """Human-readable label, with units."""
    lines = []
    if not result.valid:
        return f'INVALID: {", ".join(result.reasons)}  ({result.seconds * 1000:.0f} ms)'
    for key, value in zip(config.OBJECTIVE_KEYS, result.y):
        lines.append(f'  {key:12} {value:12.6f}  {UNITS[key]}')
    lines.append(f'  {"seconds":12} {result.seconds:12.3f}  s')
    return '\n'.join(lines)
