"""
The canonical reference cell.

Chosen for clinical realism. `f_metal = 0.248` sits in the conventional self-expanding-nitinol 
band (~0.19-0.26); the exact middle of the parameter grid (w = 0.1718) would give 0.287, 
already outside it.

Known caveat: `n_periods = 1` gives 12 crowns per ring, the low end of Kamenskiy's 16-32
struts scaled to our 6 mm diameter. `n_periods = 2` would be more central but pushes
coverage to 0.288. Another face of the same envelope question.
"""

import json

import numpy as np

import config
from geom.cell import UnitCell
from geom.parametric import crown

PARAMS = {
    'strut_width_mm': 0.1473,
    'crown_amplitude': 0.25,
    'n_periods': 1,
}

# config values the geometry actually depends on.
FINGERPRINT_KEYS = ('GRID_N', 'N_CIRC', 'D_DEPLOYED_MM', 'AXIAL_PITCH_MM',
                    'MIN_FEATURE_MM')

NPY_PATH = config.DATA_DIR / 'reference_cell.npy'
JSON_PATH = config.DATA_DIR / 'reference_cell.json'
PNG_PATH = config.FIG_DEV_DIR / 'reference_cell.png'

def fingerprint():
    """The config values this cell was built from."""
    return {k: getattr(config, k) for k in FINGERPRINT_KEYS}

def build():
    """Generate the reference cell from PARAMS and the current config."""
    return crown(**PARAMS)

def save():
    """Write the array, the recipe and the fingerprint. Returns the cell."""
    cell = build()
    NPY_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(NPY_PATH, cell.to_array())
    JSON_PATH.write_text(json.dumps({
        'params': PARAMS,
        'config': fingerprint(),
        'f_metal': cell.f_metal,
        'note': 'Regenerate with geom.reference.build().',
    }, indent=2), encoding='utf-8')
    return cell

def load(check_fingerprint=True):
    """
    Load the saved reference cell.

    Raises if the stored fingerprint no longer matches config - which means a constant the
    geometry depends on has moved and the saved array is stale. Regenerate with `save()`
    rather than suppressing this.
    """
    if not NPY_PATH.exists():
        raise FileNotFoundError(
            f'{NPY_PATH} not found - run geom.reference.save() '
            f'(or python figures/scripts/s3_2_reference.py)'
        )
    cell = UnitCell(np.load(NPY_PATH))

    if check_fingerprint:
        stored = json.loads(JSON_PATH.read_text(encoding='utf-8'))['config']
        current = fingerprint()
        drift = {k: (stored[k], current[k]) for k in current
                 if stored.get(k) != current[k]}
        if drift:
            raise ValueError(
                'reference cell is stale - config changed since it was saved: '
                + ', '.join(f'{k}: {was} -> {now}' for k, (was, now) in drift.items())
                + '. Regenerate with geom.reference.save().'
            )
    return cell
