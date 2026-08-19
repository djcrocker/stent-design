"""f_metal, and its agreement between the geometry and the FE mesh."""

import numpy as np
import pytest

import config
from geom.cell import UnitCell
from geom import reference
from geom.parametric import achievable_widths, crown
from sim2d.homogenize import homogenize

N = config.GRID_N
CELL_AREA = config.cell_extent_mm()[0] * config.cell_extent_mm()[1]

def test_f_metal_is_the_pixel_count():
    cell = reference.build()
    arr = cell.to_array()
    assert cell.f_metal == pytest.approx(arr.sum() / arr.size)

@pytest.mark.parametrize('fraction', [0.0, 0.25, 0.5, 1.0])
def test_f_metal_hits_known_fractions_exactly(fraction):
    a = np.zeros((N, N), dtype=bool)
    a[:int(fraction * N), :] = True
    assert UnitCell(a).f_metal == pytest.approx(fraction)

@pytest.mark.parametrize('w', achievable_widths()[:3])
def test_f_metal_matches_the_fe_material_area(w):
    """
    Geometry and simulator must see one structure.

    A mismatch would mean the mesh drops or duplicates material, which would then
    corrupt every label: f_metal is a component of y, and the same mesh produces
    K_radial and eps_a_max.
    """
    cell = crown(strut_width_mm=w, crown_amplitude=0.25, n_periods=1)
    fe_area = float(homogenize(cell).basis.dx.sum())
    assert fe_area / CELL_AREA == pytest.approx(cell.f_metal, abs=1e-12)

def test_f_metal_lies_inside_the_validity_guards_for_the_reference():
    assert config.F_METAL_MIN < reference.build().f_metal < config.F_METAL_MAX
