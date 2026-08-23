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

# label() WRAPPER #
from geom import handmade as H          # noqa: E402
from sim2d.fatigue import fatigue       # noqa: E402
from sim2d.label import UNITS, Label, describe, label  # noqa: E402

@pytest.fixture(scope='module')
def labeled():
    return label(reference.build())

def test_valid_cell_yields_the_full_objective_vector(labeled):
    assert labeled.valid
    assert labeled.y.shape == (len(config.OBJECTIVE_KEYS),)
    assert np.all(np.isfinite(labeled.y))

def test_vector_order_follows_the_config(labeled):
    """y is built from OBJECTIVE_KEYS, so it can't drift from the config that names it."""
    cell = reference.build()
    h = homogenize(cell)
    f = fatigue(cell, h)
    expected = {'K_radial': h.K_radial, 'eps_a_max': f.eps_a_max,
                'A_over_lim': f.A_over_lim, 'f_metal': cell.f_metal}
    for key, value in zip(config.OBJECTIVE_KEYS, labeled.y):
        assert value == pytest.approx(expected[key], rel=1e-12)

def test_every_component_has_a_declared_unit():
    assert set(UNITS) == set(config.OBJECTIVE_KEYS)

@pytest.mark.parametrize('name', list(H.BROKEN_CELLS), ids=list(H.BROKEN_CELLS))
def test_invalid_cells_return_a_result_rather_than_raising(name):
    build, _ = H.BROKEN_CELLS[name]
    result = label(build())
    assert not result.valid
    assert result.y is None
    assert result.reasons

def test_invalid_cells_skip_the_fe_solve():
    """Validity is checked first so a discarded cell never costs a solve."""
    invalid = label(H.tiny_blob())
    valid = label(reference.build())
    assert invalid.seconds < valid.seconds / 10

def test_validity_check_can_be_bypassed():
    result = label(reference.build(), check_validity=False)
    assert result.valid and result.y is not None

def test_p99_is_carried_alongside_for_spike_b(labeled):
    """Which epsilon enters y is still open; storing both now avoids re-labeling later."""
    assert 'eps_a_p99' in labeled.metrics
    assert labeled.metrics['eps_a_p99'] <= labeled.y[1]

def test_result_unpacks_as_y_and_valid(labeled):
    y, valid = labeled
    assert valid and y is labeled.y

def test_as_dict_is_a_flat_record(labeled):
    row = labeled.as_dict()
    for key in config.OBJECTIVE_KEYS:
        assert isinstance(row[key], float)
    assert row['valid'] is True
    assert 'eps_a_p99' in row

def test_invalid_as_dict_nulls_the_solved_components_but_keeps_coverage():
    """f_metal is a pixel count."""
    row = label(H.tiny_blob()).as_dict()
    assert row['valid'] is False
    assert all(row[k] is None for k in ('K_radial', 'eps_a_max', 'A_over_lim'))
    assert row['f_metal'] == pytest.approx(H.tiny_blob().f_metal)

def test_labeling_meets_the_time_budget(labeled):
    """~5 s per cell."""
    assert labeled.seconds < 5.0

def test_describe_reports_units(labeled):
    text = describe(labeled)
    for key in config.OBJECTIVE_KEYS:
        assert key in text
    assert 'N/mm^3' in text
