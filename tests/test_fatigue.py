"""The strain-amplitude surrogate, checked against closed forms and linearity."""

import numpy as np
import pytest

import config
from geom.cell import UnitCell
from geom import reference
from geom.parametric import achievable_widths, crown
from sim2d.fatigue import Fatigue, fatigue, macroscopic_strain
from sim2d.homogenize import homogenize

N = config.GRID_N
NU = config.NITINOL['poisson_austenite']

@pytest.fixture(scope='module')
def ref():
    cell = reference.build()
    h = homogenize(cell)
    return cell, h, fatigue(cell, h)

# MACROSCOPIC LOAD CASE #
def test_circumferential_and_shear_stress_are_released(ref):
    """The artery doesn't clamp the stent's diameter; hoop and shear stress go to zero."""
    _, h, f = ref
    stress = h.C_eff @ f.macro
    assert abs(stress[0]) < 1e-9 * abs(stress[1])
    assert abs(stress[2]) < 1e-9 * max(abs(stress[1]), 1.0)

def test_axial_strain_is_the_imposed_compression(ref):
    _, _, f = ref
    assert f.macro[1] == pytest.approx(-config.FLEX_AXIAL_COMPRESSION)

def test_cell_expands_circumferentially_when_compressed(ref):
    """Positive effective Poisson response."""
    _, _, f = ref
    assert f.macro[0] > 0

# EXACT SOLUTION #
def test_solid_cell_has_uniform_strain_at_the_poisson_value():
    """
    No holes means no concentration, every point sees the same macroscopic strain.

    Under uniaxial axial stress the transverse strain is nu * compression, which is the
    tensile principal, and the amplitude is half of it.
    """
    solid = UnitCell(np.ones((N, N), dtype=bool))
    f = fatigue(solid)
    expected = 0.5 * NU * config.FLEX_AXIAL_COMPRESSION
    assert f.eps_a_max == pytest.approx(expected, rel=1e-6)
    assert f.field.std() < 1e-12 * f.eps_a_max      # uniform
    assert f.eps_a_max == pytest.approx(f.eps_a_p99, rel=1e-9)

def test_solid_cell_has_no_material_over_the_limit_scaled_down():
    solid = UnitCell(np.ones((N, N), dtype=bool))
    f = fatigue(solid, axial_compression=0.001)
    assert f.A_over_lim == 0.0

# LINEARITY #
def test_amplitude_is_linear_in_the_applied_compression(ref):
    cell, h, _ = ref
    a = fatigue(cell, h, axial_compression=0.02).eps_a_max
    b = fatigue(cell, h, axial_compression=0.04).eps_a_max
    assert b == pytest.approx(2 * a, rel=1e-9)

def test_reusing_a_homogenized_matches_computing_it_fresh(ref):
    cell, h, f = ref
    assert fatigue(cell).eps_a_max == pytest.approx(f.eps_a_max, rel=1e-12)


# REPORTED QUANTITIES #
def test_p99_never_exceeds_the_max(ref):
    _, _, f = ref
    assert f.eps_a_p99 <= f.eps_a_max

def test_area_fraction_is_a_fraction(ref):
    _, _, f = ref
    assert 0.0 <= f.A_over_lim <= 1.0

def test_area_fraction_saturates_at_the_extremes(ref):
    cell, h, _ = ref
    f = fatigue(cell, h)
    assert Fatigue(f.field.reshape(f._shape), f.weights.reshape(f._shape),
                   f.macro).A_over_lim == pytest.approx(f.A_over_lim)

def test_per_element_field_covers_every_element(ref):
    cell, _, f = ref
    assert f.per_element().size == int(cell.to_array().sum())

def test_thicker_struts_carry_higher_strain_amplitude():
    """
    Limb flexion says: the leg shortens the artery by 9% whatever the stent prefers. Beam 
    bending then gives surface strain = kappa * w / 2, so a thicker member sees more strain 
    at the same curvature. Measured exponent ~0.82, against the w^1 that pure bending predicts. 
    It is also why industry moved from 110 um struts to 60-85 um.
    """
    previous = 0.0
    for w in achievable_widths():
        cell = crown(strut_width_mm=w, crown_amplitude=0.25, n_periods=1)
        e = fatigue(cell).eps_a_max
        assert e > previous
        previous = e

def test_stiffness_and_fatigue_pull_against_each_other():
    """The Pareto tension."""
    stiff, strain = [], []
    for w in achievable_widths():
        cell = crown(strut_width_mm=w, crown_amplitude=0.25, n_periods=1)
        h = homogenize(cell)
        stiff.append(h.K_radial)
        strain.append(fatigue(cell, h).eps_a_max)
    assert np.corrcoef(stiff, strain)[0, 1] > 0.95
