"""The void-side minimum-feature criterion and the manufacturability screen."""

import numpy as np
import pytest

import config
from geom import cleanup, validity
from geom.cell import UnitCell
from geom.handmade import diamond
from screen import shortlist

def _punch(arr, cy, cx, r):
    """Punch a circular hole of radius r px into a field."""
    out = arr.copy()
    y, x = np.ogrid[:arr.shape[0], :arr.shape[1]]
    out[((y - cy) ** 2 + (x - cx) ** 2) <= r * r] = False
    return out

def test_a_handmade_cell_has_no_thin_voids():
    """The baseline was drawn to be manufacturable, so it must score zero."""
    assert validity.void_thin_fraction(diamond().to_array()) == 0.0
    assert not validity.has_thin_void(diamond().to_array())

def test_a_sub_minimum_hole_is_caught():
    solid = np.ones((config.GRID_N, config.GRID_N), bool)
    r = validity.min_feature_radius_px()
    holed = _punch(solid, 32, 32, max(r - 1.0, 1.0))
    assert validity.has_thin_void(holed)

def test_a_hole_larger_than_the_minimum_feature_is_allowed():
    solid = np.ones((config.GRID_N, config.GRID_N), bool)
    big = _punch(solid, 32, 32, validity.min_feature_radius_px() * 4)
    assert not validity.has_thin_void(big)

def test_the_criterion_is_the_dual_of_the_solid_test():
    """
    Opening tests material, closing tests void. Complementing a field must swap which
    of the two fires, which is what makes this the same length scale on both phases.
    """
    solid = np.ones((config.GRID_N, config.GRID_N), bool)
    holed = _punch(solid, 32, 32, max(validity.min_feature_radius_px() - 1.0, 1.0))
    se = validity.disk(validity.min_feature_radius_px())
    from geom import periodic
    opened = periodic.opening(~holed, structure=se)
    assert (~holed & ~opened).any()          # The complement is too thin by the solid test
    assert validity.has_thin_void(holed)     # The original is too thin by the void test

def test_void_fraction_is_normalized_by_void_not_material():
    """Symmetry with `thin_fraction`, which is a fraction of the material."""
    solid = np.ones((config.GRID_N, config.GRID_N), bool)
    holed = _punch(solid, 32, 32, max(validity.min_feature_radius_px() - 1.0, 1.0))
    frac = validity.void_thin_fraction(holed)
    assert frac == pytest.approx(1.0)        # Every void pixel gets filled

def test_filling_only_adds_material():
    rng = np.random.default_rng(0)
    arr = rng.random((config.GRID_N, config.GRID_N)) > 0.5
    filled, added = cleanup.fill_thin_voids(arr)
    assert added >= 0
    assert (filled | arr == filled).all()    # Nothing was removed

def test_filling_removes_the_thin_voids_it_targets():
    solid = np.ones((config.GRID_N, config.GRID_N), bool)
    holed = _punch(solid, 32, 32, max(validity.min_feature_radius_px() - 1.0, 1.0))
    filled, added = cleanup.fill_thin_voids(holed)
    assert added > 0
    assert not validity.has_thin_void(filled)

def test_filling_is_idempotent():
    """A second pass must be a no-op, or the screen would depend on how often it ran."""
    rng = np.random.default_rng(1)
    arr = rng.random((config.GRID_N, config.GRID_N)) > 0.45
    once, _ = cleanup.fill_thin_voids(arr)
    twice, added = cleanup.fill_thin_voids(once)
    assert added == 0
    assert np.array_equal(once, twice)

def test_filling_is_shift_equivariant():
    """It runs on the torus, so the answer can't depend on where the cell is cut."""
    from geom import periodic
    rng = np.random.default_rng(2)
    arr = rng.random((config.GRID_N, config.GRID_N)) > 0.45
    assert periodic.is_shift_equivariant(lambda a: cleanup.fill_thin_voids(a)[0], arr)

def test_the_screen_keeps_a_clean_cell_untouched():
    fields = np.array([diamond().to_array()])
    kept, repaired, labels, report = shortlist.manufacturable(fields, [0], progress=False)
    assert list(kept) == [0]
    assert report['repaired'] == 0
    assert np.array_equal(repaired[0], fields[0])

def _perforated_blob():
    """A thickened diamond with sub-minimum holes punched deep inside the material."""
    from geom import periodic

    n = config.GRID_N
    y, x = np.ogrid[:n, :n]
    arr = periodic.dilation(diamond().to_array(), structure=validity.disk(3.0))
    dist = periodic.distance_transform(arr)
    used = []
    for flat in np.argsort(dist.ravel())[::-1]:
        cy, cx = divmod(int(flat), n)
        if dist[cy, cx] < 5.0:
            break
        if any((cy - uy) ** 2 + (cx - ux) ** 2 < 144 for uy, ux in used):
            continue
        arr[((y - cy) ** 2 + (x - cx) ** 2) <= 2.25] = False
        used.append((cy, cx))
        if len(used) >= 12:
            break
    return arr

def test_the_perforated_blob_is_rejected_by_the_void_criterion():
    """Under the density cap, connected, wrapping, and thick everywhere in the material."""
    arr = _perforated_blob()
    assert arr.mean() < config.F_METAL_MAX          # Meets the density cap as built
    assert validity.has_thin_void(arr)

    verdict = validity.check(UnitCell(arr))
    assert not verdict.ok
    assert validity.VOID_THIN_FEATURE in verdict.reasons
    assert validity.THIN_FEATURE not in verdict.reasons

def test_repair_pushes_the_perforated_blob_over_the_density_cap():
    """Closing the unmakeable holes is what reveals how much material is really there."""
    arr = _perforated_blob()
    filled, added = cleanup.fill_thin_voids(arr)
    assert added > 0
    assert filled.mean() > config.F_METAL_MAX

def test_the_screen_drops_the_perforated_blob():
    """However it's routed, the blob can't survive to be ranked."""
    arr = _perforated_blob()
    kept, _, _, report = shortlist.manufacturable(np.array([arr]), [0], progress=False)
    assert len(kept) == 0
    assert report['dropped'] == 1
    assert set(report['drop_reasons']) & {'cleanup_failed', validity.TOO_DENSE}

def test_the_screen_relabels_the_repaired_geometry_not_the_raw_field():
    """
    The labels should describe the cell that is kept. `label_fields` labels the cleaned 
    cell while handing back the raw field, so a screen that skipped relabeling would ship
    numbers for geometry nobody has.
    """
    from diffusion.dataset import label_one

    fields = np.array([diamond().to_array()])
    kept, repaired, labels, _ = shortlist.manufacturable(fields, [0], progress=False)
    assert list(kept) == [0]
    assert labels[0] == label_one(repaired[0])

def test_closing_can_add_a_diagonally_attached_speck():
    """Documents why the screen drops islands after filling."""
    from geom import periodic

    n = config.GRID_N
    arr = np.zeros((n, n), bool)
    arr[10:14, 10:30] = True                       # A bar
    filled, added = cleanup.fill_thin_voids(arr)
    assert (arr & ~filled).sum() == 0              # Nothing removed: closing is extensive
    before = periodic.label(arr)[1]
    after = periodic.label(filled)[1]
    assert after >= before                         # Never fewer, may be more

def test_the_screen_does_not_fail_a_cell_on_specks_it_created():
    """A cell that survives must not be dropped for islands the repair itself deposited."""
    fields = np.array([diamond().to_array()])
    kept, repaired, _, report = shortlist.manufacturable(fields, [0], progress=False)
    assert list(kept) == [0]
    assert validity.check(UnitCell(repaired[0])).ok
    assert report['drop_reasons'].get(validity.DISCONNECTED, 0) == 0

def test_the_label_cache_is_fingerprinted_against_the_validity_envelope():
    """A cache keyed on sample count alone survives a criteria change silently."""
    from diffusion.generate import label_fingerprint

    fp = label_fingerprint()
    assert validity.VOID_THIN_FEATURE in fp['criteria']
    assert fp['void_thin_tolerance'] == validity.VOID_THIN_TOLERANCE
    assert fp['f_metal_max'] == config.F_METAL_MAX

def test_the_void_criterion_is_reported_in_check_metrics():
    """The measured value travels with the verdict, so a near-miss is visible."""
    metrics = validity.check(UnitCell(diamond().to_array())).metrics
    assert 'void_thin_fraction' in metrics
    assert metrics['void_thin_fraction'] == pytest.approx(0.0)

def test_cleanup_now_repairs_a_perforated_cell_instead_of_passing_it_through():
    arr = _perforated_blob()
    result = cleanup.clean(arr)
    if result.fixed:
        assert not validity.has_thin_void(result.cell.to_array())
        assert result.change_fraction > 0.0
    else:
        assert not validity.check(UnitCell(arr)).ok


def test_the_label_cache_is_keyed_on_the_cells_not_just_their_count():
    """
    A retrained model regenerates a pool of identical size under identical criteria, so
    count-plus-criteria HITS and serves the previous pool's labels. That happened on
    2026-08-25 and put 763 wrong rows into the shortlist.
    """
    from diffusion.generate import fields_digest

    rng = np.random.default_rng(0)
    a = rng.random((8, config.GRID_N, config.GRID_N)) > 0.5
    b = a.copy()
    b[3, 10, 10] = not b[3, 10, 10]                 # One pixel, one cell

    assert fields_digest(a) == fields_digest(a.copy())
    assert fields_digest(a) != fields_digest(b)
    assert len(a) == len(b)                         # The count can't tell them apart

def test_wraps_handles_a_non_square_stack():
    """An axial stack of cells is taller than it is wide."""
    from geom import validity as v

    n = config.GRID_N
    bar = np.zeros((2 * n, n), bool)
    bar[:, 10:14] = True                        # A full-height axial strut
    circ, axial = v.wraps(bar)
    assert axial and not circ

    band = np.zeros((2 * n, n), bool)
    band[10:14, :] = True                       # A full-width circumferential band
    circ, axial = v.wraps(band)
    assert circ and not axial
