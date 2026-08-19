"""The canonical reference cell has to be valid, reproducible, and stale-checked."""

import json

import numpy as np
import pytest

import config
from geom import reference
from geom import validity as V

@pytest.fixture(autouse=True)
def saved():
    """Every test works against a freshly written reference."""
    return reference.save()

def test_reference_is_valid(saved):
    ok, reasons = V.is_valid(saved)
    assert ok, reasons

def test_reference_sits_in_the_conventional_coverage_band(saved):
    """The reason this cell was chosen over the grid-center one."""
    assert 0.19 <= saved.f_metal <= 0.26

def test_reference_has_no_thin_features(saved):
    """We have to mesh and converge on this."""
    assert V.check(saved).metrics['thin_fraction'] == pytest.approx(0.0)

def test_reference_is_deterministic():
    assert np.array_equal(reference.build().to_array(), reference.build().to_array())

def test_build_matches_what_was_saved(saved):
    assert np.array_equal(reference.load().to_array(), reference.build().to_array())

def test_saved_files_exist_and_record_the_recipe():
    assert reference.NPY_PATH.exists()
    stored = json.loads(reference.JSON_PATH.read_text(encoding='utf-8'))
    assert stored['params'] == reference.PARAMS
    assert set(stored['config']) == set(reference.FINGERPRINT_KEYS)

def test_stale_fingerprint_raises(monkeypatch):
    """A moved config constant should fail."""
    monkeypatch.setattr(config, 'D_DEPLOYED_MM', config.D_DEPLOYED_MM + 1.0)
    with pytest.raises(ValueError, match='stale'):
        reference.load()

def test_stale_check_can_be_bypassed_deliberately(monkeypatch):
    monkeypatch.setattr(config, 'D_DEPLOYED_MM', config.D_DEPLOYED_MM + 1.0)
    assert reference.load(check_fingerprint=False) is not None

def test_missing_file_raises_with_a_usable_message(monkeypatch, tmp_path):
    monkeypatch.setattr(reference, 'NPY_PATH', tmp_path / 'absent.npy')
    with pytest.raises(FileNotFoundError, match='save'):
        reference.load()

def test_params_are_an_achievable_width():
    """The stored width must land exactly on the pixel lattice, not near it."""
    from geom.parametric import snap_width_mm
    w = reference.PARAMS['strut_width_mm']
    assert snap_width_mm(w) == pytest.approx(w, abs=1e-4)
