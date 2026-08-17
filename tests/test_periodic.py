"""Torus-aware operations must commute with rolling."""

import numpy as np
import pytest
from scipy import ndimage
from geom import periodic
from geom.handmade import diamond

OPS = [
    ('erosion', periodic.erosion),
    ('dilation', periodic.dilation),
    ('opening', periodic.opening),
    ('closing', periodic.closing),
    ('distance_transform', periodic.distance_transform),
    ('label', lambda a: periodic.label(a)[0] > 0),
]

@pytest.fixture
def cell():
    return diamond().to_array()

@pytest.mark.parametrize('name,op', OPS, ids=[n for n, _ in OPS])
def test_shift_equivariant(cell, name, op):
    assert periodic.is_shift_equivariant(op, cell)

def test_scipy_erosion_is_not_shift_equivariant(cell):
    """The rectangle-based op fails this."""
    assert not periodic.is_shift_equivariant(ndimage.binary_erosion, cell)

def test_label_merges_across_the_wrap():
    """Two bars hugging opposite edges are one strut on the torus, two on a rectangle."""
    a = np.zeros((16, 16), dtype=bool)
    a[:, 0] = True
    a[:, -1] = True
    _, periodic_count = periodic.label(a)
    _, rect_count = ndimage.label(a, structure=periodic.CONN4)
    assert periodic_count == 1
    assert rect_count == 2

def test_diamond_is_one_component(cell):
    """Arms meet inside the cell."""
    _, count = periodic.label(cell)
    assert count == 1

def test_label_counts_a_genuinely_split_field():
    """Two isolated blobs stay two."""
    a = np.zeros((32, 32), dtype=bool)
    a[4:8, 4:8] = True
    a[20:24, 20:24] = True
    _, count = periodic.label(a)
    assert count == 2

def test_distance_transform_sees_through_the_wrap():
    """A pixel by the edge is close to void on the other side."""
    a = np.ones((16, 16), dtype=bool)
    a[0, 0] = False  # The only void, in the corner
    d = periodic.distance_transform(a)
    # The opposite corner is 1 pixel away across the wrap, not 15 across the middle.
    assert d[15, 15] == pytest.approx(np.sqrt(2))

def test_erosion_preserves_strut_width_at_the_boundary(cell):
    """Rectangle-based erosion messes with struts where they cross the seam."""
    ours = periodic.erosion(cell)
    theirs = ndimage.binary_erosion(cell)
    assert ours.sum() > theirs.sum()
    assert np.array_equal(ours[2:-2, 2:-2], theirs[2:-2, 2:-2])
