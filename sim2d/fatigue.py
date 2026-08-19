"""
Fatigue-strain surrogate: eps_a_max and A_over_lim.

Reuses the three fluctuation fields from sim2d.homogenize, no new solves. Linear
elasticity makes the local strain field linear in the macroscopic strain, so the fluctuation
for any load case is a linear combination of the unit-strain solutions already computed.

Limb flexion axially compresses the artery, and the stent inside it, by
config.FLEX_AXIAL_COMPRESSION (9% for the proximal SFA during walking). The cycle runs 
between the standing baseline and the flexed state, so it's proportional: the strain tensor
scales along one path rather than rotating. Two consequences:

  - The amplitude tensor is half the flexed-state tensor, and
  - Principal directions don't move, so the principal of the amplitude equals the
    amplitude of the principal.

The stent isn't held at constant diameter while this happens, so circumferential and shear
stress are released to zero rather than clamped, a uniaxial macroscopic stress state,
solved from C_eff. Clamping eps_circ = 0 instead would invent hoop stress the artery doesn't
apply and would inflate the fatigue numbers.

What's reported:Maximum principal strain amplitude, which is what nitinol constant-life
data (Pelton 2008) is expressed against. `eps_a_max` is the raw field maximum;
`eps_a_p99` is the area-weighted 99th percentile, carried alongside because one quad
element at a stair-stepped pixel corner can dominate the raw max.
"""

import numpy as np

import config
from sim2d.homogenize import UNIT_STRAINS, homogenize

def macroscopic_strain(axial_strain, C_eff):
    """
    Macroscopic strain (Voigt) for imposed axial strain with circ and shear stress free.

    Solves C[0,:] . eps = 0 and C[2,:] . eps = 0 for (eps_circ, gamma).
    """
    A = np.array([[C_eff[0, 0], C_eff[0, 2]],
                  [C_eff[2, 0], C_eff[2, 2]]])
    b = -axial_strain * np.array([C_eff[0, 1], C_eff[2, 1]])
    eps_circ, gamma = np.linalg.solve(A, b)
    return np.array([eps_circ, axial_strain, gamma])

class Fatigue:
    """Strain-amplitude read for one cell."""

    def __init__(self, eps_a, weights, macro):
        self._shape = eps_a.shape       # (elements, quadrature points)
        self.field = eps_a.ravel()      # max principal strain amplitude per quadrature point
        self.weights = weights.ravel()  # area per quadrature point, mm^2
        self.macro = macro              # the macroscopic strain applied (Voigt)

    def per_element(self):
        """Peak amplitude within each element."""
        return self.field.reshape(self._shape).max(axis=1)

    @property
    def eps_a_max(self):
        return float(self.field.max())

    @property
    def eps_a_p99(self):
        """Area-weighted 99th percentile."""
        order = np.argsort(self.field)
        f, w = self.field[order], self.weights[order]
        cdf = np.cumsum(w) / w.sum()
        return float(f[np.searchsorted(cdf, 0.99)])

    @property
    def A_over_lim(self):
        """Fraction of the structure's area with amplitude above the nitinol limit."""
        over = self.field > config.EPS_A_LIM
        return float(self.weights[over].sum() / self.weights.sum())

    @property
    def attenuation(self):
        """
        Applied amplitude divided by the peak strut amplitude.

        The mechanical job of the topology: absorb device-level deformation by folding
        rather than stretching. Above 1 means the structure attenuates; below 1 means it
        concentrates.
        """
        return abs(config.FLEX_AXIAL_AMPLITUDE) / self.eps_a_max

def fatigue(cell, homogenized=None, axial_compression=None):
    """Strain-amplitude read for a cell. Reuses a Homogenized if one is supplied."""
    h = homogenize(cell) if homogenized is None else homogenized
    compression = (config.FLEX_AXIAL_COMPRESSION if axial_compression is None
                   else axial_compression)

    macro = macroscopic_strain(-abs(compression), h.C_eff)

    # Fluctuation for this load case: linear combination of the three unit-strain solutions.
    u = sum(c * f for c, f in zip(macro, h.fluctuations))

    grad = h.basis.interpolate(u).grad
    E_mac = sum(c * E for c, E in zip(macro, UNIT_STRAINS))

    exx = grad[0][0] + E_mac[0, 0]
    eyy = grad[1][1] + E_mac[1, 1]
    exy = 0.5 * (grad[0][1] + grad[1][0]) + E_mac[0, 1]

    # Maximum principal strain of the flexed state.
    mean = 0.5 * (exx + eyy)
    radius = np.sqrt((0.5 * (exx - eyy)) ** 2 + exy ** 2)
    principal = mean + radius

    # Proportional loading from the standing baseline: amplitude is half the range.
    eps_a = 0.5 * np.abs(principal)

    return Fatigue(eps_a, h.basis.dx, macro)
