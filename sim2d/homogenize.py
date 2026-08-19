"""
Effective in-plane stiffness of a periodic unit cell.

Computational homogenisation: the total strain splits as

    eps = E + sym(grad u~)

with E the imposed macroscopic strain and u~ a periodic fluctuation. Solving three unit
macroscopic strains gives the effective plane-stress stiffness C_eff (3x3, Voigt), from
which K_radial follows and any later cyclic load case can be evaluated by recombining 
the stored fluctuation fields, with no new solves.

Material: linear-elastic austenite (config.NITINOL). We don't use the superelastic model
here, as it belongs in the 3D tier. Using it here would get accuracy that the ranking
doesn't need at a cost the throughput can't afford.

Averaging is over the full cell area including void, because the quantity wanted is the
effective stiffness of the perforated sheet, not of the metal in it.
"""

import time

import numpy as np
from skfem import (Basis, BilinearForm, ElementQuad1, ElementVector, Functional,
                   LinearForm, asm, condense, solve)
from skfem.helpers import ddot, sym_grad, trace

import config
from sim2d.mesh import cell_to_mesh

# Voigt order used throughout: (xx, yy, xy) = (circumferential, axial, shear).
VOIGT = ('circ', 'axial', 'shear')
UNIT_STRAINS = (
    np.array([[1.0, 0.0], [0.0, 0.0]]),
    np.array([[0.0, 0.0], [0.0, 1.0]]),
    np.array([[0.0, 0.5], [0.5, 0.0]]),   # engineering shear gamma = 1
)

def plane_stress_moduli(E=None, nu=None):
    """(lambda*, mu) for plane stress, so that sigma = 2 mu eps + lambda* tr(eps) I."""
    E = config.NITINOL['E_austenite_MPa'] if E is None else E
    nu = config.NITINOL['poisson_austenite'] if nu is None else nu
    return E * nu / (1.0 - nu ** 2), E / (2.0 * (1.0 + nu))

def _forms(lam, mu):
    @BilinearForm
    def stiffness(u, v, w):
        return (2.0 * mu * ddot(sym_grad(u), sym_grad(v))
                + lam * trace(sym_grad(u)) * trace(sym_grad(v)))
    return stiffness

class Homogenized:
    """Effective stiffness of one cell, plus everything needed to re-use the solves."""

    def __init__(self, C_eff, fluctuations, basis, area_mm2, seconds):
        self.C_eff = C_eff              # 3x3 MPa, Voigt (circ, axial, shear)
        self.fluctuations = fluctuations
        self.basis = basis
        self.area_mm2 = area_mm2
        self.seconds = seconds

    @property
    def E_circ(self):
        """
        Effective circumferential Young's modulus, MPa.

        Taken from the compliance so it is the modulus under uniaxial circumferential
        stress rather than the constrained C[0,0].
        """
        return 1.0 / np.linalg.inv(self.C_eff)[0, 0]

    @property
    def E_axial(self):
        return 1.0 / np.linalg.inv(self.C_eff)[1, 1]

    @property
    def K_radial(self):
        """
        Pressure per unit radial displacement, N/mm^3.

        Thin-walled cylinder: hoop stress sigma = pR/t and hoop strain = dR/R give
        p = E_circ * t * dR / R^2, so K_radial = E_circ * t / R^2. Defined this way so the
        3D tier can reproduce the same number by applying pressure and measuring dR.
        """
        R = config.D_DEPLOYED_MM / 2.0
        return self.E_circ * config.STRUT_THICKNESS_MM / R ** 2

def homogenize(cell, E=None, nu=None):
    """Compute C_eff for a unit cell. Returns a Homogenized."""
    started = time.perf_counter()
    lam, mu = plane_stress_moduli(E, nu)

    mesh = cell_to_mesh(cell)
    basis = Basis(mesh, ElementVector(ElementQuad1()))
    K = asm(_forms(lam, mu), basis)

    # Periodic BCs leave rigid translation unconstrained; pin one node.
    pinned = np.array([0, 1])

    circ_mm, axial_mm = config.cell_extent_mm()
    area = circ_mm * axial_mm

    C_eff = np.zeros((3, 3))
    fluctuations = []
    for col, E_mac in enumerate(UNIT_STRAINS):
        @LinearForm
        def rhs(v, w, E_mac=E_mac):
            e = sym_grad(v)
            trE = E_mac[0, 0] + E_mac[1, 1]
            return -(2.0 * mu * (E_mac[0, 0] * e[0, 0] + E_mac[1, 1] * e[1, 1]
                                 + 2.0 * E_mac[0, 1] * e[0, 1])
                     + lam * trE * trace(e))

        f = asm(rhs, basis)
        # solve() on a condensed system returns the full vector, constrained DOFs included.
        u = solve(*condense(K, f, D=pinned))
        fluctuations.append(u)

        # Volume-averaged stress over the whole cell (metal contributes, void does not).
        for row, comp in enumerate(((0, 0), (1, 1), (0, 1))):
            @Functional
            def avg(w, comp=comp, E_mac=E_mac):
                e = sym_grad(w['u'])
                tot = [[E_mac[0, 0] + e[0, 0], E_mac[0, 1] + e[0, 1]],
                       [E_mac[1, 0] + e[1, 0], E_mac[1, 1] + e[1, 1]]]
                tr = tot[0][0] + tot[1][1]
                s = 2.0 * mu * tot[comp[0]][comp[1]]
                if comp[0] == comp[1]:
                    s = s + lam * tr
                return s

            C_eff[row, col] = avg.assemble(basis, u=basis.interpolate(u)) / area

    return Homogenized(C_eff, fluctuations, basis, area,
                       time.perf_counter() - started)
