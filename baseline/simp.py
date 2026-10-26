"""
SIMP on the periodic unit cell.

We already established that Ferro/Carbonaro (CMAME 2023) solve our setup by
SIMP + inverse homogenization with a minimum length-scale constraint.

We reuse: `sim2d.homogenize` already solves the three periodic fluctuation problems on the 
64x64 grid, and a solid cell is the SIMP design domain, so `sim2d.mesh.cell_to_mesh` already
builds the grid with the wrap identified. This code adds the two things topology optimization 
needs: density-interpolated stiffness, and the derivative of C_eff with respect to every 
element's density.

Homogenized stiffness is self-adjoint, so with the fluctuation fields in hand

    C_ij         = (1/|Y|) * sum_e  E_e(rho) * W_ij_e
    dC_ij/drho_e = (1/|Y|) * dE_e/drho * W_ij_e

where W_ij_e is the unit-modulus mutual energy of load cases i and j over element e. The
term du/drho drops out.

C_eff is assembled here in the energy form, while `sim2d.homogenize` uses the
volume-averaged stress. The two are identical for the same discretization and the same
solved fluctuations, and `tests/test_simp.py` checks that they agree.
"""

import functools
import time

import numpy as np
from scipy.ndimage import convolve
from skfem import (Basis, BilinearForm, ElementQuad1, ElementVector, Functional, LinearForm, asm, condense, solve)
from skfem.helpers import ddot, sym_grad, trace

import config
from geom.cell import UnitCell
from sim2d.homogenize import UNIT_STRAINS, Homogenized, plane_stress_moduli
from sim2d.mesh import cell_to_mesh

PENAL = 3.0                 # SIMP exponent
E_MIN_FRAC = 1e-9           # void modulus as a fraction of solid, to keep K non-singular
PINNED = np.array([0, 1])   # periodic BCs leave rigid translation free, as in sim2d

@functools.lru_cache(maxsize=1)
def design_basis():
    """
    The full grid as one periodic design domain.
    Element e corresponds to `rho.ravel()[e]` with rho indexed [axial, circumferential].
    """
    n = config.GRID_N
    return Basis(cell_to_mesh(UnitCell(np.ones((n, n), dtype=bool))), ElementVector(ElementQuad1()))

def moduli(rho, penal=PENAL, e_min_frac=E_MIN_FRAC):
    """SIMP interpolation E(rho) = E_min + rho^p (E_0 - E_min), and dE/drho, per element."""
    e0 = config.NITINOL['E_austenite_MPa']
    e_min = e_min_frac * e0
    rho = np.asarray(rho, float).ravel()
    return (e_min + rho ** penal * (e0 - e_min),
            penal * rho ** (penal - 1.0) * (e0 - e_min))

def _at_quadrature(values, basis):
    """Element-wise values broadcast to the quadrature points a form sees."""
    return np.broadcast_to(np.asarray(values, float)[:, None], (basis.mesh.t.shape[1], basis.X.shape[1]))

class DensityHomogenized(Homogenized):
    """
    A density field's effective stiffness, plus the gradient of it.

    Subclasses `Homogenized` so `E_circ` and `K_radial` come from the one definition the
    rest of the project already uses, instead of a second copy that can drift from it.
    """

    def __init__(self, C_eff, fluctuations, basis, area_mm2, seconds, energies, dE_drho):
        super().__init__(C_eff, fluctuations, basis, area_mm2, seconds)
        self.energies = energies        # (3, 3, n_elements) unit-modulus mutual energies
        self.dE_drho = dE_drho          # (n_elements,) dE/drho from the SIMP interpolation

    def dC(self, i, j):
        """dC_eff[i, j] / drho, as an (N, N) field in the same layout as the cell."""
        n = config.GRID_N
        return (self.dE_drho * self.energies[i, j] / self.area_mm2).reshape(n, n)

    def dK_radial(self):
        """
        dK_radial / drho, as an (N, N) field.

        K_radial goes through the compliance (E_circ = 1 / inv(C)[0,0]), so the chain 
        rule runs over the whole matrix: d inv(C) = -inv(C) dC inv(C).
        """
        S = np.linalg.inv(self.C_eff)
        # d(1/S00)/dC_ij = -(1/S00^2) * dS00/dC_ij, and dS00/dC_ij = -S[0,i] S[j,0].
        dE_circ = np.einsum('i,j,ij...->...', S[0], S[:, 0],
                            np.array([[self.dC(i, j) for j in range(3)]
                                      for i in range(3)])) / S[0, 0] ** 2
        R = config.D_DEPLOYED_MM / 2.0
        return dE_circ * config.STRUT_THICKNESS_MM / R ** 2

def homogenize_density(rho, penal=PENAL, e_min_frac=E_MIN_FRAC):
    """C_eff of a density field, with the per-element mutual energies kept for the gradient."""
    started = time.perf_counter()
    basis = design_basis()
    lam1, mu1 = plane_stress_moduli(E=1.0)
    e_mod, de_mod = moduli(rho, penal, e_min_frac)
    emod = _at_quadrature(e_mod, basis)

    @BilinearForm
    def stiffness(u, v, w):
        return w['emod'] * (2.0 * mu1 * ddot(sym_grad(u), sym_grad(v))
                             + lam1 * trace(sym_grad(u)) * trace(sym_grad(v)))

    K = asm(stiffness, basis, emod=emod)

    fluctuations = []
    for E_mac in UNIT_STRAINS:
        @LinearForm
        def rhs(v, w, E_mac=E_mac):
            e = sym_grad(v)
            trE = E_mac[0, 0] + E_mac[1, 1]
            return -w['emod'] * (2.0 * mu1 * (E_mac[0, 0] * e[0, 0] 
                               + E_mac[1, 1] * e[1, 1]
                               + 2.0 * E_mac[0, 1] * e[0, 1])
                               + lam1 * trE * trace(e))

        f = asm(rhs, basis, emod=emod)
        fluctuations.append(solve(*condense(K, f, D=PINNED)))

    interp = [basis.interpolate(u) for u in fluctuations]
    energies = np.zeros((3, 3, basis.mesh.t.shape[1]))
    for i in range(3):
        for j in range(i, 3):
            @Functional
            def mutual(w, Ei=UNIT_STRAINS[i], Ej=UNIT_STRAINS[j]):
                gi, gj = sym_grad(w['ui']), sym_grad(w['uj'])
                ti = [[Ei[0, 0] + gi[0, 0], Ei[0, 1] + gi[0, 1]],
                      [Ei[1, 0] + gi[1, 0], Ei[1, 1] + gi[1, 1]]]
                tj = [[Ej[0, 0] + gj[0, 0], Ej[0, 1] + gj[0, 1]],
                      [Ej[1, 0] + gj[1, 0], Ej[1, 1] + gj[1, 1]]]
                dd = (ti[0][0] * tj[0][0] + ti[0][1] * tj[0][1]
                    + ti[1][0] * tj[1][0] + ti[1][1] * tj[1][1])
                return (2.0 * mu1 * dd
                        + lam1 * (ti[0][0] + ti[1][1]) * (tj[0][0] + tj[1][1]))

            energies[i, j] = energies[j, i] = mutual.elemental(basis, ui=interp[i], uj=interp[j])

    circ_mm, axial_mm = config.cell_extent_mm()
    area = circ_mm * axial_mm
    C_eff = np.einsum('ije,e->ij', energies, e_mod) / area
    return DensityHomogenized(C_eff, fluctuations, basis, area, time.perf_counter() - started, energies, de_mod)

# LENGTH SCALE #
def filter_kernel(radius):
    """Normalized cone weights out to `radius` pixels."""
    r = int(np.ceil(radius))
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    k = np.maximum(0.0, radius - np.hypot(dx, dy))
    return k / k.sum()

def density_filter(x, radius):
    """
    Cone-weighted average over a `radius` pixel neighborhood, wrapping at both seams.

    On a torus a normalized kernel conserves total material exactly, so the volume
    constraint stays meaningful after filtering and top88's boundary correction is not
    needed. `mode='wrap'` is the whole of the periodicity.
    """
    return convolve(np.asarray(x, float), filter_kernel(radius), mode='wrap')

def filter_chain_rule(sensitivity, radius):
    """
    Push a sensitivity in the FILTERED field back onto the design variables.

    The cone kernel is symmetric, so its transpose is itself and the same convolution
    serves both directions.
    """
    return density_filter(sensitivity, radius)

def project(x, beta, eta=0.5):
    """Smoothed Heaviside projection, which is what turns a gray filtered field crisp."""
    if beta <= 0.0:
        return np.asarray(x, float)
    return ((np.tanh(beta * eta) + np.tanh(beta * (np.asarray(x, float) - eta)))
            / (np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))))

def project_grad(x, beta, eta=0.5):
    """d project / dx, for the chain rule through the projection."""
    if beta <= 0.0:
        return np.ones_like(np.asarray(x, float))
    return (beta * (1.0 - np.tanh(beta * (np.asarray(x, float) - eta)) ** 2)
            / (np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))))
