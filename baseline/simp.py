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

@functools.lru_cache(maxsize=4)
def design_basis(n=None):
    """
    The full grid as one periodic design domain.
    Element e corresponds to `rho.ravel()[e]` with rho indexed [axial, circumferential].

    `n` below GRID_N is the same physical cell at a coarser discretization, which is how
    the optimizer loop is tested at a cost a test suite can pay.
    """
    n = config.GRID_N if n is None else n
    return Basis(cell_to_mesh(UnitCell(np.ones((n, n), dtype=bool))), ElementVector(ElementQuad1()))

def density_stiffness(lam, mu):
    """Plane-stress stiffness form weighted by a per-element modulus `w['emod']`."""
    @BilinearForm
    def form(u, v, w):
        return w['emod'] * (2.0 * mu * ddot(sym_grad(u), sym_grad(v))
                            + lam * trace(sym_grad(u)) * trace(sym_grad(v)))
    return form

def moduli(rho, penal=PENAL, e_min_frac=E_MIN_FRAC, e0=None):
    """SIMP interpolation E(rho) = E_min + rho^p (E_0 - E_min), and dE/drho, per element."""
    e0 = config.NITINOL['E_austenite_MPa'] if e0 is None else e0
    e_min = e_min_frac * e0
    rho = np.asarray(rho, float).ravel()
    return (e_min + rho ** penal * (e0 - e_min),
            penal * rho ** (penal - 1.0) * (e0 - e_min))

def at_quadrature(values, basis):
    """Element-wise values broadcast to the quadrature points a form sees."""
    return np.broadcast_to(np.asarray(values, float)[:, None], (basis.mesh.t.shape[1], basis.X.shape[1]))

class DensityHomogenized(Homogenized):
    """
    A density field's effective stiffness, plus the gradient of it.

    Subclasses `Homogenized` so `E_circ` and `K_radial` come from the one definition the
    rest of the project already uses, instead of a second copy that can drift from it.
    """

    def __init__(self, C_eff, fluctuations, basis, area_mm2, seconds, energies, dE_drho, shape):
        super().__init__(C_eff, fluctuations, basis, area_mm2, seconds)
        self.energies = energies        # (3, 3, n_elements) unit-modulus mutual energies
        self.dE_drho = dE_drho          # (n_elements,) dE/drho from the SIMP interpolation
        self.shape = shape              # the density field's grid shape

    def dC(self, i, j):
        """dC_eff[i, j] / drho, as an (N, N) field in the same layout as the cell."""
        return (self.dE_drho * self.energies[i, j] / self.area_mm2).reshape(self.shape)

    def dbulk(self):
        """
        d of the 2D bulk modulus (C[:2, :2].sum() / 4) / drho, as an (N, N) field.

        The objective validation (c) maximizes, since the Hashin-Shtrikman bound it is
        checked against is a bound on exactly this quantity.
        """
        return sum(self.dC(i, j) for i in range(2) for j in range(2)) / 4.0

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

def bulk(C_eff):
    """
    2D (area) bulk modulus of a plane-stress C_eff, MPa.

    Under hydrostatic macro strain, p = (sigma_xx + sigma_yy) / 2 and the area strain is
    e_xx + e_yy, so K = C[:2, :2].sum() / 4. For solid isotropic material this is
    E / (2 (1 - nu)), which `tests/test_optimize.py` checks.
    """
    return np.asarray(C_eff)[:2, :2].sum() / 4.0

def hs_bulk_bound(volfrac, e0=None, nu=None):
    """
    Hashin-Shtrikman upper bound on the 2D bulk modulus of a porous material, MPa.

    For a solid/void two-phase composite the general 2D bound collapses to

        K_HS = f K_s G_s / ((1 - f) K_s + G_s)

    with K_s = E / (2 (1 - nu)) and G_s = E / (2 (1 + nu)) the solid's 2D moduli. This is
    the bound optimal microstructures are known to attain in 2D, so the optimizer
    approaching it is a real test of the loop.
    """
    e0 = config.NITINOL['E_austenite_MPa'] if e0 is None else e0
    nu = config.NITINOL['poisson_austenite'] if nu is None else nu
    k_s, g_s = e0 / (2.0 * (1.0 - nu)), e0 / (2.0 * (1.0 + nu))
    return volfrac * k_s * g_s / ((1.0 - volfrac) * k_s + g_s)

def homogenize_density(rho, penal=PENAL, e_min_frac=E_MIN_FRAC):
    """C_eff of a density field, with the per-element mutual energies kept for the gradient."""
    started = time.perf_counter()
    rho = np.asarray(rho, float)
    basis = design_basis(rho.shape[0])
    lam1, mu1 = plane_stress_moduli(E=1.0)
    e_mod, de_mod = moduli(rho, penal, e_min_frac)
    emod = at_quadrature(e_mod, basis)

    K = asm(density_stiffness(lam1, mu1), basis, emod=emod)

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
    return DensityHomogenized(C_eff, fluctuations, basis, area, time.perf_counter() - started,
                              energies, de_mod, rho.shape)

# LENGTH SCALE #
def filter_kernel(radius):
    """Normalized cone weights out to `radius` pixels."""
    r = int(np.ceil(radius))
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    k = np.maximum(0.0, radius - np.hypot(dx, dy))
    return k / k.sum()

def _boundary_weight(shape, radius, mode):
    """top88's Hs: how much of the kernel stays inside a non-periodic domain."""
    return convolve(np.ones(shape), filter_kernel(radius), mode=mode, cval=0.0)

def density_filter(x, radius, mode='wrap'):
    """
    Cone-weighted average over a `radius` pixel neighborhood, wrapping at both seams.

    On a torus a normalized kernel conserves total material exactly, so the volume
    constraint stays meaningful after filtering and top88's boundary correction is not
    needed. `mode='wrap'` is the whole of the periodicity.
    """
    x = np.asarray(x, float)
    out = convolve(x, filter_kernel(radius), mode=mode, cval=0.0)
    return out if mode == 'wrap' else out / _boundary_weight(x.shape, radius, mode)

def filter_chain_rule(sensitivity, radius, mode='wrap'):
    """
    Push a sensitivity in the FILTERED field back onto the design variables.

    The cone kernel is symmetric, so its transpose is itself and the same convolution
    serves both directions. With the boundary correction the operator is diag(1/Hs) K, whose
    transpose is K diag(1/Hs), so the division comes first on the way back.
    """
    g = np.asarray(sensitivity, float)
    if mode != 'wrap':
        g = g / _boundary_weight(g.shape, radius, mode)
    return convolve(g, filter_kernel(radius), mode=mode, cval=0.0)

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
