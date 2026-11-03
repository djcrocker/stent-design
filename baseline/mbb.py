"""
The top88 MBB beam: validation (b) for the SIMP baseline.

The domain is a beam, not a torus, so the filter runs in its boundary-corrected mode. Using
the periodic filter here would be the mistake in the other direction.
"""

import functools

import numpy as np
from skfem import (Basis, ElementQuad1, ElementVector, Functional, MeshQuad1, asm, condense, solve)
from skfem.helpers import ddot, sym_grad, trace

from baseline import optimize, simp
from sim2d.homogenize import plane_stress_moduli

E0 = 1.0                # benchmark modulus
NU = 0.3                # benchmark Poisson ratio
NELX, NELY = 60, 20
VOLFRAC = 0.5
RADIUS = 0.04 * NELX    # 2.4 - the paper's rmin for this mesh, so its number applies
PUBLISHED = 233.71      # Andreassen et al. (2011) Fig. 3, sensitivity filter, same case
                        # (Fig. 4, PDE density filter: 237.60)

@functools.lru_cache(maxsize=2)
def _setup(nelx=NELX, nely=NELY):
    """Mesh, basis, element-to-grid index, load vector and constrained DOFs."""
    mesh = MeshQuad1.init_tensor(np.arange(nelx + 1, dtype=float),
                                 np.arange(nely + 1, dtype=float))
    basis = Basis(mesh, ElementVector(ElementQuad1()))

    # Element order is skfem's.
    cx, cy = mesh.p[:, mesh.t].mean(axis=1)
    index = (np.floor(cy).astype(int), np.floor(cx).astype(int))

    def node_at(x, y):
        return int(np.argmin(np.hypot(mesh.p[0] - x, mesh.p[1] - y)))

    symmetry = basis.nodal_dofs[0, np.nonzero(mesh.p[0] == 0.0)[0]]       # ux = 0 on x = 0
    roller = basis.nodal_dofs[1, node_at(nelx, 0.0)]                      # uy = 0, far corner
    constrained = np.append(symmetry, roller)

    f = np.zeros(basis.N)
    f[basis.nodal_dofs[1, node_at(0.0, nely)]] = -1.0                    # unit load, downward
    return basis, index, f, constrained

def compliance(rho, penal=simp.PENAL, nelx=NELX, nely=NELY):
    """Compliance f.u of the density field, and its sensitivity, both on the (nely, nelx) grid."""
    basis, index, f, constrained = _setup(nelx, nely)
    lam1, mu1 = plane_stress_moduli(E=1.0, nu=NU)
    e_mod, de_mod = simp.moduli(np.asarray(rho, float)[index], penal=penal, e0=E0)

    K = asm(simp.density_stiffness(lam1, mu1), basis, emod=simp.at_quadrature(e_mod, basis))
    u = solve(*condense(K, f, D=constrained))

    @Functional
    def unit_energy(w):
        e = sym_grad(w['u'])
        return 2.0 * mu1 * ddot(e, e) + lam1 * trace(e) ** 2

    # c = sum_e E_e(rho) * u_e.k0_e.u_e, so the sensitivity is self-adjoint here too.
    energy = unit_energy.elemental(basis, u=basis.interpolate(u))
    dc = np.zeros((nely, nelx))
    dc[index] = -de_mod * energy
    return float(f @ u), dc

def run(volfrac=VOLFRAC, radius=RADIUS, nelx=NELX, nely=NELY, **kw):
    """Minimize compliance at the benchmark's volume fraction. Returns an optimize.Result."""
    def grad_fn(rho):
        c, dc = compliance(rho, nelx=nelx, nely=nely)
        return -c, -dc  # the loop maximizes

    result = optimize.optimize(grad_fn, (nely, nelx), volfrac, radius=radius, mode='constant', **kw)
    for row in result.history:
        row['value'] = -row['value']    # report compliance, not the maximized negative
    return result._replace(value=-result.value)
