"""
The optimizer loop for the SIMP baseline: optimality criteria on filtered, projected
densities.

Formulation: maximize homogenized stiffness at a fixed volume fraction, and reach a prescribed
`K_radial` by bisecting the volume fraction.
"""

import time
from collections import namedtuple

import numpy as np

import config
from baseline import simp
from geom.cell import UnitCell

Result = namedtuple('Result', 'x rho value volume history seconds')

def default_radius(n=None):
    """
    Filter radius in pixels: half the minimum feature, floored at 1.5 px.

    The filter sets a length scale but doesn't guarantee our criterion, so the delivered
    cell is still checked against `geom.validity`, which measures minimum feature directly.
    """
    n = config.GRID_N if n is None else n
    px_mm = config.cell_extent_mm()[0] / n
    return max(1.5, (config.MIN_FEATURE_MM / px_mm) / 2.0)

def initial_design(shape, volfrac, hole_frac=0.25):
    """
    A starting design whose symmetry is already broken.

    Uniform density is a stationary point of stiffness maximization on a periodic cell: the
    gradient of a symmetric field is symmetric, so the OC update rescales it back to the
    same field and the loop reports convergence having moved nothing.

    A soft centered disc is the standard choice (Sigmund's inverse-homogenization setup) and
    is deterministic, so runs are reproducible without carrying a seed.
    """
    ii, jj = np.indices(shape) + 0.5
    radius = hole_frac * min(shape)
    x = np.full(shape, float(volfrac))
    x[np.hypot(ii - shape[0] / 2.0, jj - shape[1] / 2.0) < radius] *= 0.5
    return x

def beta_at(iteration, iters, beta_max=16.0):
    """No projection for the first half, then doubling."""
    if beta_max <= 0.0 or iteration < iters // 2:
        return 0.0
    stage = max(1, iters // 10)
    return float(min(beta_max, 2.0 ** ((iteration - iters // 2) // stage)))

def oc_step(x, benefit, dv, volume_fn, volfrac, move=0.2, damping=0.5):
    """
    `benefit` is d(objective)/dx for an objective being maximized, so compliance
    minimization passes `-dc`.

    Bisection runs on log10 of the multiplier because the three objectives in this project
    differ by orders of magnitude (compliance ~2e2, K_radial ~4e1, bulk ~1e4) and a linear
    bracket tuned to one of them would fail to converge on another.
    """
    target = volfrac * x.size
    benefit = np.maximum(np.asarray(benefit, float), 0.0)   # OC needs a sign it can invert
    dv = np.maximum(np.asarray(dv, float), 1e-30)
    lo, hi, x_new = -40.0, 40.0, x
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        step = (benefit / (10.0 ** mid * dv)) ** damping
        x_new = np.clip(np.clip(x * step, x - move, x + move), 0.0, 1.0)
        if volume_fn(x_new) > target:
            lo = mid
        else:
            hi = mid
    return x_new

def optimize(grad_fn, shape, volfrac, radius=None, iters=60, move=0.2, damping=0.5,
             beta_max=16.0, mode='wrap', x0=None, tol=1e-3, callback=None, eta=0.5):
    """
    Maximize `grad_fn` at a fixed volume fraction.

    `grad_fn(rho) -> (value, dvalue_drho)` on the physical (filtered, projected) field.
    Returns the design variables, the physical field, and a per-iteration history, so a
    broken loop is diagnosable from the history instead of only from its answer.

    `eta` is the projection threshold, and raising it above 0.5 is the robust formulation:
    the field this loop optimizes and constrains is then the eroded design, so a member has
    to stay load-bearing after erosion, and the caller delivers the intermediate (eta = 0.5)
    design, which is thicker by the erosion margin.
    """
    started = time.perf_counter()
    radius = default_radius(shape[0]) if radius is None else radius
    x = np.full(shape, float(volfrac)) if x0 is None else np.clip(np.asarray(x0, float), 0.0, 1.0)
    history = []
    value = rho = None

    for iteration in range(iters):
        beta = beta_at(iteration, iters, beta_max)
        x_tilde = simp.density_filter(x, radius, mode)
        rho = simp.project(x_tilde, beta, eta)
        value, dvalue = grad_fn(rho)

        dproject = simp.project_grad(x_tilde, beta, eta)
        benefit = simp.filter_chain_rule(dvalue * dproject, radius, mode)
        dv = simp.filter_chain_rule(dproject, radius, mode)

        def volume_fn(candidate):
            return simp.project(simp.density_filter(candidate, radius, mode), beta, eta).sum()

        x_next = oc_step(x, benefit, dv, volume_fn, volfrac, move, damping)
        change = float(np.max(np.abs(x_next - x)))
        history.append({'iteration': iteration, 'value': float(value), 'beta': beta,
                        'volume': float(rho.mean()), 'change': change,
                        'gray': float(np.mean(4.0 * rho * (1.0 - rho)))})
        if callback is not None:
            callback(history[-1])
        x = x_next
        if change < tol and beta >= beta_max:
            break

    rho = simp.project(simp.density_filter(x, radius, mode),
                       beta_at(iters, iters, beta_max), eta)
    value = grad_fn(rho)[0]
    return Result(x, rho, float(value), float(rho.mean()), history, time.perf_counter() - started)

def optimize_cell(volfrac, objective='k_radial', n=None, **kw):
    """Inverse homogenization on the periodic cell."""
    n = config.GRID_N if n is None else n
    if kw.get('x0') is None:    # also catches an explicit x0=None warm start
        kw['x0'] = initial_design((n, n), volfrac)

    def grad_fn(rho):
        solved = simp.homogenize_density(rho)
        if objective == 'k_radial':
            return solved.K_radial, solved.dK_radial()
        if objective == 'bulk':
            return simp.bulk(solved.C_eff), solved.dbulk()
        raise ValueError(f"objective must be 'k_radial' or 'bulk', got {objective!r}")

    return optimize(grad_fn, (n, n), volfrac, **kw)

def cell_for_k_radial(target, bounds=(0.10, 0.90), rtol=0.02, max_outer=6, n=None, **kw):
    """
    One cell at a prescribed `K_radial`.

    K_radial rises monotonically with the volume fraction, so the constrained problem is
    solved by bisecting the volume constraint. That costs one full optimization per outer
    step, hence the warm start from the previous bracket's design variables.

    Returns the closest run, which could miss `target` if it lies outside `bounds`.
    """
    lo, hi = bounds
    best = last = None
    for _ in range(max_outer):
        mid = 0.5 * (lo + hi)
        last = optimize_cell(mid, n=n, x0=(None if last is None else last.x), **kw)
        if best is None or abs(last.value - target) < abs(best.value - target):
            best = last
        if abs(last.value - target) <= rtol * target:
            break
        lo, hi = (mid, hi) if last.value < target else (lo, mid)
    return best

def to_cell(rho, threshold=0.5):
    """The physical density field as the binary cell the rest of the project consumes."""
    return UnitCell(np.asarray(rho) > threshold)
