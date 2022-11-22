import jax.numpy as jnp
from jax.lax import scan
import jax
from jax import vmap

from flux import Flux
from model import stencil_flux_DG_1D_advection


def _upwind_flux_DG_1D_advection(a, core_params):
    """
    Computes the upwind flux F(f_{j+1/2}).

    For c > 0, F(f_{j+1/2}) = f_{j+1/2}^-
    where + = outside and - = inside.
    We always set c > 0.

    Inputs
    a: nx, p array

    Outputs
    F: (nx) array equal to f upwind.
    """
    c = 1.0
    if c > 0:
        F = jnp.sum(a, axis=1)
    else:
        alt = (jnp.ones(core_params.order+1) * -1) ** jnp.arange(core_params.order+1)
        a = jnp.pad(a, ((0, 1), (0, 0)), "wrap")
        F = jnp.sum(alt[None, :] * a[1:, :], axis=1)
    return F


def _centered_flux_DG_1D_advection(a, core_params):
    """
    Computes the centered flux F(f_{j+1/2}).

    F(f_{j+1/2}) = (f_{j+1/2}^- + f_{j+1/2}^+) / 2
    where + = outside and - = inside.

    Inputs
    a: (nx, p) array

    Outputs
    F: (nx) array equal to f averaged.
    """
    a = jnp.pad(a, ((0, 1), (0, 0)), "wrap")
    alt = (jnp.ones(core_params.order+1) * -1) ** jnp.arange(core_params.order+1)
    return 0.5 * (jnp.sum(a[:-1], axis=-1) + jnp.sum(alt[None, :] * a[1:], axis=-1))


def _global_stabilization(f0, a):
    raise NotImplementedError



def _flux_term_DG_1D_advection(a, core_params, global_stabilization=False, model=None, params=None):
    negonetok = (jnp.ones(core_params.order+1) * -1) ** jnp.arange(core_params.order+1)
    if core_params.flux == Flux.UPWIND:
        flux_right = _upwind_flux_DG_1D_advection(a, core_params)
    elif core_params.flux == Flux.CENTERED:
        flux_right = _centered_flux_DG_1D_advection(a, core_params)
    else:
        raise NotImplementedError

    if params is not None:
        delta_flux = stencil_flux_DG_1D_advection(a, model, params)
        flux_right = flux_right + delta_flux

    flux_left = jnp.roll(flux_right, 1, axis=0)
    return negonetok[None, :] * flux_left[:, None] - flux_right[:, None]


def _volume_integral_DG_1D_advection(a):
    volume_sum = jnp.zeros(a.shape).at[:, 1::2].add(2 * jnp.cumsum(a[:, :-1:2], axis=1))
    return volume_sum.at[:, 2::2].add(2 * jnp.cumsum(a[:, 1:-1:2], axis=1))


def time_derivative_DG_1D_advection(core_params, global_stabilization=False, model=None, params=None):
    """
    Compute da_j^m/dt given the matrix a_j^m which represents the solution,
    for a given flux. The time-derivative is given by a Galerkin minimization
    of the residual squared, with Legendre polynomial basis functions.
    For the 1D advection equation
            df/dt + c df/dx = 0
    with f_j = \sum a_j^m \phi_m, the time derivatives equal

    da_j^m/dt = ((2m+1)*c/deltax) [ (-1)^m F(f_{j-1/2}) - F(f_{j+1/2}) ]
                            + c(2m+1) (a_j^{m-1} + a_j^{m-3} + ...)

    Inputs
    a: (nx, p) array of coefficients
    t: time, scalar, not used here
    c: speed (scalar)
    flux: Enum, decides which flux will be used for the boundary

    Outputs
    da_j^m/dt: (nx, p) array of time derivatives
    """
    c = 1.0 
    twokplusone = 2 * jnp.arange(0, core_params.order+1) + 1

    def dadt(a):
        nx, _ = a.shape
        dx = core_params.Lx / nx
        C = (c / dx) * twokplusone
        flux_term = _flux_term_DG_1D_advection(a, core_params, global_stabilization=global_stabilization, model=model, params=params)
        volume_integral = _volume_integral_DG_1D_advection(a)
        return C[None, :] * (flux_term + volume_integral)

    return dadt
