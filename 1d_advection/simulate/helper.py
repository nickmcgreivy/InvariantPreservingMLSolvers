import jax.numpy as jnp
from jax import vmap, jit
from functools import partial, lru_cache
import numpy as onp
from scipy.special import comb

vmap_polyval = vmap(jnp.polyval, (0, None), -1)


@lru_cache(maxsize=10)
def generate_legendre(p):
    """
    Returns a (p, p) array which represents
    p length-p polynomials. legendre_poly[k] gives
    an array which represents the kth Legendre
    polynomial. The polynomials are represented
    from highest degree of x (x^{p-1}) to lowest degree (x^0),
    as is standard in numpy poly1d.
    Inputs
    p: the number of Legendre polynomials
    Outputs
    poly: (p, p) array representing the Legendre polynomials
    """
    assert p >= 1
    poly = onp.zeros((p, p))
    poly[0, -1] = 1.0
    twodpoly = onp.asarray([0.5, -0.5])
    for n in range(1, p):
        for k in range(n + 1):
            temp = onp.asarray([1.0])
            for j in range(k):
                temp = onp.polymul(temp, twodpoly)
            temp *= comb(n, k) * comb(n + k, k)
            poly[n] = onp.polyadd(poly[n], temp)

    return poly


def _fixed_quad(f, a, b, n=5):
    """
    Single quadrature of a given order.

    Inputs
    f: function which takes a vector of positions of length n
    and returns a (possibly) multivariate output of length (n, p)
    a: beginning of integration
    b: end of integration
    n: order of quadrature. max n is 8.
    """
    assert isinstance(n, int) and n <= 8 and n > 0
    w = {
        1: jnp.asarray([2.0]),
        2: jnp.asarray([1.0, 1.0]),
        3: jnp.asarray(
            [
                0.5555555555555555555556,
                0.8888888888888888888889,
                0.555555555555555555556,
            ]
        ),
        4: jnp.asarray(
            [
                0.3478548451374538573731,
                0.6521451548625461426269,
                0.6521451548625461426269,
                0.3478548451374538573731,
            ]
        ),
        5: jnp.asarray(
            [
                0.2369268850561890875143,
                0.4786286704993664680413,
                0.5688888888888888888889,
                0.4786286704993664680413,
                0.2369268850561890875143,
            ]
        ),
        6: jnp.asarray(
            [
                0.1713244923791703450403,
                0.3607615730481386075698,
                0.4679139345726910473899,
                0.4679139345726910473899,
                0.3607615730481386075698,
                0.1713244923791703450403,
            ]
        ),
        7: jnp.asarray(
            [
                0.1294849661688696932706,
                0.2797053914892766679015,
                0.38183005050511894495,
                0.417959183673469387755,
                0.38183005050511894495,
                0.279705391489276667901,
                0.129484966168869693271,
            ]
        ),
        8: jnp.asarray(
            [
                0.1012285362903762591525,
                0.2223810344533744705444,
                0.313706645877887287338,
                0.3626837833783619829652,
                0.3626837833783619829652,
                0.313706645877887287338,
                0.222381034453374470544,
                0.1012285362903762591525,
            ]
        ),
    }[n]

    xi_i = {
        1: jnp.asarray([0.0]),
        2: jnp.asarray([-0.5773502691896257645092, 0.5773502691896257645092]),
        3: jnp.asarray([-0.7745966692414833770359, 0.0, 0.7745966692414833770359]),
        4: jnp.asarray(
            [
                -0.861136311594052575224,
                -0.3399810435848562648027,
                0.3399810435848562648027,
                0.861136311594052575224,
            ]
        ),
        5: jnp.asarray(
            [
                -0.9061798459386639927976,
                -0.5384693101056830910363,
                0.0,
                0.5384693101056830910363,
                0.9061798459386639927976,
            ]
        ),
        6: jnp.asarray(
            [
                -0.9324695142031520278123,
                -0.661209386466264513661,
                -0.2386191860831969086305,
                0.238619186083196908631,
                0.661209386466264513661,
                0.9324695142031520278123,
            ]
        ),
        7: jnp.asarray(
            [
                -0.9491079123427585245262,
                -0.7415311855993944398639,
                -0.4058451513773971669066,
                0.0,
                0.4058451513773971669066,
                0.7415311855993944398639,
                0.9491079123427585245262,
            ]
        ),
        8: jnp.asarray(
            [
                -0.9602898564975362316836,
                -0.7966664774136267395916,
                -0.5255324099163289858177,
                -0.1834346424956498049395,
                0.1834346424956498049395,
                0.5255324099163289858177,
                0.7966664774136267395916,
                0.9602898564975362316836,
            ]
        ),
    }[n]

    x_i = (b + a) / 2 + (b - a) / 2 * xi_i
    wprime = w * (b - a) / 2
    return jnp.sum(wprime[:, None] * f(x_i), axis=0)


def evalf_1D_right(a):
    """
    Returns the representation of f at the right end of
    each of the nx gridpoints

    Inputs
    a: (nx, p) ndarray

    Outputs
    f: (nx,) ndarray
    """
    return jnp.sum(a, axis=1)


def evalf_1D_left(a, p):
    """
    Returns the representation of f at the left end of
    each of the nx gridpoints

    Inputs
    a: (nx, p) ndarray

    Outputs
    f: (nx,) ndarray
    """
    negonetok = (jnp.ones(p) * -1) ** jnp.arange(p)
    return jnp.sum(a * negonetok, axis=1)


def evalf_1D(x, a, dx, leg_poly):
    """
    Returns the value of DG representation of the
    solution at x.

    Inputs:
    x: 1D array of points
    a: DG representation, (nx, p) ndarray

    Ouputs:
    f: 1d array of points, equal to sum over p polynomials
    """
    j = jnp.floor(x / dx).astype(int)
    x_j = dx * (0.5 + j)
    xi = (x - x_j) / (0.5 * dx)
    poly_eval = vmap_polyval(leg_poly, xi)  # nx, p array
    return jnp.sum(poly_eval * a[j, :], axis=-1)


def map_f_to_DG(f, t, p, nx, dx, quad_func=_fixed_quad, n=5):
    twokplusone = 2 * jnp.arange(0, p) + 1
    return (
        twokplusone[None, :]
        / dx
        * inner_prod_with_legendre(f, t, p, nx, dx, quad_func=quad_func, n=n)
    )


def map_f_to_FV(f, t, nx, dx, quad_func=_fixed_quad, n=5):
    return map_f_to_DG(f, t, 1, nx, dx, quad_func=quad_func, n=n)[..., 0]


def inner_prod_with_legendre(f, t, p, nx, dx, quad_func=_fixed_quad, n=5):
    """
    Takes a function f of type lambda x, t: f(x,t) and
    takes the inner product of the solution with p
    legendre polynomials over all nx grid cells,
    resulting in an array of size (nx, p).

    Inputs
    f: lambda x, t: f(x, t), the value of f
    t: the current time
    leg_poly: the legendre coefficients

    Outputs
    integral: The inner product representation of f(x, t) at t=t
    """

    leg_poly = generate_legendre(p)

    _vmap_fixed_quad = vmap(
        lambda f, a, b: quad_func(f, a, b, n=n), (None, 0, 0), 0
    )  # is n = p+1 high enough order?
    twokplusone = jnp.arange(p) * 2 + 1
    j = jnp.arange(nx)
    a = dx * j
    b = dx * (j + 1)

    def xi(x):
        j = jnp.floor(x / dx)
        x_j = dx * (0.5 + j)
        return (x - x_j) / (0.5 * dx)

    to_int_func = lambda x: f(x, t)[:, None] * vmap_polyval(leg_poly, xi(x))

    return _vmap_fixed_quad(to_int_func, a, b)


@partial(
    jit,
    static_argnums=(
        1,
        2,
    ),
)
def convert_DG_representation(a, p_new, nx_new, Lx):
    """
    # Converts one DG representation to another. Starts by writing a function
    # which does the mapping for a single timestep, then vmaps for many timesteps.

    # Inputs
    # a: (nx, p_old), high-resolution DG representation
    # p_new: The order of the new representation
    # upsampling: Spatial upsampling of new resolution

    # Outputs
    # a_new: (nx//upsampling, p_new), low-resolution DG representation
    """
    nx_old, p_old = a.shape
    if p_new == p_old and nx_new == nx_old:
        return a
    leg_poly_old = generate_legendre(p_old)

    dx_new = Lx / nx_new
    dx_old = Lx / nx_old

    def convert_repr(a):
        """
        Same function except a is (nx, p_old) and a_new is (nx//upsampling, p_new)
        """

        def f_old(x, t):
            res = evalf_1D(x, a, dx_old, leg_poly_old)
            return res

        a_pre = map_f_to_DG(
            f_old,
            0.0,
            p_new,
            nx_new,
            dx_new,
            quad_func=_quad_two_per_interval,
            n=8,
        )
        return a_pre

    return convert_repr(a)


@partial(
    jit,
    static_argnums=(1,),
)
def convert_FV_representation(a, nx_new, Lx):
    """
    Converts one FV representation to another. Starts by writing a function
    which does the mapping for a single grid cell, then vmaps for many grid cells.
    """
    nx_old = a.shape[0]
    if nx_old >= nx_new and nx_old % nx_new == 0:
        return jnp.mean(a.reshape(-1, nx_old // nx_new), axis=-1)

    return convert_DG_representation(a[..., None], 1, nx_new, Lx)[..., 0]
