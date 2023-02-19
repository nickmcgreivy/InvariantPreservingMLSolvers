import jax.numpy as np
from functools import partial
import numpy as onp
from basisfunctions import (
    legendre_npbasis,
    node_locations,
    legendre_inner_product,
)
from jax import vmap, hessian, jit


def _trapezoidal_integration(f, xi, xf, yi, yf, n=None):
    return (xf - xi) * (yf - yi) * (f(xi, yi) + f(xf, yi) + f(xi, yf) + f(xf, yf)) / 4


def _2d_fixed_quad(f, xi, xf, yi, yf, n=3):
    """
    Takes a 2D-valued function of two 2D inputs
    f(x,y) and four scalars xi, xf, yi, yf, and
    integrates f over the 2D domain to order n.
    """

    w_1d = {
        1: np.asarray([2.0]),
        2: np.asarray([1.0, 1.0]),
        3: np.asarray(
            [
                0.5555555555555555555556,
                0.8888888888888888888889,
                0.555555555555555555556,
            ]
        ),
        4: np.asarray(
            [
                0.3478548451374538573731,
                0.6521451548625461426269,
                0.6521451548625461426269,
                0.3478548451374538573731,
            ]
        ),
        5: np.asarray(
            [
                0.2369268850561890875143,
                0.4786286704993664680413,
                0.5688888888888888888889,
                0.4786286704993664680413,
                0.2369268850561890875143,
            ]
        ),
        6: np.asarray(
            [
                0.1713244923791703450403,
                0.3607615730481386075698,
                0.4679139345726910473899,
                0.4679139345726910473899,
                0.3607615730481386075698,
                0.1713244923791703450403,
            ]
        ),
        7: np.asarray(
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
        8: np.asarray(
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

    xi_i_1d = {
        1: np.asarray([0.0]),
        2: np.asarray([-0.5773502691896257645092, 0.5773502691896257645092]),
        3: np.asarray([-0.7745966692414833770359, 0.0, 0.7745966692414833770359]),
        4: np.asarray(
            [
                -0.861136311594052575224,
                -0.3399810435848562648027,
                0.3399810435848562648027,
                0.861136311594052575224,
            ]
        ),
        5: np.asarray(
            [
                -0.9061798459386639927976,
                -0.5384693101056830910363,
                0.0,
                0.5384693101056830910363,
                0.9061798459386639927976,
            ]
        ),
        6: np.asarray(
            [
                -0.9324695142031520278123,
                -0.661209386466264513661,
                -0.2386191860831969086305,
                0.238619186083196908631,
                0.661209386466264513661,
                0.9324695142031520278123,
            ]
        ),
        7: np.asarray(
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
        8: np.asarray(
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

    x_w, y_w = np.meshgrid(w_1d, w_1d)
    x_w = x_w.reshape(-1)
    y_w = y_w.reshape(-1)
    w_2d = x_w * y_w

    xi_x, xi_y = np.meshgrid(xi_i_1d, xi_i_1d)
    xi_x = xi_x.reshape(-1)
    xi_y = xi_y.reshape(-1)

    x_i = (xf + xi) / 2 + (xf - xi) / 2 * xi_x
    y_i = (yf + yi) / 2 + (yf - yi) / 2 * xi_y
    wprime = w_2d * (xf - xi) * (yf - yi) / 4
    return np.sum(wprime * f(x_i, y_i))


def evalf_2D(x, y, a, dx, dy, order):
    """
    Returns the value of DG representation of the
    solution at x, y, where x,y is a 2D array of points

    Inputs:
    x, y: 2D array of points
    a: DG representation, (nx, ny, num_elements) ndarray

    Ouputs:
    f: 2d array of points, equal to sum over num_elements polynomials
    """
    j = np.floor(x / dx).astype(int)
    k = np.floor(y / dy).astype(int)
    x_j = dx * (0.5 + j)
    y_k = dy * (0.5 + k)
    xi_x = (x - x_j) / (0.5 * dx)
    xi_y = (y - y_k) / (0.5 * dy)
    f_eval = _eval_legendre(order)
    legendre_val = np.transpose(f_eval(xi_x, xi_y), (1, 2, 0))
    return np.sum(a[j, k, :] * legendre_val, axis=-1)


def _evalf_2D_integrate(x, y, a, dx, dy, order):
    """
    Returns the value of DG representation of the
    solution at x, y, where x and y are a 1d array of points

    Inputs:
    x, y: 1D array of points
    a: DG representation, (nx, ny, num_elements) ndarray

    Ouputs:
    f: 2d array of points, equal to sum over num_elements polynomials
    """
    j = np.floor(x / dx).astype(int)
    k = np.floor(y / dy).astype(int)
    x_j = dx * (0.5 + j)
    y_k = dy * (0.5 + k)
    xi_x = (x - x_j) / (0.5 * dx)
    xi_y = (y - y_k) / (0.5 * dy)
    f_eval = _eval_legendre(order)
    return np.sum(a[j, k, :] * f_eval(xi_x, xi_y).T, axis=-1)


def _eval_legendre(order):
    """
    Takes two 1D vectors xi_x and xi_y, outputs
    the 2D legendre basis at (xi_x, xi_y)
    """
    polybasis = legendre_npbasis(order)  # (order+1, k, 2) matrix
    _vmap_polyval = vmap(np.polyval, (1, None), 0)

    def f(xi_x, xi_y):
        return _vmap_polyval(polybasis[:, :, 0], xi_x) * _vmap_polyval(
            polybasis[:, :, 1], xi_y
        )  # (k,) array

    return f


def inner_prod_with_legendre(sim_params, func, quad_func=_2d_fixed_quad, n=5):

    i = np.arange(sim_params.nx)
    x_i = sim_params.dx * i
    x_f = sim_params.dx * (i + 1)
    j = np.arange(sim_params.ny)
    y_i = sim_params.dy * j
    y_f = sim_params.dy * (j + 1)

    def xi_x(x):
        k = np.floor(x / sim_params.dx)
        x_k = sim_params.dx * (0.5 + k)
        return (x - x_k) / (0.5 * sim_params.dx)

    def xi_y(y):
        k = np.floor(y / sim_params.dy)
        y_k = sim_params.dy * (0.5 + k)
        return (y - y_k) / (0.5 * sim_params.dy)

    quad_lambda = lambda f, xi, xf, yi, yf: quad_func(f, xi, xf, yi, yf, n=n)

    _vmap_integrate = vmap(
        vmap(quad_lambda, (None, 0, 0, None, None), 0), (None, None, None, 0, 0), 1
    )
    to_int_func = lambda x, y: func(x, y) * _eval_legendre(sim_params.order)(xi_x(x), xi_y(y))
    return _vmap_integrate(to_int_func, x_i, x_f, y_i, y_f)


def integrate_fn_fv(sim_params, func, quad_func=_2d_fixed_quad, n=5):

    i = np.arange(sim_params.nx)
    x_i = sim_params.dx * i
    x_f = sim_params.dx * (i + 1)
    j = np.arange(sim_params.ny)
    y_i = sim_params.dy * j
    y_f = sim_params.dy * (j + 1)


    denom = sim_params.dx * sim_params.dy
    quad_lambda = lambda f, xi, xf, yi, yf: quad_func(f, xi, xf, yi, yf, n=n)

    _vmap_integrate = vmap(
        vmap(quad_lambda, (None, 0, 0, None, None), 0), (None, None, None, 0, 0), 1
    )
    return _vmap_integrate(func, x_i, x_f, y_i, y_f) / denom


def f_to_DG(sim_params, func, quad_func=_2d_fixed_quad, n=8):
    inner_prod = legendre_inner_product(sim_params.order)

    return inner_prod_with_legendre(sim_params, func, quad_func=quad_func, n=n) / (
        inner_prod[None, None, :] * sim_params.dx * sim_params.dy
    )


def f_to_FE(nx, ny, Lx, Ly, order, func, t):
    dx = Lx / nx
    dy = Ly / ny
    i = np.arange(nx)
    x_i = dx * i + dx / 2
    j = np.arange(ny)
    y_i = dy * j + dx / 2
    nodes = np.asarray(node_locations(order), dtype=float)

    x_eval = (
        np.ones((nx, ny, nodes.shape[0])) * x_i[:, None, None]
        + nodes[None, None, :, 0] * dx / 2
    )
    y_eval = (
        np.ones((nx, ny, nodes.shape[0])) * y_i[None, :, None]
        + nodes[None, None, :, 1] * dy / 2
    )
    FE_repr = np.zeros((nx, ny, nodes.shape[0]))

    _vmap_evaluate = vmap(vmap(vmap(func, (0, 0, None)), (0, 0, None)), (0, 0, None))
    return _vmap_evaluate(x_eval, y_eval, t)


def convert_DG_representation(
    zeta, old_order, sim_params, n = 8
):
    nx_high, ny_high, _ = zeta.shape
    dx_high = sim_params.Lx / nx_high
    dy_high = sim_params.Ly / ny_high

    def convert_repr(zeta):
        def f_high(x, y):
            return _evalf_2D_integrate(x, y, zeta, dx_high, dy_high, old_order)

        return f_to_DG(sim_params, f_high, n=n)

    return convert_repr(zeta)


def convert_FV_to_DG_representation(zeta, sim_params, n=8):
    return convert_DG_representation(zeta[:,:,None], 0, sim_params, n=n)


def convert_FV_representation(zeta, sim_params, n=8):
    nx_high, ny_high = zeta.shape
    dx_high = sim_params.Lx / nx_high
    dy_high = sim_params.Ly / ny_high

    nx_new = sim_params.nx
    ny_new = sim_params.ny 
    
    if nx_high % nx_new == 0 and ny_high % ny_new == 0:
        return np.mean(zeta.reshape(nx_new, nx_high // nx_new, ny_new, ny_high // ny_new) , axis=(1, 3))   

    def convert_repr(zeta):
        def f_high(x, y):
            return _evalf_2D_integrate(x, y, zeta[:,:, None], dx_high, dy_high, 0)

        return integrate_fn_fv(sim_params, f_high, n=n)

    return convert_repr(zeta)

def batch_convert_DG_representation(*args):
    return vmap(convert_DG_representation, in_axes=(0,None,None))(*args)


def vorticity_to_velocity(Lx, Ly, a, f_poisson):
    H = f_poisson(a)
    nx, ny, _ = H.shape
    dx = Lx / nx
    dy = Ly / ny

    u_y = -(H[:,:,2] - H[:,:,3]) / dx
    u_x = (H[:,:,2] - H[:,:,1]) / dy
    return u_x, u_y


def nabla(f):
    """
    Takes a function of type f(x,y) and returns a function del^2 f(x,y)
    """
    H = hessian(f)
    return lambda x, y: np.trace(H(x, y))


def minmod(r):
    return np.maximum(0, np.minimum(1, r))


def minmod_2(z1, z2):
    s = 0.5 * (np.sign(z1) + np.sign(z2))
    return s * np.minimum(np.absolute(z1), np.absolute(z2))


def minmod_3(z1, z2, z3):
    s = (
        0.5
        * (np.sign(z1) + np.sign(z2))
        * np.absolute(0.5 * ((np.sign(z1) + np.sign(z3))))
    )
    return s * np.minimum(np.absolute(z1), np.minimum(np.absolute(z2), np.absolute(z3)))
