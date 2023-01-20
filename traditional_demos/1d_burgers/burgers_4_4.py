import jax.numpy as np
import numpy as onp
from jax import vmap, jit
from jax.lax import scan
import matplotlib.pyplot as plt
from jax.config import config
from scipy.special import comb
from functools import lru_cache, partial
from sympy import legendre, diff, integrate, symbols
from scipy.special import eval_legendre
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

vmap_polyval = vmap(np.polyval, (0, None), -1)
config.update("jax_enable_x64", True)


#################
# LEGENDRE
#################


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


#################
# POLYNOMIAL RECOVERY
#################


def upper_B(m, k):
    x = symbols("x")
    expr = x ** k * (x + 0.5) ** m
    return integrate(expr, (x, -1, 0))


def lower_B(m, k):
    x = symbols("x")
    expr = x ** k * (x - 0.5) ** m
    return integrate(expr, (x, 0, 1))


def A(m, k):
    x = symbols("x")
    expr = legendre(k, x) * x ** m
    return integrate(expr, (x, -1, 1)) / (2 ** (m + 1))


@lru_cache(maxsize=10)
def get_B_matrix(p):
    res = onp.zeros((2 * p, 2 * p))
    for m in range(p):
        for k in range(2 * p):
            res[m, k] = upper_B(m, k)
    for m in range(p):
        for k in range(2 * p):
            res[m + p, k] = lower_B(m, k)
    return res


@lru_cache(maxsize=10)
def get_inverse_B(p):
    B = get_B_matrix(p)
    return onp.linalg.inv(B)


@lru_cache(maxsize=10)
def get_A_matrix(p):
    res = onp.zeros((2 * p, 2 * p))
    for m in range(p):
        for k in range(p):
            res[m, k] = A(m, k)
            res[m + p, k + p] = A(m, k)
    return res


def get_b_coefficients(a):
    """
    Inputs:
    a: (nx, p) array of coefficients

    Outputs:
    b: (nx, 2p) array of coefficients for the right boundary
    """
    p = a.shape[1]
    a_jplusone = np.roll(a, -1, axis=0)
    a_bar = np.concatenate((a, a_jplusone), axis=1)
    A = get_A_matrix(p)
    B_inv = get_inverse_B(p)
    rhs = np.einsum("ml,jl->jm", A, a_bar)
    b = np.einsum("km,jm->jk", B_inv, rhs)
    return b


def recovery_slope(a, p):
    """
    Inputs:
    a: (nx, p) array of coefficients

    Outputs:
    b: (nx,) array of slopes of recovery polynomial at right boundary
    """
    a_jplusone = np.roll(a, -1, axis=0)
    a_bar = np.concatenate((a, a_jplusone), axis=1)
    A = get_A_matrix(p)
    B_inv = get_inverse_B(p)[1, :]
    rhs = np.einsum("ml,jl->jm", A, a_bar)
    slope = np.einsum("m,jm->j", B_inv, rhs)
    return slope


#################
# RUNGE KUTTA
#################


def forward_euler(a_n, t_n, F, dt, dl_dt=None):
    a_1 = a_n + dt * F(a_n, t_n, dl_dt=dl_dt)
    return a_1, t_n + dt


def ssp_rk2(a_n, t_n, F, dt, dl_dt=None):
    a_1 = a_n + dt * F(a_n, t_n, dl_dt=dl_dt)
    a_2 = 0.5 * a_n + 0.5 * a_1 + 0.5 * dt * F(a_1, t_n + dt, dl_dt=dl_dt)
    return a_2, t_n + dt


def ssp_rk3(a_n, t_n, F, dt, dl_dt=None):
    a_1 = a_n + dt * F(a_n, t_n, dl_dt=dl_dt)
    a_2 = 0.75 * a_n + 0.25 * (a_1 + dt * F(a_1, t_n + dt, dl_dt=dl_dt))
    a_3 = 1 / 3 * a_n + 2 / 3 * (a_2 + dt * F(a_2, dt + dt / 2, dl_dt=dl_dt))
    return a_3, t_n + dt


###############
# HELPER FUNCTIONS
###############


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

    xi_i = {
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

    x_i = (b + a) / 2 + (b - a) / 2 * xi_i
    wprime = w * (b - a) / 2
    return np.sum(wprime[:, None] * f(x_i), axis=0)



def evalf_1D_right(a):
    """
    Returns the representation of f at the right end of
    each of the nx gridpoints

    Inputs
    a: (nx, p) ndarray

    Outputs
    f: (nx,) ndarray
    """
    return np.sum(a, axis=1)


def evalf_1D_left(a, p):
    """
    Returns the representation of f at the left end of
    each of the nx gridpoints

    Inputs
    a: (nx, p) ndarray

    Outputs
    f: (nx,) ndarray
    """
    negonetok = (np.ones(p) * -1) ** np.arange(p)
    return np.sum(a * negonetok, axis=1)


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
    j = np.floor(x / dx).astype(int)
    x_j = dx * (0.5 + j)
    xi = (x - x_j) / (0.5 * dx)
    poly_eval = vmap_polyval(leg_poly, xi)  # nx, p array
    return np.sum(poly_eval * a[j, :], axis=-1)


def map_f_to_DG(f, t, p, nx, dx, leg_poly, quad_func=_fixed_quad, n=5):
    """
    Takes a function f of type lambda x, t: f(x,t) and
    generates the DG representation of the solution, an
    array of size (nx, p).

    Computes the inner product of f with p Legendre polynomials
    over nx regions, to produce an array of size (nx, p)

    Inputs
    f: lambda x, t: f(x, t), the value of f
    t: the current time

    Outputs
    a0: The DG representation of f(x, t) at t=t
    """
    twokplusone = 2 * np.arange(0, p) + 1
    return (
        twokplusone[None, :]
        / dx
        * inner_prod_with_legendre(f, t, p, nx, dx, leg_poly, quad_func=quad_func, n=n)
    )

def inner_prod_with_legendre(f, t, p, nx, dx, leg_poly, quad_func=_fixed_quad, n=5):
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

    _vmap_fixed_quad = vmap(
        lambda f, a, b: quad_func(f, a, b, n=n), (None, 0, 0), 0
    )  # is n = p+1 high enough order?
    twokplusone = np.arange(p) * 2 + 1
    j = np.arange(nx)
    a = dx * j
    b = dx * (j + 1)

    def xi(x):
        j = np.floor(x / dx)
        x_j = dx * (0.5 + j)
        return (x - x_j) / (0.5 * dx)

    to_int_func = lambda x: f(x, t)[:, None] * vmap_polyval(leg_poly, xi(x))

    return _vmap_fixed_quad(to_int_func, a, b)


def l2_norm(a):
    """
    a should be (nx, p)
    """
    twokplusone = 2 * np.arange(0, p) + 1
    return 1/2 * np.mean(np.sum(a**2 / twokplusone, axis=-1))

vmap_l2_norm = vmap(l2_norm)


###################
# TIME DERIVATIVE
###################


def _centered_flux_DG_1D_burgers(a, p):
    """
    Computes the centered flux F(f_{j+1/2}) where
    + is the right side.
    F(f_{j+1/2}) = (f_{j+1/2}^- + f_{j+1/2}^+) / 2
    where + = outside and - = inside.

    Inputs
    a: (nx, p) array

    Outputs
    F: (nx) array equal to f averaged.
    """
    a = np.pad(a, ((0, 1), (0, 0)), "wrap")
    alt = (np.ones(p) * -1) ** np.arange(p)
    u_left = np.sum(a[:-1], axis=-1)
    u_right = np.sum(alt[None, :] * a[1:], axis=-1)
    return ((u_left + u_right) / 2) ** 2 / 2
    #return ((u_left**2 + u_right**2) / 4) # THIS ONE IS STABLE ON BURGERS


def _godunov_flux_DG_1D_burgers(a, p):
    """
    Computes the Godunov flux F(f_{j+1/2}) where
    + is the right side.

    Inputs
    a: (nx, p) array

    Outputs
    F: (nx) array equal to the godunov flux
    """
    a = np.pad(a, ((0, 1), (0, 0)), "wrap")
    alt = (np.ones(p) * -1) ** np.arange(p)
    u_left = np.sum(a[:-1], axis=-1)
    u_right = np.sum(alt[None, :] * a[1:], axis=-1)
    zero_out = 0.5 * np.abs(np.sign(u_left) + np.sign(u_right))
    compare = np.less(u_left, u_right)
    F = lambda u: u ** 2 / 2
    return compare * zero_out * np.minimum(F(u_left), F(u_right)) + (
        1 - compare
    ) * np.maximum(F(u_left), F(u_right))


def _flux_term_DG_1D_burgers(a, t, p, flux):
    negonetok = (np.ones(p) * -1) ** np.arange(p)
    if flux == "centered":
        flux_right = _centered_flux_DG_1D_burgers(a, p)
    elif flux == "godunov":
        flux_right = _godunov_flux_DG_1D_burgers(a, p)
    else:
        raise NotImplementedError

    """
    if dl_dt is not None:
        G = np.repeat(recovery_slope(a, p)[:,None], p, axis=1)
        #if p > 1:
        #    G = G.at[:,0].set(0.0)
        diff = np.roll(a * negonetok, -1, axis=0) - a
        denom = np.sum(G * diff)
        V = np.sum(a * _volume_integral_DG_1D_burgers(a, t, p))
        B = np.sum(flux_right * np.sum(diff, axis=-1))
        dl_dt_old = B + V
        flux_right = flux_right[:, None] + (dl_dt - dl_dt_old) * G / denom
        flux_left = np.roll(flux_right, 1, axis=0)
        return negonetok[None, :] * flux_left - flux_right

    else:
    """
    flux_left = np.roll(flux_right, 1, axis=0)
    return negonetok[None, :] * flux_left[:, None] - flux_right[:, None]


def _volume_integral_DG_1D_burgers(a, t, p):
    if p == 1:
        volume_sum = np.zeros(a.shape)
        return volume_sum
    elif p == 2:
        volume_sum = np.zeros(a.shape).at[:, 1].add(1.0 * a[:, 0] * a[:, 0] + 0.3333333333333333 * a[:, 1] * a[:, 1])
        return volume_sum
    elif p == 3:
        volume_sum = np.zeros(a.shape).at[:, 1].add(1.0 * a[:, 0] * a[:, 0] + 0.3333333333333333 * a[:, 1] * a[:, 1] + 0.2 * a[:, 2] * a[:, 2])
        volume_sum = volume_sum.at[:, 2].add(
            1.0 * a[:, 0] * a[:, 1]
            + 1.0 * a[:, 1] * a[:, 0]
            + 0.4 * a[:, 1] * a[:, 2]
            + 0.4 * a[:, 2] * a[:, 1],
        )
        return volume_sum
    elif p == 4:
        volume_sum = np.zeros(a.shape).at[:, 1].add(1.0 * a[:, 0] * a[:, 0]
            + 0.3333333333333333 * a[:, 1] * a[:, 1]
            + 0.2 * a[:, 2] * a[:, 2]
            + 0.14285714285714285 * a[:, 3] * a[:, 3],
        )   
        volume_sum = volume_sum.at[:, 2].add(
            1.0 * a[:, 0] * a[:, 1]
            + 1.0 * a[:, 1] * a[:, 0]
            + 0.4 * a[:, 1] * a[:, 2]
            + 0.4 * a[:, 2] * a[:, 1]
            + 0.2571428571428571 * a[:, 2] * a[:, 3]
            + 0.2571428571428571 * a[:, 3] * a[:, 2],
        )
        volume_sum = volume_sum.at[:, 3].add(
            1.0 * a[:, 0] * a[:, 0]
            + 1.0 * a[:, 0] * a[:, 2]
            + 1.0 * a[:, 1] * a[:, 1]
            + 0.42857142857142855 * a[:, 1] * a[:, 3]
            + 1.0 * a[:, 2] * a[:, 0]
            + 0.4857142857142857 * a[:, 2] * a[:, 2]
            + 0.42857142857142855 * a[:, 3] * a[:, 1]
            + 0.3333333333333333 * a[:, 3] * a[:, 3],
        )
        return volume_sum
    else:
        raise NotImplementedError


def _diffusion_flux_term_DG_1D_burgers(a, t, p, dx):
    negonetok = (np.ones(p) * -1) ** np.arange(p)
    slope_right = recovery_slope(a, p) / dx
    slope_left = np.roll(slope_right, 1)
    return (slope_right[:, None] - negonetok[None, :] * slope_left[:, None])


def _diffusion_volume_integral_DG_1D_burgers(a, t, p, dx):
    coeff = -2 / dx
    if p == 1:
        volume_sum = np.zeros(a.shape)
    elif p == 2:
        volume_sum = np.zeros(a.shape).at[:, 1].add(2.0 * a[:, 1])
    elif p == 3:
        volume_sum = np.zeros(a.shape).at[:, 1].add(2.0 * a[:, 1])
        volume_sum = volume_sum.at[:, 2].add(6.0 * a[:, 2])
    elif p == 4:
        volume_sum = np.zeros(a.shape).at[:, 1].add(2.0 * a[:, 1] + 2.0 * a[:, 3])
        volume_sum = volume_sum.at[:, 2].add(6.0 * a[:, 2])
        volume_sum = volume_sum.at[:, 3].add(2.0 * a[:, 1] + 12.0 * a[:, 3])
    else:
        raise NotImplementedError
    return coeff * volume_sum


def time_derivative_DG_1D_burgers(
    a, t, p, flux, nx, dx, leg_poly, forcing_func=None, nu=0.0, dl_dt=None,
):
    """
    Compute da_j^m/dt given the matrix a_j^m which represents the solution,
    for a given flux. The time-derivative is given by a Galerkin minimization
    of the residual squared, with Legendre polynomial basis functions.
    For the 1D burgers equation
            df/dt + c df/dx = 0
    with f_j = \sum a_j^m \phi_m, the time derivatives equal

    da_j^m/dt = ...

    Inputs
    a: (nx, p) array of coefficients
    t: time, scalar, not used here
    c: speed (scalar)
    flux: Enum, decides which flux will be used for the boundary

    Outputs
    da_j^m/dt: (nx, p) array of time derivatives
    """
    twokplusone = 2 * np.arange(0, p) + 1
    flux_term = _flux_term_DG_1D_burgers(a, t, p, flux)
    volume_integral = _volume_integral_DG_1D_burgers(a, t, p)
    dif_flux_term = _diffusion_flux_term_DG_1D_burgers(a, t, p, dx)
    dif_volume_integral = _diffusion_volume_integral_DG_1D_burgers(a, t, p, dx)

    if dl_dt is not None:
        dl_dt_old = np.sum(a * (flux_term + volume_integral))
        dl_dt_diffusion = np.sum(a * (dif_flux_term + dif_volume_integral))
        nu = nu + (dl_dt - dl_dt_old) / dl_dt_diffusion



    if forcing_func is not None:
        forcing_term = inner_prod_with_legendre(forcing_func, t, p, nx, dx, leg_poly, n = 2 * p - 1)
    else:
        forcing_term = 0.0
    return (twokplusone[None, :] / dx) * (
        flux_term + volume_integral + nu * (dif_flux_term + dif_volume_integral) + forcing_term
    )


#######################
# SIMULATE
#######################



def _scan(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t, x)
    return (a_f, t_f), None


def _scan_output(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t, x)
    return (a_f, t_f), (a, t)




def simulate_1D(
    a0,
    t0,
    p,
    flux,
    nx,
    dx,
    dt,
    nt,
    leg_poly,
    output=False,
    forcing_func=None,
    nu=0.0,
    rk=ssp_rk3,
    dl_dt = None,
):

    dadt = lambda a, t, dl_dt: time_derivative_DG_1D_burgers(
        a,
        t,
        p,
        flux,
        nx,
        dx,
        leg_poly,
        forcing_func=forcing_func,
        nu=nu,
        dl_dt = dl_dt,
    )

    rk_F = lambda a, t, dl_dt: rk(a, t, dadt, dt, dl_dt=dl_dt)


    if output:
        scanf = jit(lambda sol, x: _scan_output(sol, x, rk_F))
        _, data = scan(scanf, (a0, t0), dl_dt, length=nt)
        return data
    else:
        scanf = jit(lambda sol, x: _scan(sol, x, rk_F))
        (a_f, t_f), _ = scan(scanf, (a0, t0), dl_dt, length=nt)
        return (a_f, t_f)


#################
# PLOTTING
#################


def plot_subfig(
    a, subfig, L, color="blue", linewidth=0.5, linestyle="solid", label=None
):
    def evalf(x, a, j, dx, leg_poly):
        x_j = dx * (0.5 + j)
        xi = (x - x_j) / (0.5 * dx)
        vmap_polyval = vmap(np.polyval, (0, None), -1)
        poly_eval = vmap_polyval(leg_poly, xi)  # nx, p array
        return np.sum(poly_eval * a, axis=-1)

    nx = a.shape[0]
    p = a.shape[1]
    dx = L / nx
    xjs = np.arange(nx) * L / nx
    xs = xjs[None, :] + np.linspace(0.0, dx, 10)[:, None]
    vmap_eval = vmap(evalf, (1, 0, 0, None, None), 1)
    subfig.plot(
        xs,
        vmap_eval(xs, a, np.arange(nx), dx, generate_legendre(p)),
        color=color,
        linewidth=linewidth,
        label=label,
        linestyle=linestyle,
    )
    return



#################
# INIT
#################



#####
# order = 1, p = 2
#####


p = 2
nx = 16
UPSAMPLE = 10
T = 0.5
L = 1.0
t0 = 0.0
Np = 4
f_init = lambda x, t: np.sin(2 * np.pi * x)
cfl_safety = 0.2
dx = L/nx
dt = cfl_safety * dx / (2 * p - 1)
nt = T // dt + 1
leg_poly = generate_legendre(p)
nx_exact = UPSAMPLE * nx
dx_exact = L / nx_exact

a0 = map_f_to_DG(f_init, t0, p, nx, dx, leg_poly, n=8)
a0_exact = map_f_to_DG(f_init, t0, p, nx_exact, dx_exact, leg_poly, n=8)

dt_exact = cfl_safety * dx_exact / (2 * p - 1)
nt_exact = nt * UPSAMPLE + 1

a_data, t_data = simulate_1D(a0_exact, t0, p, "godunov", nx_exact, dx_exact, dt_exact, nt_exact, leg_poly, output=True)
dl_dt_exact = (vmap_l2_norm(a_data[1:]) - vmap_l2_norm(a_data[:-1])) / dt_exact

dl_dt_upsample = np.mean(dl_dt_exact.reshape(-1, UPSAMPLE), axis=-1)
a_godunov = a_data[:-1][::UPSAMPLE]
t_godunov = t_data[:-1][::UPSAMPLE]




a_centered, _ = simulate_1D(a0, t0, p, "centered", nx, dx, dt, nt, leg_poly, output="True")
a_stabilized, _ = simulate_1D(a0, t0, p, "centered", nx, dx, dt, nt, leg_poly, output="True", dl_dt = np.zeros(dl_dt_upsample.shape))
a_stabilized2, _ = simulate_1D(a0, t0, p, "centered", nx, dx, dt, nt, leg_poly, output="True", dl_dt = dl_dt_upsample)


num = a_godunov.shape[0]
js = [0, int(0.20 * num), int(0.40 * num), int(0.60 * num)]

assert len(js) == Np

a_godunov_list = []
a_centered_list = []
a_stabilized_list = []
a_stabilized2_list = []


for j in js:
    a_godunov_list.append(a_godunov[j])
    a_centered_list.append(a_centered[j])
    a_stabilized_list.append(a_stabilized[j])
    a_stabilized2_list.append(a_stabilized2[j])


for i in range(len(a_godunov_list)):
    print("Godunov {}: {}".format(i, l2_norm(a_godunov_list[i])))
    print("Centered {}: {}".format(i, l2_norm(a_centered_list[i])))
    print("Stabilized {}: {}".format(i, l2_norm(a_stabilized_list[i])))
    print("Stabilized2 {}: {}".format(i, l2_norm(a_stabilized2_list[i])))

fig, axs = plt.subplots(1, Np, sharex=True, sharey=True, squeeze=True, figsize=(8,3))

for j in range(Np):
    plot_subfig(a_godunov_list[j], axs[j], L, color="grey", label="Exact\nsolution", linewidth=1.2)
    plot_subfig(a_centered_list[j], axs[j], L, color="#ff5555", label="Centered flux\n(unstable)", linewidth=1.5)
    plot_subfig(a_stabilized_list[j], axs[j], L, color="#003366", label='$\ell_2$-norm conserving\ncentered flux', linewidth=1.5)
    plot_subfig(a_stabilized2_list[j], axs[j], L, color="#007733", label='$\ell_2$-norm decaying\ncentered flux', linewidth=1.5)
    axs[j].plot(np.zeros(len(a_stabilized)), '--',  color="black", linewidth=0.4)

axs[0].set_xlim([0, 1])
axs[0].set_ylim([-2.0, 2.0])

axs[0].spines['left'].set_visible(False)
axs[Np-1].spines['right'].set_visible(False)
for j in range(Np):
    axs[j].set_yticklabels([])
    axs[j].set_xticklabels([])
    axs[j].spines['top'].set_visible(False)
    axs[j].spines['bottom'].set_visible(False)
    axs[j].tick_params(bottom=False)
    axs[j].tick_params(left=False)

#for j in range(Np):
#axs[j].set_axis_off()
#for j in range(Np):
#    axs[j].grid(True, which="both")


plt.style.use('seaborn')

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axs[1].text(0.15, 0.15, r'$\frac{\partial u}{\partial t} + \frac{\partial}{\partial x} (\frac{u^2}{2}) = 0$', transform=axs[1].transAxes, fontsize=14, verticalalignment='top', bbox=props)

# place a text box in upper left in axes coords 
axs[0].text(0.02, 0.95, "$t={:.1f}$".format(t_godunov[js[0]]), transform=axs[0].transAxes, fontsize=13,
        verticalalignment='top')
axs[1].text(0.02, 0.95, "$t={:.1f}$".format(t_godunov[js[1]]), transform=axs[1].transAxes, fontsize=13,
        verticalalignment='top')
axs[2].text(0.02, 0.95, "$t={:.1f}$".format(t_godunov[js[2]]), transform=axs[2].transAxes, fontsize=13,
        verticalalignment='top')
axs[3].text(0.02, 0.95, "$t={:.1f}$".format(t_godunov[js[3]]), transform=axs[3].transAxes, fontsize=13,
        verticalalignment='top')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),loc=(0.003,0.002), prop={'size': 10})

#fig.suptitle("")

fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)


plt.savefig("burgers_dg_order{}_demo.eps".format(p-1))
plt.savefig("burgers_dg_order{}_demo.png".format(p-1))

plt.show()


