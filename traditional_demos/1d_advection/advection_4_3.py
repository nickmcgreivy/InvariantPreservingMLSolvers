import jax.numpy as np
from jax import vmap, jit
from jax.lax import scan
import matplotlib.pyplot as plt
from jax.config import config

config.update("jax_enable_x64", True)

vmap_polyval = vmap(np.polyval, (0, None), -1)

import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)


def forward_euler(a_n, t_n, F, dt, delta_l2=None):
    update = dt * F(a_n, t_n)
    if delta_l2 is not None:
        update = update - np.mean(update)
        G = np.roll(a_n, -1, axis=0) + np.roll(a_n, 1, axis=0) - 2 * a_n
        coeff = np.sum((a_n + update) * G) / np.sum(G**2)
        sqrt_arg = (
            (np.sum(G**2))
            * (2 * np.sum(a_n * update) + np.sum(update**2) - delta_l2)
            / ((np.sum((a_n + update) * G)) ** 2)
        )
        epsilon = coeff * (-1 + np.sqrt(1 - sqrt_arg))
        update = update + epsilon * G
    a_1 = a_n + update
    return a_1, t_n + dt


def ssp_rk2(a_n, t_n, F, dt, delta_l2=None):
    a_1 = a_n + dt * F(a_n, t_n)
    a_2 = 0.5 * a_n + 0.5 * a_1 + 0.5 * dt * F(a_1, t_n + dt)
    return a_2, t_n + dt


def ssp_rk3(a_n, t_n, F, dt, delta_l2=None):
    a_1 = a_n + dt * F(a_n, t_n)
    a_2 = 0.75 * a_n + 0.25 * (a_1 + dt * F(a_1, t_n + dt))
    a_3 = 1 / 3 * a_n + 2 / 3 * (a_2 + dt * F(a_2, dt + dt / 2))
    return a_3, t_n + dt


def _quad_two_per_interval(f, a, b, n=5):
    mid = (a + b) / 2
    return _fixed_quad(f, a, mid, n) + _fixed_quad(f, mid, b, n)


def _fixed_quad(f, a, b, n=5):
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


def inner_prod_with_legendre(f, t, nx, dx, quad_func=_fixed_quad, n=5):
    _vmap_fixed_quad = vmap(lambda f, a, b: quad_func(f, a, b, n=n), (None, 0, 0), 0)
    j = np.arange(nx)
    a = dx * j
    b = dx * (j + 1)

    def xi(x):
        j = np.floor(x / dx)
        x_j = dx * (0.5 + j)
        return (x - x_j) / (0.5 * dx)

    to_int_func = lambda x: f(x, t)[:, None] * vmap_polyval(np.asarray([[1.0]]), xi(x))

    return _vmap_fixed_quad(to_int_func, a, b)


def map_f_to_FV(f, t, nx, dx, quad_func=_fixed_quad, n=5):
    return inner_prod_with_legendre(f, t, nx, dx, quad_func=quad_func, n=n) / dx


def evalf_1D(x, a, dx, leg_poly):
    j = np.floor(x / dx).astype(int)
    x_j = dx * (0.5 + j)
    xi = (x - x_j) / (0.5 * dx)
    poly_eval = vmap_polyval(np.asarray([[1.0]]), xi)  # nx, p array
    return np.sum(poly_eval * a[j, :], axis=-1)


def _scan(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t)
    return (a_f, t_f), None


def _scan_output(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t)
    return (a_f, t_f), (a, t)


def _muscl_flux_1D_advection(a):
    raise NotImplementedError


def _centered_flux_1D_advection(a):
    u = np.sum(a, axis=-1)
    return (u + np.roll(u, -1)) / 2


def _upwind_flux_1D_advection(a):
    return np.sum(a, axis=-1)


def time_derivative_1D_advection(a, t, nx, dx, flux):
    if flux == "centered":
        flux_right = _centered_flux_1D_advection(a)
    elif flux == "upwind":
        flux_right = _upwind_flux_1D_advection(a)
    else:
        raise Exception

    flux_left = np.roll(flux_right, 1, axis=0)
    return (flux_left[:, None] - flux_right[:, None]) / dx


def simulate_1D(
    a0,
    t0,
    nx,
    dx,
    dt,
    nt,
    output=False,
    rk=ssp_rk3,
    delta_l2=None,
    flux="centered",
):
    dadt = lambda a, t: time_derivative_1D_advection(
        a,
        t,
        nx,
        dx,
        flux,
    )

    rk_F = lambda a, t: rk(a, t, dadt, dt, delta_l2=delta_l2)

    if output:
        scanf = jit(lambda sol, x: _scan_output(sol, x, rk_F))
        _, data = scan(scanf, (a0, t0), None, length=nt)
        return data
    else:
        scanf = jit(lambda sol, x: _scan(sol, x, rk_F))
        (a_f, t_f), _ = scan(scanf, (a0, t0), None, length=nt)
        return (a_f, t_f)


def plot_subfig(
    a, subfig, L, color="blue", linewidth=0.5, linestyle="solid", label=None
):
    def evalf(x, a, j, dx):
        x_j = dx * (0.5 + j)
        xi = (x - x_j) / (0.5 * dx)
        vmap_polyval = vmap(np.polyval, (0, None), -1)
        poly_eval = vmap_polyval(np.asarray([[1.0]]), xi)
        return np.sum(poly_eval * a, axis=-1)

    nx = a.shape[0]
    dx = L / nx
    xjs = np.arange(nx) * L / nx
    xs = xjs[None, :] + np.linspace(0.0, dx, 10)[:, None]
    vmap_eval = vmap(evalf, (1, 0, 0, None), 1)
    subfig.plot(
        xs,
        vmap_eval(xs, a, np.arange(nx), dx),
        color=color,
        linewidth=linewidth,
        label=label,
        linestyle=linestyle,
    )
    return


def plot_subfig_oscillations(
    a, subfig, L, color="blue", linewidth=0.5, linestyle="solid", label=None
):
    def evalf(x, a, j, dx):
        x_j = dx * (0.5 + j)
        xi = (x - x_j) / (0.5 * dx)
        vmap_polyval = vmap(np.polyval, (0, None), -1)
        poly_eval = vmap_polyval(np.asarray([[1.0]]), xi)
        return np.sum(poly_eval * a, axis=-1)

    nx = a.shape[0]
    dx = L / nx
    xjs = np.arange(nx) * L / nx
    xs = xjs[None, :] + np.linspace(0.0, dx, 10)[:, None]
    vmap_eval = vmap(evalf, (1, 0, 0, None), 1)
    res = vmap_eval(xs, a, np.arange(nx), dx)
    subfig.plot(
        xs.T.reshape(-1),
        res.T.reshape(-1),
        color=color,
        linewidth=linewidth,
        label=label,
        linestyle=linestyle,
    )
    return


def l2_norm(a):
    """
    a should be (nx, 1)
    """
    return 1 / 2 * np.mean(a**2)


vmap_l2_norm = vmap(l2_norm)


nx = 20
Np = 3
T = 1.0
L = 1.0
t0 = 0.0
fs = 14
f_init = lambda x, t: np.sin(2 * np.pi * x)
f_exact = lambda x, t: np.sin(2 * np.pi * (x - t))
cfl_safety = 0.5

dx = L / nx
dt = cfl_safety * dx
nt = T // dt + 1
a0 = map_f_to_FV(f_init, t0, nx, dx)

UPSAMPLE = 20

nx_exact = nx * UPSAMPLE
dx_exact = L / nx_exact


#### fluxes

a_centered, t_centered = simulate_1D(
    a0, t0, nx, dx, dt, nt, flux="centered", output="True", rk=forward_euler
)
a_update, _ = simulate_1D(
    a0,
    t0,
    nx,
    dx,
    dt,
    nt,
    flux="centered",
    output="True",
    rk=forward_euler,
    delta_l2=0.0,
)

num = a_centered.shape[0]
js = [0, int(num * 0.5), -1]
assert Np == len(js)

a_centered_list = []
a_update_list = []
a_exact_list = []


for j in js:
    a_centered_list.append(a_centered[j])
    a_update_list.append(a_update[j])
    a_exact_j = map_f_to_FV(f_exact, t_centered[j], nx_exact, dx_exact)
    a_exact_list.append(a_exact_j)


fig, axs = plt.subplots(1, Np, sharex=True, sharey=True, squeeze=True, figsize=(8, 3))

for j in range(Np):
    plot_subfig_oscillations(
        a_exact_list[j], axs[j], L, color="grey", label="Exact\nsolution", linewidth=1.2
    )
    plot_subfig_oscillations(
        a_centered_list[j],
        axs[j],
        L,
        color="#ff5555",
        label="FTCS\n(unstable)",
        linewidth=1.5,
    )
    plot_subfig_oscillations(
        a_update_list[j],
        axs[j],
        L,
        color="#5555ff",
        label="Modified FTCS\n(stable)",
        linewidth=1.5,
    )


axs[0].set_xlim([0, 1])
axs[0].set_ylim([-2.0, 2.0])

axs[0].spines["left"].set_visible(False)
axs[Np - 1].spines["right"].set_visible(False)
for j in range(Np):
    axs[j].set_yticklabels([])
    axs[j].set_xticklabels([])
    axs[j].spines["top"].set_visible(False)
    axs[j].spines["bottom"].set_visible(False)
    axs[j].tick_params(bottom=False)
    axs[j].tick_params(left=False)

# for j in range(Np):
# axs[j].set_axis_off()
# for j in range(Np):
#    axs[j].grid(True, which="both")


plt.style.use("seaborn")

props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
axs[1].text(
    0.5,
    0.15,
    r"$\frac{\partial u}{\partial t} + \frac{\partial u}{\partial x} = 0$",
    transform=axs[1].transAxes,
    fontsize=fs,
    verticalalignment="top",
    bbox=props,
)

# place a text box in upper left in axes coords
axs[0].text(
    0.02,
    0.97,
    "$t=0.0$",
    transform=axs[0].transAxes,
    fontsize=fs,
    verticalalignment="top",
)
axs[1].text(
    0.02,
    0.97,
    "$t=0.5$",
    transform=axs[1].transAxes,
    fontsize=fs,
    verticalalignment="top",
)
axs[2].text(
    0.02,
    0.97,
    "$t=1.0$",
    transform=axs[2].transAxes,
    fontsize=fs,
    verticalalignment="top",
)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc=(0.003, 0.002), prop={"size": fs})

# fig.suptitle("")


fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)


plt.savefig("advection_demo.eps")
plt.savefig("advection_demo.png")

plt.show()
