import jax.numpy as np
from jax import vmap, jit
from jax.lax import scan
import matplotlib.pyplot as plt
from jax.config import config
config.update("jax_enable_x64", True)

vmap_polyval = vmap(np.polyval, (0, None), -1)

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


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
    
    _vmap_fixed_quad = vmap(
        lambda f, a, b: quad_func(f, a, b, n=n), (None, 0, 0), 0
    ) 
    j = np.arange(nx)
    a = dx * j
    b = dx * (j + 1)

    def xi(x):
        j = np.floor(x / dx)
        x_j = dx * (0.5 + j)
        return (x - x_j) / (0.5 * dx)

    to_int_func = lambda x: f(x, t)[:, None] * vmap_polyval(np.asarray([[1.]]), xi(x))

    return _vmap_fixed_quad(to_int_func, a, b)





def map_f_to_FV(f, t, nx, dx, quad_func=_fixed_quad, n=5):
    return (
        inner_prod_with_legendre(f, t, nx, dx, quad_func=quad_func, n=n) / dx
    )



def evalf_1D(x, a, dx, leg_poly):
    j = np.floor(x / dx).astype(int)
    x_j = dx * (0.5 + j)
    xi = (x - x_j) / (0.5 * dx)
    poly_eval = vmap_polyval(np.asarray([[1.]]), xi)  # nx, p array
    return np.sum(poly_eval * a[j, :], axis=-1)


def _scan(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t, x)
    return (a_f, t_f), None


def _scan_output(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t, x)
    return (a_f, t_f), (a, t)


def _centered_flux_1D_burgers(a):
    a = np.pad(a, ((0, 1), (0, 0)), "wrap")
    u_left = np.sum(a[:-1], axis=-1)
    u_right = np.sum(a[1:], axis=-1)
    return ((u_left + u_right) / 2) ** 2 / 2
    #return ((u_left**2 + u_right**2) / 2)

def _stabilized_flux_1D_burgers(a):
    f0 = _centered_flux_1D_burgers(a)
    diff = (np.roll(a[:,0], -1) - a[:,0])
    S = np.sum(f0 * diff)
    return f0 - S * diff / np.sum(diff**2)

def _stabilized2_flux_1D_burgers(a, dl_dt):
    f0 = _centered_flux_1D_burgers(a)
    diff = (np.roll(a[:,0], -1) - a[:,0])
    dl_dt_old = np.sum(f0 * diff)
    return f0 + (dl_dt - dl_dt_old) * diff / np.sum(diff**2)


def _godunov_flux_1D_burgers(a):
    a = np.pad(a, ((0, 1), (0, 0)), "wrap")
    u_left = np.sum(a[:-1], axis=-1)
    u_right = np.sum(a[1:], axis=-1)
    zero_out = 0.5 * np.abs(np.sign(u_left) + np.sign(u_right))
    compare = np.less(u_left, u_right)
    F = lambda u: u ** 2 / 2
    return compare * zero_out * np.minimum(F(u_left), F(u_right)) + (
        1 - compare
    ) * np.maximum(F(u_left), F(u_right))



def time_derivative_1D_burgers(
    a, t, nx, dx, flux, dl_dt = None,
):
    if flux == "centered":
        flux_right = _centered_flux_1D_burgers(a)
    elif flux == "godunov":
        flux_right = _godunov_flux_1D_burgers(a)
    elif flux == "stabilized":
        flux_right = _stabilized_flux_1D_burgers(a)
    elif flux == "stabilized2":
        flux_right = _stabilized2_flux_1D_burgers(a, dl_dt)
    else:
        raise Exception

    flux_left = np.roll(flux_right, 1, axis=0)
    return (flux_left[:, None] - flux_right[:, None]) / dx
    


def time_derivative_1D_burgers_non_conservative(
    a, t, nx, dx, dl_dt = None, conservative = False,
):
    greater_than_zero = np.less(0.0, a)
    a_backwards = (a - np.roll(a, 1, axis=0)) / dx

    a_forwards = (np.roll(a, -1, axis=0) - a) / dx
    N = - a * (greater_than_zero * a_backwards + (1 - greater_than_zero) * a_forwards)
    if conservative is True:
        U = a - np.mean(a)
        M = N - np.mean(N)
        dl_dt_old = np.sum(U * M) * dx
        G = np.roll(a, -1, axis=0) + np.roll(a, 1, axis=0) - 2 * a
        denom = np.sum(U * G) * dx
        if dl_dt is None:
            dl_dt = dl_dt_old
        N = M + (dl_dt - dl_dt_old) * G / denom
    return N
    

def simulate_1D_nonconservative(
    a0,
    t0,
    nx,
    dx,
    dt,
    nt,
    output=False,
    rk=ssp_rk3,
    dl_dt = None,
    conservative = False,
):

    dadt = lambda a, t, dl_dt: time_derivative_1D_burgers_non_conservative(
        a,
        t,
        nx,
        dx,
        dl_dt = dl_dt,
        conservative = conservative,
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

def simulate_1D(
    a0,
    t0,
    nx,
    dx,
    dt,
    nt,
    output=False,
    rk=ssp_rk3,
    flux="centered",
    dl_dt = None,
):

    dadt = lambda a, t, dl_dt: time_derivative_1D_burgers(
        a,
        t,
        nx,
        dx,
        flux,
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


def plot_subfig(
    a, subfig, L, color="blue", linewidth=0.5, linestyle="solid", label=None
):
    def evalf(x, a, j, dx):
        x_j = dx * (0.5 + j)
        xi = (x - x_j) / (0.5 * dx)
        vmap_polyval = vmap(np.polyval, (0, None), -1)
        poly_eval = vmap_polyval(np.asarray([[1.]]), xi)
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
        poly_eval = vmap_polyval(np.asarray([[1.]]), xi)
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
    return 1/2 * np.mean(a**2)

vmap_l2_norm = vmap(l2_norm)


nx = 20
Np = 4
T = 0.5
fs = 14
L = 1.0
t0 = 0.0
f_init = lambda x, t: 0.5 + np.sin(2 * np.pi * x)
cfl_safety = 0.2

dx = L/nx
dt = cfl_safety * dx
nt = T // dt + 1
a0 = map_f_to_FV(f_init, t0, nx, dx)

##########
# Calculate dl/dt
##########


##### exact 

UPSAMPLE = 25
nx_exact = nx * UPSAMPLE
dx_exact = L / nx_exact
dt_exact = cfl_safety * dx_exact
a0_exact = map_f_to_FV(f_init, t0, nx_exact, dx_exact)
nt_exact = T // dt_exact + 2

a_data, t_data = simulate_1D(a0_exact, t0, nx_exact, dx_exact, dt_exact, nt_exact, output=True, flux="godunov")
dl_dt_exact = (vmap_l2_norm(a_data[1:]) - vmap_l2_norm(a_data[:-1])) / dt_exact

dl_dt_upsample = np.mean(dl_dt_exact.reshape(-1, UPSAMPLE), axis=-1)
a_godunov = a_data[:-1][::UPSAMPLE]
t_godunov = t_data[:-1][::UPSAMPLE]


#### fluxes

a_finite_difference, _ = simulate_1D_nonconservative(a0, t0, nx, dx, dt, nt, output="True", conservative = False)
a_stabilized, _ = simulate_1D_nonconservative(a0, t0, nx, dx, dt, nt, output="True", conservative = True, dl_dt = None) #np.zeros(dl_dt_upsample.shape))
a_stabilized2, _ = simulate_1D_nonconservative(a0, t0, nx, dx, dt, nt, output="True", conservative = True, dl_dt = dl_dt_upsample)

"""
print(a_godunov.shape)
print(a_finite_difference.shape)

print("Mass of a_godunov is {}".format(np.sum(a_godunov, axis=(1,2)) * dx_exact))
print("Mass of a_non_conservative is {}".format(np.sum(a_finite_difference, axis=(1,2)) * dx))
print("Mass of a_stabilized is {}".format(np.sum(a_stabilized, axis=(1,2)) * dx))
print("Mass of a_stabilized_dldt is {}".format(np.sum(a_stabilized2, axis=(1,2)) * dx))

print("L2 norm of a_godunov is {}".format(0.5 * np.sum(a_godunov**2, axis=(1,2)) * dx_exact))
print("L2 norm of a_non_conservative is {}".format(0.5 * np.sum(a_finite_difference**2, axis=(1,2)) * dx))
print("L2 norm of a_stabilized is {}".format(0.5 * np.sum(a_stabilized**2, axis=(1,2)) * dx))
print("L2 norm of a_stabilized_dldt is {}".format(0.5 * np.sum(a_stabilized2**2, axis=(1,2)) * dx))
"""

num = a_finite_difference.shape[0]
js = [0, int(0.33 * num), int(0.66 * num), num-1]

a_fd_list = []
a_stabilized_list = []
a_stabilized2_list = []
a_godunov_list = []


for j in js:
    a_godunov_list.append(a_godunov[j])
    a_fd_list.append(a_finite_difference[j])
    a_stabilized_list.append(a_stabilized[j])
    a_stabilized2_list.append(a_stabilized2[j])




fig, axs = plt.subplots(1, Np, sharex=True, sharey=True, squeeze=True, figsize=(8,3))

for j in range(Np):
    plot_subfig_oscillations(a_godunov_list[j], axs[j], L, color="grey", label="Exact\nsolution", linewidth=1.2)
    plot_subfig_oscillations(a_fd_list[j], axs[j], L, color="#ff5555", label="Non-conservative\nfinite-difference", linewidth=1.5)
    plot_subfig_oscillations(a_stabilized_list[j], axs[j], L, color="#003366", label=r'$\frac{d\ell_2^{new}}{dt} = \frac{d\ell_2^{old}}{dt}$', linewidth=1.5)
    plot_subfig_oscillations(a_stabilized2_list[j], axs[j], L, color="#007733", label=r'$\frac{d\ell_2^{new}}{dt} = \frac{d\ell_2^{exact}}{dt}$', linewidth=1.5)
    axs[j].plot(np.zeros(len(a_stabilized)), '--',  color="black", linewidth=0.4)

axs[0].set_xlim([0, 1])
axs[0].set_ylim([-1.5, 2.0])

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
#axs[0].text(0.4, 0.95, r'$\frac{\partial u}{\partial t} + \frac{\partial}{\partial x} \big(\frac{u^2}{2}\big) = 0$', transform=axs[0].transAxes, fontsize=10, verticalalignment='top', bbox=props)
axs[2].text(0.15, 0.15, r'$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = 0$', transform=axs[2].transAxes, fontsize=14, verticalalignment='top', bbox=props)

# place a text box in upper left in axes coords
axs[0].text(0.02, 0.95, "$t=0.0$", transform=axs[0].transAxes, fontsize=fs,
        verticalalignment='top')
axs[1].text(0.02, 0.95, "$t=0.167$", transform=axs[1].transAxes, fontsize=fs,
        verticalalignment='top')
axs[2].text(0.02, 0.95, "$t=0.333$", transform=axs[2].transAxes, fontsize=fs,
        verticalalignment='top')
axs[3].text(0.02, 0.95, "$t=0.5$", transform=axs[3].transAxes, fontsize=fs,
        verticalalignment='top')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(),loc=(0.00,0.00), prop={'size': 0.95 * fs}, ncol=2)

#fig.suptitle("")

fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)


plt.savefig("burgers_nonconservative_demo.eps")
plt.savefig("burgers_nonconservative_demo.png")

plt.show()

