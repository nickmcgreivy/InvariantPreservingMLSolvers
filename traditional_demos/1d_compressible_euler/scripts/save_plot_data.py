basedir = "/Users/nickm/thesis/InvariantPreservingMLSolvers/traditional_demos/1d_compressible_euler"

import sys

sys.path.append("{}/core".format(basedir))
sys.path.append("{}/simulate".format(basedir))
sys.path.append("{}/ml".format(basedir))


# In[ ]:


# import external packages
import jax
import jax.numpy as jnp
import numpy as onp
from jax import config, vmap

config.update("jax_enable_x64", True)
import xarray
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


from initialconditions import get_a0, shock_tube_problem_1
from simparams import CoreParams, SimulationParams
from simulations import EulerFVSim
from helper import get_rho, get_u, get_p, get_entropy, get_w
from trajectory import get_trajectory_fn, get_inner_fn


# In[ ]:


def get_core_params(Lx=1.0, gamma=5 / 3, bc="periodic", fluxstr="laxfriedrichs"):
    return CoreParams(Lx, gamma, bc, fluxstr)


def get_sim_params(name="test", cfl_safety=0.3, rk="ssp_rk3"):
    return SimulationParams(name, basedir, None, cfl_safety, rk)


def plot_a(a, core_params, mins=[0.0 - 2e-2] * 3, maxs=[1.0 + 5e-2] * 3):
    x = jnp.linspace(0.0, core_params.Lx, a.shape[1])

    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    axs[0].plot(x, get_rho(a, core_params))
    axs[0].set_title(r"$\rho$")
    axs[0].set_ylim([mins[0], maxs[0]])

    axs[1].plot(x, get_u(a, core_params))
    axs[1].set_title(r"$v$")
    axs[1].set_ylim([mins[1], maxs[1]])

    axs[2].plot(x, get_p(a, core_params))
    axs[2].set_title(r"$p$")
    axs[2].set_ylim([mins[2], maxs[2]])


def plot_trajectory(
    trajectory, core_params, mins=[0.0 - 2e-2] * 3, maxs=[1.0 + 5e-2] * 3
):
    nx = trajectory.shape[2]
    xs = jnp.arange(nx) * core_params.Lx / nx
    xs = xs.T.reshape(-1)
    coords = {"x": xs, "time": t_inner * jnp.arange(trajectory.shape[0])}
    rhos = trajectory[:, 0, :]
    xarray.DataArray(rhos, dims=["time", "x"], coords=coords).plot(
        col="time", col_wrap=5
    )

    us = trajectory[:, 1, :] / trajectory[:, 0, :]
    xarray.DataArray(us, dims=["time", "x"], coords=coords).plot(col="time", col_wrap=5)

    ps = (core_params.gamma - 1) * (
        trajectory[:, 2, :] - 0.5 * trajectory[:, 1, :] ** 2 / trajectory[:, 0, :]
    )
    xarray.DataArray(ps, dims=["time", "x"], coords=coords).plot(col="time", col_wrap=5)


# In[ ]:


kwargs_core_params = {
    "Lx": 1.0,
    "gamma": 1.4,
    "bc": "open",
    "fluxstr": "musclcharacteristic",
}
kwargs_sim = {"name": "test_euler", "cfl_safety": 0.2, "rk": "ssp_rk3"}

core_params = get_core_params(**kwargs_core_params)
sim_params = get_sim_params(**kwargs_sim)

nx_exact = 100
nx = 100
f_init = shock_tube_problem_1(core_params)
a0_exact = get_a0(f_init, core_params, nx_exact)
a0 = get_a0(f_init, core_params, nx)

# simulate for some time
t_inner = 0.1
outer_steps = 3
ratio = 0.0


# In[ ]:


def G_primitive(a, core_params):
    p = get_p(a, core_params)
    rho = a[0]
    zeros = jnp.zeros(rho.shape)
    u = a[1] / a[0]
    G = jnp.concatenate([zeros[None], u[None], p[None]], axis=0)
    return G[:, 1:] - G[:, :-1]


def G_w(a, core_params):
    w = get_w(a, core_params)
    return w[:, 1:] - w[:, :-1]


ratio1 = 1.0
ratio2 = 0.0
ratio3 = 2.0
G_select = G_primitive

sim = EulerFVSim(core_params, sim_params, deta_dt_ratio=None, G=None)
inner_fn = get_inner_fn(sim.step_fn, sim.dt_fn, t_inner)
trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
trajectory_exact = trajectory_fn(a0)

sim = EulerFVSim(core_params, sim_params, deta_dt_ratio=ratio1, G=G_select)
inner_fn = get_inner_fn(sim.step_fn, sim.dt_fn, t_inner)
trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
trajectory_zero = trajectory_fn(a0)

sim = EulerFVSim(core_params, sim_params, deta_dt_ratio=ratio2, G=G_select)
inner_fn = get_inner_fn(sim.step_fn, sim.dt_fn, t_inner)
trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
trajectory_half = trajectory_fn(a0)

sim = EulerFVSim(core_params, sim_params, deta_dt_ratio=ratio3, G=G_select)
inner_fn = get_inner_fn(sim.step_fn, sim.dt_fn, t_inner)
trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
trajectory_double = trajectory_fn(a0)


# In[ ]:


label0 = r"R=1"  # r'$\frac{d\eta^{new}}{dt} = 0$'
label1 = r"R=0"  # r'$\frac{d\eta^{new}}{dt} = \frac{1}{2}\frac{d\eta^{old}}{dt}$'
label2 = r"R=2"  # r'$\frac{d\eta^{new}}{dt} = \frac{5}{4}\frac{d\eta^{old}}{dt}$'
label3 = r"R=1"  # r'$\frac{d\eta^{new}}{dt} = \frac{d\eta^{old}}{dt}$'
color0 = "red"
color1 = "blue"
color2 = "green"
color3 = "black"

i = 2
R = 1.0
fs = 14
lw = 2.0
lw_zero = 1.0
figsize = (4 * R, 3 * R)
fig0, ax0 = plt.subplots(1, 1, figsize=figsize)
fig1, ax1 = plt.subplots(1, 1, figsize=figsize)
fig2, ax2 = plt.subplots(1, 1, figsize=figsize)

axs = [ax0, ax1, ax2]

x = jnp.linspace(0, 1.0, trajectory_exact[i].shape[1])
# axs[0].plot(x,get_rho(trajectory_zero[i],core_params),label=label0,color=color0)
axs[0].plot(
    x,
    get_rho(trajectory_exact[i], core_params),
    label=label3,
    color=color3,
    linewidth=lw,
)
axs[0].plot(
    x,
    get_rho(trajectory_double[i], core_params),
    label=label2,
    color=color2,
    linewidth=lw,
)
axs[0].plot(
    x,
    get_rho(trajectory_half[i], core_params),
    label=label1,
    color=color1,
    linewidth=lw_zero,
)
axs[0].set_ylim([0.0, 1.06])
axs[0].set_xlim([0.0, 1.0])

# axs[1].plot(x,get_u(trajectory_zero[i],core_params),label=label0,color=color0)
axs[1].plot(
    x, get_u(trajectory_exact[i], core_params), label=label3, color=color3, linewidth=lw
)
axs[1].plot(
    x,
    get_u(trajectory_double[i], core_params),
    label=label2,
    color=color2,
    linewidth=lw,
)
axs[1].plot(
    x,
    get_u(trajectory_half[i], core_params),
    label=label1,
    color=color1,
    linewidth=lw_zero,
)
axs[1].set_ylim([-0.25, 1.95])
axs[1].set_xlim([0.0, 1.0])

# axs[2].plot(x,get_p(trajectory_zero[i],core_params),label=label0,color=color0)
axs[2].plot(
    x, get_p(trajectory_exact[i], core_params), label=label3, color=color3, linewidth=lw
)
axs[2].plot(
    x,
    get_p(trajectory_double[i], core_params),
    label=label2,
    color=color2,
    linewidth=lw,
)
axs[2].plot(
    x,
    get_p(trajectory_half[i], core_params),
    label=label1,
    color=color1,
    linewidth=lw_zero,
)
axs[2].set_ylim([0.0, 1.06])
axs[2].set_xlim([0.0, 1.0])

j = 0
axs[j].set_xticks([0.0, 1.0])
axs[j].set_xticklabels(["0", "L"], fontsize=fs)
axs[j].set_yticks([0.0, 1.0])
axs[j].set_yticklabels(["0", "1"], fontsize=fs)
axs[j].spines["top"].set_visible(False)
axs[j].spines["right"].set_visible(False)
axs[j].set_xlabel(r"$\rho$", fontsize=fs)

j = 1
axs[j].set_xticks([0.0, 1.0])
axs[j].set_xticklabels(["0", "L"], fontsize=fs)
axs[j].set_yticks([0.0, 1.5])
axs[j].set_yticklabels(["0", "1.5"], fontsize=fs)
axs[j].spines["top"].set_visible(False)
axs[j].spines["right"].set_visible(False)
axs[j].set_xlabel(r"$v$", fontsize=fs)

j = 2
axs[j].set_xticks([0.0, 1.0])
axs[j].set_xticklabels(["0", "L"], fontsize=fs)
axs[j].set_yticks([0.0, 1.0])
axs[j].set_yticklabels(["0", "1"], fontsize=fs)
axs[j].spines["top"].set_visible(False)
axs[j].spines["right"].set_visible(False)
axs[j].set_xlabel(r"$p$", fontsize=fs)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig0.legend(
    by_label.values(),
    by_label.keys(),
    loc=(0.63, 0.6),
    prop={"size": 1.1 * fs},
    frameon=False,
)
fig0.tight_layout()
fig1.tight_layout()
fig2.tight_layout()


fig0.savefig("compressible_euler_rho_demo.eps")
fig0.savefig("compressible_euler_rho_demo.png")
fig1.savefig("compressible_euler_v_demo.eps")
fig1.savefig("compressible_euler_v_demo.png")
fig2.savefig("compressible_euler_p_demo.eps")
fig2.savefig("compressible_euler_p_demo.png")
plt.show()
