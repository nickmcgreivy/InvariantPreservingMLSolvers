import sys

basedir = "/Users/nickm/thesis/InvariantPreservingMLSolvers/1d_advection"
readwritedir = "/Users/nickm/thesis/InvariantPreservingMLSolvers/1d_advection"

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


# import internal packages
from flux import Flux
from initialconditions import get_a0, get_initial_condition_fn, get_a
from simparams import CoreParams, SimulationParams
from helper import generate_legendre
from simulations import AdvectionFVSim
from trajectory import get_trajectory_fn, get_inner_fn
from trainingutils import save_training_data
from mlparams import TrainingParams, ModelParams
from model import LearnedFluxOutput
from trainingutils import (
    get_loss_fn,
    get_batch_fn,
    get_idx_gen,
    train_model,
    compute_losses_no_model,
    init_params,
    save_training_params,
    load_training_params,
)


# In[ ]:


# helper functions


def plot_fv(a, core_params, color="blue"):
    plot_dg(a[..., None], core_params, color=color)


def plot_fv_trajectory(trajectory, core_params, t_inner, color="blue"):
    plot_dg_trajectory(trajectory[..., None], core_params, t_inner, color=color)


def plot_dg(a, core_params, color="blue"):
    p = a.shape[-1]

    def evalf(x, a, j, dx, leg_poly):
        x_j = dx * (0.5 + j)
        xi = (x - x_j) / (0.5 * dx)
        vmap_polyval = vmap(jnp.polyval, (0, None), -1)
        poly_eval = vmap_polyval(leg_poly, xi)  # nx, p array
        return jnp.sum(poly_eval * a, axis=-1)

    NPLOT = [2, 2, 5, 7][p - 1]
    nx = a.shape[0]
    dx = core_params.Lx / nx
    xjs = jnp.arange(nx) * core_params.Lx / nx
    xs = xjs[None, :] + jnp.linspace(0.0, dx, NPLOT)[:, None]
    vmap_eval = vmap(evalf, (1, 0, 0, None, None), 1)

    a_plot = vmap_eval(xs, a, jnp.arange(nx), dx, generate_legendre(p))
    a_plot = a_plot.T.reshape(-1)
    xs = xs.T.reshape(-1)
    coords = {("x"): xs}
    data = xarray.DataArray(a_plot, coords=coords)
    data.plot(color=color)


def plot_dg_trajectory(trajectory, core_params, t_inner, color="blue"):
    p = trajectory.shape[-1]
    NPLOT = [2, 2, 5, 7][p - 1]
    nx = trajectory.shape[1]
    dx = core_params.Lx / nx
    xjs = jnp.arange(nx) * core_params.Lx / nx
    xs = xjs[None, :] + jnp.linspace(0.0, dx, NPLOT)[:, None]

    def get_plot_repr(a):
        def evalf(x, a, j, dx, leg_poly):
            x_j = dx * (0.5 + j)
            xi = (x - x_j) / (0.5 * dx)
            vmap_polyval = vmap(jnp.polyval, (0, None), -1)
            poly_eval = vmap_polyval(leg_poly, xi)  # nx, p array
            return jnp.sum(poly_eval * a, axis=-1)

        vmap_eval = vmap(evalf, (1, 0, 0, None, None), 1)
        return vmap_eval(xs, a, jnp.arange(nx), dx, generate_legendre(p)).T

    get_trajectory_plot_repr = vmap(get_plot_repr)
    trajectory_plot = get_trajectory_plot_repr(trajectory)

    outer_steps = trajectory.shape[0]

    trajectory_plot = trajectory_plot.reshape(outer_steps, -1)
    xs = xs.T.reshape(-1)
    coords = {"x": xs, "time": t_inner * jnp.arange(outer_steps)}
    xarray.DataArray(trajectory_plot, dims=["time", "x"], coords=coords).plot(
        col="time", col_wrap=5, color=color
    )


def plot_multiple_fv_trajectories(trajectories, core_params, t_inner, ylim=[-1.5, 1.5]):
    plot_multiple_dg_trajectories(
        [trajectory[..., None] for trajectory in trajectories],
        core_params,
        t_inner,
        ylim=ylim,
    )


def plot_multiple_dg_trajectories(trajectories, core_params, t_inner, ylim=[-1.5, 1.5]):
    outer_steps = trajectories[0].shape[0]
    nx = trajectories[0].shape[1]
    p = trajectories[0].shape[-1]
    NPLOT = [2, 2, 5, 7][p - 1]
    dx = core_params.Lx / nx
    xjs = jnp.arange(nx) * core_params.Lx / nx
    xs = xjs[None, :] + jnp.linspace(0.0, dx, NPLOT)[:, None]

    def get_plot_repr(a):
        def evalf(x, a, j, dx, leg_poly):
            x_j = dx * (0.5 + j)
            xi = (x - x_j) / (0.5 * dx)
            vmap_polyval = vmap(jnp.polyval, (0, None), -1)
            poly_eval = vmap_polyval(leg_poly, xi)  # nx, p array
            return jnp.sum(poly_eval * a, axis=-1)

        vmap_eval = vmap(evalf, (1, 0, 0, None, None), 1)
        return vmap_eval(xs, a, jnp.arange(nx), dx, generate_legendre(p)).T

    get_trajectory_plot_repr = vmap(get_plot_repr)
    trajectory_plots = []
    for trajectory in trajectories:
        trajectory_plots.append(
            get_trajectory_plot_repr(trajectory).reshape(outer_steps, -1)
        )

    xs = xs.T.reshape(-1)
    coords = {"x": xs, "time": t_inner * jnp.arange(outer_steps)}
    xarray.DataArray(
        trajectory_plots, dims=["stack", "time", "x"], coords=coords
    ).plot.line(col="time", hue="stack", col_wrap=5, ylim=ylim)


def get_core_params(flux="upwind"):
    Lx = 1.0
    return CoreParams(Lx, flux)


def get_sim_params(name="test", cfl_safety=0.3, rk="ssp_rk3"):
    return SimulationParams(name, basedir, readwritedir, cfl_safety, rk)


def get_training_params(
    n_data,
    train_id="test",
    batch_size=4,
    learning_rate=1e-3,
    num_epochs=10,
    optimizer="sgd",
):
    return TrainingParams(
        n_data, num_epochs, train_id, batch_size, learning_rate, optimizer
    )


def get_model_params(kernel_size=3, kernel_out=4, depth=3, width=16):
    return ModelParams(kernel_size, kernel_out, depth, width)


def l2_norm_trajectory(trajectory):
    return jnp.mean(trajectory**2, axis=1)


def get_model(core_params, model_params):
    features = [model_params.width for _ in range(model_params.depth - 1)]
    return LearnedFluxOutput(
        features, model_params.kernel_size, model_params.kernel_out
    )


# ### Finite Volume
#
# ##### Training Loop
#
# First, we will generate the data.

# In[ ]:


# training hyperparameters
init_description = "sum_sin"
kwargs_init = {
    "min_num_modes": 1,
    "max_num_modes": 6,
    "min_k": 1,
    "max_k": 4,
    "amplitude_max": 1.0,
}
kwargs_sim = {"name": "paper_test", "cfl_safety": 0.3, "rk": "ssp_rk3"}
kwargs_model = {"kernel_size": 3, "kernel_out": 4, "depth": 3, "width": 24}
n_runs = 100
t_inner_train = 0.02
BS = 32
NE = 200  # num epochs
outer_steps_train = int(1.0 / t_inner_train)
nx_exact = 512
nxs = [8, 16, 32, 64]
learning_rate_list = [1e-3, 1e-3, 1e-3, 1e-3]
assert len(nxs) == len(learning_rate_list)
key = jax.random.PRNGKey(12)


# In[ ]:


##### Setup for Generating Training Data
core_params_muscl = get_core_params(flux="muscl")
sim_params = get_sim_params(**kwargs_sim)
n_data = n_runs * outer_steps_train
sim = AdvectionFVSim(core_params_muscl, sim_params)
init_fn = lambda key: get_initial_condition_fn(
    core_params_muscl, init_description, key=key, **kwargs_init
)


# In[ ]:


save_training_data(
    key,
    init_fn,
    core_params_muscl,
    sim_params,
    sim,
    t_inner_train,
    outer_steps_train,
    n_runs,
    nx_exact,
    nxs,
)


# In[ ]:


##### Setup for training models

model_params = get_model_params(**kwargs_model)
model = get_model(core_params_muscl, model_params)
key = jax.random.PRNGKey(42)
i_params = init_params(key, model)

core_params_learned = get_core_params(flux="learned")
kwargs_train_FV = {
    "train_id": "flux_predicting",
    "batch_size": BS,
    "optimizer": "adam",
    "num_epochs": NE,
}
training_params_list_learned = [
    get_training_params(n_data, **kwargs_train_FV, learning_rate=lr)
    for lr in learning_rate_list
]

core_params_limiter = get_core_params(flux="learnedlimiter")
kwargs_train_FV = {
    "train_id": "flux_limited",
    "batch_size": BS,
    "optimizer": "adam",
    "num_epochs": NE,
}
training_params_list_limited = [
    get_training_params(n_data, **kwargs_train_FV, learning_rate=lr)
    for lr in learning_rate_list
]

core_params_combo = get_core_params(flux="combination_learned")
kwargs_train_FV = {
    "train_id": "combo_learned",
    "batch_size": BS,
    "optimizer": "adam",
    "num_epochs": NE,
}
training_params_list_combo = [
    get_training_params(n_data, **kwargs_train_FV, learning_rate=lr)
    for lr in learning_rate_list
]


# Next, we run a training loop for each value of nx. The learning rate undergoes a prespecified decay.

# In[ ]:


#### First, train original ML Model

for i, nx in enumerate(nxs):
    print(nx)
    training_params = training_params_list_learned[i]
    idx_fn = lambda key: get_idx_gen(key, training_params)
    batch_fn = get_batch_fn(core_params_learned, sim_params, training_params, nx)
    loss_fn = get_loss_fn(model, core_params_learned)
    losses, params = train_model(
        model, i_params, training_params, key, idx_fn, batch_fn, loss_fn
    )
    save_training_params(nx, sim_params, training_params, params, losses)


# In[ ]:


#### Second, train flux-limited model

for i, nx in enumerate(nxs):
    print(nx)
    training_params = training_params_list_limited[i]
    idx_fn = lambda key: get_idx_gen(key, training_params)
    batch_fn = get_batch_fn(core_params_limiter, sim_params, training_params, nx)
    loss_fn = get_loss_fn(model, core_params_limiter)
    losses, params = train_model(
        model, i_params, training_params, key, idx_fn, batch_fn, loss_fn
    )
    save_training_params(nx, sim_params, training_params, params, losses)


# In[ ]:


#### Third, train combination model

for i, nx in enumerate(nxs):
    print(nx)
    training_params = training_params_list_combo[i]
    idx_fn = lambda key: get_idx_gen(key, training_params)
    batch_fn = get_batch_fn(core_params_combo, sim_params, training_params, nx)
    loss_fn = get_loss_fn(model, core_params_combo)
    losses, params = train_model(
        model, i_params, training_params, key, idx_fn, batch_fn, loss_fn
    )
    save_training_params(nx, sim_params, training_params, params, losses)


# Next, we load and plot the losses for each nx to check that the simulation trained properly.

# In[ ]:


for i, nx in enumerate(nxs):
    losses, _ = load_training_params(
        nx, sim_params, training_params_list_learned[i], model
    )
    plt.plot(losses, label="learned {}".format(nx))
    print(losses)

    losses, _ = load_training_params(
        nx, sim_params, training_params_list_limited[i], model
    )
    plt.plot(losses, label="limited {}".format(nx))

    losses, _ = load_training_params(
        nx, sim_params, training_params_list_combo[i], model
    )
    plt.plot(losses, label="combination {}".format(nx))

plt.ylim([0, 100])
plt.legend()
plt.show()


# In[ ]:


N = 25

mses = onp.zeros((len(nxs), 7))


def normalized_mse(traj, traj_exact):
    assert len(traj_exact.shape) == 2
    return jnp.mean(
        (traj - traj_exact) ** 2 / jnp.mean(traj_exact**2, axis=1)[:, None]
    )


for i, nx in enumerate(nxs):
    print(nx)

    key = jax.random.PRNGKey(10)

    t_inner = 0.1
    outer_steps = 11

    for k in range(N):
        print(k)

        ########
        # Generate Exact Data
        ########

        core_params = get_core_params(flux="muscl")
        f_init = get_initial_condition_fn(
            core_params, init_description, key=key, **kwargs_init
        )
        trajectory_exact = onp.zeros((outer_steps, nx))
        for k in range(outer_steps):
            t = k * t_inner
            trajectory_exact[k] = get_a(f_init, t, core_params, nx)
        trajectory_exact = jnp.asarray(trajectory_exact)

        # Initial conditions
        a0 = get_a(f_init, 0, core_params, nx)

        ########
        # Flux 1: Centered
        ########
        core_params = get_core_params(flux="centered")
        sim = AdvectionFVSim(core_params, sim_params)
        inner_fn = get_inner_fn(sim.step_fn, sim.dt_fn, t_inner)
        trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
        trajectory_centered = trajectory_fn(a0)

        ########
        # Flux 2: Upwind
        ########
        core_params = get_core_params(flux="upwind")
        sim = AdvectionFVSim(core_params, sim_params)
        inner_fn = get_inner_fn(sim.step_fn, sim.dt_fn, t_inner)
        trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
        trajectory_upwind = trajectory_fn(a0)

        ########
        # Flux 3: MUSCL
        ########
        core_params = get_core_params(flux="muscl")
        sim = AdvectionFVSim(core_params, sim_params)
        inner_fn = get_inner_fn(sim.step_fn, sim.dt_fn, t_inner)
        trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
        trajectory_muscl = trajectory_fn(a0)

        ########
        # Flux 4: Learned
        ########
        core_params = get_core_params(flux="learned")
        _, params = load_training_params(
            nx, sim_params, training_params_list_learned[i], model
        )
        sim = AdvectionFVSim(core_params, sim_params, model=model, params=params)
        inner_fn = get_inner_fn(sim.step_fn, sim.dt_fn, t_inner)
        trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
        trajectory_learned = trajectory_fn(a0)

        ########
        # Flux 5: Upwind + Centered
        ########
        core_params = get_core_params(flux="combination_learned")
        _, params = load_training_params(
            nx, sim_params, training_params_list_learned[i], model
        )
        sim = AdvectionFVSim(core_params, sim_params, model=model, params=params)
        inner_fn = get_inner_fn(sim.step_fn, sim.dt_fn, t_inner)
        trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
        trajectory_combination = trajectory_fn(a0)

        ########
        # Flux 6: Learned Limiter
        ########
        core_params = get_core_params(flux="learnedlimiter")
        _, params = load_training_params(
            nx, sim_params, training_params_list_limited[i], model
        )
        sim = AdvectionFVSim(core_params, sim_params, model=model, params=params)
        inner_fn = get_inner_fn(sim.step_fn, sim.dt_fn, t_inner)
        trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
        trajectory_limiter = trajectory_fn(a0)

        ########
        # Flux 7: Invariant-Preserving Learned
        ########
        core_params = get_core_params(flux="learned")
        _, params = load_training_params(
            nx, sim_params, training_params_list_learned[i], model
        )
        sim = AdvectionFVSim(
            core_params,
            sim_params,
            model=model,
            params=params,
            global_stabilization=True,
        )
        inner_fn = get_inner_fn(sim.step_fn, sim.dt_fn, t_inner)
        trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
        trajectory_invariant_learned = trajectory_fn(a0)

        mses[i, 0] += normalized_mse(trajectory_centered, trajectory_exact) / N
        mses[i, 1] += normalized_mse(trajectory_upwind, trajectory_exact) / N
        mses[i, 2] += normalized_mse(trajectory_muscl, trajectory_exact) / N
        mses[i, 3] += normalized_mse(trajectory_learned, trajectory_exact) / N
        mses[i, 4] += normalized_mse(trajectory_combination, trajectory_exact) / N
        mses[i, 5] += normalized_mse(trajectory_limiter, trajectory_exact) / N
        mses[i, 6] += normalized_mse(trajectory_invariant_learned, trajectory_exact) / N

        key, _ = jax.random.split(key)


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

print(mses.shape)
fig, axs = plt.subplots(1, 1, figsize=(8, 3.25))
axs.spines["top"].set_visible(False)
axs.spines["right"].set_visible(False)
linewidth = 2

labels = [
    "Centered",
    "Upwind",
    "MUSCL MC Limiter",
    "ML",
    "ML Upwind-Biased",
    "ML MC Limiter",
    "ML Invariant-\nPreserving",
]
colors = [
    "blue",
    "red",
    "green",
    "black",
    "red",
    "green",
    "black",
]  # ["black", "#1232ED", "#E619D6", "red", "#EDCD12", "#19E629", "black"]
markers = ["P", "o", "^", "s", "o", "^", "s"]
linestyles = ["solid", "solid", "solid", "solid", "--", "--", "--"]

nxs = [8, 16, 32, 64]
nxs_rev = [64, 32, 16, 8]


for k in range(7):
    plt.loglog(
        nxs_rev,
        (mses[:, k]),
        label=labels[k],
        color=colors[k],
        linewidth=linewidth,
        linestyle=linestyles[k],
        marker=markers[k],
        markersize=8,
    )


axs.set_xticks(nxs)
axs.set_xticklabels(["N=64", "N=32", "N=16", "N=8"], fontsize=16)
axs.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
axs.set_yticklabels(
    ["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^0$"], fontsize=18
)
axs.minorticks_off()
axs.set_ylabel("Normalized MSE", fontsize=18)


handles = []
for k in range(7):
    handles.append(
        mlines.Line2D(
            [],
            [],
            color=colors[k],
            linewidth=linewidth,
            label=labels[k],
            linestyle=linestyles[k],
            marker=markers[k],
            markersize=8,
        )
    )
axs.legend(handles=handles, loc=(0.97, 0.02), prop={"size": 16}, frameon=False)
plt.ylim([2.5e-5, 2e0 + 1e-1])

fig.tight_layout()
plt.savefig("invariant_preserving_mse_vs_nx.png")
plt.savefig("invariant_preserving_mse_vs_nx.eps")
plt.show()


# In[ ]:
