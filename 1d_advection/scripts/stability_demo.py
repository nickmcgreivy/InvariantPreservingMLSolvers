# setup paths
import sys
basedir = '/Users/nickm/thesis/icml2023paper/1d_advection'
readwritedir = '/Users/nickm/thesis/icml2023paper/1d_advection'

sys.path.append('{}/core'.format(basedir))
sys.path.append('{}/simulate'.format(basedir))
sys.path.append('{}/ml'.format(basedir))

# import external packages
import jax
import jax.numpy as jnp
import numpy as onp
from jax import config, vmap
config.update("jax_enable_x64", True)
import xarray
import seaborn as sns
import matplotlib.pyplot as plt

# import internal packages
from initialconditions import get_a0, get_initial_condition_fn, get_a
from simparams import CoreParams, CoreParamsDG, SimulationParams
from legendre import generate_legendre
from simulations import AdvectionFVSim, AdvectionDGSim
from trajectory import get_trajectory_fn, get_inner_fn
from trainingutils import save_training_data
from mlparams import TrainingParams, StencilParams
from model import LearnedStencil
from trainingutils import (get_loss_fn, get_batch_fn, get_idx_gen, train_model, 
                           compute_losses_no_model, init_params, save_training_params, load_training_params)

# helper functions

def plot_fv(a, core_params, color="blue"):
    plot_dg(a[...,None], core_params, color=color)
    
def plot_fv_trajectory(trajectory, core_params, t_inner, color='blue'):
    plot_dg_trajectory(trajectory[...,None], core_params, t_inner, color=color)
    
def plot_dg(a, core_params, color='blue'):
    if core_params.order is None:
        p = 1
    else:
        p = core_params.order + 1
    def evalf(x, a, j, dx, leg_poly):
        x_j = dx * (0.5 + j)
        xi = (x - x_j) / (0.5 * dx)
        vmap_polyval = vmap(jnp.polyval, (0, None), -1)
        poly_eval = vmap_polyval(leg_poly, xi)  # nx, p array
        return jnp.sum(poly_eval * a, axis=-1)

    NPLOT = [2,2,5,7][p-1]
    nx = a.shape[0]
    dx = core_params.Lx / nx
    xjs = jnp.arange(nx) * core_params.Lx / nx
    xs = xjs[None, :] + jnp.linspace(0.0, dx, NPLOT)[:, None]
    vmap_eval = vmap(evalf, (1, 0, 0, None, None), 1)

    a_plot = vmap_eval(xs, a, jnp.arange(nx), dx, generate_legendre(p))
    a_plot = a_plot.T.reshape(-1)
    xs = xs.T.reshape(-1)
    coords = {('x'): xs}
    data = xarray.DataArray(a_plot, coords=coords)
    data.plot(color=color)

def plot_dg_trajectory(trajectory, core_params, t_inner, color='blue'):
    if core_params.order is None:
        p = 1
    else:
        p = core_params.order + 1
    NPLOT = [2,2,5,7][p-1]
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
    coords = {
        'x': xs,
        'time': t_inner * jnp.arange(outer_steps)
    }
    xarray.DataArray(trajectory_plot, dims=["time", "x"], coords=coords).plot(
        col='time', col_wrap=5, color=color)
    
def plot_multiple_fv_trajectories(trajectories, core_params, t_inner):
    plot_multiple_dg_trajectories([trajectory[..., None] for trajectory in trajectories], core_params, t_inner)

def plot_multiple_dg_trajectories(trajectories, core_params, t_inner):
    outer_steps = trajectories[0].shape[0]
    nx = trajectories[0].shape[1]
    
    if core_params.order is None:
        p = 1
    else:
        p = core_params.order + 1
    NPLOT = [2,2,5,7][p-1]
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
        trajectory_plots.append(get_trajectory_plot_repr(trajectory).reshape(outer_steps, -1))
        
    xs = xs.T.reshape(-1)
    coords = {
        'x': xs,
        'time': t_inner * jnp.arange(outer_steps)
    }
    xarray.DataArray(trajectory_plots, dims=["stack", "time", "x"], coords=coords).plot.line(
        col='time', hue="stack", col_wrap=5)
    

def get_core_params(order, flux='upwind'):
    Lx = 1.0
    if order == 0:
        return CoreParams(Lx, flux)
    else:
        return CoreParamsDG(Lx, flux, order)

def get_sim_params(name = "test", cfl_safety=0.3, rk='ssp_rk3'):
    return SimulationParams(name, basedir, readwritedir, cfl_safety, rk)

def get_training_params(n_data, train_id="test", batch_size=4, learning_rate=1e-3, num_epochs = 10, optimizer='sgd'):
    return TrainingParams(n_data, num_epochs, train_id, batch_size, learning_rate, optimizer)

def get_stencil_params(kernel_size = 3, kernel_out = 4, stencil_width=4, depth = 3, width = 16):
    return StencilParams(kernel_size, kernel_out, stencil_width, depth, width)


def l2_norm_trajectory(trajectory):
    return (jnp.mean(trajectory**2, axis=1))
    
def get_model(core_params, stencil_params):
    if core_params.order is None:
        p = 1
    else:
        p = core_params.order + 1
    features = [stencil_params.width for _ in range(stencil_params.depth - 1)]
    return LearnedStencil(features, stencil_params.kernel_size, stencil_params.kernel_out, stencil_params.stencil_width, p)


# training hyperparameters
init_description = 'sum_sin'
kwargs_init = {'min_num_modes': 1, 'max_num_modes': 6, 'min_k': 1, 'max_k': 4, 'amplitude_max': 1.0}
kwargs_sim = {'name' : "larger", 'cfl_safety' : 0.3, 'rk' : 'ssp_rk3'}
kwargs_train_FV = {'train_id': "larger", 'batch_size' : 32, 'optimizer': 'adam', 'learning_rate' : 1e-3,  'num_epochs' : 200}
#kwargs_train_DG = {'train_id': "test", 'batch_size' : 8, 'optimizer': 'adam', 'learning_rate' : 1e-5, 'num_epochs' : 10}
kwargs_stencil = {'kernel_size' : 3, 'kernel_out' : 4, 'stencil_width' : 4, 'depth' : 3, 'width' : 16}
n_runs = 200
t_inner_train = 0.01
outer_steps_train = int(1.0/t_inner_train)
fv_flux_baseline = 'muscl' # learning a correction to the MUSCL scheme
nx_exact = 256
nxs = [8, 16, 32]
plot_colors = ['blue', 'green', 'orange', 'purple']
key = jax.random.PRNGKey(12)

# setup
core_params = get_core_params(0, flux=fv_flux_baseline)
sim_params = get_sim_params(**kwargs_sim)
n_data = n_runs * outer_steps_train
training_params = get_training_params(n_data, **kwargs_train_FV)
stencil_params = get_stencil_params(**kwargs_stencil)
sim = AdvectionFVSim(core_params, sim_params)
init_fn = lambda key: get_initial_condition_fn(core_params, init_description, key=key, **kwargs_init)
model = get_model(core_params, stencil_params)



nx = 16
key = jax.random.PRNGKey(13) # new key, same initial key for each nx

t_inner = 1.0
outer_steps = 100
_, params = load_training_params(nx, sim_params, training_params, model)
f_init = get_initial_condition_fn(core_params, init_description, key=key, **kwargs_init)
a0 = get_a0(f_init, core_params, nx)

# ML model
sim_model = AdvectionFVSim(core_params, sim_params, model=model, params=params)
inner_fn_model = get_inner_fn(sim_model.step_fn, sim_model.dt_fn, t_inner)
trajectory_fn_model = get_trajectory_fn(inner_fn_model, outer_steps)
trajectory_model = trajectory_fn_model(a0)
# ML global stabilized model
sim_model_gs = AdvectionFVSim(core_params, sim_params, global_stabilization=True, model=model, params=params)
inner_fn_model_gs = get_inner_fn(sim_model_gs.step_fn, sim_model_gs.dt_fn, t_inner)
trajectory_fn_model_gs = get_trajectory_fn(inner_fn_model_gs, outer_steps)
trajectory_model_gs = trajectory_fn_model_gs(a0)
# exact trajectory
nx_exact = 256
exact_trajectory = onp.zeros((trajectory_model.shape[0],nx_exact))
for n in range(outer_steps):
    t = n * t_inner
    exact_trajectory[n] = get_a(f_init, t, core_params, nx_exact)



def plot_subfig(
    a, subfig, L, color="blue", linewidth=0.5, linestyle="solid", label=None
):
    def evalf(x, a, j, dx):
        x_j = dx * (0.5 + j)
        xi = (x - x_j) / (0.5 * dx)
        vmap_polyval = vmap(jnp.polyval, (0, None), -1)
        poly_eval = vmap_polyval(jnp.asarray([[1.]]), xi)
        return jnp.sum(poly_eval * a, axis=-1)

    nx = a.shape[0]
    dx = L / nx
    xjs = jnp.arange(nx) * L / nx
    xs = xjs[None, :] + jnp.linspace(0.0, dx, 10)[:, None]
    vmap_eval = vmap(evalf, (1, 0, 0, None), 1)
    subfig.plot(
        xs,
        vmap_eval(xs, a, jnp.arange(nx), dx),
        color=color,
        linewidth=linewidth,
        label=label,
        linestyle=linestyle,
    )
    return

L = core_params.Lx
Np = 4
step = 12
fig, axs = plt.subplots(1, Np, sharex=True, sharey=True, squeeze=True, figsize=(8,8/2.5))

lw = 2.0

plot_j = [0, 50, 58, 59]

for j in range(Np):
    plot_subfig(exact_trajectory[plot_j[j]], axs[j], L, color="grey", label="Exact", linewidth=1.0)
    plot_subfig(trajectory_model[plot_j[j]], axs[j], L, color="#ff5555", label="ML", linewidth=lw)
    if j == 0:
        plot_subfig(trajectory_model_gs[plot_j[j]], axs[j], L, color="#003366", label="ML (Stabilized)", linewidth=lw/1.67)
    else:
        plot_subfig(trajectory_model_gs[plot_j[j]], axs[j], L, color="#003366", label="ML (Stabilized)", linewidth=lw)
    axs[j].plot(onp.zeros(len(exact_trajectory[plot_j[j]])), '--',  color="black", linewidth=0.33)

axs[0].set_xlim([0, 1])
axs[0].set_ylim([-3.7, 3.7])


axs[0].spines['left'].set_visible(False)
axs[Np-1].spines['right'].set_visible(False)
for j in range(Np):
    axs[j].set_yticklabels([])
    axs[j].set_xticklabels([])
    axs[j].spines['top'].set_visible(False)
    axs[j].spines['bottom'].set_visible(False)
    axs[j].tick_params(bottom=False)
    axs[j].tick_params(left=False)
    #axs[j].tick_params(top=False)
    #axs[j].tick_params(right=False)




props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
props_2 = dict(boxstyle='round', facecolor='wheat', alpha=0.9)


axs[0].text(0.41, 0.95, r'$\frac{\partial u}{\partial t} + \frac{\partial u}{\partial x} = 0$', transform=axs[0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props_2)
# place a text box in upper left in axes coords
axs[0].text(0.00, 0.95, "$t=0.0$", transform=axs[0].transAxes, fontsize=14,
        verticalalignment='top')
axs[1].text(0.05, 0.95, "$t=50.0$", transform=axs[1].transAxes, fontsize=14,
        verticalalignment='top')
axs[2].text(0.05, 0.95, "$t=58.0$", transform=axs[2].transAxes, fontsize=14,
        verticalalignment='top')
axs[3].text(0.05, 0.95, "$t=59.0$", transform=axs[3].transAxes, fontsize=14,
        verticalalignment='top')

axs[3].text(0.05, 0.25, 'The ML-PDE solver (red)\nbecomes unstable\nbetween $58<t<59$.\nIn this work, we show\nhow to guarantee\nstability (blue).', transform=axs[3].transAxes, fontsize=8,
        verticalalignment='top', bbox=props_2)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc=(0.0, 0.02), prop={'size': 13.0}, frameon=False)
#fig.legend()

fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.show()