#!/usr/bin/env python
# coding: utf-8

# In this Jupyter notebook, we will train a machine learned FV solver to solve the 1D Burgers' equation at reduced resolution. Our objective is to study whether is it better to use the a linear flux correction or non-linear stencil for for the ML flux.
# 
# The linear stencil is given by
# 
# $$f_{j+1/2} = \sum_{k} s_{j+1/2,k} f_{j+k}$$
# 
# where $f_{j+k} = u_{j+k}^2/2$. The non-linear stencil is given by
# 
# $$f_{j+1/2} = u_{j+1/2}^2/2, \hspace{0.5cm} u_{j+1/2} = \sum_{k} s_{j+1/2, k} u_{j+k}.$$

# In[ ]:


# setup paths
import sys
basedir = '/Users/nickm/thesis/InvariantPreservingMLSolvers/1d_burgers'
readwritedir = '/Users/nickm/thesis/InvariantPreservingMLSolvers/1d_burgers'

sys.path.append('{}/core'.format(basedir))
sys.path.append('{}/simulate'.format(basedir))
sys.path.append('{}/ml'.format(basedir))


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
from initialconditions import get_a0, get_initial_condition_fn, get_a, forcing_func_sum_of_modes
from simparams import CoreParams, SimulationParams
from legendre import generate_legendre
from simulations import BurgersFVSim
from trajectory import get_trajectory_fn, get_inner_fn
from trainingutils import save_training_data
from mlparams import TrainingParams, StencilParams
from model import LearnedStencil
from trainingutils import (get_loss_fn, get_batch_fn, get_idx_gen, train_model, 
                           compute_losses_no_model, init_params, save_training_params, load_training_params)
from helper import convert_FV_representation


# In[ ]:


# helper functions

def plot_fv(a, core_params, color="blue"):
    plot_dg(a[...,None], core_params, color=color)
    
def plot_fv_trajectory(trajectory, core_params, t_inner, color='blue'):
    plot_dg_trajectory(trajectory[...,None], core_params, t_inner, color=color)
    
def plot_dg(a, core_params, color='blue'):
    p = 1
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
    p = 1
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
    p = 1
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
    
def get_core_params(Lx = 1.0, flux='godunov', nu = 0.0):
    return CoreParams(Lx, flux, nu)

def get_sim_params(name = "test", cfl_safety=0.3, rk='ssp_rk3'):
    return SimulationParams(name, basedir, readwritedir, cfl_safety, rk)

def get_training_params(n_data, train_id="test", batch_size=4, learning_rate=1e-3, num_epochs = 10, optimizer='sgd'):
    return TrainingParams(n_data, num_epochs, train_id, batch_size, learning_rate, optimizer)

def get_stencil_params(kernel_size = 3, kernel_out = 4, stencil_width=4, depth = 3, width = 16):
    return StencilParams(kernel_size, kernel_out, stencil_width, depth, width)


def l2_norm_trajectory(trajectory):
    return (jnp.mean(trajectory**2, axis=1))
    
def get_model(core_params, stencil_params, delta=True):
    features = [stencil_params.width for _ in range(stencil_params.depth - 1)]
    return LearnedStencil(features, stencil_params.kernel_size, stencil_params.kernel_out, stencil_params.stencil_width, delta)


# ### Finite Volume
# 
# ##### Training Loop
# 
# First, we will generate the data.

# In[ ]:


# training hyperparameters
init_description = 'zeros'
omega_max = 0.4
kwargs_init = {'min_num_modes': 2, 'max_num_modes': 6, 'min_k': 0, 'max_k': 3, 'amplitude_max': 1.0}
kwargs_core = {'Lx': 2 * jnp.pi, 'flux': 'weno', 'nu': 0.01}
kwargs_core_god = {'Lx': 2 * jnp.pi, 'flux': 'godunov', 'nu': 0.01}
kwargs_core_bad = {'Lx': 2 * jnp.pi, 'flux': 'godunovbad', 'nu': 0.01}
kwargs_core_bad2 = {'Lx': 2 * jnp.pi, 'flux': 'wenobad', 'nu': 0.01}
kwargs_forcing = {'min_num_modes': 20, 'max_num_modes': 20, 'min_k': 3, 'max_k': 6, 'amplitude_max': 0.5, 'omega_max': omega_max}
kwargs_sim = {'name' : "debug", 'cfl_safety' : 0.3, 'rk' : 'ssp_rk3'}
kwargs_stencil = {'kernel_size' : 5, 'kernel_out' : 4, 'stencil_width' : 6, 'depth' : 3, 'width' : 32}

#################
# HYPERPARAMETERS
#################

n_runs = 800
datapoints_per_run = 10
time_between_datapoints = 0.5

nx_exact = 512
training_steps =  40000
nxs = [16, 32, 64, 128, 256]
BASEBATCHSIZE = 128
train_id = 'reproduce'

key = jax.random.PRNGKey(13)

#################
# END HYPERPARAMS
#################


t_inner_train = time_between_datapoints
Tf = int(t_inner_train * (datapoints_per_run))
outer_steps_train = int(Tf/t_inner_train)
n_data = n_runs * outer_steps_train
lr = 3e-3
optimizer = 'adam'
delta = False


# setup
core_params = get_core_params(**kwargs_core)
core_params_god = get_core_params(**kwargs_core_god)
core_params_bad = get_core_params(**kwargs_core_bad)
core_params_bad2 = get_core_params(**kwargs_core_bad2)
sim_params = get_sim_params(**kwargs_sim)
n_data = n_runs * outer_steps_train
stencil_params = get_stencil_params(**kwargs_stencil)
sim = BurgersFVSim(core_params, sim_params, delta=delta, omega_max = omega_max)
sim_god = BurgersFVSim(core_params_god, sim_params, delta=delta, omega_max = omega_max)
sim_bad = BurgersFVSim(core_params_bad, sim_params, delta=delta, omega_max = omega_max)
sim_bad2 = BurgersFVSim(core_params_bad2, sim_params, delta=delta, omega_max = omega_max)
init_fn = lambda key: get_initial_condition_fn(core_params, init_description, key=key, **kwargs_init)
forcing_fn = forcing_func_sum_of_modes(core_params.Lx, **kwargs_forcing)
model = get_model(core_params, stencil_params, delta=delta)


# In[ ]:


save_training_data(key, init_fn, forcing_fn, core_params, sim_params, sim, t_inner_train, outer_steps_train, n_runs, nx_exact, nxs, delta=delta)


# In[ ]:


key = jax.random.PRNGKey(43)
i_params = init_params(key, model)


# In[ ]:


def training_params(nx):
    batch_size = int(BASEBATCHSIZE * (nx_exact / nx))
    num_epochs = int(training_steps * batch_size / n_data)
    return get_training_params(n_data, train_id = train_id, batch_size = batch_size, optimizer = optimizer, num_epochs = num_epochs, learning_rate = lr)

for i, nx in enumerate(nxs):
    print(nx)
    idx_fn = lambda key: get_idx_gen(key, training_params(nx))
    batch_fn = get_batch_fn(sim_params, training_params(nx), nx, delta=delta)
    loss_fn = get_loss_fn(model, core_params, forcing_fn, delta=delta)
    losses, params = train_model(model, i_params, training_params(nx), key, idx_fn, batch_fn, loss_fn)
    save_training_params(nx, sim_params, training_params(nx), params, losses)


# In[ ]:


print("Equation is 1D Burgers")
for i, nx in enumerate(nxs):
    losses, _ = load_training_params(nx, sim_params, training_params(nx), model)
    plt.plot(losses, label=nx)
    print("nx is {}, average loss is {}".format(nx, (losses)))
plt.ylim([0,0.1])
plt.legend()
plt.show()


# In[ ]:


# pick a key that gives something nice
key = jax.random.PRNGKey(29)

key1, key2 = jax.random.split(key)


for i, nx in enumerate(nxs):
    print("nx is {}".format(nx))
    f_init = get_initial_condition_fn(core_params, 'zeros', key=key1, **kwargs_init)
    f_forcing = forcing_fn(key2)
    t0 = 0.0
    a0 = get_a0(f_init, core_params, nx)
    a0_exact = get_a0(f_init, core_params, nx_exact)
    x0 = (a0, t0)
    x0_exact = (a0_exact, t0)
    
    t_inner = 2.0
    outer_steps = 10
    
    # Weno
    step_fn = lambda a, t, dt: sim.step_fn(a, t, dt, forcing_func = f_forcing)
    inner_fn = get_inner_fn(step_fn, sim.dt_fn, t_inner)
    trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
    trajectory_weno, _ = trajectory_fn(x0)

    # exact trajectory
    trajectory_exact, _ = trajectory_fn(x0_exact)
    trajectory_exact_ds = vmap(convert_FV_representation, (0, None, None), 0)(trajectory_exact, nx, core_params.Lx)
    
    
    # Godunov
    step_fn = lambda a, t, dt: sim_god.step_fn(a, t, dt, forcing_func = f_forcing)
    inner_fn = get_inner_fn(step_fn, sim.dt_fn, t_inner)
    trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
    trajectory, _ = trajectory_fn(x0)

    # Godunov Bad
    step_fn = lambda a, t, dt: sim_bad.step_fn(a, t, dt, forcing_func = f_forcing)
    inner_fn = get_inner_fn(step_fn, sim.dt_fn, t_inner)
    trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
    trajectory_godunov_bad, _ = trajectory_fn(x0)
    
    # Weno Bad
    step_fn = lambda a, t, dt: sim_bad2.step_fn(a, t, dt, forcing_func = f_forcing)
    inner_fn = get_inner_fn(step_fn, sim.dt_fn, t_inner)
    trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
    trajectory_weno_bad, _ = trajectory_fn(x0)
    
    
    # with params
    _, params = load_training_params(nx, sim_params, training_params(nx), model)
    sim_model = BurgersFVSim(core_params, sim_params, model=model, params=params, delta=delta, omega_max = omega_max)
    step_fn_model = lambda a, t, dt: sim_model.step_fn(a, t, dt, forcing_func = f_forcing)
    inner_fn_model = get_inner_fn(step_fn_model, sim_model.dt_fn, t_inner)
    trajectory_fn_model = get_trajectory_fn(inner_fn_model, outer_steps)
    trajectory_model, _ = trajectory_fn_model(x0)    

    # with gs
    sim_model_gs = BurgersFVSim(core_params, sim_params, model=model, params=params, global_stabilization = True, delta=delta, omega_max = omega_max)
    step_fn_model_gs = lambda a, t, dt: sim_model_gs.step_fn(a, t, dt, forcing_func = f_forcing)
    inner_fn_model_gs = get_inner_fn(step_fn_model_gs, sim_model_gs.dt_fn, t_inner)
    trajectory_fn_model_gs = get_trajectory_fn(inner_fn_model_gs, outer_steps)
    trajectory_model_gs, _ = trajectory_fn_model_gs(x0)    
    
    
    plot_multiple_fv_trajectories([trajectory_weno_bad, trajectory_model, trajectory_exact_ds], core_params, t_inner)
    plt.ylim([-1.5, 1.5])
    plt.show()
    
    plt.plot(l2_norm_trajectory(trajectory))
    plt.plot(l2_norm_trajectory(trajectory_model))
    plt.plot(l2_norm_trajectory(trajectory_exact_ds))
    plt.show()
"""

# In[ ]:


N = 2

#mae_weno = []
mae_god = []
mae_learned = []

#mae_bad = []
#mae_bad2 = []
#mae_zeros = []

def mae_loss(v, v_ex):
    diff = v - v_ex
    return jnp.mean(jnp.absolute(diff))

for i, nx in enumerate(nxs):
    
    key = jax.random.PRNGKey(11)
    
    _, params = load_training_params(nx, sim_params, training_params(nx), model)
    t_inner = 0.1
    outer_steps = 150
    outer_steps_warmup = 100
    
    mae_weno_nx = 0.0
    mae_god_nx = 0.0
    mae_learned_nx = 0.0
    mae_bad_nx = 0.0
    mae_bad2_nx = 0.0
    mae_zeros_nx = 0.0
    
    vmap_convert = vmap(convert_FV_representation, (0, None, None), 0)
    
    for n in range(N):
        print(n)
        
        key, key1, key2 = jax.random.split(key, 3)
        
        f_init = get_initial_condition_fn(core_params, 'zeros', key=key1, **kwargs_init)
        f_forcing = forcing_fn(key2)
        
        step_fn = lambda a, t, dt: sim.step_fn(a, t, dt, forcing_func = f_forcing)
        inner_fn = get_inner_fn(step_fn, sim.dt_fn, t_inner)
        trajectory_fn_weno = get_trajectory_fn(inner_fn, outer_steps)


        
        t0 = 0.0
        a0_exact = get_a0(f_init, core_params, nx_exact)
        x0_exact = (a0_exact, t0)
    

        #warmup
        trajectory_exact, trajectory_t = trajectory_fn_weno(x0_exact)
        a0_exact = trajectory_exact[-1]
        t0 = trajectory_t[-1]
        a0 = convert_FV_representation(a0_exact, nx, core_params.Lx)
        x0 = (a0, t0)
        x0_exact = (a0_exact, t0)
        
        
        
        # exact trajectory
        trajectory_exact, _ = trajectory_fn_weno(x0_exact)
        trajectory_exact_ds = vmap(convert_FV_representation, (0, None, None), 0)(trajectory_exact, nx, core_params.Lx)
    
        
        # WENO
        trajectory_weno, _ = trajectory_fn_weno(x0)
        

        # Godunov
        step_fn = lambda a, t, dt: sim_god.step_fn(a, t, dt, forcing_func = f_forcing)
        inner_fn = get_inner_fn(step_fn, sim.dt_fn, t_inner)
        trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
        trajectory_god, _ = trajectory_fn(x0)

        # Godunov Bad
        step_fn = lambda a, t, dt: sim_bad.step_fn(a, t, dt, forcing_func = f_forcing)
        inner_fn = get_inner_fn(step_fn, sim.dt_fn, t_inner)
        trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
        trajectory_godunov_bad, _ = trajectory_fn(x0)

        # Weno Bad
        step_fn = lambda a, t, dt: sim_bad2.step_fn(a, t, dt, forcing_func = f_forcing)
        inner_fn = get_inner_fn(step_fn, sim.dt_fn, t_inner)
        trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
        trajectory_weno_bad, _ = trajectory_fn(x0)


        # with params
        _, params = load_training_params(nx, sim_params, training_params(nx), model)
        sim_model = BurgersFVSim(core_params, sim_params, model=model, params=params, delta=delta, omega_max = omega_max)
        step_fn_model = lambda a, t, dt: sim_model.step_fn(a, t, dt, forcing_func = f_forcing)
        inner_fn_model = get_inner_fn(step_fn_model, sim_model.dt_fn, t_inner)
        trajectory_fn_model = get_trajectory_fn(inner_fn_model, outer_steps)
        trajectory_model, _ = trajectory_fn_model(x0)    

        # with gs
        sim_model_gs = BurgersFVSim(core_params, sim_params, model=model, params=params, global_stabilization = True, delta=delta, omega_max = omega_max)
        step_fn_model_gs = lambda a, t, dt: sim_model_gs.step_fn(a, t, dt, forcing_func = f_forcing)
        inner_fn_model_gs = get_inner_fn(step_fn_model_gs, sim_model_gs.dt_fn, t_inner)
        trajectory_fn_model_gs = get_trajectory_fn(inner_fn_model_gs, outer_steps)
        trajectory_model_gs, _ = trajectory_fn_model_gs(x0)    



    
        
        #mae_weno_nx += mae_loss(trajectory_weno, trajectory_exact_ds) / N
        mae_god_nx += mae_loss(trajectory_model, trajectory_exact_ds) / N
        mae_learned_nx += mae_loss(trajectory_god, trajectory_exact_ds) / N
        #mae_bad_nx += mae_loss(trajectory_bad, trajectory_exact_ds) / N
        #mae_bad2_nx += mae_loss(trajectory_bad2, trajectory_exact_ds) / N
        #mae_zeros_nx += mae_loss(jnp.zeros(trajectory_exact_ds.shape), trajectory_exact_ds) / N
        
    #mae_weno.append(mae_weno_nx)
    mae_god.append(mae_god_nx)
    mae_learned.append(mae_learned_nx)
    #mae_bad.append(mae_bad_nx)
    #mae_bad2.append(mae_bad2_nx)
    #mae_zeros.append(mae_zeros_nx)


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
print(mae_weno)
print(mae_god)
print(mae_bad)
print(mae_bad2)
print(mae_zeros)
fig, axs = plt.subplots(1, 1, figsize=(7, 3.25))
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
linewidth = 3

maes = [mae_weno, mae_god, mae_bad, mae_bad2, mae_zeros]
labels = ["WENO", "GODUNOV", "GODUNOV Bad", "WENO Bad", "zeros loss"]
colors = ["blue", "red", "purple", "green", "pink"]
linestyles = ["solid", "solid", "solid", "solid", "solid"]

for k, mae in enumerate(maes):
    plt.loglog(nxs, mae, label = labels[k], color=colors[k], linewidth=linewidth, linestyle=linestyles[k])
plt.loglog(nxs, [1e-1] * len(nxs), color='black', linewidth=0.5)
plt.loglog(nxs, [1e-2] * len(nxs), color='black', linewidth=0.5)
plt.loglog(nxs, [1e-3] * len(nxs), color='black', linewidth=0.5)
axs.set_xticks([64, 32, 16, 8])
axs.set_xticklabels(["N=64", "N=32", "N=16", "N=8"], fontsize=18)
axs.set_yticks([1e-3, 1e-2, 1e-1, 1e0])
axs.set_yticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^0$"], fontsize=18)
axs.minorticks_off()
axs.set_ylabel("MAE", fontsize=18)
axs.text(0.3, 0.95, '$t=1$', transform=axs.transAxes, fontsize=18, verticalalignment='top')


handles = []
for k, mae in enumerate(maes):
    handles.append(
        mlines.Line2D(
            [],
            [],
            color=colors[k],
            linewidth=linewidth,
            label=labels[k],
            linestyle=linestyles[k]
        )
    )
axs.legend(handles=handles,loc=(0.655,0.45) , prop={'size': 15}, frameon=False)
plt.ylim([2.5e-5, 1e0+1e-1])
fig.tight_layout()


#plt.savefig('mse_vs_nx.png')
#plt.savefig('mse_vs_nx.eps')
plt.show()


# In[ ]:

"""


