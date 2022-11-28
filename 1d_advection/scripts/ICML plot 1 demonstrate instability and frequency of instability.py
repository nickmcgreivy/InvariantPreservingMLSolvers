#!/usr/bin/env python
# coding: utf-8

# In this Jupyter notebook, we will train a machine learned FV solver and machine learned DG solver to solve the 1D advection equation at reduced resolution. Our objective is to study the frequency of instability, and to demonstrate that global stabilization eliminates this instability.

# In[1]:


# setup paths
import sys
basedir = '/Users/nickm/thesis/icml2023paper/1d_advection'
readwritedir = '/Users/nickm/thesis/icml2023paper/1d_advection'

sys.path.append('{}/core'.format(basedir))
sys.path.append('{}/simulate'.format(basedir))
sys.path.append('{}/ml'.format(basedir))


# In[2]:


# import external packages
import jax
import jax.numpy as jnp
import numpy as onp
from jax import config, vmap
config.update("jax_enable_x64", True)
import xarray
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


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


# In[4]:


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
    plot_multiple_dg_trajectory([trajectory[..., None] for trajectory in trajectories], core_params, t_inner)

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


# ### Finite Volume
# 
# ##### Training Loop
# 
# First, we will generate the data.

# In[ ]:


# training hyperparameters
init_description = 'sum_sin'
kwargs_init = {'min_num_modes': 1, 'max_num_modes': 6, 'min_k': 1, 'max_k': 4, 'amplitude_max': 1.0}
kwargs_sim = {'name' : "test", 'cfl_safety' : 0.3, 'rk' : 'ssp_rk3'}
kwargs_train_FV = {'train_id': "test", 'batch_size' : 32, 'optimizer': 'adam', 'learning_rate' : 1e-3,  'num_epochs' : 1000}
#kwargs_train_DG = {'train_id': "test", 'batch_size' : 8, 'optimizer': 'adam', 'learning_rate' : 1e-5, 'num_epochs' : 10}
kwargs_stencil = {'kernel_size' : 3, 'kernel_out' : 4, 'stencil_width' : 4, 'depth' : 3, 'width' : 16}
n_runs = 1000
t_inner_train = 0.02
outer_steps_train = int(1.0/t_inner_train)
fv_flux_baseline = 'muscl' # learning a correction to the MUSCL scheme
nx_exact = 256
nxs = [8, 16, 32, 64]
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


# In[ ]:


# save training data
save_training_data(key, init_fn, core_params, sim_params, sim, t_inner_train, outer_steps_train, n_runs, nx_exact, nxs)


# Next, we initialize the model parameters.

# In[ ]:


key = jax.random.PRNGKey(42)
i_params = init_params(key, model)


# Next, we run a training loop for each value of nx. The learning rate undergoes a prespecified decay.

# In[ ]:


for nx in nxs:
    print(nx)
    idx_fn = lambda key: get_idx_gen(key, training_params)
    batch_fn = get_batch_fn(core_params, sim_params, training_params, nx)
    loss_fn = get_loss_fn(model, core_params)
    losses, params = train_model(model, i_params, training_params, key, idx_fn, batch_fn, loss_fn)
    save_training_params(nx, sim_params, training_params, params, losses)


# Next, we load and plot the losses for each nx to check that the simulation trained properly.

# In[ ]:


for nx in nxs:
    losses, _ = load_training_params(nx, sim_params, training_params, model)
    plt.plot(losses, label=nx)
plt.ylim([0,1])
plt.legend()
plt.show()


# Next, we plot the accuracy of the trained model on a few simple test cases to qualitatively evaluate the success of the training. We will eventually quantify the accuracy of the trained model.

# In[ ]:


# pick a key that gives something nice
key = jax.random.PRNGKey(18)

for i, nx in enumerate(nxs):
    print("nx is {}".format(nx))
    _, params = load_training_params(nx, sim_params, training_params, model)
    
    f_init = get_initial_condition_fn(core_params, init_description, key=key, **kwargs_init)
    a0 = get_a0(f_init, core_params, nx)
    t_inner = 0.2
    outer_steps = 50
    # with params
    sim_model = AdvectionFVSim(core_params, sim_params, model=model, params=params)
    inner_fn_model = get_inner_fn(sim_model.step_fn, sim_model.dt_fn, t_inner)
    trajectory_fn_model = get_trajectory_fn(inner_fn_model, outer_steps)
    trajectory_model = trajectory_fn_model(a0)
    #plot_fv_trajectory(trajectory_model, core_params, t_inner, color = plot_colors[i])
    
    # with global stabilization
    sim_model_gs = AdvectionFVSim(core_params, sim_params, global_stabilization=True, epsilon_gs=0.0, model=model, params=params)
    inner_fn_model_gs = get_inner_fn(sim_model_gs.step_fn, sim_model_gs.dt_fn, t_inner)
    trajectory_fn_model_gs = get_trajectory_fn(inner_fn_model_gs, outer_steps)
    trajectory_model_gs = trajectory_fn_model_gs(a0)
    #plot_fv_trajectory(trajectory_model, core_params, t_inner, color = plot_colors[i])

    # without params
    inner_fn = get_inner_fn(sim.step_fn, sim.dt_fn, t_inner)
    trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
    trajectory = trajectory_fn(a0)
    #plot_fv_trajectory(trajectory, core_params, t_inner, color = 'red')
    

    plot_multiple_fv_trajectories([trajectory, trajectory_model_gs, trajectory_model], core_params, t_inner)
    
    #exact_trajectory = a0[None, ...] * jnp.ones(trajectory.shape[0])[:, None]
    #plot_fv_trajectory(exact_trajectory, core_params, t_inner, color='red')
    plt.show()
    plt.plot(l2_norm_trajectory(trajectory))
    plt.plot(l2_norm_trajectory(trajectory_model_gs))
    plt.plot(l2_norm_trajectory(trajectory_model))
    plt.show()


# We see from above that the baseline (red) has a large amount of numerical diffusion for small number of gridpoints, while is more accurate for more gridpoints. We also see that the machine learned model learns to accurately evolve the solution for nx > 8. So far, so good. Let's now look at a different initial condition.

# In[ ]:


# pick a key that gives something nice
key = jax.random.PRNGKey(20)

for i, nx in enumerate(nxs):
    print("nx is {}".format(nx))
    _, params = load_training_params(nx, sim_params, training_params, model)
    
    f_init = get_initial_condition_fn(core_params, init_description, key=key, **kwargs_init)
    a0 = get_a0(f_init, core_params, nx)
    t_inner = 1.0
    outer_steps = 5
    # with params
    
    sim_model = AdvectionFVSim(core_params, sim_params, model=model, params=params)
    inner_fn_model = get_inner_fn(sim_model.step_fn, sim_model.dt_fn, t_inner)
    trajectory_fn_model = get_trajectory_fn(inner_fn_model, outer_steps)
    trajectory = trajectory_fn_model(a0)
    plot_fv_trajectory(trajectory, core_params, t_inner, color = plot_colors[i])
    

    # without params
    inner_fn = get_inner_fn(sim.step_fn, sim.dt_fn, t_inner)
    trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
    
    trajectory = trajectory_fn(a0)
    plot_fv_trajectory(trajectory, core_params, t_inner, color = 'red')
    
    
    plt.show()


# Oh no! We can see that for nx=8 and nx=32, the solution goes unstable between 1.0 < t < 2.0 and 0.0 < t < 1.0 respectively.
# 
# This is not good. Even though the machine learned PDE solver gives accurate solution for certain initial conditions, the solution blows up unexpectedly for certain initial conditions. Let's next ask: how frequently does an instability arise?

# In[ ]:


N = 25

for nx in nxs:
    
    key = jax.random.PRNGKey(10) # new key, same initial key for each nx
    _, params = load_training_params(nx, sim_params, training_params, model)
    t_inner = 1.0
    outer_steps = 5
    sim_model = AdvectionFVSim(core_params, sim_params, model=model, params=params)
    inner_fn_model = get_inner_fn(sim_model.step_fn, sim_model.dt_fn, t_inner)
    trajectory_fn_model = get_trajectory_fn(inner_fn_model, outer_steps)
    
    inner_fn = get_inner_fn(sim.step_fn, sim.dt_fn, t_inner)
    trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
    
    num_nan = 0
    
    for n in range(N):
        f_init = get_initial_condition_fn(core_params, init_description, key=key, **kwargs_init)
        a0 = get_a0(f_init, core_params, nx)
        trajectory_model = trajectory_fn_model(a0)
        num_nan += jnp.isnan(trajectory_model[-1]).any()
        key, _ = jax.random.split(key)
        
    print("nx is {}, num_nan is {} out of {}".format(nx, num_nan, N))
    
    


# So we see that a large percentage of the simulations go unstable. We could use various tips and tricks to decrease the number of simulations that go unstable, such as to increase the size of the training set or the duration of training or use a different loss function. But we are interested in something else entirely: eliminating the ability of the solution to go unstable. What happens when we set global stabilization to True?

# In[ ]:


N = 25

for nx in nxs:
    
    key = jax.random.PRNGKey(10)
    _, params = load_training_params(nx, sim_params, training_params, model)
    t_inner = 1.0
    outer_steps = 5
    sim_model = AdvectionFVSim(core_params, sim_params, global_stabilization = True, epsilon_gs = 0.0, model=model, params=params)
    inner_fn_model = get_inner_fn(sim_model.step_fn, sim_model.dt_fn, t_inner)
    trajectory_fn_model = get_trajectory_fn(inner_fn_model, outer_steps)
    
    num_nan = 0
    
    for n in range(N):
        f_init = get_initial_condition_fn(core_params, init_description, key=key, **kwargs_init)
        a0 = get_a0(f_init, core_params, nx)
        trajectory_model = trajectory_fn_model(a0)
        num_nan += jnp.isnan(trajectory_model[-1]).any()
        key, _ = jax.random.split(key)
        
    print("nx is {}, num_nan is {} out of {}".format(nx, num_nan, N))


# We see that, as expected, the global stabilization method eliminates NaNs from the final solution. This is a demonstration of our claim that the solution is provably stable (in the time-continuous limit).  

# ### Demonstrate Stabilization of Solution
# 
# Now we will plot (a) the exact solution (b) the ML solution (c) the stabilized ML solution.

# In[ ]:


nx = 32
key = jax.random.PRNGKey(10) # new key, same initial key for each nx

t_inner = 0.1
outer_steps = 5
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
exact_trajectory = onp.zeros(trajectory_model.shape)
for n in range(outer_steps):
    t = n * t_inner
    exact_trajectory[n] = get_a(f_init, t, core_params, nx)


# In[ ]:


import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

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
fig, axs = plt.subplots(1, Np, sharex=True, sharey=True, squeeze=True, figsize=(8,8/4))


for j in range(Np):
    plot_subfig(exact_trajectory[j], axs[j], L, color="grey", label="Exact\nsolution", linewidth=0.75)
    plot_subfig(trajectory_model[j], axs[j], L, color="#ff5555", label="ML Solver", linewidth=2.0)
    plot_subfig(trajectory_model_gs[j], axs[j], L, color="#003366", label="Stabilized\nML Solver", linewidth=2.0)
    axs[j].plot(onp.zeros(len(exact_trajectory[j])), '--',  color="black", linewidth=0.25)

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




props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

"""
axs[0].text(0.4, 0.95, r'$\frac{\partial u}{\partial t} + \frac{\partial u}{\partial x} = 0$', transform=axs[0].transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
# place a text box in upper left in axes coords
axs[0].text(0.05, 0.95, "$t=0.0$", transform=axs[0].transAxes, fontsize=12,
        verticalalignment='top')
axs[1].text(0.05, 0.95, "$t=0.1$", transform=axs[1].transAxes, fontsize=12,
        verticalalignment='top')
axs[2].text(0.05, 0.95, "$t=0.2$", transform=axs[2].transAxes, fontsize=12,
        verticalalignment='top')
axs[3].text(0.05, 0.95, "$t=0.3$", transform=axs[3].transAxes, fontsize=12,
        verticalalignment='top')
"""
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
#fig.legend(by_label.values(), by_label.keys(),loc=(0.003,0.001), prop={'size': 9.5})

fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.show()


# ### Demonstrate that Global Stabilization Doesn't Degrade Accuracy
# 
# We want to compare four different numerical algorithms for solving the 1D advection equation. We compare: (a) MUSCL (b) Machine Learned (ML) (c) Machine learned with global stabilization and (d) Machine learned with MC limiter. 

# In[ ]:


N = 10

mse_muscl = []
mse_ml = []
mse_mlgs = []
mse_mlmc = []

for nx in nxs:
    
    key = jax.random.PRNGKey(10)
    
    _, params = load_training_params(nx, sim_params, training_params, model)
    t_inner = 1.0
    outer_steps = 5
    sim_model = AdvectionFVSim(core_params, sim_params, global_stabilization = True, model=model, params=params)
    inner_fn_model = get_inner_fn(sim_model.step_fn, sim_model.dt_fn, t_inner)
    trajectory_fn_model = get_trajectory_fn(inner_fn_model, outer_steps)
    
    mse_muscl = 0.0
    mse_ml = 0.0
    mse_mlgs = 0.0
    mse_mlmc = 0.0
    
    for n in range(N):
    
        # MUSCL
        
        
        
        key, _ = jax.random.split(key)

