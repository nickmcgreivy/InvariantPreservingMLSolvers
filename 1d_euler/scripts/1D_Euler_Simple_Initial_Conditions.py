#!/usr/bin/env python
# coding: utf-8

# In this Jupyter notebook, we will train a machine learned FV solver to solve the 1D euler equations at reduced resolution. Our objective is to first study how and whether an ML model can learn to solve these equations, then to study whether global stabilization ensures entropy increase.

# In[ ]:





# In[ ]:


# setup paths
import sys
#basedir = '/Users/nickm/thesis/InvariantPreservingMLSolvers/1d_euler'
#readwritedir = '/Users/nickm/thesis/InvariantPreservingMLSolvers/1d_euler'
basedir = '/home/mcgreivy/InvariantPreservingMLSolvers/1d_euler'
readwritedir = 'scratch/gpfs/mcgreivy/InvariantPreservingMLSolvers/1d_euler'

sys.path.append('{}/core'.format(basedir))
sys.path.append('{}/simulate'.format(basedir))
sys.path.append('{}/ml'.format(basedir))


# In[ ]:


# import external packages
import jax
import jax.numpy as jnp
import numpy as onp
from jax import config, vmap, jit
config.update("jax_enable_x64", True)
import xarray
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial


# In[ ]:


# import internal packages
from flux import Flux
from initialconditions import get_a0, f_init_sum_of_amplitudes, shock_tube_problem_1
from simparams import CoreParams, SimulationParams
from simulations import EulerFVSim
from trajectory import get_trajectory_fn, get_inner_fn
from helper import get_rho, get_u, get_p, get_c, get_entropy, convert_FV_representation
from trainingutils import save_training_data
from mlparams import TrainingParams, ModelParams
from model import LearnedFlux
from trainingutils import (get_loss_fn, get_batch_fn, get_idx_gen, train_model, 
                           compute_losses_no_model, init_params, save_training_params, 
                           load_training_params)


# In[ ]:


# helper functions
def plot_a(a, core_params, mins = [0.0 - 2e-2] * 3, maxs= [1.0 + 5e-2] * 3, title = ""):
    x = jnp.linspace(0.0, core_params.Lx, a.shape[1])
    
    fig, axs = plt.subplots(1, 3, figsize=(11, 3))
    axs[0].plot(x, get_rho(a, core_params))
    axs[0].set_ylabel(r'$\rho$')
    axs[0].set_ylim([mins[0], maxs[0]])
    
    axs[1].plot(x, get_u(a, core_params))
    axs[1].set_ylabel(r'$u$')
    axs[1].set_ylim([mins[1], maxs[1]])
    
    axs[2].plot(x, get_p(a, core_params))
    axs[2].set_ylabel(r'$p$')
    axs[2].set_ylim([mins[2], maxs[2]])
    
    fig.suptitle(title)
    fig.tight_layout()

def plot_ac(a, core_params, mins = [0.0 - 2e-2] * 3, maxs= [1.0 + 5e-2] * 3):
    x = jnp.linspace(0.0, core_params.Lx, a.shape[1])
    
    fig, axs = plt.subplots(1, 4, figsize=(11, 3))
    axs[0].plot(x, get_rho(a, core_params))
    axs[0].set_title(r'$\rho$')
    axs[0].set_ylim([mins[0], maxs[0]])
    
    axs[1].plot(x, get_u(a, core_params))
    axs[1].set_ylabel(r'$u$')
    axs[1].set_ylim([mins[1], maxs[1]])
    
    axs[2].plot(x, get_p(a, core_params))
    axs[2].set_ylabel(r'$p$')
    axs[2].set_ylim([mins[2], maxs[2]])
    
    axs[3].plot(x, get_c(a, core_params))
    axs[3].set_ylabel(r'$c$')
    axs[3].set_ylim([mins[1], maxs[1]])

def plot_trajectory(trajectory, core_params, mins = [0.0 - 2e-2] * 3, maxs= [1.0 + 5e-2] * 3):
    nx = trajectory.shape[2]
    xs = jnp.arange(nx) * core_params.Lx / nx
    xs = xs.T.reshape(-1)
    outer_steps = trajectory.shape[0]
    coords = {
        'x': xs,
        'time': t_inner * jnp.arange(outer_steps)
    }
    rhos = trajectory[:, 0, :]
    g = xarray.DataArray(rhos, dims=["time", "x"], coords=coords).plot(
        col='time', col_wrap=5)
    plt.ylim([mins[0], maxs[0]])
    g.axes[0][0].set_ylabel(r'$\rho$', fontsize=18)

    us = trajectory[:, 1, :] / trajectory[:, 0, :]
    g = xarray.DataArray(us, dims=["time", "x"], coords=coords).plot(
        col='time', col_wrap=5)
    plt.ylim([mins[1], maxs[1]])
    g.axes[0][0].set_ylabel(r'$u$', fontsize=18)

    ps = (core_params.gamma - 1) * (trajectory[:, 2, :] - 0.5 * trajectory[:, 1, :]**2 / trajectory[:, 0, :])
    g = xarray.DataArray(ps, dims=["time", "x"], coords=coords).plot(
        col='time', col_wrap=5)
    plt.ylim([mins[2], maxs[2]])
    g.axes[0][0].set_ylabel(r'$p$', fontsize=18)

def get_core_params(Lx = 1.0, gamma = 5/3, bc = 'periodic', fluxstr = 'laxfriedrichs'):
    return CoreParams(Lx, gamma, bc, fluxstr)

def get_sim_params(name = "test", cfl_safety=0.3, rk='ssp_rk3'):
    return SimulationParams(name, basedir, readwritedir, cfl_safety, rk)
    
def get_training_params(n_data, train_id="test", batch_size=4, learning_rate=1e-3, num_epochs = 10, optimizer='adam'):
    return TrainingParams(n_data, num_epochs, train_id, batch_size, learning_rate, optimizer)

def get_model_params(kernel_size = 3, kernel_out = 4, depth = 3, width = 16):
    return ModelParams(kernel_size, kernel_out, depth, width)

def entropy_trajectory(trajectory, core_params):
    return jnp.sum(vmap(get_entropy, (0, None))(trajectory, core_params), axis=-1)

def get_model(core_params, model_params):
    features = [model_params.width for _ in range(model_params.depth - 1)]
    return LearnedFlux(features, model_params.kernel_size, model_params.kernel_out, core_params.bc)


# ### Finite Volume
# 
# ##### Training Loop
# 
# First, we will generate the data.

# In[ ]:


####### HYPERPARAMS

Lx = 1.0
gamma = 1.4

flux_exact = 'musclcharacteristic'
flux_learned = 'learned'
n_runs = 10000
t_inner_train = 0.01
Tf = 0.2
BC = 'open'
sim_id = "euler_{}_simple".format(BC)
train_id = "euler_{}_simple".format(BC)
DEPTH=5

nxs = [4, 8, 16, 32]
nx_exact = 256

key_data = jax.random.PRNGKey(13)
key_train = jax.random.PRNGKey(43)
key_init_params = jax.random.PRNGKey(31)

BASEBATCHSIZE = 64
WIDTH = 32
learning_rate = 1e-4
NUM_TRAINING_ITERATIONS = 200000

###### END HYPERPARAMS



kwargs_init = {'min_num_modes': 1, 'max_num_modes': 1, 'min_k': 1, 'max_k': 1, 'amplitude_max': 1.0, 'background_rho' : 1.0, 'min_rho' : 0.75, 'background_p' : 1.0, 'min_p' : 0.5}
kwargs_core_exact = {'Lx': Lx, 'gamma': gamma, 'bc': BC, 'fluxstr': flux_exact}
kwargs_core_learned = {'Lx': Lx, 'gamma': gamma, 'bc': BC, 'fluxstr': flux_learned}
kwargs_sim = {'name' : sim_id, 'cfl_safety' : 0.3, 'rk' : 'ssp_rk3'}
kwargs_train_FV = {'train_id': train_id, 'optimizer': 'adam'}


kwargs_model = {'kernel_size' : 5, 'kernel_out' : 4, 'depth' : DEPTH, 'width' : WIDTH}


outer_steps_train = int(Tf/t_inner_train)

# setup
core_params_exact = get_core_params(**kwargs_core_exact)
core_params_learned = get_core_params(**kwargs_core_learned)
sim_params = get_sim_params(**kwargs_sim)
n_data = n_runs * outer_steps_train
model_params = get_model_params(**kwargs_model)
sim_exact = EulerFVSim(core_params_exact, sim_params)
model = get_model(core_params_learned, model_params)

def training_params(nx):
    UPSAMPLE = nx_exact // nx
    batch_size = BASEBATCHSIZE * UPSAMPLE
    num_epochs = 1 + int(NUM_TRAINING_ITERATIONS * batch_size / n_data)
    return get_training_params(n_data, **kwargs_train_FV, num_epochs = num_epochs, batch_size=batch_size, learning_rate = learning_rate)


sim_exact = lambda aL, aR: EulerFVSim(core_params_exact, sim_params, aL=aL, aR=aR)

init_fn = lambda key: f_init_sum_of_amplitudes(core_params_exact, key, **kwargs_init)
save_training_data(key_data, init_fn, core_params_exact, sim_params, sim_exact, t_inner_train, outer_steps_train, n_runs, nx_exact, nxs)


i_params = init_params(key_init_params, model)

for i, nx in enumerate(nxs):
    print(nx)
    idx_fn = lambda key: get_idx_gen(key, training_params(nx))
    batch_fn = get_batch_fn(core_params_learned, sim_params, training_params(nx), nx)
    loss_fn = get_loss_fn(model, core_params_learned)
    losses, params = train_model(model, i_params, training_params(nx), key_train, idx_fn, batch_fn, loss_fn)
    save_training_params(nx, sim_params, training_params(nx), params, losses)












N_test = 100

key = jax.random.PRNGKey(45)

t_inner = 0.01
outer_steps = 11
    
# exact trajectory setup


convert_trajectory_fn = vmap(convert_FV_representation, (0, None, None))

def RMSE_trajectory(traj, traj_ex):
    return jnp.sqrt(jnp.mean((traj - traj_ex)**2))

# trajectory setup

@partial(jit, static_argnums=(4,5,))
def get_trajectory_ML(a0, aL, aR, params, invariant_preserving=False, cfl_safety=0.3):
    sim_params = get_sim_params(name=sim_id, cfl_safety=cfl_safety, rk = 'ssp_rk3')
    sim_model = EulerFVSim(core_params_learned, sim_params, model=model, params=params, invariant_preserving = invariant_preserving, aL=aL, aR=aR)
    inner_fn_model = get_inner_fn(sim_model.step_fn, sim_model.dt_fn, t_inner)
    trajectory_fn_model = get_trajectory_fn(inner_fn_model, outer_steps)
    return trajectory_fn_model(a0)

errors = onp.zeros((len(nxs), 3))

num_not_nan = onp.zeros(len(nxs))

for n in range(N_test):
    
    print(n)
    
    key, _ = jax.random.split(key)
    
    f_init = f_init_sum_of_amplitudes(core_params_exact, key, **kwargs_init)
    aL = f_init(0.0, 0.0)
    aR = f_init(Lx, 0.0)
    a0_exact = get_a0(f_init, core_params_exact, nx_exact)

    sim_exact = EulerFVSim(core_params_exact, sim_params, aL=aL, aR=aR)
    inner_fn_exact = get_inner_fn(sim_exact.step_fn, sim_exact.dt_fn, t_inner)
    trajectory_fn_exact = jit(get_trajectory_fn(inner_fn_exact, outer_steps))
    trajectory_exact = trajectory_fn_exact(a0_exact)
    
    for i, nx in enumerate(nxs):
        
        print(nx)
        
        _, params = load_training_params(nx, sim_params, training_params(nx), model)
        
    
        a0 = convert_FV_representation(a0_exact, nx, core_params_exact.Lx)
        
        # exact trajectory
        trajectory_exact_ds = convert_trajectory_fn(trajectory_exact, nx, core_params_exact.Lx)
        
        # MUSCL trajectory
        trajectory_muscl = trajectory_fn_exact(a0)
        
        # ML trajectory
        trajectory_ML = get_trajectory_ML(a0, aL, aR, params)
        
        # Invariant-preserving ML trajectory
        trajectory_invariant_ML = get_trajectory_ML(a0, aL, aR, params, invariant_preserving=True, cfl_safety = 0.05)
        
        error_muscl = RMSE_trajectory(trajectory_muscl, trajectory_exact_ds)
        error_ml = RMSE_trajectory(trajectory_ML, trajectory_exact_ds)
        error_ml_gs = RMSE_trajectory(trajectory_invariant_ML, trajectory_exact_ds)
        if not onp.isnan(error_ml) and error_ml < 1.0:
            errors[i, 0] += error_muscl
            errors[i, 1] += error_ml
            errors[i, 2] += error_ml_gs
            num_not_nan[i] += 1
    
errors = errors / num_not_nan[:, None]
print("number nan: {}".format(N_test - num_not_nan))


# In[ ]:

"""
with open('mses_simple_{}.npy'.format(BC), 'wb') as f:
    onp.save(f, errors)


# In[ ]:


mses = onp.load('mses_simple_{}.npy'.format(BC))
mses = onp.nan_to_num(mses, nan=1e5)
print(mses)


# In[ ]:


import matplotlib.patches as mpatches
import matplotlib.lines as mlines


fig, axs = plt.subplots(1, 1, figsize=(7, 3.5))
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
linewidth = 3

labels = ["MUSCL", "ML", "ML (Positivity- &\nEntropy-Preserving)"]
colors = ["blue", "red", "green", "green"]
linestyles = ["solid", "solid", "dashed", "solid"]

for k in range(3):
    plt.loglog(nxs, mses[:,k], label = labels[k], color=colors[k], linewidth=linewidth, linestyle=linestyles[k])

axs.set_xticks(list(reversed(nxs)))
axs.set_xticklabels(["N=32", "N=16", "N=8", "N=4"], fontsize=18)
axs.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
axs.set_yticklabels(["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"], fontsize=18)
axs.minorticks_off()
axs.set_ylabel("Normalized MSE", fontsize=18)
#axs.text(0.15, 0.9, '$t=0.1$', transform=axs.transAxes, fontsize=18, verticalalignment='top')


handles = []
for k in range(3):
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
axs.legend(handles=handles,loc=(0.4,0.45) , prop={'size': 18}, frameon=True)
plt.ylim([5e-4, 4e-1])
fig.suptitle('1D Compressible Euler, Ghost Boundary Conditions', fontsize=18)


fig.tight_layout()

plt.savefig('mse_vs_nx_euler_{}.png'.format(BC))
plt.savefig('mse_vs_nx_euler_{}.eps'.format(BC))
plt.show()


# In[ ]:



"""
