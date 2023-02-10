#!/usr/bin/env python
# coding: utf-8

# In this Jupyter notebook, we will train a machine learned FV solver to solve the 1D euler equations at reduced resolution. Our objective is to first study how and whether an ML model can learn to solve these equations, then to study whether global stabilization ensures entropy increase.

# In[ ]:


# setup paths
import sys
basedir = '/Users/nickm/thesis/InvariantPreservingMLSolvers/1d_euler'
readwritedir = '/Users/nickm/thesis/InvariantPreservingMLSolvers/1d_euler'

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
BATCHSIZE = 8 #128
NUMEPOCHS = 10
flux_exact = 'musclcharacteristic'
flux_learned = 'learned'
n_runs = 25 # 1000
t_inner_train = 0.01 # 0.01
Tf = 0.2
sim_id = "euler_test"
train_id = "euler_ghost_test"
BC = 'periodic'

nxs = [16, 32, 64, 128]
nx_exact = 512

key_data = jax.random.PRNGKey(12)
key_train = jax.random.PRNGKey(42)
key_init_params = jax.random.PRNGKey(30)

BASEBATCHSIZE = 4

learning_rate = 1e-3

###### END HYPERPARAMS



kwargs_init = {'min_num_modes': 1, 'max_num_modes': 4, 'min_k': 0, 'max_k': 3, 'amplitude_max': 1.0, 'background_rho' : 1.0, 'min_rho' : 0.5, 'background_p' : 1.0, 'min_p' : 0.1}
kwargs_core_exact = {'Lx': Lx, 'gamma': gamma, 'bc': BC, 'fluxstr': flux_exact}
kwargs_core_learned = {'Lx': Lx, 'gamma': gamma, 'bc': BC, 'fluxstr': flux_learned}
kwargs_sim = {'name' : sim_id, 'cfl_safety' : 0.3, 'rk' : 'ssp_rk3'}
kwargs_train_FV = {'train_id': train_id, 'optimizer': 'adam', 'num_epochs' : NUMEPOCHS}


kwargs_model = {'kernel_size' : 5, 'kernel_out' : 4, 'depth' : 3, 'width' : 32}


outer_steps_train = int(Tf/t_inner_train)

# setup
core_params_exact = get_core_params(**kwargs_core_exact)
core_params_learned = get_core_params(**kwargs_core_learned)
sim_params = get_sim_params(**kwargs_sim)
n_data = n_runs * outer_steps_train
model_params = get_model_params(**kwargs_model)
sim_exact = EulerFVSim(core_params_exact, sim_params)
model = get_model(core_params_learned, model_params)


# ### Test Initial Conditions

# In[ ]:


nx = 100
key_test = jax.random.PRNGKey(31)
f_init = f_init_sum_of_amplitudes(core_params_exact, key_test, **kwargs_init)
a0 = get_a0(f_init, core_params_exact, nx)
t_inner = 0.02
outer_steps = 10

inner_fn = get_inner_fn(sim_exact.step_fn, sim_exact.dt_fn, t_inner)
trajectory_fn = get_trajectory_fn(inner_fn, outer_steps)
trajectory = trajectory_fn(a0)


# In[ ]:


maxs = [4.0, 2.0, 3.0]
mins = [-0.05, -2.0, -0.05]
#plot_a(a0, core_params, maxs=maxs, mins=mins)
plot_trajectory(trajectory, core_params_exact, maxs=maxs, mins=mins)


# ### Save Training Data

# In[ ]:


init_fn = lambda key: f_init_sum_of_amplitudes(core_params_exact, key_data, **kwargs_init)
save_training_data(key_data, init_fn, core_params_exact, sim_params, sim_exact, t_inner_train, outer_steps_train, n_runs, nx_exact, nxs)


# ### Train

# Next, we initialize the model parameters.

# In[ ]:


i_params = init_params(key_init_params, model)


# In[ ]:


def training_params(nx):
    UPSAMPLE = nx_exact // nx
    batch_size = BASEBATCHSIZE * UPSAMPLE
    return get_training_params(n_data, **kwargs_train_FV, batch_size=batch_size, learning_rate = learning_rate)


# Next, we run a training loop for each value of nx. The learning rate undergoes a prespecified decay.

# In[ ]:


for i, nx in enumerate(nxs):
    print(nx)
    idx_fn = lambda key: get_idx_gen(key, training_params(nx))
    batch_fn = get_batch_fn(core_params_learned, sim_params, training_params(nx), nx)
    loss_fn = get_loss_fn(model, core_params_learned)
    losses, params = train_model(model, i_params, training_params(nx), key_train, idx_fn, batch_fn, loss_fn)
    save_training_params(nx, sim_params, training_params(nx), params, losses)


# Next, we load and plot the losses for each nx to check that the simulation trained properly.

# In[ ]:


for i, nx in enumerate(nxs):
    losses, _ = load_training_params(nx, sim_params, training_params(nx), model)
    plt.plot(losses, label=nx)
    print(losses)
plt.ylim([0,25])
plt.legend()
plt.show()


# Next, we plot the accuracy of the trained model on a few simple test cases to qualitatively evaluate the success of the training. We will eventually quantify the accuracy of the trained model.

# In[ ]:


key_plot_eval = jax.random.PRNGKey(18)

for i, nx in enumerate([64]):
    print("nx is {}".format(nx))
    
    _, params = load_training_params(nx, sim_params, training_params(nx), model)
    
    f_init = f_init_sum_of_amplitudes(core_params_exact, key_plot_eval, **kwargs_init)
    a0 = get_a0(f_init, core_params_exact, nx)
    a0_exact = get_a0(f_init, core_params_exact, nx_exact)
    t_inner = 0.06666667
    outer_steps = 4
    
    # exact trajectory
    
    sim_exact = EulerFVSim(core_params_exact, sim_params)
    inner_fn_exact = get_inner_fn(sim_exact.step_fn, sim_exact.dt_fn, t_inner)
    trajectory_fn_exact = get_trajectory_fn(inner_fn_exact, outer_steps)
    trajectory_exact = trajectory_fn_exact(a0_exact)
    trajectory_exact_ds = vmap(convert_FV_representation, (0, None, None))(trajectory_exact, nx, core_params_exact.Lx)
    
    
    # characteristic flux
    trajectory_characteristic = trajectory_fn_exact(a0)
    
    
    # with params
    sim_model = EulerFVSim(core_params_learned, sim_params, model=model, params=params)
    inner_fn_model = get_inner_fn(sim_model.step_fn, sim_model.dt_fn, t_inner)
    trajectory_fn_model = get_trajectory_fn(inner_fn_model, outer_steps)
    trajectory_model = trajectory_fn_model(a0)
    
    
    # params with invariant preserving
    sim_model_gs = EulerFVSim(core_params_learned, sim_params, model=model, params=params, invariant_preserving=True)
    inner_fn_model_gs = get_inner_fn(sim_model_gs.step_fn, sim_model_gs.dt_fn, t_inner)
    trajectory_fn_model_gs = get_trajectory_fn(inner_fn_model_gs, outer_steps)
    trajectory_model_gs = trajectory_fn_model_gs(a0)
    
    maxs = [4.0, 2.0, 3.0]
    mins = [-0.05, -2.0, -0.05]
    
    """
    print("Exact")
    plot_trajectory(trajectory_exact_ds, core_params_exact, mins=mins, maxs=maxs)
    plt.show()
    
    print("MUSCL")
    plot_trajectory(trajectory_characteristic, core_params_exact, mins=mins, maxs=maxs)
    plt.show()
    """
    
    print("model")
    plot_trajectory(trajectory_model, core_params_exact, mins=mins, maxs=maxs)
    plt.show()
    
    print("positive model")
    plot_trajectory(trajectory_model_gs, core_params_exact, mins=mins, maxs=maxs)
    plt.show()
    
    
    plt.plot(entropy_trajectory(trajectory_exact_ds, core_params_exact), label="Exact")
    plt.plot(entropy_trajectory(trajectory_characteristic, core_params_exact), label="MUSCL")
    plt.plot(entropy_trajectory(trajectory_model, core_params_learned), label="Model")
    plt.plot(entropy_trajectory(trajectory_model_gs, core_params_learned), label="Invariant-Preserving", linestyle='dotted')
    plt.legend()
    plt.show()


# In[ ]:


N = 100

mse_muscl = []
mse_ml = []
mse_mlgs = []

def normalized_mse(traj, traj_exact):
    return jnp.mean((traj - traj_exact)**2 / jnp.mean(traj_exact**2, axis=1)[:, None])


for i, nx in enumerate(nxs):
    
    key = jax.random.PRNGKey(10)
    
    _, params = load_training_params(nx, sim_params, training_params_list[i], model)
    t_inner = 0.1
    outer_steps = 10
    
    mse_muscl_nx = 0.0
    mse_ml_nx = 0.0
    mse_mlgs_nx = 0.0
    
    
    # MUSCL
    inner_fn_muscl = get_inner_fn(sim.step_fn, sim.dt_fn, t_inner)
    trajectory_fn_muscl = get_trajectory_fn(inner_fn_muscl, outer_steps)

    # Model without GS
    sim_model = EulerFVSim(core_params, sim_params, global_stabilization = False, model=model, params=params)
    inner_fn_model = get_inner_fn(sim_model.step_fn, sim_model.dt_fn, t_inner)
    trajectory_fn_model = get_trajectory_fn(inner_fn_model, outer_steps)

    # Model with GS
    sim_model_gs = EulerFVSim(core_params, sim_params, global_stabilization = True, model=model, params=params)
    inner_fn_model_gs = get_inner_fn(sim_model_gs.step_fn, sim_model_gs.dt_fn, t_inner)
    trajectory_fn_model_gs = get_trajectory_fn(inner_fn_model_gs, outer_steps)
   
    
    for n in range(N):
        
        f_init = f_init_sum_of_amplitudes(core_params, key, **kwargs_init)
        a0 = get_a0(f_init, core_params, nx)
        a0_exact = get_a0(f_init, core_params, nx_exact)
        
        trajectory_muscl = trajectory_fn_muscl(a0)
        trajectory_model = trajectory_fn_model(a0)
        trajectory_model_gs = trajectory_fn_model_gs(a0)
        
        
        # Exact trajectory
        exact_trajectory = trajectory_fn_muscl(a0_exact)
        raise Exception # TODO
        exact_trajectory_ds = ...
        
        mse_muscl_nx += normalized_mse(trajectory_muscl, exact_trajectory) / N
        mse_ml_nx += normalized_mse(trajectory_model, exact_trajectory) / N
        mse_mlgs_nx += normalized_mse(trajectory_model_gs, exact_trajectory) / N
        
        key, _ = jax.random.split(key)
        
    mse_muscl.append(mse_muscl_nx)
    mse_ml.append(mse_ml_nx)
    mse_mlgs.append(mse_mlgs_nx)


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
print(mse_muscl)
print(mse_ml)
print(mse_mlgs)
print(mse_mlmc)
fig, axs = plt.subplots(1, 1, figsize=(7, 3.25))
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
linewidth = 3

mses = [mse_ml, mse_mlgs, mse_muscl]
labels = ["ML", "ML (Stabilized)", "MUSCL"]
colors = ["blue", "red", "purple", "green"]
linestyles = ["solid", "dashed", "solid", "solid"]

for k, mse in enumerate(mses):
    plt.loglog(nxs, mse, label = labels[k], color=colors[k], linewidth=linewidth, linestyle=linestyles[k])

axs.set_xticks([64, 32, 16, 8])
axs.set_xticklabels(["N=64", "N=32", "N=16", "N=8"], fontsize=18)
axs.set_yticks([1e-3, 1e-2, 1e-1, 1e0])
axs.set_yticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^0$"], fontsize=18)
axs.minorticks_off()
axs.set_ylabel("Normalized MSE", fontsize=18)
axs.text(0.3, 0.95, '$t=1$', transform=axs.transAxes, fontsize=18, verticalalignment='top')


handles = []
for k, mse in enumerate(mses):
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
plt.ylim([2.5e-4, 1e0+1e-1])
fig.tight_layout()


#plt.savefig('mse_vs_nx_euler.png')
#plt.savefig('mse_vs_nx_euler.eps')
plt.show()


# ### Demonstrate that Global Stabilization Improves Accuracy over Time
# 
# For nx = 16, plot the accuracy of global stabilization versus ML MC limiter vs ML on the y-axis, with time on the x-axis.

# In[ ]:


nx = 16
N = 25

Ts = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

def normalized_mse(traj, traj_exact):
    return jnp.mean((traj - traj_exact)**2 / jnp.mean(traj_exact**2, axis=1)[:, None])

mse_muscl = []
mse_ml = []
mse_mlgs = []

_, params = load_training_params(nx, sim_params, training_params_list[0], model)


for T in Ts:
    
    key = jax.random.PRNGKey(20)
    
    t_inner = 0.1
    outer_steps = int(T / t_inner)
    
    # MUSCL
    inner_fn_muscl = get_inner_fn(sim.step_fn, sim.dt_fn, t_inner)
    trajectory_fn_muscl = get_trajectory_fn(inner_fn_muscl, outer_steps)

    # Model without GS
    sim_model = EulerFVSim(core_params, sim_params, global_stabilization = False, model=model, params=params)
    inner_fn_model = get_inner_fn(sim_model.step_fn, sim_model.dt_fn, t_inner)
    trajectory_fn_model = get_trajectory_fn(inner_fn_model, outer_steps)

    # Model with GS
    sim_model_gs = EulerFVSim(core_params, sim_params, global_stabilization = True, model=model, params=params)
    inner_fn_model_gs = get_inner_fn(sim_model_gs.step_fn, sim_model_gs.dt_fn, t_inner)
    trajectory_fn_model_gs = get_trajectory_fn(inner_fn_model_gs, outer_steps)
    
    mse_muscl_nx = 0.0
    mse_ml_nx = 0.0
    mse_mlgs_nx = 0.0
    
    for n in range(N):
    
        f_init = f_init_sum_of_amplitudes(core_params, key, **kwargs_init)
        a0 = get_a0(f_init, core_params, nx)
        a0_exact = get_a0(f_init, core_params, nx_exact)
        
        trajectory_muscl = trajectory_fn_muscl(a0)
        trajectory_model = trajectory_fn_model(a0)
        trajectory_model_gs = trajectory_fn_model_gs(a0)
        
        # Exact trajectory
        exact_trajectory = trajectory_fn_muscl(a0_exact)
        raise Exception # TODO
        exact_trajectory_ds = ...
        
        mse_muscl_nx += normalized_mse(trajectory_muscl, exact_trajectory_ds) / N
        mse_ml_nx += normalized_mse(trajectory_model, exact_trajectory_ds) / N
        mse_mlgs_nx += normalized_mse(trajectory_model_gs, exact_trajectory_ds) / N
    
        key, _ = jax.random.split(key)
    
    
    mse_muscl.append(mse_muscl_nx)
    mse_ml.append(mse_ml_nx)
    mse_mlgs.append(mse_mlgs_nx)


# In[ ]:


fig, axs = plt.subplots(1, 1, figsize=(7, 3.25))
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
linewidth = 3

Ts = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

mses = [mse_ml, mse_mlgs, mse_muscl]
labels = ["ML", "ML (Stabilized)", "MUSCL"]
colors = ["blue", "red", "purple", "green"]
linestyles = ["solid", "dashed", "solid", "solid"]

for k, mse in enumerate(mses):
    plt.loglog(Ts, [jnp.nan_to_num(error, nan=1e7) for error in mse], label = labels[k], color=colors[k], linewidth=linewidth, linestyle=linestyles[k])

Ts = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
axs.set_xticks(Ts)
axs.set_xticklabels(["t=1", "2", "5", "10", "20", "50", "t=100"], fontsize=18)
axs.set_yticks([1e-3, 1e-2, 1e-1, 1e0])
axs.set_yticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^0$"], fontsize=18)
axs.minorticks_off()
axs.set_ylabel("Normalized MSE", fontsize=18)
axs.text(0.15, 0.8, '$N=16$', transform=axs.transAxes, fontsize=18, verticalalignment='top')


handles = []
for k, mse in enumerate(mses):
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
axs.legend(handles=handles,loc=(0.6,0.1) , prop={'size': 15}, frameon=False)
plt.ylim([2.5e-4, 1e0+1e-1])
fig.tight_layout()


#plt.savefig('mse_vs_time_euler.png')
#plt.savefig('mse_vs_time_euler.eps')
plt.show()


# In[ ]:





# In[ ]:




