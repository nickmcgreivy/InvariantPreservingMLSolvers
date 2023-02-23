#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append('/home/mcgreivy/InvariantPreservingMLSolvers/2d_incompressible_euler/ml')
sys.path.append('/home/mcgreivy/InvariantPreservingMLSolvers/2d_incompressible_euler/baselines')
sys.path.append('/home/mcgreivy/InvariantPreservingMLSolvers/2d_incompressible_euler/simulate')
#sys.path.append('/Users/nickm/thesis/InvariantPreservingMLSolvers/2d_incompressible_euler/ml')
#sys.path.append('/Users/nickm/thesis/InvariantPreservingMLSolvers/2d_incompressible_euler/baselines')
#sys.path.append('/Users/nickm/thesis/InvariantPreservingMLSolvers/2d_incompressible_euler/simulate')



import jax
import jax.numpy as jnp
import numpy as onp
from jax import config
config.update("jax_enable_x64", True)
import xarray
import seaborn as sns
import matplotlib.pyplot as plt

from initialconditions import init_fn_FNO, init_fn_jax_cfd
from simulations import KolmogorovFiniteVolumeSimulation
from simparams import FiniteVolumeSimulationParams

from helper import convert_FV_representation
from trajectory import get_trajectory_fn, get_inner_fn
from flux import Flux

from model import LearnedFlux2D
from mlparams import ModelParams, TrainingParams
from trainingutils import init_params, save_training_data, save_training_params, load_training_params
from trainingutils import get_loss_fn, get_batch_fn, get_idx_gen, train_model, compute_losses_no_model


#########################
# HYPERPARAMS
#########################


simname = "first"
train_id = "first"


cfl_safety=0.3
Lx = Ly = 2 * jnp.pi
viscosity=1/1000
forcing_coeff = 1.0
drag = 0.1
max_velocity = 7.0
ic_wavenumber = 2

batch_size= 100
learning_rate=1e-4
num_epochs = 1000
kernel_size = 5
depth = 6
width = 64

outer_steps = 100
n_runs = 100
t_inner = 0.1



t_burnin = 20.0
nx_exact = ny_exact = 128
nxs = [32] # [32, 64]

basedir = "/home/mcgreivy/InvariantPreservingMLSolvers/2d_incompressible_euler"
readwritedir = "/scratch/gpfs/mcgreivy/InvariantPreservingMLSolvers/2d_incompressible_euler"
#basedir = '/Users/nickm/thesis/InvariantPreservingMLSolvers/2d_incompressible_euler'
#readwritedir = '/Users/nickm/thesis/InvariantPreservingMLSolvers/2d_incompressible_euler'

#########################
# END HYPERPARAMS
#########################

plot_dir = '{}/data/plots'.format(readwritedir)


# In[ ]:


def get_sim_params(nx, ny, global_stabilization=False, energy_conserving=False):
    rk='ssp_rk3'
    flux=Flux.VANLEER
    return FiniteVolumeSimulationParams(simname, basedir, readwritedir, nx, ny, Lx, Ly, cfl_safety, rk, flux, global_stabilization, energy_conserving)

def get_simulation(sim_params, model=None, params=None):
    return KolmogorovFiniteVolumeSimulation(sim_params, viscosity, forcing_coeff, drag, model=model, params=params)

def get_trajectory(sim_params, v0):
    v_init = convert_FV_representation(v0, sim_params)
    sim = get_simulation(sim_params)
    rollout_fn = get_trajectory_fn(sim.step_fn, outer_steps)
    return rollout_fn(v_init)

def get_ml_params():
    return ModelParams(train_id, batch_size, learning_rate, num_epochs, kernel_size, depth, width)

def get_model():
    model_params = get_ml_params()
    return LearnedFlux2D(model_params)

def plot_fv(zeta, sim_params): 
    nx = zeta.shape[0]
    spatial_coord = jnp.arange(nx) * sim_params.Lx / nx # same for x and y
    coords = {
      'x': spatial_coord,
      'y': spatial_coord,
    }
    xarray.DataArray(zeta, dims=["x", "y"], coords=coords).plot.imshow(cmap=sns.cm.icefire, robust=True)

def plot_trajectory_fv(trajectory, sim_params, t_inner):
    nx = trajectory.shape[1]
    spatial_coord = jnp.arange(nx) * sim_params.Lx / sim_params.nx # same for x and y
    coords = {
      'x': spatial_coord,
      'y': spatial_coord,
        'time': t_inner * jnp.arange(outer_steps)
    }
    xarray.DataArray(trajectory, dims=["time", "x", "y"], coords=coords).plot.imshow(
        col='time', col_wrap=5, 
        cmap=sns.cm.icefire, robust=True)


# In[ ]:


sim_params_exact = get_sim_params(nx_exact, ny_exact)

model = get_model()
params = init_params(jax.random.PRNGKey(0), model)

sim_params_ds = []
simulations_ds = []
for nx in nxs:
    sim_params = get_sim_params(nx, nx)
    sim_params_ds.append(sim_params)
    simulations_ds.append(get_simulation(sim_params))



key_data = jax.random.PRNGKey(0)
key_eval = jax.random.PRNGKey(105)
simulation_exact = get_simulation(sim_params_exact)

#save_training_data(key_data, sim_params_exact, simulation_exact, t_burnin, t_inner, outer_steps, n_runs, sim_params_ds, simulations_ds, max_velocity=max_velocity, ic_wavenumber=ic_wavenumber)


# In[ ]:

key_init = jax.random.PRNGKey(42)
key_train = jax.random.PRNGKey(43)
n_data = n_runs * outer_steps
ml_params = get_ml_params()
model = get_model()

i_params = init_params(key_init, model)

"""
for nx in nxs:
    sim_params = get_sim_params(nx, nx)
    idx_fn = lambda key: get_idx_gen(key, ml_params, n_data)
    batch_fn = get_batch_fn(sim_params, n_data)
    loss_fn = get_loss_fn(model, sim_params)
    losses, params = train_model(model, i_params, key_train, idx_fn, batch_fn, loss_fn)

    save_training_params(sim_params, params, losses)
    print(losses)
"""


####################
# Time for evaluation
####################


t_inner = 0.5
outer_steps = 20






# In[ ]:


for i, nx in enumerate(nxs):
    sim_params = get_sim_params(nx, nx)
    model = get_model()
    losses, _ = load_training_params(sim_params, model)
    
    plt.plot(losses, label="nx={}".format(nx))

plt.yscale('log')
plt.legend()
plt.savefig('{}/losses_{}.png'.format(plot_dir, train_id))
plt.savefig('{}/losses_{}.eps'.format(plot_dir, train_id))
plt.close()

# In[ ]:




def errors_sim(key):

    key, subkey = jax.random.split(key)

    vorticity0 = init_fn_jax_cfd(subkey, sim_params_exact, 7.0, 2)

    inner_fn_burnin = get_inner_fn(simulation_exact.step_fn, simulation_exact.dt_fn, t_burnin)
    rollout_burnin_fn = jax.jit(get_trajectory_fn(inner_fn_burnin, 1, start_with_input = False))
    v_burnin = rollout_burnin_fn(vorticity0)[0]

    inner_fn = get_inner_fn(simulation_exact.step_fn, simulation_exact.dt_fn, t_inner)
    rollout_fn = jax.jit(get_trajectory_fn(inner_fn, outer_steps))
    exact_trajectory = rollout_fn(v_burnin)

    model = get_model()

    ######

    errors_vanleer = onp.zeros((len(nxs), exact_trajectory.shape[0]))
    errors_model = onp.zeros((len(nxs), exact_trajectory.shape[0]))
    
    for i, nx in enumerate(nxs):
        
        sim_params = get_sim_params(nx, nx)
        simulation = get_simulation(sim_params)
        _, params = load_training_params(sim_params, model)
        convert_fn = lambda v: convert_FV_representation(v, sim_params)

        for t, a in enumerate(exact_trajectory):

            
            a_ds = convert_fn(a)
            
            dadt_exact = jax.jit(simulation_exact.F)(a)
            dadt_exact_ds = convert_fn(dadt_exact)

            
            dadt_model = jax.jit(lambda a: simulation.F_params(a, model, params))(a_ds)
            dadt_vanleer = jax.jit(simulation.F)(a_ds)

            errors_vanleer[i, t] = jnp.mean((dadt_vanleer - dadt_exact_ds)**2)
            errors_model[i, t] = jnp.mean((dadt_model - dadt_exact_ds)**2)

    return errors_vanleer, errors_model

errors_vanleer_train, errors_model_train = errors_sim(key_data)
errors_vanleer_eval, errors_model_eval = errors_sim(key_eval)
    
for i, nx in enumerate(nxs):
    plt.plot(errors_vanleer_train[i], color='orange',                      label="Training Data, MUSCL".format(nx))
    plt.plot(errors_model_train[i], color='red',                        label="Training Data, ML".format(nx))
    plt.plot(errors_vanleer_eval[i], color='orange',  linestyle='dotted', label="Evalulation Data, MUSCL".format(nx))
    plt.plot(errors_model_eval[i],  color = 'red', linestyle='dotted', label="Evaluation Data, ML".format(nx))
    plt.title('nx={}'.format(nx))
    plt.yscale('log')
    plt.legend()

    plt.savefig('{}/MSE_{}_{}.png'.format(plot_dir, nx, train_id))
    plt.savefig('{}/MSE_{}_{}.eps'.format(plot_dir, nx, train_id))
    plt.close()

# In[ ]:



def compute_correlation(trajectory_exact, trajectory, sim_params):
    trajectory_exact_ds = jax.vmap(convert_FV_representation, in_axes=(0,None))(trajectory_exact, sim_params)
    nt = trajectory.shape[0]
    M = jnp.concatenate([trajectory.reshape(nt, -1)[:,None,:], trajectory_exact_ds.reshape(nt, -1)[:,None,:]],axis=1)
    return jax.vmap(jnp.corrcoef)(M)[:,0,1]


def correlations(key):
    
    key, subkey = jax.random.split(key)
    

    vorticity0 = init_fn_jax_cfd(subkey, sim_params_exact, 7.0, 2)

    inner_fn_burnin = get_inner_fn(simulation_exact.step_fn, simulation_exact.dt_fn, t_burnin)
    rollout_burnin_fn = jax.jit(get_trajectory_fn(inner_fn_burnin, 1, start_with_input = False))
    v_burnin = rollout_burnin_fn(vorticity0)[0]

    inner_fn = get_inner_fn(simulation_exact.step_fn, simulation_exact.dt_fn, t_inner)
    rollout_fn = jax.jit(get_trajectory_fn(inner_fn, outer_steps))
    exact_trajectory = rollout_fn(v_burnin)
    
    
    model = get_model()
    
    
    corrs_vanleer = onp.zeros((len(nxs), exact_trajectory.shape[0]))
    corrs_model = onp.zeros((len(nxs), exact_trajectory.shape[0]))
    corrs_model_gs = onp.zeros((len(nxs), exact_trajectory.shape[0]))
    corrs_model_ec = onp.zeros((len(nxs), exact_trajectory.shape[0]))
    
    for i, nx in enumerate(nxs):
        
        sim_params = get_sim_params(nx, nx)
        convert_fn = lambda v: convert_FV_representation(v, sim_params)
        v_burnin_ds = convert_fn(v_burnin)
        
        _, params = load_training_params(sim_params, model)
        
        
        # MUSCL
        simulation = get_simulation(sim_params)
        inner_fn = get_inner_fn(simulation.step_fn, simulation.dt_fn, t_inner)
        rollout_fn = jax.jit(get_trajectory_fn(inner_fn, outer_steps))
        traj_vanleer = rollout_fn(v_burnin_ds)
        
        # Model
        simulation = get_simulation(sim_params, model=model, params=params)
        inner_fn = get_inner_fn(simulation.step_fn, simulation.dt_fn, t_inner)
        rollout_fn = jax.jit(get_trajectory_fn(inner_fn, outer_steps))
        traj_model = rollout_fn(v_burnin_ds)
        
        # Model GS
        sim_params_gs = get_sim_params(nx, nx, global_stabilization=True)
        simulation_gs = get_simulation(sim_params_gs, model=model, params=params)
        inner_fn_gs = get_inner_fn(simulation_gs.step_fn, simulation_gs.dt_fn, t_inner)
        rollout_fn_gs = jax.jit(get_trajectory_fn(inner_fn_gs, outer_steps))
        traj_model_gs = rollout_fn_gs(v_burnin_ds)
        
        # Model EC
        sim_params_ec = get_sim_params(nx, nx, global_stabilization=True, energy_conserving=True)
        simulation_ec = get_simulation(sim_params_ec, model=model, params=params)
        inner_fn_ec = get_inner_fn(simulation_ec.step_fn, simulation_ec.dt_fn, t_inner)
        rollout_fn_ec = jax.jit(get_trajectory_fn(inner_fn_ec, outer_steps))
        traj_model_ec = rollout_fn_ec(v_burnin_ds)
        
        
        corrs_vanleer[i] = compute_correlation(exact_trajectory, traj_vanleer, sim_params)
        corrs_model[i] = compute_correlation(exact_trajectory, traj_model, sim_params)
        corrs_model_gs[i] = compute_correlation(exact_trajectory, traj_model_gs, sim_params)
        corrs_model_ec[i] = compute_correlation(exact_trajectory, traj_model_ec, sim_params)
        
        
    return corrs_vanleer, corrs_model, corrs_model_gs, corrs_model_ec
    

corrs_vanleer_train, corrs_model_train, corrs_model_gs_train, corrs_model_ec_train = correlations(key_data)
corrs_vanleer_eval, corrs_model_eval, corrs_model_gs_eval, corrs_model_ec_eval = correlations(key_eval)

Ts = jnp.arange(outer_steps) * t_inner
    
for i, nx in enumerate(nxs):
    plt.plot(Ts, corrs_vanleer_train[i], color='orange', label="Training Data, MUSCL".format(nx))
    plt.plot(Ts, corrs_model_train[i], color='red', label="Training Data, ML".format(nx))
    plt.plot(Ts, corrs_model_gs_train[i], color='blue', label="Training Data, ML GS".format(nx))
    plt.plot(Ts, corrs_model_ec_train[i], color='green',  label="Training Data, ML EC".format(nx))
    
    plt.plot(Ts, corrs_vanleer_eval[i], color='orange', linestyle='dotted',  label="Evaluation Data, MUSCL".format(nx))
    plt.plot(Ts, corrs_model_eval[i],  color = 'red', linestyle='dotted', label="Evaluation Data, ML".format(nx))
    plt.plot(Ts, corrs_model_gs_eval[i],  color = 'blue', linestyle='dotted', label="Evaluation Data, ML GS".format(nx))
    plt.plot(Ts, corrs_model_ec_eval[i],  color = 'green', linestyle='dotted', label="Evaluation Data, ML EC".format(nx))


    plt.title('nx={}'.format(nx))
    plt.legend()

    plt.savefig('{}/corr_{}_{}.png'.format(plot_dir, nx, train_id))
    plt.savefig('{}/corr_{}_{}.eps'.format(plot_dir, nx, train_id))
    plt.close()

# In[ ]:



key, subkey = jax.random.split(key_data)


vorticity0 = init_fn_jax_cfd(subkey, sim_params_exact, 7.0, 2)

inner_fn_burnin = get_inner_fn(simulation_exact.step_fn, simulation_exact.dt_fn, t_burnin)
rollout_burnin_fn = jax.jit(get_trajectory_fn(inner_fn_burnin, 1, start_with_input = False))
v_burnin = rollout_burnin_fn(vorticity0)[0]

inner_fn = get_inner_fn(simulation_exact.step_fn, simulation_exact.dt_fn, t_inner)
rollout_fn = jax.jit(get_trajectory_fn(inner_fn, outer_steps))
exact_trajectory = rollout_fn(v_burnin)



for i, nx in enumerate(nxs):

    sim_params = get_sim_params(nx, nx)
    convert_fn = lambda v: convert_FV_representation(v, sim_params)
    v_burnin_ds = convert_fn(v_burnin)

    _, params = load_training_params(sim_params, model)


    # MUSCL
    simulation = get_simulation(sim_params)
    inner_fn = get_inner_fn(simulation.step_fn, simulation.dt_fn, t_inner)
    rollout_fn = jax.jit(get_trajectory_fn(inner_fn, outer_steps))
    traj_vanleer = rollout_fn(v_burnin_ds)

    # Model
    simulation = get_simulation(sim_params, model=model, params=params)
    inner_fn = get_inner_fn(simulation.step_fn, simulation.dt_fn, t_inner)
    rollout_fn = jax.jit(get_trajectory_fn(inner_fn, outer_steps))
    traj_model = rollout_fn(v_burnin_ds)

    # Model GS
    sim_params_gs = get_sim_params(nx, nx, global_stabilization=True)
    simulation_gs = get_simulation(sim_params_gs, model=model, params=params)
    inner_fn_gs = get_inner_fn(simulation_gs.step_fn, simulation_gs.dt_fn, t_inner)
    rollout_fn_gs = jax.jit(get_trajectory_fn(inner_fn_gs, outer_steps))
    traj_model_gs = rollout_fn_gs(v_burnin_ds)

    # Model EC
    sim_params_ec = get_sim_params(nx, nx, global_stabilization=True, energy_conserving=True)
    simulation_ec = get_simulation(sim_params_ec, model=model, params=params)
    inner_fn_ec = get_inner_fn(simulation_ec.step_fn, simulation_ec.dt_fn, t_inner)
    rollout_fn_ec = jax.jit(get_trajectory_fn(inner_fn_ec, outer_steps))
    traj_model_ec = rollout_fn_ec(v_burnin_ds)

    plot_trajectory_fv(exact_trajectory, sim_params, t_inner)
    plt.savefig('{}/exactrollout_{}_{}.png'.format(plot_dir, nx, train_id))
    plt.savefig('{}/exactrollout_{}_{}.eps'.format(plot_dir, nx, train_id))
    plt.close()
    plot_trajectory_fv(traj_vanleer, sim_params, t_inner)
    plt.savefig('{}/musclrollout_{}_{}.png'.format(plot_dir, nx, train_id))
    plt.savefig('{}/musclrollout_{}_{}.eps'.format(plot_dir, nx, train_id))
    plt.close()
    plot_trajectory_fv(traj_model, sim_params, t_inner)
    plt.savefig('{}/modelrollout_{}_{}.png'.format(plot_dir, nx, train_id))
    plt.savefig('{}/modelrollout_{}_{}.eps'.format(plot_dir, nx, train_id))
    plt.close()
    plot_trajectory_fv(traj_model_gs, sim_params, t_inner)
    plt.savefig('{}/gsrollout_{}_{}.png'.format(plot_dir, nx, train_id))
    plt.savefig('{}/gsrollout_{}_{}.eps'.format(plot_dir, nx, train_id))
    plt.close()
    plot_trajectory_fv(traj_model_ec, sim_params, t_inner)
    plt.savefig('{}/ecrollout_{}_{}.png'.format(plot_dir, nx, train_id))
    plt.savefig('{}/ecrollout_{}_{}.eps'.format(plot_dir, nx, train_id))
    plt.close()
