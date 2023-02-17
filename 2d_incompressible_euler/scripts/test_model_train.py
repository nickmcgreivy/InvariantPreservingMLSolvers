#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('/Users/nickm/thesis/InvariantPreservingMLSolvers/2d_incompressible_euler/ml')
sys.path.append('/Users/nickm/thesis/InvariantPreservingMLSolvers/2d_incompressible_euler/baselines')
sys.path.append('/Users/nickm/thesis/InvariantPreservingMLSolvers/2d_incompressible_euler/simulate')


import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import xarray
import seaborn as sns
import matplotlib.pyplot as plt

from initialconditions import init_fn_FNO, init_fn_jax_cfd
from simulations import KolmogorovFiniteVolumeSimulation
from simparams import FiniteVolumeSimulationParams

from helper import convert_FV_representation
from trajectory import get_trajectory_fn
from flux import Flux

from model import LearnedFlux2D
from mlparams import ModelParams, TrainingParams
from trainingutils import init_params, save_training_data
from trainingutils import get_loss_fn, get_batch_fn, get_idx_gen, train_model, compute_losses_no_model

def plot_fv(zeta, sim_params): 
    nx = zeta.shape[0]
    spatial_coord = jnp.arange(nx) * sim_params.Lx / nx # same for x and y
    coords = {
      'x': spatial_coord,
      'y': spatial_coord,
    }
    xarray.DataArray(zeta, dims=["x", "y"], coords=coords).plot.imshow(cmap=sns.cm.icefire, robust=True)

def plot_trajectory_fv(trajectory, sim_params):
    nx = trajectory.shape[1]
    spatial_coord = jnp.arange(nx) * sim_params.Lx / sim_params.nx # same for x and y
    coords = {
      'x': spatial_coord,
      'y': spatial_coord,
        'time': sim_params.dt * sim_params.inner_steps * jnp.arange(outer_steps)
    }
    xarray.DataArray(trajectory, dims=["time", "x", "y"], coords=coords).plot.imshow(
        col='time', col_wrap=5, 
        cmap=sns.cm.icefire, robust=True)

def get_sim_params(nx, ny):
    name = "test"
    basedir = "/Users/nickm/thesis/InvariantPreservingMLSolvers/2d_incompressible_euler"
    readwritedir = "/Users/nickm/thesis/InvariantPreservingMLSolvers/2d_incompressible_euler"
    Lx = Ly = 2 * jnp.pi
    cfl_safety=0.3
    rk='ssp_rk3'
    flux=Flux.VANLEER
    global_stabilization=False
    return FiniteVolumeSimulationParams(name, basedir, readwritedir, nx, ny, Lx, Ly, cfl_safety, rk, flux, global_stabilization)

def get_simulation(sim_params, model=None, params=None):
    viscosity=1/1000
    forcing_coeff=1.0
    drag=0.1
    return KolmogorovFiniteVolumeSimulation(sim_params, viscosity, forcing_coeff, drag, model=model, params=params)

def get_trajectory(sim_params, v0):
    v_init = convert_FV_representation(v0, sim_params)
    sim = get_simulation(sim_params)
    rollout_fn = get_trajectory_fn(sim.step_fn, outer_steps)
    return rollout_fn(v_init)

def get_ml_params():
    unique_id = "test"
    batch_size=4
    learning_rate=1e-3
    num_epochs = 10
    kernel_size = 5
    depth = 4
    width = 16
    return ModelParams(unique_id, batch_size, learning_rate, num_epochs, kernel_size, depth, width)

def get_model():
    model_params = get_ml_params()
    return LearnedFlux2D(model_params)



model = get_model()
params = init_params(jax.random.PRNGKey(0), model)


nx_exact = ny_exact = 128
sim_params_exact = get_sim_params(nx_exact, ny_exact)
outer_steps = 10
n_runs = 5
t_inner = 0.1
nxs = [32, 64]
nxs_all = [32, 64]
sim_params_ds = []
simulations_ds = []
for nx in nxs:
    sim_params = get_sim_params(nx, nx)
    sim_params_ds.append(sim_params)
    simulations_ds.append(get_simulation(sim_params))
sim_params_ds_all = []
for nx in nxs_all:
    sim_params_ds_all.append(get_sim_params(nx, nx))
key_data = jax.random.PRNGKey(0)



simulation_exact = get_simulation(sim_params_exact)
save_training_data(key_data, sim_params_exact, simulation_exact, t_inner, outer_steps, n_runs, sim_params_ds, simulations_ds, max_velocity=7.0, ic_wavenumber=2)


key = jax.random.PRNGKey(42)
n_data = n_runs * outer_steps
ml_params = get_ml_params()
model = get_model()

i_params = init_params(key, model)

params_list = []


for sim_params in sim_params_ds:
    idx_fn = lambda key: get_idx_gen(key, ml_params, n_data)
    batch_fn = get_batch_fn(sim_params, n_data)
    loss_fn = get_loss_fn(model, sim_params)
    losses, params = train_model(model, i_params, key, idx_fn, batch_fn, loss_fn)
    plt.plot(losses)

    save_training_params(sim_params, params, losses)
