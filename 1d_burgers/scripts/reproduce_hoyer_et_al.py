# setup paths
import sys
base = '/Users/nickm/thesis'
basereadwrite = '/Users/nickm/thesis'
basedir = '{}/icml2023paper/1d_burgers'.format(base)
readwritedir = '{}/icml2023paper/1d_burgers'.format(basereadwrite)

sys.path.append('{}/core'.format(basedir))
sys.path.append('{}/simulate'.format(basedir))
sys.path.append('{}/ml'.format(basedir))


# In[ ]:


# import external packages
import jax
import jax.numpy as jnp
from jax import config, vmap
config.update("jax_enable_x64", True)


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


# In[ ]:


# training hyperparameters
init_description = 'zeros'
fv_flux_baseline = 'weno' # learning a correction to the weno scheme
omega_max = 0.4
kwargs_init = {'min_num_modes': 2, 'max_num_modes': 6, 'min_k': 0, 'max_k': 3, 'amplitude_max': 1.0}
kwargs_core = {'Lx': 2 * jnp.pi, 'flux': fv_flux_baseline, 'nu': 0.01}
kwargs_forcing = {'min_num_modes': 20, 'max_num_modes': 20, 'min_k': 3, 'max_k': 6, 'amplitude_max': 0.5, 'omega_max': omega_max}
kwargs_sim = {'name' : "burgers_test1", 'cfl_safety' : 0.3, 'rk' : 'ssp_rk3'}
kwargs_train_FV = {'train_id': "burgers_test1", 'batch_size' : 128, 'optimizer': 'adam', 'num_epochs' : 100}
kwargs_stencil = {'kernel_size' : 5, 'kernel_out' : 4, 'stencil_width' 6: , 'depth' : 3, 'width' : 32}
n_runs = 800
t_inner_train = 0.1
Tf = 1.0
outer_steps_train = int(Tf/t_inner_train)
nx_exact = 512
nxs = [16, 32, 64, 128, 256] # [8, 16, 32, 64]
learning_rate_list = [3e-3,3e-3, 3e-3, 3e-3, 3e-3] #[1e-2, 1e-2, 1e-4, 1e-5]
assert len(nxs) == len(learning_rate_list)
key = jax.random.PRNGKey(12)

delta = True

# setup
core_params = get_core_params(**kwargs_core)
sim_params = get_sim_params(**kwargs_sim)
n_data = n_runs * outer_steps_train
training_params_list = [get_training_params(n_data, **kwargs_train_FV, learning_rate = lr) for lr in learning_rate_list]
stencil_params = get_stencil_params(**kwargs_stencil)
sim = BurgersFVSim(core_params, sim_params, delta=delta, omega_max = omega_max)
init_fn = lambda key: get_initial_condition_fn(core_params, init_description, key=key, **kwargs_init)
forcing_fn = forcing_func_sum_of_modes(core_params.Lx, **kwargs_forcing)
model = get_model(core_params, stencil_params, delta=delta)


# In[ ]:


save_training_data(key, init_fn, forcing_fn, core_params, sim_params, sim, t_inner_train, outer_steps_train, n_runs, nx_exact, nxs, delta=delta)


# In[ ]:


key = jax.random.PRNGKey(43)
i_params = init_params(key, model)


# In[ ]:


for i, nx in enumerate(nxs):
    print(nx)
    training_params = training_params_list[i]
    idx_fn = lambda key: get_idx_gen(key, training_params)
    batch_fn = get_batch_fn(core_params, sim_params, training_params, nx, delta=delta)
    loss_fn = get_loss_fn(model, core_params, delta=delta)
    losses, params = train_model(model, i_params, training_params, key, idx_fn, batch_fn, loss_fn)
    save_training_params(nx, sim_params, training_params, params, losses)