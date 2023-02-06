import jax
from jax import device_put
import h5py
import jax.numpy as jnp
import numpy as onp
import optax
from flax import serialization
from flax.training import train_state  # Useful dataclass to keep train state
from optax import polynomial_schedule

from helper import convert_FV_representation
from trajectory import get_inner_fn, get_trajectory_fn
from initialconditions import get_initial_condition_fn, get_a0
from model import stencil_delta_flux_FV_1D_burgers, stencil_flux_FV_1D_burgers
from lossfunctions import mse_loss_FV
from simulations import BurgersFVSim
from timederivative import time_derivative_FV_1D_burgers_train

def create_training_data(sim_params, core_params, nxs, N, delta=True):
	for nx in nxs:
		f = h5py.File(
			"{}/data/traindata/{}_nx{}.hdf5".format(sim_params.readwritedir, sim_params.name, nx),
			"w",
		)
		if delta:
			word = "delta_dadt"
		else:
			word = "dadt"
		f.create_dataset("a", (N, nx), dtype="float64")
		f.create_dataset("t", (N,), dtype="float64")
		f.create_dataset("key", (N, 2), dtype="uint32")
		f.create_dataset(word, (N, nx), dtype="float64")
		f.close()


def write_trajectory(sim_params, core_params, nx, a_trajectory, t_trajectory, dadt_trajectory, forcing_key, n, outer_steps, delta=True):
	f = h5py.File(
		"{}/data/traindata/{}_nx{}.hdf5".format(sim_params.readwritedir, sim_params.name, nx),
		"r+",
	)
	j_begin = n * outer_steps
	j_end = (n+1) * outer_steps
	if delta:
		word = "delta_dadt"
	else:
		word = "dadt"
	f["a"][j_begin:j_end] = a_trajectory
	f["t"][j_begin:j_end] = t_trajectory
	f["key"][j_begin:j_end] = forcing_key[None]
	f[word][j_begin:j_end] = dadt_trajectory
	f.close()


def save_training_data(key, init_fn, forcing_fn, core_params, sim_params, sim, t_inner, outer_steps, n_runs, nx_exact, nxs, delta=True, **kwargs):


	# initialize files for saving training data
	create_training_data(sim_params, core_params, nxs, outer_steps * n_runs, delta=delta)

	for n in range(n_runs):
		print(n)
		key, initkey, forcing_key = jax.random.split(key, 3)
		# init solution
		f_init = init_fn(initkey)
		f_forcing = forcing_fn(forcing_key)
		a0 = get_a0(f_init, core_params, nx_exact)
		t0 = 0.0
		x0 = (a0, t0)

		step_fn = lambda a, t, dt: sim.step_fn(a, t, dt, forcing_func = f_forcing)
		inner_fn = get_inner_fn(step_fn, sim.dt_fn, t_inner)
		rollout_fn = jax.jit(get_trajectory_fn(inner_fn, outer_steps, start_with_input=True))

		# get exact trajectory
		trajectory, trajectory_t = rollout_fn(x0)
		# get dadt along trajectory
		time_derivative_fn = jax.vmap(jax.jit(lambda a, t: sim.F(a, t, forcing_func = f_forcing)))
		dadt_trajectory = time_derivative_fn(trajectory, trajectory_t)

		for j, nx in enumerate(nxs):
			convert_fn = jax.jit(jax.vmap(lambda a: convert_FV_representation(a, nx, core_params.Lx)))
			# downsample exact trajectory to nx
			trajectory_ds = convert_fn(trajectory)
			# get exact dadt
			dadt_trajectory_exact_ds = convert_fn(dadt_trajectory)
			# get downsampled dadt
			dadt_trajectory_ds = time_derivative_fn(trajectory_ds, trajectory_t)
			# get delta(dadt), which is the correction ML will have to learn
			if delta:
				dadt_to_store = dadt_trajectory_exact_ds - dadt_trajectory_ds # delta_dadt_trajectory
			else:
				dadt_to_store = dadt_trajectory_exact_ds

			write_trajectory(sim_params, core_params, nx, trajectory_ds, trajectory_t, dadt_to_store, forcing_key, n, outer_steps, delta=delta)



def save_training_params(nx, sim_params, training_params, params, losses):
    bytes_output = serialization.to_bytes(params)
    with open("{}/data/params/{}_nx{}_params".format(sim_params.readwritedir, training_params.train_id, nx), "wb") as f:
        f.write(bytes_output)
    with open(
        "{}/data/params/{}_nx{}_losses.npy".format(sim_params.readwritedir, training_params.train_id, nx), "wb"
    ) as f:
        onp.save(f, losses)


def load_training_params(nx, sim_params, training_params, model):
    losses = onp.load("{}/data/params/{}_nx{}_losses.npy".format(sim_params.readwritedir, training_params.train_id, nx))
    with open(
        "{}/data/params/{}_nx{}_params".format(sim_params.readwritedir, training_params.train_id, nx), "rb"
    ) as f:
        param_bytes = f.read()

    params = serialization.from_bytes(
        init_params(jax.random.PRNGKey(0), model), param_bytes
    )
    return losses, params


def init_params(key, model):
	NX_NO_MEANING = 128  # params doesn't depend on this
	zeros = jnp.zeros((NX_NO_MEANING))
	return model.init(
		key, zeros
	)

def create_train_state(model, params, training_params):
	schedule_fn = polynomial_schedule(
		init_value=training_params.learning_rate,
		end_value=training_params.learning_rate / 10,
		power=1,
		transition_steps=1,
		transition_begin=training_params.num_training_iterations // 2,
	)
	if training_params.optimizer == "adam":
		tx = optax.chain(optax.adam(schedule_fn), optax.zero_nans())
	elif training_params.optimizer == "sgd":
		tx = optax.chain(optax.sgd(schedule_fn), optax.zero_nans())
	else:
		raise ValueError("Incorrect Optimizer")
	return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)



def get_batch_fn(sim_params, training_params, nx, delta=True):
	f = h5py.File(
		"{}/data/traindata/{}_nx{}.hdf5".format(sim_params.readwritedir, sim_params.name, nx),
		"r",
	)
	if delta:
		word = "delta_dadt"
	else:
		word = "dadt"

	trajectory_a = device_put(jnp.asarray(f["a"][:training_params.n_data]), jax.devices()[0])
	trajectory_t = device_put(jnp.asarray(f["t"][:training_params.n_data]), jax.devices()[0])
	trajectory_key = device_put(jnp.asarray(f["key"][:training_params.n_data]), jax.devices()[0])
	dadt = device_put(jnp.asarray(f[word][:training_params.n_data]), jax.devices()[0])
	f.close()

	def batch_fn(idxs):
		return {"a": trajectory_a[idxs], "t": trajectory_t[idxs], "key": trajectory_key[idxs], word: dadt[idxs]}

	return batch_fn


def get_loss_fn(model, core_params, forcing_fn, delta=True):
	if delta:
		def delta_dadt_fn(a, params):
			nx = a.shape[0]
			dx = core_params.Lx / nx
			delta_flux_R = stencil_delta_flux_FV_1D_burgers(a, model, params)
			delta_flux_L = jnp.roll(delta_flux_R, 1, axis=0)
			return (delta_flux_L - delta_flux_R) / dx

		batch_delta_dadt_fn = jax.vmap(delta_dadt_fn, in_axes=(0, None), out_axes=0)

		@jax.jit
		def loss_fn(params, batch):
			delta_dadt = batch_delta_dadt_fn(batch["a"], params)
			return mse_loss_FV(delta_dadt, batch["delta_dadt"])
		
	else:
		
		f_dadt = time_derivative_FV_1D_burgers_train(core_params, model=model, delta=delta)
			
		def dadt_fn(a, t, forcing_key, params):
			f_forcing = forcing_fn(forcing_key)
			return f_dadt(a, t, params=params, forcing_func = f_forcing)

		batch_dadt_fn = jax.vmap(dadt_fn, in_axes=(0, 0, 0, None), out_axes=0)

		@jax.jit
		def loss_fn(params, batch):
			dadt = batch_dadt_fn(batch["a"], batch["t"], batch["key"], params)
			return mse_loss_FV(dadt, batch["dadt"])	

	return loss_fn


def get_idx_gen(key, training_params):
	possible_idxs = jnp.arange(training_params.n_data)
	shuffle_idxs = jax.random.permutation(key, possible_idxs)

	counter = 0
	while counter + training_params.batch_size <= training_params.n_data:
		yield jnp.sort(shuffle_idxs[counter : counter + training_params.batch_size])
		counter += training_params.batch_size


def train_model(model, params, training_params, key, idx_fn, batch_fn, loss_fn, **kwargs):
	"""
	idx_fn: lambda subkey -> idx_gen
	batch_fn: lambda idxs -> batch
	loss_fn: lambda params, batch -> loss
	"""
	state = create_train_state(model, params, training_params)
	grad_fn = jax.value_and_grad(loss_fn)

	@jax.jit
	def train_step(state, batch):
		loss, grads = grad_fn(state.params, batch, **kwargs)
		state = state.apply_gradients(grads=grads)
		return state, loss

	def batch_step(state, key):
		losses = []
		idx_gen = idx_fn(key)
		for idxs in idx_gen:
			batch = batch_fn(idxs)
			state, loss = train_step(state, batch)
			losses.append(loss)
		return state, losses
	
	n_loss_per_batch = 0
	for idxs in idx_fn(jax.random.PRNGKey(0)):
		n_loss_per_batch += 1
	num_losses = n_loss_per_batch * training_params.num_epochs
	losses_all = onp.zeros(num_losses)

	for n in range(training_params.num_epochs):
		print(n)
		key, _ = jax.random.split(key)
		state, losses = batch_step(state, key)
		losses_all[n * n_loss_per_batch : (n+1) * n_loss_per_batch] = onp.asarray(losses)

	return jnp.asarray(losses_all), state.params


def compute_losses_no_model(key, idx_fn, batch_fn, loss_fn):
	"""
	idx_fn: lambda subkey -> idx_gen
	batch_fn: lambda idxs -> batch
	loss_fn: lambda batch -> loss
	"""

	@jax.jit
	def train_step(batch):
		return loss_fn(None, batch)

	losses = []
	for _ in range(1):
		key, subkey = jax.random.split(key)
		idxs = idx_fn(subkey)
		batch = batch_fn(idxs)
		loss = train_step(batch)
		losses.append(loss)

	return losses

