import jax
from jax import device_put
import h5py
import jax.numpy as jnp
import numpy as onp
import optax
from flax import serialization
from flax.training import train_state  # Useful dataclass to keep train state
from optax import polynomial_schedule

from helper import convert_FV_representation, convert_DG_representation
from trajectory import get_inner_fn, get_trajectory_fn
from initialconditions import get_initial_condition_fn, get_a0
from model import stencil_flux_FV_1D_advection, stencil_flux_DG_1D_advection
from lossfunctions import mse_loss_DG, mse_loss_FV

def create_training_data(sim_params, core_params, nxs, N):
	for nx in nxs:
		if core_params.order is None:
			f = h5py.File(
				"{}/data/traindata/{}_nx{}.hdf5".format(sim_params.readwritedir, sim_params.name, nx),
				"w",
			)
			f.create_dataset("a", (N, nx), dtype="float64")
			f.create_dataset("delta_dadt", (N, nx), dtype="float64")
			f.close()
		else:
			f = h5py.File(
				"{}/data/traindata/{}_order{}_nx{}.hdf5".format(sim_params.readwritedir, sim_params.name, core_params.order, nx),
				"w",
			)
			f.create_dataset("a", (N, nx, core_params.order + 1), dtype="float64")
			f.create_dataset("delta_dadt", (N, nx, core_params.order + 1), dtype="float64")
			f.close()

def write_trajectory(sim_params, core_params, nx, trajectory, delta_dadt_trajectory, n, outer_steps):
	if core_params.order is None:
		f = h5py.File(
			"{}/data/traindata/{}_nx{}.hdf5".format(sim_params.readwritedir, sim_params.name, nx),
			"r+",
		)
	else:
		f = h5py.File(
			"{}/data/traindata/{}_order{}_nx{}.hdf5".format(sim_params.readwritedir, sim_params.name, core_params.order, nx),
			"r+",
		)
	j_begin = n * outer_steps
	j_end = (n+1) * outer_steps
	f["a"][j_begin:j_end] = trajectory
	f["delta_dadt"][j_begin:j_end] = delta_dadt_trajectory
	f.close()


def save_training_data(key, init_fn, core_params, sim_params, sim, t_inner, outer_steps, n_runs, nx_exact, nxs, **kwargs):
	"""
	One peculiarity of this function is that it isn't written to have order_exact = 2 and then downsample to order = 1
	"""

	inner_fn = get_inner_fn(sim.step_fn, sim.dt_fn, t_inner)
	rollout_fn = jax.jit(get_trajectory_fn(inner_fn, outer_steps))
	time_derivative_fn = jax.vmap(jax.jit(sim.F))

	# initialize files for saving training data
	create_training_data(sim_params, core_params, nxs, outer_steps * n_runs)

	for n in range(n_runs):
		print(n)
		key, subkey = jax.random.split(key)
		# init solution
		f_init = init_fn(subkey)
		a0 = get_a0(f_init, core_params, nx_exact)
		# get exact trajectory
		trajectory = rollout_fn(a0)
		# get dadt along trajectory
		dadt_trajectory = time_derivative_fn(trajectory)

		for j, nx in enumerate(nxs):
			if core_params.order is None:
				convert_fn = jax.jit(jax.vmap(lambda a: convert_FV_representation(a, nx, core_params.Lx)))
			else:
				convert_fn = jax.jit(jax.vmap(lambda a: convert_DG_representation(a, core_params.order + 1, nx, core_params.Lx)))
			time_derivative_fn_ds = jax.jit(jax.vmap(sim.F))

			# downsample exact trajectory to nx
			trajectory_ds = convert_fn(trajectory)
			# get exact dadt
			dadt_trajectory_exact_ds = convert_fn(dadt_trajectory)
			# get downsampled dadt
			dadt_trajectory_ds = time_derivative_fn_ds(trajectory_ds)
			# get delta(dadt), which is the correction ML will have to learn
			delta_dadt_trajectory = dadt_trajectory_exact_ds - dadt_trajectory_ds

			write_trajectory(sim_params, core_params, nx, trajectory_ds, delta_dadt_trajectory, n, outer_steps)




def init_params(key, model):
	NX_NO_MEANING = 128  # params doesn't depend on this
	zeros = jnp.zeros((NX_NO_MEANING, model.p))
	return model.init(
		key, zeros
	)

def create_train_state(model, params, training_params):
	schedule_fn = polynomial_schedule(
		init_value=training_params.learning_rate,
		end_value=training_params.learning_rate / 10,
		power=1,
		transition_steps=training_params.num_training_iterations // 2,
		transition_begin=training_params.num_training_iterations // 4,
	)
	if training_params.optimizer == "adam":
		tx = optax.chain(optax.adam(schedule_fn), optax.zero_nans())
	elif training_params.optimizer == "sgd":
		tx = optax.chain(optax.sgd(schedule_fn), optax.zero_nans())
	else:
		raise ValueError("Incorrect Optimizer")
	return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)



def get_batch_fn(core_params, sim_params, training_params, nx):
	if core_params.order is None:
		f = h5py.File(
			"{}/data/traindata/{}_nx{}.hdf5".format(sim_params.readwritedir, sim_params.name, nx),
			"r",
		)
	else:
		f = h5py.File(
			"{}/data/traindata/{}_order{}_nx{}.hdf5".format(sim_params.readwritedir, sim_params.name, core_params.order, nx),
			"r",
		)
	
	trajectory = device_put(jnp.asarray(f["a"][:training_params.n_data]), jax.devices()[0])
	delta_dadt = device_put(jnp.asarray(f["delta_dadt"][:training_params.n_data]), jax.devices()[0])
	f.close()

	def batch_fn(idxs):
		return {"a": trajectory[idxs], "delta_dadt": delta_dadt[idxs]}

	return batch_fn


def get_loss_fn(model, core_params):    
	c = 1.0

	def delta_dadt_fn(a, params):
		nx = a.shape[0]
		dx = core_params.Lx / nx
		C = c / dx
		if core_params.order is None:
			flux_R = stencil_flux_FV_1D_advection(a, model, params)
			flux_L = jnp.roll(flux_R, 1, axis=0)
			return (flux_L - flux_R) * C
		else:
			twokplusone = 2 * jnp.arange(0, core_params.order+1) + 1
			negonetok = (jnp.ones(core_params.order+1) * -1) ** jnp.arange(core_params.order+1)
			flux_R = stencil_flux_DG_1D_advection(a, model, params)
			flux_L = jnp.roll(flux_R, 1, axis=0)
			return (negonetok[None, :] * flux_L[:, None] - flux_R[:, None]) * C * twokplusone[None, :] 

	batch_delta_dadt_fn = jax.vmap(delta_dadt_fn, in_axes=(0, None), out_axes=0)

	@jax.jit
	def loss_fn(params, batch):
		delta_dadt = batch_delta_dadt_fn(batch["a"], params)
		if core_params.order is None:
			return mse_loss_FV(delta_dadt, batch["delta_dadt"])
		else:
			return mse_loss_DG(delta_dadt, batch["delta_dadt"])
		
	return loss_fn


def get_idx_gen(key, training_params):
	possible_idxs = jnp.arange(training_params.n_data)
	shuffle_idxs = jax.random.permutation(key, possible_idxs)

	counter = 0
	while counter + training_params.batch_size <= training_params.n_data:
		yield jnp.sort(shuffle_idxs[counter : counter + training_params.batch_size])
		counter += training_params.batch_size








def train_model(model, params, training_params, key, idx_fn, batch_fn, loss_fn):
	"""
	idx_fn: lambda subkey -> idx_gen
	batch_fn: lambda idxs -> batch
	loss_fn: lambda params, batch -> loss
	"""
	state = create_train_state(model, params, training_params)
	grad_fn = jax.value_and_grad(loss_fn)

	@jax.jit
	def train_step(state, batch):
		loss, grads = grad_fn(state.params, batch)
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

