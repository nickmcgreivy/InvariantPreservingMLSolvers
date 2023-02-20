from model import LearnedFlux2D
from flax.training import train_state  # Useful dataclass to keep train state
import optax
import jax
import jax.numpy as jnp
from jax import device_put
import numpy as onp
from optax import polynomial_schedule
import h5py
from flax import serialization
from helper import convert_FV_representation
from trajectory import get_trajectory_fn, get_inner_fn
from initialconditions import init_fn_jax_cfd
from lossfunctions import MSE_loss
from model import output_flux


def init_params(key, model):
    NX_NO_MEANING = 128  # params doesn't depend on this
    NY_NO_MEANING = 128
    zeros = jnp.zeros((NX_NO_MEANING, NY_NO_MEANING))
    return model.init(
        key, zeros, zeros, zeros
    )

def create_train_state(model, params, optimizer="adam"):
    if optimizer == "adam":
        tx = optax.adam(model.ml_params.learning_rate)
    elif optimizer == "sgd":
        tx = optax.sgd(model.ml_params.learning_rate)
    else:
        raise ValueError("Incorrect Optimizer")
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def create_training_data(sim_params_ds, N):
    for sim_params in sim_params_ds:
        nx = sim_params.nx
        ny = sim_params.ny
        f = h5py.File(
            "{}/data/traindata/{}_nx{}_ny{}.hdf5".format(sim_params.readwritedir, sim_params.name, nx, ny),
            "w",
        )
        f.create_dataset("vorticity", (N, nx, ny), dtype="float64")
        f.create_dataset("dadt_diff", (N, nx, ny), dtype="float64")
        f.create_dataset("alpha_R", (N, nx, ny), dtype="float64")
        f.create_dataset("alpha_T", (N, nx, ny), dtype="float64")
        f.close()

def write_trajectory(sim_params, vorticity, dadt_diff, alpha_R, alpha_T, n, outer_steps):
    nx = sim_params.nx
    ny = sim_params.ny
    f = h5py.File(
        "{}/data/traindata/{}_nx{}_ny{}.hdf5".format(sim_params.readwritedir, sim_params.name, nx, ny),
        "r+",
    )
    j_begin = n * outer_steps
    j_end = (n+1) * outer_steps
    f["vorticity"][j_begin:j_end] = vorticity
    f["dadt_diff"][j_begin:j_end] = dadt_diff
    f["alpha_R"][j_begin:j_end] = alpha_R
    f["alpha_T"][j_begin:j_end] = alpha_T
    f.close()

def save_training_data(key, sim_params_exact, simulation_exact, t_burn_in, t_inner, outer_steps, n_runs, sim_params_ds, simulation_ds, max_velocity=7.0, ic_wavenumber=2):
    inner_fn = get_inner_fn(simulation_exact.step_fn, simulation_exact.dt_fn, t_inner)
    inner_fn_burnin = get_inner_fn(simulation_exact.step_fn, simulation_exact.dt_fn, t_burn_in)
    rollout_burnin_fn = jax.jit(get_trajectory_fn(inner_fn_burnin, 1, start_with_input = False))
    rollout_fn = jax.jit(get_trajectory_fn(inner_fn, outer_steps))
    time_derivative_fn = jax.vmap(jax.jit(simulation_exact.F))

    create_training_data(sim_params_ds, outer_steps * n_runs)

    for n in range(n_runs):
        print(n)
        key, subkey = jax.random.split(key)
        v0 = init_fn_jax_cfd(subkey, sim_params_exact, max_velocity, ic_wavenumber)
        v_burnin = rollout_burnin_fn(v0)[0]
        vorticity_trajectory = rollout_fn(v_burnin)
        dadt_trajectory = time_derivative_fn(vorticity_trajectory)

        for j, sim_params in enumerate(sim_params_ds):
            convert_fn = jax.jit(jax.vmap(lambda v: convert_FV_representation(v, sim_params)))
            time_derivative_fn_ds = jax.vmap(jax.jit(simulation_ds[j].F))
            alpha_fn_ds = jax.vmap(jax.jit(simulation_ds[j].alpha_fn))
            vorticity_trajectory_ds = convert_fn(vorticity_trajectory)
            dadt_trajectory_exact_ds = convert_fn(dadt_trajectory)
            dadt_trajectory_ds = time_derivative_fn_ds(vorticity_trajectory_ds)
            dadt_diff = dadt_trajectory_exact_ds - dadt_trajectory_ds
            alpha_R_trajectory_ds, alpha_T_trajectory_ds = alpha_fn_ds(vorticity_trajectory_ds)
            write_trajectory(sim_params, vorticity_trajectory_ds, dadt_diff, alpha_R_trajectory_ds, alpha_T_trajectory_ds, n, outer_steps)


def save_training_params(sim_params, params, losses):
    nx = sim_params.nx
    ny = sim_params.ny
    bytes_output = serialization.to_bytes(params)
    with open("{}/data/params/{}_nx{}_ny{}_params".format(sim_params.readwritedir, sim_params.name, nx, ny), "wb") as f:
        f.write(bytes_output)
    with open(
        "{}/data/params/{}_nx{}_ny{}_losses.npy".format(sim_params.readwritedir, sim_params.name, nx, ny), "wb"
    ) as f:
        onp.save(f, losses)


def load_training_params(sim_params, model):
    nx = sim_params.nx
    ny = sim_params.ny
    losses = onp.load("{}/data/params/{}_nx{}_ny{}_losses.npy".format(sim_params.readwritedir, sim_params.name, nx, ny))
    with open(
        "{}/data/params/{}_nx{}_ny{}_params".format(sim_params.readwritedir, sim_params.name, nx, ny), "rb"
    ) as f:
        param_bytes = f.read()

    params = serialization.from_bytes(
        init_params(jax.random.PRNGKey(0), model), param_bytes
    )
    return losses, params


def get_idx_gen(key, ml_params, n_data):
    possible_idxs = jnp.arange(n_data)
    shuffle_idxs = jax.random.permutation(key, possible_idxs)

    counter = 0
    while counter + ml_params.batch_size <= n_data:
        yield jnp.sort(shuffle_idxs[counter : counter + ml_params.batch_size])
        counter += ml_params.batch_size

def get_batch_fn(sim_params, n_data):
    nx = sim_params.nx
    ny = sim_params.ny
    f = h5py.File(
        "{}/data/traindata/{}_nx{}_ny{}.hdf5".format(sim_params.readwritedir, sim_params.name, nx, ny),
        "r",
    )
    vorticity = device_put(jnp.asarray(f["vorticity"][:n_data]), jax.devices()[0])
    alpha_R = device_put(jnp.asarray(f["alpha_R"][:n_data]), jax.devices()[0])
    alpha_T = device_put(jnp.asarray(f["alpha_T"][:n_data]), jax.devices()[0])
    dadt_diff = device_put(jnp.asarray(f["dadt_diff"][:n_data]), jax.devices()[0])
    f.close()

    @jax.jit
    def batch_fn(idxs):
        return {"vorticity": vorticity[idxs], "alpha_R": alpha_R[idxs], "alpha_T": alpha_T[idxs], "dadt_diff": dadt_diff[idxs]}

    return batch_fn


def get_loss_fn(model, sim_params):

    denominator = sim_params.dx * sim_params.dy

    def delta_dadt_fn(zeta, alpha_R, alpha_T, params):
        #flux_R, flux_T = stencil_flux(zeta, alpha_R, alpha_T, model, params)
        flux_R, flux_T = output_flux(zeta, alpha_R, alpha_T, model, params)
        flux_L = jnp.roll(flux_R, 1, axis=0)
        flux_B = jnp.roll(flux_T, 1, axis=1)

        return (flux_L + flux_B - flux_R - flux_T) / denominator

    batch_delta_dadt_fn = jax.vmap(delta_dadt_fn, in_axes=(0, 0, 0, None), out_axes=0)


    if model is not None:

        @jax.jit
        def loss_fn(params, batch):
            dadt_diff = batch["dadt_diff"]
            delta_dadt = batch_delta_dadt_fn(batch["vorticity"], batch["alpha_R"], batch["alpha_T"], params)
            return MSE_loss(dadt_diff, delta_dadt)

    else:

        @jax.jit
        def loss_fn(params, batch):
            dadt_diff = batch["dadt_diff"]
            return MSE_loss(dadt_diff, jnp.zeros(dadt_diff.shape))


    return loss_fn


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


def train_model(model, params, key, idx_fn, batch_fn, loss_fn):
    """
    idx_fn: lambda subkey -> idx_gen
    batch_fn: lambda idxs -> batch
    loss_fn: lambda params, batch -> loss
    """
    state = create_train_state(model, params)
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
    num_losses = n_loss_per_batch * model.ml_params.num_epochs
    losses_all = onp.zeros(num_losses)

    for n in range(model.ml_params.num_epochs):
        print(n)
        key, _ = jax.random.split(key)
        state, losses = batch_step(state, key)
        losses_all[n * n_loss_per_batch : (n+1) * n_loss_per_batch] = onp.asarray(losses)

    return jnp.asarray(losses_all), state.params
