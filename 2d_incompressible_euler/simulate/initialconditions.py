import math
import jax.numpy as jnp
import jax
from jax import config
import jax_cfd.base as jax_cfd
import jax_cfd.base.grids as grids

config.update("jax_enable_x64", True)


def init_sum_sines(key):
    Lx = 2 * PI
    Ly = 2 * PI

    max_k = 5
    min_k = 1
    num_init_modes = 6
    amplitude_max = 4.0

    def sum_modes(x, y, amplitudes, ks_x, ks_y, phases_x, phases_y):
        return jnp.sum(
            amplitudes[None, :]
            * jnp.sin(ks_x[None, :] * 2 * PI / Lx * x[:, None] + phases_x[None, :])
            * jnp.sin(ks_y[None, :] * 2 * PI / Ly * y[:, None] + phases_y[None, :]),
            axis=1,
        )

    key1, key2, key3, key4, key5 = jax.random.split(key, 5)
    phases_x = jax.random.uniform(key1, (num_init_modes,)) * 2 * PI
    phases_y = jax.random.uniform(key2, (num_init_modes,)) * 2 * PI
    ks_x = jax.random.randint(key3, (num_init_modes,), min_k, max_k)
    ks_y = jax.random.randint(key4, (num_init_modes,), min_k, max_k)
    amplitudes = jax.random.uniform(key5, (num_init_modes,)) * amplitude_max
    return lambda x, y, t: sum_modes(x, y, amplitudes, ks_x, ks_y, phases_x, phases_y)


def get_u0_jax_cfd(key, sim_params, max_velocity, ic_wavenumber):
    grid = grids.Grid(
        (sim_params.nx, sim_params.ny), domain=((0, sim_params.Lx), (0, sim_params.Ly))
    )
    return jax_cfd.initial_conditions.filtered_velocity_field(
        key, grid, max_velocity, ic_wavenumber
    )


def vorticity(u):
    return jax_cfd.finite_differences.curl_2d(u).data


def init_fn_jax_cfd(*args):
    u0 = get_u0_jax_cfd(*args)
    return vorticity(u0)
