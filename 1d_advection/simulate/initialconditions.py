import jax.numpy as jnp
import jax
from jax import random

from legendre import generate_legendre
from helper import map_f_to_DG

PI = jnp.pi


def get_a0(f_init, core_params, nx):
	dx = core_params.Lx / nx
	if core_params.order is None:
		p = 1
		res = map_f_to_DG(f_init, 0.0, p, nx, dx, generate_legendre(p))
		return res[:,0]
	else:
		p = core_params.order+1
		res = map_f_to_DG(f_init, 0.0, p, nx, dx, generate_legendre(p))
		return res


def f_init_sum_of_amplitudes(Lx, key=random.PRNGKey(0), min_num_modes=1, max_num_modes=6, min_k = 1, max_k = 4, amplitude_max=1.0):
	key1, key2, key3, key4 = random.split(key, 4)
	phases = random.uniform(key1, (max_num_modes,)) * 2 * PI
	ks = random.randint(
		key3, (max_num_modes,), min_k, max_k
	)
	num_nonzero_modes = random.randint(
		key2, (1,), min_num_modes, max_num_modes + 1
	)[0]
	mask = jnp.arange(max_num_modes) < num_nonzero_modes
	amplitudes = jax.random.uniform(key4, (max_num_modes,)) * amplitude_max
	amplitudes = amplitudes * mask
	c = 1.0

	def sum_modes(x, t):
		return jnp.sum(
			amplitudes[None, :]
			* jnp.sin(
				ks[None, :] * 2 * PI / Lx * (x[:, None] - c * t) + phases[None, :]
			),
			axis=1,
		)

	return sum_modes


def get_initial_condition_fn(core_params, ic_string, **kwargs):
	Lx = core_params.Lx

	def f_init_zeros(x, t):
		return jnp.zeros(x.shape)

	def f_sawtooth(x, t):
		return 1 - 4 * jnp.abs(((x - t) % Lx) / Lx - 1 / 2)

	def f_sin(x, t):
		return -jnp.cos(2 * PI / Lx * (x - t))

	def f_gaussian(x, t):
		return jnp.exp(-32 * (((x - t) % Lx) - Lx / 2) ** 2 / (Lx ** 2))

	if ic_string == "zero" or ic_string == "zeros":
		return f_init_zeros
	elif ic_string == "sin_wave" or ic_string == "sin":
		return f_sin
	elif ic_string == "sum_sin":
		return f_init_sum_of_amplitudes(Lx, **kwargs)
	elif ic_string == "sawtooth":
		return f_sawtooth
	elif ic_string == "gaussian":
		return f_gaussian
	else:
		raise NotImplementedError