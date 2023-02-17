import jax.numpy as jnp
from sympy import legendre, diff, integrate, symbols
from functools import lru_cache

from sympy.matrices import Matrix, zeros
from scipy.special import eval_legendre

def get_diffusion_fn_fv(sim_params, viscosity):

	def f_boundary(Lx, Ly, zeta):
		return (Ly/Lx) * (jnp.roll(zeta, 1, axis=0) + jnp.roll(zeta, -1, axis=0) - 2*zeta) + (Lx/Ly) * (jnp.roll(zeta, 1, axis=1) + jnp.roll(zeta, -1, axis=1) - 2*zeta)

	def f_diffusion(zeta):
		return viscosity * f_boundary(sim_params.Lx, sim_params.Ly, zeta)
	
	return f_diffusion