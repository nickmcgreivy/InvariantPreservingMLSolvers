import jax.numpy as jnp
import jax
from jax import vmap

from flux import Flux
from boundaryconditions import BoundaryCondition
from helper import get_p, get_u, get_E

def f_j(a_j, core_params):
	rho_u = a_j[0]
	rho_u_sq = a_j[1]**2 / a_j[0]
	u = get_u(a_j, core_params)
	E = get_E(a_j, core_params)
	p = get_p(a_j, core_params)
	return jnp.asarray([rho_u, rho_u_sq + p, u * (p + E)])

def lax_friedrichs_flux_periodic(a, core_params, dt, dx):
	a_j_plus_one = jnp.roll(a, -1, axis=1)
	F_R = 0.5 * (f_j(a, core_params) + f_j(a_j_plus_one, core_params)) - 0.5 * (dx / dt) * (a_j_plus_one - a)
	F_L = jnp.roll(F_R, 1, axis=1)
	return -(F_R - F_L)

def lax_friedrichs_flux_ghost(a, core_params, dt, dx):
	a = jnp.pad(a, ((0,0), (1,1)), mode='edge')
	a_j = a[:, :-1]
	a_j_plus_one = a[:, 1:]
	F = 0.5 * (f_j(, core_params) + f_j(, core_params)) - 0.5 * (dx / dt) * (a_j_plus_one - a)
	F_R = F[:, 1:]
	F_L = F[:, :-1]
	return -(F_R - F_L)


def time_derivative_FV_1D_euler(core_params, model=None, params=None, dt_fn = None):

	if core_params.flux == Flux.LAXFRIEDRICHS:
		assert dt_fn is not None
		if core_params.bc == BoundaryCondition.GHOST:
			flux_term = lambda a: lax_friedrichs_flux_ghost(a, core_params, dt_fn(a), core_params.Lx / a.shape[0])
		elif core_params.bc == BoundaryCondition.PERIODIC:
			flux_term = lambda a: lax_friedrichs_flux_periodic(a, core_params, dt_fn(a), core_params.Lx / a.shape[0])
		else:
			raise NotImplementedError
	elif core_params.flux == Flux.ROE:
		raise NotImplementedError
	else:
		raise NotImplementedError

	def dadt(a):
		nx = a.shape[0]
		dx = core_params.Lx / nx
		return flux_term(a) / dx