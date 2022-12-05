import jax.numpy as jnp
import jax
from jax import vmap

from flux import Flux
from boundaryconditions import BoundaryCondition
from helper import get_p, get_u, get_H

def f_j(a_j, core_params):
	rho_u = a_j[1]
	rho_usq = rho_u**2 / a_j[0]
	E = a_j[2]
	u = get_u(a_j, core_params)
	p = get_p(a_j, core_params)
	return jnp.asarray([rho_u, rho_usq + p, u * (p + E)])

def lax_friedrichs_flux_periodic(a, core_params, dt, dx):
	a_j_plus_one = jnp.roll(a, -1, axis=1)
	F_R = 0.5 * (f_j(a, core_params) + f_j(a_j_plus_one, core_params)) - 0.5 * (dx / dt) * (a_j_plus_one - a)
	F_L = jnp.roll(F_R, 1, axis=1)
	return -(F_R - F_L)

def lax_friedrichs_flux_ghost(a, core_params, dt, dx):
	a = jnp.pad(a, ((0,0), (1,1)), mode='edge')
	a_j = a[:, :-1]
	a_j_plus_one = a[:, 1:]
	F = 0.5 * (f_j(a_j, core_params) + f_j(a_j_plus_one, core_params)) - 0.5 * (dx / dt) * (a_j_plus_one - a_j)
	F_R = F[:, 1:]
	F_L = F[:, :-1]
	return -(F_R - F_L)

def flux_roe(aL, aR, core_params):
	rhoL = aL[0]
	rhoR = aR[0]

	deltap = get_p(aR, core_params) - get_p(aL, core_params)
	deltau = get_u(aR, core_params) - get_u(aL, core_params)
	deltarho = rhoR - rhoL

	rhoRoe = jnp.sqrt(rhoL * rhoR)
	denom = jnp.sqrt(rhoL) + jnp.sqrt(rhoR)
	uRoe = (jnp.sqrt(rhoL) * get_u(aL, core_params) + jnp.sqrt(rhoR) * get_u(aR, core_params)) / denom
	HRoe = (jnp.sqrt(rhoL) * get_H(aL, core_params) + jnp.sqrt(rhoR) * get_H(aR, core_params)) / denom
	cRoe = jnp.sqrt( (core_params.gamma - 1) * (HRoe - uRoe ** 2 / 2) )

	V1 = (deltap - rhoRoe * cRoe * deltau) / (2 * cRoe**2)
	V2 = -(deltap - cRoe**2 * deltarho)     / (cRoe**2)
	V3 = (deltap + rhoRoe * cRoe * deltau) / (2 * cRoe**2)

	ones = jnp.ones(aL.shape[1])
	r1 = jnp.asarray([ones, uRoe - cRoe, HRoe - uRoe * cRoe])
	r2 = jnp.asarray([ones, uRoe,        uRoe**2 / 2       ])
	r3 = jnp.asarray([ones, uRoe + cRoe, HRoe + uRoe * cRoe])

	eig1 = uRoe - cRoe
	eig2 = uRoe
	eig3 = uRoe + cRoe

	corr1 = jnp.abs(eig1) * V1 * r1
	corr2 = jnp.abs(eig2) * V2 * r2
	corr3 = jnp.abs(eig3) * V3 * r3

	return 0.5 * (f_j(aL, core_params) + f_j(aR, core_params)) - 0.5 * (corr1 + corr2 + corr3)

def flux_periodic(a, core_params, flux_fn):
	a_j = a
	a_j_plus_one = jnp.roll(a, -1, axis=1)
	F = flux_fn(a_j, a_j_plus_one, core_params)
	F_R = F
	F_L = jnp.roll(F_R, 1, axis=1)
	return -(F_R - F_L)

def flux_ghost(a, core_params, flux_fn):
	a = jnp.pad(a, ((0,0), (1,1)), mode='edge')
	a_j = a[:, :-1]
	a_j_plus_one = a[:, 1:]
	F = flux_fn(a_j, a_j_plus_one, core_params)
	F_R = F[:, 1:]
	F_L = F[:, :-1]
	return -(F_R - F_L)


def time_derivative_FV_1D_euler(core_params, model=None, params=None, dt_fn = None):

	if core_params.flux == Flux.LAXFRIEDRICHS:
		assert dt_fn is not None
		if core_params.bc == BoundaryCondition.GHOST:
			flux_term = lambda a: lax_friedrichs_flux_ghost(a, core_params, dt_fn(a), core_params.Lx / a.shape[1])
		elif core_params.bc == BoundaryCondition.PERIODIC:
			flux_term = lambda a: lax_friedrichs_flux_periodic(a, core_params, dt_fn(a), core_params.Lx / a.shape[1])
		else:
			raise NotImplementedError
	elif core_params.flux == Flux.ROE:
		flux_fn = flux_roe
	else:
		raise NotImplementedError

	if core_params.flux != Flux.LAXFRIEDRICHS:
		if core_params.bc == BoundaryCondition.GHOST:
			flux_term = lambda a: flux_ghost(a, core_params, flux_fn)
		elif core_params.bc == BoundaryCondition.PERIODIC:
			flux_term = lambda a: flux_periodic(a, core_params, flux_fn)
		else:
			raise NotImplementedError

	def dadt(a):
		nx = a.shape[1]
		dx = core_params.Lx / nx
		return flux_term(a) / dx

	return dadt