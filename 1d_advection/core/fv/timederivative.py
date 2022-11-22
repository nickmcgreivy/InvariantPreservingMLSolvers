import jax.numpy as jnp
from jax.lax import scan
import jax
from jax import vmap

from flux import Flux
from model import stencil_flux_FV_1D_advection

def _upwind_flux_FV_1D_advection(a, core_params):
	c = 1.0
	if c > 0:
		F = a
	else:
		F = jnp.roll(a, -1)
	return F


def _centered_flux_FV_1D_advection(a, core_params):
	return 0.5 * (a + jnp.roll(a, -1))


def minmod_3(z1, z2, z3):
	s = (
		0.5
		* (jnp.sign(z1) + jnp.sign(z2))
		* jnp.absolute(0.5 * ((jnp.sign(z1) + jnp.sign(z3))))
	)
	return s * jnp.minimum(jnp.absolute(z1), jnp.minimum(jnp.absolute(z2), jnp.absolute(z3)))

def _muscl_flux_FV_1D_advection(u, core_params):
	du_j_minus = u - jnp.roll(u, 1)
	du_j_plus = jnp.roll(u, -1) - u
	du_j = minmod_3(du_j_minus, (du_j_plus + du_j_minus) / 4, du_j_plus)
	return u + du_j

def _global_stabilization(f0, a):
	raise NotImplementedError




def _flux_term_FV_1D_advection(a, core_params, global_stabilization=False, model=None, params=None):
	if core_params.flux == Flux.UPWIND:
		flux_right = _upwind_flux_FV_1D_advection(a, core_params)
	elif core_params.flux == Flux.CENTERED:
		flux_right = _centered_flux_FV_1D_advection(a, core_params)
	elif core_params.flux == Flux.MUSCL:
		flux_right = _muscl_flux_FV_1D_advection(a, core_params)
	else:
		raise NotImplementedError

	if params is not None:
		delta_flux = stencil_flux_FV_1D_advection(a, model, params)
		flux_right = flux_right + delta_flux


	flux_left = jnp.roll(flux_right, 1, axis=0)
	return flux_left - flux_right



def time_derivative_FV_1D_advection(core_params, global_stabilization=False, model=None, params=None):
	
	c = 1.0

	def dadt(a):
		nx = a.shape[0]
		dx = core_params.Lx / nx
		C = c / dx
		flux_term = _flux_term_FV_1D_advection(a, core_params, global_stabilization=global_stabilization, model=model, params=params)
		return C * flux_term

	return dadt
