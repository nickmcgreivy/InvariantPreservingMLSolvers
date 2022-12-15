import jax.numpy as jnp

from flux import Flux
from model import stencil_delta_flux_FV_1D_burgers, stencil_flux_FV_1D_burgers
from helper import integrate_f

def minmod_3(z1, z2, z3):
	s = (
		0.5
		* (jnp.sign(z1) + jnp.sign(z2))
		* jnp.absolute(0.5 * ((jnp.sign(z1) + jnp.sign(z3))))
	)
	return s * jnp.minimum(jnp.absolute(z1), jnp.minimum(jnp.absolute(z2), jnp.absolute(z3)))


def _global_stabilization(f0, a, epsilon_gs=0.0, G = lambda f, u: jnp.roll(u, -1) - u):
	diff = (jnp.roll(a, -1) - a)
	dl2_old_dt = jnp.sum(f0 * diff)
	g = G(f0, a)
	return f0 - jnp.nan_to_num((dl2_old_dt > 0.) * (dl2_old_dt * (1. + epsilon_gs)) * g / jnp.sum(diff * g))


def _godunov_flux_FV_1D_burgers(a):
	"""
	Computes the Godunov flux F(f_{j+1/2}) where
	+ is the right side.

	Inputs
	a: (nx) array

	Outputs
	F: (nx) array equal to the godunov flux
	"""
	u_left = a
	u_right = jnp.roll(a, -1, axis=0)
	zero_out = 0.5 * jnp.abs(jnp.sign(u_left) + jnp.sign(u_right))
	compare = jnp.less(u_left, u_right)
	F = lambda u: u ** 2 / 2
	return compare * zero_out * jnp.minimum(F(u_left), F(u_right)) + (
		1 - compare
	) * jnp.maximum(F(u_left), F(u_right))


def _weno_flux_FV_1D_burgers(a):
    epsilon = 1e-6
    d0 = 1/10
    d1 = 6/10
    d2 = 3/10
    a_minus2 = jnp.roll(a, 2)
    a_minus1 = jnp.roll(a, 1)
    a_plus1 = jnp.roll(a, -1)
    a_plus2 = jnp.roll(a, -2)
    a_plus3 = jnp.roll(a, -3)

    f = lambda u: u ** 2 / 2

    f_a_minus2 = f(a_minus2)
    f_a_minus1 = f(a_minus1)
    f_a = f(a)
    f_a_plus1 = f(a_plus1)
    f_a_plus2 = f(a_plus2)
    f_a_plus3 = f(a_plus3)

    # Moving to right, a > 0, f_plus
    f0 = (2/6) * f_a_minus2 - (7/6) * f_a_minus1 + (11/6) * f_a
    f1 = (-1/6) * f_a_minus1 + (5/6) * f_a + (2/6) * f_a_plus1
    f2 = (2/6) * f_a + (5/6) * f_a_plus1 + (-1/6) * f_a_plus2
    beta0 = (13/12) * (f_a_minus2 - 2 * f_a_minus1 + f_a)**2 + (1/4) * (f_a_minus2 - 4 * f_a_minus1 + 3 * f_a)**2
    beta1 = (13/12) * (f_a_minus1 - 2 * f_a  + f_a_plus1)**2 + (1/4) * (- f_a_minus1 + f_a_plus1)**2
    beta2 = (13/12) * (f_a - 2 * f_a_plus1  + f_a_plus2)**2 + (1/4) * (3 * f_a - 4 * f_a_plus1  + f_a_plus2)**2
    alpha0 = d0 / (epsilon + beta0)**2
    alpha1 = d1 / (epsilon + beta1)**2
    alpha2 = d2 / (epsilon + beta2)**2
    f_plus = (alpha0 * f0 + alpha1 * f1 + alpha2 * f2) / (alpha0 + alpha1 + alpha2)

    

    # Moving to left, a < 0, f_minus
    f0 = (2/6)  * f_a_plus3 - (7/6) * f_a_plus2 + (11/6) * f_a_plus1
    f1 = (-1/6) * f_a_plus2 + (5/6) * f_a_plus1 + (2/6)  * f_a
    f2 = (2/6)  * f_a_plus1 + (5/6) * f_a       + (-1/6) * f_a_minus1
    beta0 = (13/12) * (f_a_plus3 - 2 * f_a_plus2 + f_a_plus1)**2  + (1/4) * (     f_a_plus3 - 4 * f_a_plus2  + 3 * f_a_plus1)**2
    beta1 = (13/12) * (f_a_plus2 - 2 * f_a_plus1 + f_a)**2        + (1/4) * ( -   f_a_plus2                  +     f_a)**2
    beta2 = (13/12) * (f_a_plus1 - 2 * f_a       + f_a_minus1)**2 + (1/4) * ( 3 * f_a_plus1 - 4 * f_a        +     f_a_minus1)**2
    alpha0 = d0 / (epsilon + beta0)**2
    alpha1 = d1 / (epsilon + beta1)**2
    alpha2 = d2 / (epsilon + beta2)**2
    f_minus = (alpha0 * f0 + alpha1 * f1 + alpha2 * f2) / (alpha0 + alpha1 + alpha2)



    
    compare = jnp.less(a, a_plus1)
    zero_out = 0.5 * jnp.abs(jnp.sign(a) + jnp.sign(a_plus1))
    return compare * zero_out * jnp.minimum(f_minus, f_plus) + (
        1 - compare
    ) * jnp.maximum(f_minus, f_plus)



def _flux_term_FV_1D_burgers(a, core_params, global_stabilization=False, G = lambda f, u: jnp.roll(u, -1) - u, epsilon_gs=0.0, model=None, params=None, delta=True):
	if core_params.flux == Flux.GODUNOV:
		flux_right = _godunov_flux_FV_1D_burgers(a)
	elif core_params.flux == Flux.WENO:
		flux_right = _weno_flux_FV_1D_burgers(a)
	else:
		raise NotImplementedError

	if params is not None:
		if delta:
			delta_flux = stencil_delta_flux_FV_1D_burgers(a, model, params)
			flux_right = flux_right + delta_flux
		else:
			flux_right = stencil_flux_FV_1D_burgers(a, model, params)

	if global_stabilization:
		flux_right = _global_stabilization(flux_right, a, epsilon_gs=epsilon_gs, G = G)


	flux_left = jnp.roll(flux_right, 1, axis=0)
	return flux_left - flux_right

def _diffusion_term_FV(a, dx):
	return (jnp.roll(a, 1) + jnp.roll(a, -1) - 2 * a) / dx


def time_derivative_FV_1D_burgers(core_params, **kwargs):

	def dadt(a, t, forcing_func = None):
		nx = a.shape[0]
		dx = core_params.Lx / nx
		flux_term = _flux_term_FV_1D_burgers(a, core_params, **kwargs)
		if forcing_func is not None:
			forcing_term = integrate_f(forcing_func, t, nx, dx, n = 1)
		else:
			forcing_term = 0.0

		if core_params.nu > 0.0:
			diffusion_term = core_params.nu * _diffusion_term_FV(a, dx)
		else:
			diffusion_term = 0.0
		return (flux_term + forcing_term + diffusion_term) / dx

	return dadt
