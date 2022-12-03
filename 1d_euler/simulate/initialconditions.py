import jax.numpy as jnp

from helper import map_f_to_FV

def get_a0(f_init, core_params, nx, n = 8):
	dx = core_params.Lx / nx
	return map_f_to_FV(f_init, nx, dx, t = 0.0, n = n)


def f_init_lax_shock_tube(core_params):
	Lx = core_params.Lx
	rho_L = 1.0
	p_L = 1.0
	u_L = 0.0
	rho_R = 1/8
	p_R = 0.1
	u_R = 0.0

	E_L = p_L / (core_params.gamma - 1) + 1/2 * rho_L * u_L**2 
	E_R = p_R / (core_params.gamma - 1) + 1/2 * rho_R * u_R**2 

	def f_init(x, t):
		return (x <= Lx/2) * jnp.asarray([rho_L, rho_L * u_L, E_L]) + (x > Lx/2) * jnp.asarray([rho_R, rho_R * u_R, E_R])

	return f_init