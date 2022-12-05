import jax.numpy as jnp

from helper import get_sound_speed, get_u
from timederivative import time_derivative_FV_1D_euler

class ShockTubeSim():

	def __init__(self, core_params, sim_params, model = None, params = None):
		self.dt_fn = self.get_dt_fn(core_params, sim_params)
		self.step_fn = self.get_step_fn(core_params, sim_params, model=model, params=params)
		

	def get_step_fn(self, core_params, sim_params, model, params):
		self.F = time_derivative_FV_1D_euler(core_params, model=model, params=params, dt_fn = self.dt_fn)
		return lambda a, dt: sim_params.rk_fn(a, self.F, dt)


	def get_dt_fn(self, core_params, sim_params):

		def get_dt(a):
			max_speed = jnp.max(jnp.abs(get_u(a, core_params)) + get_sound_speed(a, core_params))
			nx = a.shape[1]
			dx = core_params.Lx / nx
			return sim_params.cfl_safety * dx / max_speed
			
		return get_dt
