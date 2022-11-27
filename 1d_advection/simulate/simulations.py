import jax.numpy as jnp
from fv.timederivative import time_derivative_FV_1D_advection
from dg.timederivative import time_derivative_DG_1D_advection

class AdvectionFVSim():

	def __init__(self, core_params, sim_params, model=None, params=None):
		self.model = model
		self.params = params
		self.step_fn = self.get_step_fn(core_params, sim_params, model=model, params=params)
		self.dt_fn = self.get_dt_fn(core_params, sim_params)


	def get_step_fn(self, core_params, sim_params, model, params):
		self.F = time_derivative_FV_1D_advection(core_params, global_stabilization=sim_params.global_stabilization, model=model, params=params)
		return lambda a, dt: sim_params.rk_fn(a, self.F, dt)

	def get_dt_fn(self, core_params, sim_params):
		c = 1.0

		def get_dt(a):
			nx = a.shape[0]
			dx = core_params.Lx / nx
			return sim_params.cfl_safety * dx / c
			
		return get_dt




class AdvectionDGSim():

	def __init__(self, core_params, sim_params, model=None, params=None):
		self.model = model
		self.params = params
		self.step_fn = self.get_step_fn(core_params, sim_params, model=model, params=params)
		self.dt_fn = self.get_dt_fn(core_params, sim_params)


	def get_step_fn(self, core_params, sim_params, model, params):
		self.F = time_derivative_DG_1D_advection(core_params, global_stabilization=sim_params.global_stabilization, model=model, params=params)
		return lambda a, dt: sim_params.rk_fn(a, self.F, dt)

	def get_dt_fn(self, core_params, sim_params):
		c = 1.0

		def get_dt(a):
			nx = a.shape[0]
			dx = core_params.Lx / nx
			return sim_params.cfl_safety * dx / c / (2 * core_params.order + 1)

		return get_dt