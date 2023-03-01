import jax.numpy as jnp

from poissonbracket import (
    get_poisson_bracket_fn_fv,
    load_alpha_right_matrix_twice,
    load_alpha_top_matrix_twice,
)
from poissonsolver import get_poisson_solve_fn_fv
from diffusion import get_diffusion_fn_fv
from forcings import kolmogorov_forcing_fn_fv
from timederivative import time_derivative_fn_fv


class KolmogorovFiniteVolumeSimulation:
    def __init__(
        self, sim_params, viscosity, forcing_coeff, drag, model=None, params=None
    ):
        self.viscosity = viscosity
        self.forcing_coeff = forcing_coeff
        self.drag = drag
        self.model = model
        self.params = params
        self.step_fn = self.get_step_fn(sim_params, model=model, params=params)
        self.dt_fn = self.get_dt_fn(sim_params)
        self.alpha_fn = self.get_alpha_fn(sim_params)

    def get_step_fn(self, sim_params, model=None, params=None):
        poisson_bracket_fn = get_poisson_bracket_fn_fv(sim_params)
        self.poisson_solve_fn = get_poisson_solve_fn_fv(sim_params)
        diffusion_fn = get_diffusion_fn_fv(sim_params, self.viscosity)
        forcing_fn = kolmogorov_forcing_fn_fv(sim_params, self.forcing_coeff, self.drag)
        dadt = time_derivative_fn_fv(
            sim_params,
            poisson_bracket_fn,
            self.poisson_solve_fn,
            forcing_fn=forcing_fn,
            diffusion_fn=diffusion_fn,
        )
        self.F_params = lambda a, model, params: dadt(a, model=model, params=params)
        self.F = lambda a: dadt(a, model=model, params=params)
        return lambda a, dt: sim_params.rk_fn(a, self.F, dt)

    def get_alpha_fn(self, sim_params):
        R = load_alpha_right_matrix_twice(sim_params.basedir, 0)[0]
        T = load_alpha_top_matrix_twice(sim_params.basedir, 0)[0]

        def get_alpha(H):
            alpha_R = H @ R
            alpha_T = H @ T
            return alpha_R, alpha_T

        def alpha_fn(zeta):
            phi = self.poisson_solve_fn(zeta)
            return get_alpha(phi)

        return alpha_fn

    def get_dt_fn(self, sim_params):
        R = load_alpha_right_matrix_twice(sim_params.basedir, 0)[0] / sim_params.dy
        T = load_alpha_top_matrix_twice(sim_params.basedir, 0)[0] / sim_params.dx

        def get_alpha(H):
            alpha_R = H @ R
            alpha_T = H @ T
            return alpha_R, alpha_T

        def get_dt(a):
            phi = self.poisson_solve_fn(a)
            alpha_R, alpha_T = get_alpha(phi)
            max_R_standard = jnp.amax(jnp.abs(alpha_R))
            max_T_standard = jnp.amax(jnp.abs(alpha_T))

            max_R_new = (
                jnp.amax(jnp.abs(jnp.roll(a, 1, axis=1) - a)) * sim_params.dx * 2
            )
            max_T_new = (
                jnp.amax(jnp.abs(jnp.roll(a, 1, axis=0) - a)) * sim_params.dy * 2
            )

            max_R = jnp.maximum(max_R_standard, max_R_new)
            max_T = jnp.maximum(max_T_standard, max_T_new)
            return (
                sim_params.cfl_safety
                * (sim_params.dx * sim_params.dy)
                / (max_R * sim_params.dy + max_T * sim_params.dx)
            )

        return get_dt
