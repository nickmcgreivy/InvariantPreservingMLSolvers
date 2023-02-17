import jax.numpy as jnp
from helper import integrate_fn_fv

PI = jnp.pi

def kolmogorov_forcing_fn_fv(sim_params, forcing_coeff, drag):
    ff = lambda x, y: 4 * (2 * PI / sim_params.Ly) * jnp.cos(4 * (2 * PI / sim_params.Ly) * y)
    y_term = integrate_fn_fv(sim_params, ff)

    drag_coeff = sim_params.dx * sim_params.dy * drag
    constant_term = forcing_coeff * y_term

    def f_forcing(zeta):
        return constant_term - drag_coeff * zeta

    return f_forcing