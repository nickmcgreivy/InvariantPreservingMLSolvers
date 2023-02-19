import jax.numpy as jnp
from helper import integrate_fn_fv

PI = jnp.pi

def kolmogorov_forcing_fn_fv(sim_params, forcing_coeff, drag):
    denom = sim_params.dx * sim_params.dy

    ff = lambda x, y: 4 * (2 * PI / sim_params.Ly) * jnp.cos(4 * (2 * PI / sim_params.Ly) * y) * jnp.ones(x.shape)
    y_term = integrate_fn_fv(sim_params, ff, n=8) * denom

    drag_coeff = denom * drag
    constant_term = forcing_coeff * y_term

    def f_forcing(zeta):
        return constant_term - drag_coeff * zeta

    return f_forcing