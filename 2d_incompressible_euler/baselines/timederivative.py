import jax.numpy as jnp

def time_derivative_fn_fv(sim_params, f_poisson_bracket, f_phi, forcing_fn=None, diffusion_fn = None
):
    denominator = sim_params.dx * sim_params.dy

    def dadt(zeta, model=None, params=None):
        phi = f_phi(zeta)

        pb_term = f_poisson_bracket(zeta, phi, model=model, params=params) 

        if forcing_fn is not None:
            forcing_term = forcing_fn(zeta)
        else:
            forcing_term = 0.0
        if diffusion_fn is not None:
            diffusion_term = diffusion_fn(zeta)
        else:
            diffusion_term = 0.0

        return (pb_term + forcing_term + diffusion_term) / denominator
        
    return dadt