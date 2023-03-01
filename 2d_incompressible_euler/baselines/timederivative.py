import jax.numpy as jnp


def nabla_FV(W):
    return (
        jnp.roll(W, 1, axis=0)
        + jnp.roll(W, -1, axis=0)
        + jnp.roll(W, 1, axis=1)
        + jnp.roll(W, -1, axis=1)
        - 4 * W
    )


def time_derivative_fn_fv(
    sim_params, f_poisson_bracket, f_phi, forcing_fn=None, diffusion_fn=None
):
    denominator = sim_params.dx * sim_params.dy

    def dadt(zeta, model=None, params=None):
        phi = f_phi(zeta)

        pb_term = f_poisson_bracket(zeta, phi, model=model, params=params)

        if sim_params.global_stabilization:
            if sim_params.energy_conserving:
                M = pb_term  # mass is already conserved
                U = zeta - jnp.mean(zeta)
                psi_bar = jnp.mean(phi, axis=-1)
                phi_bar = psi_bar - jnp.mean(psi_bar)
                P = M - jnp.sum(M * phi_bar) / jnp.sum(phi_bar**2) * phi_bar
                W = U - jnp.sum(U * phi_bar) / jnp.sum(phi_bar**2) * phi_bar
                dldt_old = jnp.sum(W * P)
                nabla = nabla_FV(W)
                G = nabla - jnp.sum(nabla * phi_bar) / jnp.sum(phi_bar**2) * phi_bar
                pb_term = P - (dldt_old > 0.0) * dldt_old * G / jnp.sum(W * G)
            else:
                M = pb_term  # mass is already conserved
                U = zeta - jnp.mean(zeta)
                dldt_old = jnp.sum(U * M)
                G = nabla_FV(U)
                pb_term = M - (dldt_old > 0.0) * dldt_old * G / jnp.sum(U * G)

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
