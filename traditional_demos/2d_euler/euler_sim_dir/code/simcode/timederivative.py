import jax.numpy as np
from flux import Flux

def time_derivative_euler(zeta, t, dx, dy, f_poisson_bracket, f_phi, flux, dldt = None):
    return np.zeros(zeta.shape)
    """
    H = f_phi(zeta, t)
    pb_term = f_poisson_bracket(zeta, H) / (dx * dy)

    if flux == Flux.CONSERVATION:
        break
    elif flux == Flux.ENERGYCONSERVATION:
        M = pb_term # mass is already conserved, M = N
        U = zeta - np.mean(zeta)
        psi_bar = np.mean(H, axis=-1)[...,None]
        psi_bar = psi_bar - np.mean(psi_bar)
        P = M - np.sum(M * psi_bar) / np.sum(psi_bar**2) * psi_bar
        W = U - np.sum(U * psi_bar) / np.sum(psi_bar**2) * psi_bar
        dSdt = -np.sum(pb_term * zeta)
        #dSdt = -np.sum(W * P)
        pb_term = P - np.sum(W * P) / np.sum(W**2) * W - (1.0 - REDUC) * dSdt * P / np.sum(W * P)

    return pb_term
    """