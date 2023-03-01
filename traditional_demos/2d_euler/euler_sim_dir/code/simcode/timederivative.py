import jax.numpy as np
from flux import Flux


def nabla_W(W):
    return (
        np.roll(W, 1, axis=0)
        + np.roll(W, -1, axis=0)
        + np.roll(W, 1, axis=1)
        + np.roll(W, -1, axis=1)
        - 4 * W
    )


def time_derivative_euler(zeta, t, dx, dy, f_poisson_bracket, f_phi, flux, dldt=None):
    H = f_phi(zeta, t)

    F_R, F_T = f_poisson_bracket(zeta, H)
    F_R = F_R / dy
    F_T = F_T / dx

    if flux == Flux.CONSERVATION:
        diff_x = np.roll(zeta, -1, axis=0) - zeta
        diff_y = np.roll(zeta, -1, axis=1) - zeta
        G_x = diff_x
        G_y = diff_y
        dldt_old_x = dy * np.sum(F_R * diff_x)
        dldt_old_y = dx * np.sum(F_T * diff_y)
        if dldt is None:
            dldt_new_x = dldt_old_x
            dldt_new_y = dldt_old_y
        else:
            dldt_new_x = dldt / 2
            dldt_new_y = dldt / 2

        denom_x = dy * np.sum(G_x * diff_x)
        denom_y = dx * np.sum(G_y * diff_y)

        F_R = F_R + (dldt_new_x - dldt_old_x) * G_x / denom_x
        F_T = F_T + (dldt_new_y - dldt_old_y) * G_y / denom_y

    pb_term = -(F_R - np.roll(F_R, 1, axis=0)) / (dx) - (
        F_T - np.roll(F_T, 1, axis=1)
    ) / (dy)

    if flux == Flux.ENERGYCONSERVATION:
        M = pb_term  # mass is already conserved, M = N
        U = zeta - np.mean(zeta)
        psi_bar = np.mean(H, axis=-1)[..., None]
        phi_bar = psi_bar - np.mean(psi_bar)
        P = M - np.sum(M * phi_bar) / np.sum(phi_bar**2) * phi_bar
        W = U - np.sum(U * phi_bar) / np.sum(phi_bar**2) * phi_bar
        dldt_old = np.sum(W * P) * dx * dy
        nabla = nabla_W(W)
        G = nabla - np.sum(nabla * phi_bar) / np.sum(phi_bar**2) * phi_bar
        if dldt is None:
            dldt = dldt_old
        pb_term = P + (dldt - dldt_old) * G / (np.sum(W * G) * dx * dy)

    return pb_term
