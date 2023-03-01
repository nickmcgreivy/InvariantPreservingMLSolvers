import jax.numpy as jnp
from functools import lru_cache
import os
from jax import config
import numpy as onp
import time
import jax

from flux import Flux
from helper import f_to_DG, minmod_3
from basisfunctions import alpha_right_matrix_twice, alpha_top_matrix_twice
from model import output_flux

config.update("jax_enable_x64", True)


def load_alpha_right_matrix_twice(basedir, order):
    if os.path.exists(
        "{}/data/poissonmatrices/alpha_right_matrix_{}.npy".format(basedir, order)
    ):
        R = onp.load(
            "{}/data/poissonmatrices/alpha_right_matrix_{}.npy".format(basedir, order)
        )
    else:
        R = alpha_right_matrix_twice(order)
        onp.save(
            "{}/data/poissonmatrices/alpha_right_matrix_{}.npy".format(basedir, order),
            R,
        )
    return R


def load_alpha_top_matrix_twice(basedir, order):
    if os.path.exists(
        "{}/data/poissonmatrices/alpha_top_matrix_{}.npy".format(basedir, order)
    ):
        T = onp.load(
            "{}/data/poissonmatrices/alpha_top_matrix_{}.npy".format(basedir, order)
        )
    else:
        T = alpha_top_matrix_twice(order)
        onp.save(
            "{}/data/poissonmatrices/alpha_top_matrix_{}.npy".format(basedir, order),
            T,
        )
    return T


def get_poisson_bracket_fn_fv(sim_params):
    basedir = sim_params.basedir
    flux = sim_params.flux

    R = load_alpha_right_matrix_twice(basedir, 0)[0]
    T = load_alpha_top_matrix_twice(basedir, 0)[0]

    def get_alpha(H):
        alpha_R = H @ R
        alpha_T = H @ T
        return alpha_R, alpha_T

    def get_zeta_plus_minus(zeta):
        return zeta, jnp.roll(zeta, -1, axis=0), zeta, jnp.roll(zeta, -1, axis=1)

    def upwind_flux(zeta, alpha_R, alpha_T):
        zeta_R_minus, zeta_R_plus, zeta_T_minus, zeta_T_plus = get_zeta_plus_minus(zeta)

        flux_R = (alpha_R > 0) * alpha_R * zeta_R_minus + (
            alpha_R <= 0
        ) * alpha_R * zeta_R_plus
        flux_T = (alpha_T > 0) * alpha_T * zeta_T_minus + (
            alpha_T <= 0
        ) * alpha_T * zeta_T_plus
        return flux_R, flux_T

    def centered_flux(zeta, alpha_R, alpha_T):
        zeta_R_minus, zeta_R_plus, zeta_T_minus, zeta_T_plus = get_zeta_plus_minus(zeta)

        zeta_R = (zeta_R_minus + zeta_R_plus) / 2
        zeta_T = (zeta_T_minus + zeta_T_plus) / 2
        flux_R = alpha_R * zeta_R
        flux_T = alpha_T * zeta_T
        return flux_R, flux_T

    def vanleer_flux(zeta, alpha_R, alpha_T):
        zeta_R_minus, zeta_R_plus, zeta_T_minus, zeta_T_plus = get_zeta_plus_minus(zeta)

        s_R_right = zeta_R_plus - zeta
        s_R_left = zeta - jnp.roll(zeta, 1, axis=0)
        s_R_centered = (s_R_right + s_R_left) / 2
        s_R_minus = minmod_3(2 * s_R_left, s_R_centered, 2 * s_R_right)
        s_R_plus = jnp.roll(s_R_minus, -1, axis=0)
        flux_R = (alpha_R > 0) * alpha_R * (zeta_R_minus + s_R_minus / 2) + (
            alpha_R <= 0
        ) * alpha_R * (zeta_R_plus - s_R_plus / 2)

        s_T_right = zeta_T_plus - zeta
        s_T_left = zeta - jnp.roll(zeta, 1, axis=1)
        s_T_centered = (s_T_right + s_T_left) / 2
        s_T_minus = minmod_3(2 * s_T_left, s_T_centered, 2 * s_T_right)
        s_T_plus = jnp.roll(s_T_minus, -1, axis=1)
        flux_T = (alpha_T > 0) * alpha_T * (zeta_T_minus + s_T_minus / 2) + (
            alpha_T <= 0
        ) * alpha_T * (zeta_T_plus - s_T_plus / 2)

        return flux_R, flux_T

    if flux == Flux.UPWIND:
        flux_fn = upwind_flux
    elif flux == Flux.CENTERED:
        flux_fn = centered_flux
    elif flux == Flux.VANLEER:
        flux_fn = vanleer_flux
    else:
        raise Exception

    def poisson_bracket(zeta, H, model=None, params=None):
        alpha_R, alpha_T = get_alpha(H)

        flux_R, flux_T = flux_fn(zeta, alpha_R, alpha_T)

        if model != None:
            assert params is not None
            delta_flux_R, delta_flux_T = output_flux(
                zeta, alpha_R, alpha_T, model, params
            )

            flux_R = flux_R + delta_flux_R
            flux_T = flux_T + delta_flux_T

        flux_L = jnp.roll(flux_R, 1, axis=0)
        flux_B = jnp.roll(flux_T, 1, axis=1)
        return flux_L + flux_B - flux_R - flux_T

    return poisson_bracket
