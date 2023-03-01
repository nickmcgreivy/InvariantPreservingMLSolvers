import jax.numpy as np


def forward_euler(a_n, t_n, F, dt, dldt=None):
    a_1 = a_n + dt * F(a_n, t_n, dldt=dldt)
    return a_1, t_n + dt


def ssp_rk2(a_n, t_n, F, dt, dldt=None):
    a_1 = a_n + dt * F(a_n, t_n, dldt=dldt)
    a_2 = 0.5 * a_n + 0.5 * a_1 + 0.5 * dt * F(a_1, t_n + dt, dldt=dldt)
    return a_2, t_n + dt


def ssp_rk3(a_n, t_n, F, dt, dldt=None):
    a_1 = a_n + dt * F(a_n, t_n, dldt=dldt)
    a_2 = 0.75 * a_n + 0.25 * (a_1 + dt * F(a_1, t_n + dt, dldt=dldt))
    a_3 = 1 / 3 * a_n + 2 / 3 * (a_2 + dt * F(a_2, dt + dt / 2, dldt=dldt))
    return a_3, t_n + dt


FUNCTION_MAP = {
    "FE": forward_euler,
    "fe": forward_euler,
    "forward_euler": forward_euler,
    "rk2": ssp_rk2,
    "RK2": ssp_rk2,
    "ssp_rk2": ssp_rk2,
    "rk3": ssp_rk3,
    "RK3": ssp_rk3,
    "ssp_rk3": ssp_rk3,
}
