import jax.numpy as jnp
from jax import random

from helper import map_f_to_FV, get_conserved_from_primitive

PI = jnp.pi


def get_a0(f_init, core_params, nx, n=8):
    dx = core_params.Lx / nx
    return map_f_to_FV(f_init, nx, dx, t=0.0, n=n)


def f_init_lax_shock_tube(core_params):
    Lx = core_params.Lx
    X_SHOCK = Lx * 0.5

    rho_L = 1.0
    p_L = 1.0
    u_L = 0.0
    rho_R = 1 / 8
    p_R = 0.1
    u_R = 0.0

    E_L = p_L / (core_params.gamma - 1) + 1 / 2 * rho_L * u_L**2
    E_R = p_R / (core_params.gamma - 1) + 1 / 2 * rho_R * u_R**2

    def f_init(x, t):
        return (x <= X_SHOCK) * jnp.asarray([rho_L, rho_L * u_L, E_L]) + (
            x > X_SHOCK
        ) * jnp.asarray([rho_R, rho_R * u_R, E_R])

    return f_init


def shock_tube_problem_1(core_params):
    Lx = core_params.Lx
    X_SHOCK = Lx * 0.3

    rho_L = 1.0
    p_L = 1.0
    u_L = 0.75
    rho_R = 1 / 8
    p_R = 0.1
    u_R = 0.0

    E_L = p_L / (core_params.gamma - 1) + 1 / 2 * rho_L * u_L**2
    E_R = p_R / (core_params.gamma - 1) + 1 / 2 * rho_R * u_R**2

    def f_init(x, t):
        return (x <= X_SHOCK) * jnp.asarray([rho_L, rho_L * u_L, E_L]) + (
            x > X_SHOCK
        ) * jnp.asarray([rho_R, rho_R * u_R, E_R])

    return f_init


def shock_tube_problem_2(core_params):
    Lx = core_params.Lx
    X_SHOCK = Lx * 0.5

    rho_L = 1.0
    u_L = -2.0
    p_L = 0.4
    rho_R = 1.0
    u_R = 2.0
    p_R = 0.4

    E_L = p_L / (core_params.gamma - 1) + 1 / 2 * rho_L * u_L**2
    E_R = p_R / (core_params.gamma - 1) + 1 / 2 * rho_R * u_R**2

    def f_init(x, t):
        return (x <= X_SHOCK) * jnp.asarray([rho_L, rho_L * u_L, E_L]) + (
            x > X_SHOCK
        ) * jnp.asarray([rho_R, rho_R * u_R, E_R])

    return f_init


def shock_tube_problem_3(core_params):
    Lx = core_params.Lx
    X_SHOCK = Lx * 0.5

    rho_L = 1.0
    u_L = 1.0
    p_L = 1e-6
    rho_R = 1.0
    u_R = -1.0
    p_R = 1e-6

    E_L = p_L / (core_params.gamma - 1) + 1 / 2 * rho_L * u_L**2
    E_R = p_R / (core_params.gamma - 1) + 1 / 2 * rho_R * u_R**2

    def f_init(x, t):
        return (x <= X_SHOCK) * jnp.asarray([rho_L, rho_L * u_L, E_L]) + (
            x > X_SHOCK
        ) * jnp.asarray([rho_R, rho_R * u_R, E_R])

    return f_init


def shock_tube_problem_4(core_params):
    Lx = core_params.Lx
    X_SHOCK = Lx * 0.8

    rho_L = 1.0
    u_L = -19.59745
    p_L = 1000
    rho_R = 1.0
    u_R = -19.59745
    p_R = 0.01

    E_L = p_L / (core_params.gamma - 1) + 1 / 2 * rho_L * u_L**2
    E_R = p_R / (core_params.gamma - 1) + 1 / 2 * rho_R * u_R**2

    def f_init(x, t):
        return (x <= X_SHOCK) * jnp.asarray([rho_L, rho_L * u_L, E_L]) + (
            x > X_SHOCK
        ) * jnp.asarray([rho_R, rho_R * u_R, E_R])

    return f_init


def shock_tube_problem_5(core_params):
    Lx = core_params.Lx
    X_SHOCK = Lx * 0.4

    rho_L = 5.99924
    u_L = 19.5975
    p_L = 460.894
    rho_R = 5.99242
    u_R = -6.19633
    p_R = 46.0895

    E_L = p_L / (core_params.gamma - 1) + 1 / 2 * rho_L * u_L**2
    E_R = p_R / (core_params.gamma - 1) + 1 / 2 * rho_R * u_R**2

    def f_init(x, t):
        return (x <= X_SHOCK) * jnp.asarray([rho_L, rho_L * u_L, E_L]) + (
            x > X_SHOCK
        ) * jnp.asarray([rho_R, rho_R * u_R, E_R])

    return f_init


def shock_tube_problem_6(core_params):
    Lx = core_params.Lx
    X_SHOCK = Lx * 0.5

    rho_L = 1.4
    u_L = 0.0
    p_L = 1.0
    rho_R = 1.0
    u_R = 0.0
    p_R = 1.0

    E_L = p_L / (core_params.gamma - 1) + 1 / 2 * rho_L * u_L**2
    E_R = p_R / (core_params.gamma - 1) + 1 / 2 * rho_R * u_R**2

    def f_init(x, t):
        return (x <= X_SHOCK) * jnp.asarray([rho_L, rho_L * u_L, E_L]) + (
            x > X_SHOCK
        ) * jnp.asarray([rho_R, rho_R * u_R, E_R])

    return f_init


def shock_tube_problem_7(core_params):
    Lx = core_params.Lx
    X_SHOCK = Lx * 0.5

    rho_L = 1.4
    u_L = 0.1
    p_L = 1.0
    rho_R = 1.0
    u_R = 0.1
    p_R = 1.0

    E_L = p_L / (core_params.gamma - 1) + 1 / 2 * rho_L * u_L**2
    E_R = p_R / (core_params.gamma - 1) + 1 / 2 * rho_R * u_R**2

    def f_init(x, t):
        return (x <= X_SHOCK) * jnp.asarray([rho_L, rho_L * u_L, E_L]) + (
            x > X_SHOCK
        ) * jnp.asarray([rho_R, rho_R * u_R, E_R])

    return f_init


def shock_tube_problem_8(core_params):
    Lx = core_params.Lx
    X_SHOCK = Lx * 0.8

    rho_L = 0.1261192
    u_L = 8.9047029
    p_L = 782.92899
    rho_R = 6.591493
    u_R = 2.2654207
    p_R = 3.1544874

    E_L = p_L / (core_params.gamma - 1) + 1 / 2 * rho_L * u_L**2
    E_R = p_R / (core_params.gamma - 1) + 1 / 2 * rho_R * u_R**2

    def f_init(x, t):
        return (x <= X_SHOCK) * jnp.asarray([rho_L, rho_L * u_L, E_L]) + (
            x > X_SHOCK
        ) * jnp.asarray([rho_R, rho_R * u_R, E_R])

    return f_init


def shock_tube_blast_wave(core_params):
    Lx = core_params.Lx
    X_SHOCK_1 = Lx * 0.1
    X_SHOCK_2 = Lx * 0.9

    rho_L = 1.0
    u_L = 0.0
    p_L = 1000.0
    rho_M = 1.0
    u_M = 0.0
    p_M = 0.01
    rho_R = 1.0
    u_R = 0.0
    p_R = 100.0

    E_L = p_L / (core_params.gamma - 1) + 1 / 2 * rho_L * u_L**2
    E_M = p_M / (core_params.gamma - 1) + 1 / 2 * rho_M * u_M**2
    E_R = p_R / (core_params.gamma - 1) + 1 / 2 * rho_R * u_R**2

    def f_init(x, t):
        return (
            (x <= X_SHOCK_1) * jnp.asarray([rho_L, rho_L * u_L, E_L])
            + ((x > X_SHOCK_1) & (x <= X_SHOCK_2))
            * jnp.asarray([rho_M, rho_M * u_M, E_M])
            + (x > X_SHOCK_2) * jnp.asarray([rho_R, rho_R * u_R, E_R])
        )

    return f_init


def f_init_sum_of_amplitudes(
    core_params,
    key,
    min_num_modes=1,
    max_num_modes=6,
    min_k=0,
    max_k=3,
    amplitude_max=1.0,
    background_rho=1.0,
    min_rho=0.5,
    background_p=1.0,
    min_p=0.01,
):
    """
    We have 3 primitive variables of interest: [rho, u, p].
    We want rho > 0 and p > 0.
    We initialize with a background density for rho and p,
    then add a sum of sine waves to this background density.
    We ensure that the initial rho and p are below constant values.
    """

    Lx = core_params.Lx

    def sum_modes_single(key):
        key1, key2, key3, key4 = random.split(key, 4)
        phases = random.uniform(key1, (max_num_modes,)) * 2 * PI
        ks = random.randint(key3, (max_num_modes,), min_k, max_k + 1)
        num_nonzero_modes = random.randint(
            key2, (1,), min_num_modes, max_num_modes + 1
        )[0]
        mask = jnp.arange(max_num_modes) < num_nonzero_modes
        amplitudes = random.uniform(key4, (max_num_modes,)) * amplitude_max
        amplitudes = amplitudes * mask
        c = 1.0

        def sum_modes(x, t):
            return jnp.sum(
                amplitudes * jnp.sin(ks * 2 * PI / Lx * (x - c * t) + phases),
                axis=-1,
            )

        return sum_modes

    key_rho, key_u, key_p = random.split(key, 3)

    sum_modes_rho = sum_modes_single(key_rho)
    sum_modes_u = sum_modes_single(key_u)
    sum_modes_p = sum_modes_single(key_p)

    def f_init(x, t):
        rho = jnp.maximum(min_rho, background_rho + sum_modes_rho(x, t))
        u = sum_modes_u(x, t)
        p = jnp.maximum(min_p, background_p + sum_modes_p(x, t))
        V = jnp.asarray([rho, u, p])
        return get_conserved_from_primitive(V, core_params)

    return f_init
