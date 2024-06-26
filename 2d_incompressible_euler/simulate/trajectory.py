import jax
import jax.numpy as jnp


def get_inner_fn(step_fn, dt_fn, t_inner):
    def cond_fun(x):
        a, t = x
        return jnp.logical_and(t < t_inner, jnp.logical_not(jnp.isnan(a).any()))

    def body_fun(x):
        a, t = x
        dt = dt_fn(a)
        dt = jnp.minimum(dt, t_inner - t)
        a_f = step_fn(a, dt)
        return (a_f, t + dt)

    @jax.jit
    def inner_fn(a):
        t = 0.0
        x = (a, t)
        a, _ = jax.lax.while_loop(cond_fun, body_fun, x)
        return a

    return inner_fn


def trajectory_fn(inner_fn, steps, start_with_input=True):
    def step(carry_in, _):
        carry_out = inner_fn(carry_in)
        frame = carry_in if start_with_input else carry_out
        return carry_out, frame

    @jax.jit
    def multistep(x_init):
        return jax.lax.scan(step, x_init, xs=None, length=steps)

    return multistep


def get_trajectory_fn(inner_fn, outer_steps, start_with_input=True):
    rollout_fn = trajectory_fn(inner_fn, outer_steps, start_with_input=start_with_input)

    def get_rollout(x_init):
        _, trajectory = rollout_fn(x_init)
        return trajectory

    return get_rollout
