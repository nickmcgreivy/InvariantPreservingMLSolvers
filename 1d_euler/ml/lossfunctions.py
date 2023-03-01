import jax.numpy as jnp
import jax


def mae_loss(v, v_ex):
    diff = v - v_ex
    MAE = jnp.mean(jnp.absolute(diff))
    return MAE


def mse_loss_FV(v, v_ex):
    diff = v - v_ex
    MSE = jnp.mean(diff**2)
    return MSE


def one_norm_grad_f_model(a, params, model_flux_fn):
    def one_norm(a):
        f = model_flux_fn(a, params)
        return jnp.mean(jnp.abs(f))

    grad_one_norm = jax.grad(one_norm)

    grad_norm = grad_one_norm(a)
    return jnp.mean(grad_norm**2)
