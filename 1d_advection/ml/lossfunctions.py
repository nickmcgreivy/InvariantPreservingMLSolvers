import jax.numpy as jnp


def mae_loss(v, v_ex):
    diff = v - v_ex
    MAE = jnp.mean(jnp.absolute(diff))
    return MAE


def mse_loss_FV(v, v_ex):
    diff = v - v_ex
    MSE = jnp.mean(diff**2)
    return MSE
