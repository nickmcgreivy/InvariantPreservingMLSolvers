import jax.numpy as jnp


def mae_loss(v, v_ex):
	diff = v - v_ex
	MAE = jnp.mean(jnp.absolute(diff))
	return MAE 

def mse_loss_FV(v, v_ex):
	diff = v - v_ex
	MSE = jnp.mean(diff**2)
	return MSE

def mse_loss_DG(v, v_ex):
	p = v.shape[-1]
	assert v_ex.shape[-1] == v.shape[-1] and p > 1
	twokplusone = 2 * jnp.arange(0, p) + 1
	return jnp.mean(jnp.sum((v - v_ex) ** 2 / (twokplusone[None, :]), axis=-1))