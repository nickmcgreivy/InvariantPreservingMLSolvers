import jax.numpy as jnp

def mae_loss(v, v_ex):
	diff = v - v_ex
	MAE = jnp.mean(jnp.absolute(diff))
	return MAE 

def mse_loss(v, v_ex):
	diff = v - v_ex
	MSE = jnp.mean(diff**2)
	return MSE

def mse_loss_DG(v, v_ex):
	p = v.shape[-1]
	assert v_ex.shape[-1] == v.shape[-1] and p > 1
	twokplusone = 2 * np.arange(0, p) + 1
	return np.mean(np.sum((a - a_exact) ** 2 / (twokplusone[None, :]), axis=-1))
