import jax.numpy as jnp

def MSE_loss(v, v_e):
	diff = v - v_e
	MSE = jnp.mean(diff**2)
	return MSE