from jax import config

config.update("jax_enable_x64", True)
from flax import linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as numpy
from flux import Flux
from typing import Sequence

from helper import f_burgers

def stencil_delta_flux_FV_1D_burgers(a, model, params):
	"""
	A correction on the flux due to the learned model.

	Adds \Delta f = \sum_k s_k f_{j+k}(u) where f = u^2/2.
	Linear sum of non-linear functions.
	"""

	s = model.apply(params, a)  # (nx, S)
	a_pad = jnp.pad(
		a,
		(((model.stencil_width - 1) // 2, (model.stencil_width - 1) // 2 + 1)),
		"wrap",
	)
	flux = f_burgers(a_pad)
	P = jax.lax.conv_general_dilated_patches(
		flux[None, ..., None],
		iter([model.stencil_width]),
		iter([1]),
		"VALID",
		dimension_numbers=("CWN", "OIW", "WCN"),
	)[...,0]
	delta_F = jnp.sum(P * s, axis=1)
	return delta_F


def stencil_flux_FV_1D_burgers(a, model, params):
	"""
	A correction on the flux due to the learned model.

	Sets u_{j+1/2} = \sum_k s_k u_j+k
	and f_{j+1/2} = u_{j+1/2}^2/2.

	Linear sum of linear stencil gives u, which results in non-linear stencil.
	"""
	s = model.apply(params, a)  # (nx, S)
	a_pad = jnp.pad(
		a,
		(((model.stencil_width - 1) // 2, (model.stencil_width - 1) // 2 + 1)),
		"wrap",
	)
	P = jax.lax.conv_general_dilated_patches(
		a_pad[None, ..., None],
		iter([model.stencil_width]),
		iter([1]),
		"VALID",
		dimension_numbers=("CWN", "OIW", "WCN"),
	)[...,0]
	a_new = jnp.sum(P * s, axis=1)
	return f_burgers(a_new)


class LearnedStencil(nn.Module):
	"""
	For a single set of DG coefficients, applies a NN to produce
	the stencil s_{j+\frac{1}{2}, l, k}

	Inputs
	features: A sequence of ints, which gives the number of features in each convolution. The
					  length of features gives the number of convolutions (until the last convolution)
	kernel_size: The size of the convolutional kernel, should be even
	S: The number of outputs for each position, which is the size of the stencil applied to the network.
			   Should be even.
	p_out: Should equal p
	inputs: (nx, p) array of DG coefficients

	Outputs
	learned_stencil: A (nx, S, p) or (nx, p, S, p) array of the finite-difference coefficients
									 to compute F, i.e. the learned stencil
	"""

	features: Sequence[int]
	kernel_size: int = 5
	kernel_out: int = 4
	stencil_width: int = 4  # S
	delta: bool = True

	def setup(self):
		assert self.stencil_width % 2 == 0 and self.stencil_width > 0
		self.conv = CNNPeriodic1D(
			self.features,
			kernel_size=self.kernel_size,
			kernel_out=self.kernel_out,
			N_out=self.stencil_width,
		)
		self.base_stencil = jnp.zeros(self.stencil_width).at[self.stencil_width//2].set(0.5).at[self.stencil_width//2-1].set(0.5)

	def __call__(self, inputs):
		x = self.conv(inputs) # (nx, s)
		x = x - jnp.mean(x, axis=-1)[:, None]
		if self.delta == False:
			x = x + self.base_stencil[None, :]
		return x


class CNNPeriodic1D(nn.Module):
	"""
	1D convolutional neural network which takes an array in (nx, p)
	and returns an array of size (nx, N_out).

	The convolutional network has num_layers = len(features), with
	len(features) + 1 total convolutions. The last convolution outputs an
	array of size (nx, N_out).
	"""

	features: Sequence[int]
	kernel_size: int = 5
	kernel_out: int = 4
	N_out: int = 6

	def setup(self):
		assert self.kernel_out % 2 == 0 and self.kernel_out > 0
		assert self.kernel_size % 2 == 1 and self.kernel_size > 0
		dtype = jnp.float64
		kernel_init = nn.initializers.lecun_normal(dtype=dtype)
		bias_init = nn.initializers.zeros
		zeros_init = nn.initializers.zeros
		self.layers = [
			nn.Conv(
				features=feat,
				kernel_size=(self.kernel_size,),
				padding="VALID",
				dtype=dtype,
				kernel_init=kernel_init,
				bias_init=bias_init,
			)
			for feat in self.features
		]
		self.output = nn.Conv(
			features=self.N_out,
			kernel_size=(self.kernel_out,),
			padding="VALID",
			dtype=dtype,
			kernel_init=zeros_init,
			bias_init=bias_init,
		)

	def __call__(self, inputs):
		"""
		inputs: (nx)
		outputs: (nx, n_out)
		"""
		x = inputs[..., None]
		# TODO : DO ALL THE PADDING AT ONCE
		for lyr in self.layers:
			x = jnp.pad(
				x,
				(((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2), (0, 0)),
				"wrap",
			)
			x = lyr(x)
			x = nn.relu(x)
		x = jnp.pad(
			x, ((self.kernel_out // 2 - 1, self.kernel_out // 2), (0, 0)), "wrap"
		)
		x = self.output(x)
		return x
