from jax import config

config.update("jax_enable_x64", True)
from flax import linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as numpy
from flux import Flux
from typing import Sequence


def model_output_FV_1D_advection(a, model, params):
	return model.apply(params, a)


class LearnedFluxOutput(nn.Module):
	"""
	For a single set of FV coefficients, applies a NN to produce
	the flux f_{j+1/2}

	Inputs
	features: A sequence of ints, which gives the number of features in each convolution. The
					  length of features gives the number of convolutions (until the last convolution)
	kernel_size: The size of the convolutional kernel, should be odd
	kernel_out: the size of the kernel in the final layer, should be even
	inputs: (nx) array of FV coefficients

	Outputs
	coefficients: A (nx) array of flux coefficients
	"""

	features: Sequence[int]
	kernel_size: int = 5
	kernel_out: int = 4 

	def setup(self):
		self.conv = CNNPeriodic1D(
			self.features,
			kernel_size=self.kernel_size,
			kernel_out=self.kernel_out,
			N_out=1,
		)

	def __call__(self, inputs):
		out = self.conv(inputs) # input is (nx), output is (nx, 1)
		return out[...,0]


class CNNPeriodic1D(nn.Module):
	"""
	1D convolutional neural network which takes an array of shape (nx)
	and returns an array of shape (nx, N_out).

	The convolutional network has num_layers = len(features), with
	len(features) + 1 total convolutions. The last convolution outputs an
	array of size (nx, N_out) without a relu layer.
	"""

	features: Sequence[int]
	kernel_size: int = 5
	kernel_out: int = 4
	N_out: int = 1

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
			kernel_init=kernel_init,
			bias_init=bias_init,
		)

	def __call__(self, inputs):
		"""
		inputs: (nx)
		outputs: (nx, n_out)
		"""
		x = inputs[...,None]
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
