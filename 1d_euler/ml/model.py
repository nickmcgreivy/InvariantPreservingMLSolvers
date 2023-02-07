from jax import config

config.update("jax_enable_x64", True)
from flax import linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as numpy
from flux import Flux
from typing import Sequence

from boundaryconditions import BoundaryCondition


def stencil_flux_FV_1D_euler(a, model, params):
	"""
	A correction on the flux due to the learned model
	"""
	s = model.apply(params, a)  # (3, nx, S)
	a_pad = jnp.pad(
		a,
		((0,0), ((model.stencil_width - 1) // 2, (model.stencil_width - 1) // 2 + 1)),
		"wrap",
	)
	P = jax.lax.conv_general_dilated_patches(
		a_pad[...,None],
		iter([model.stencil_width]),
		iter([1]),
		"VALID", 
		dimension_numbers=("NWC", "OIW", "NWC"),
	)
	F = jnp.sum(P * s, axis=-1)
	return F


def model_flux_FV_1D_euler(a, model, params):
	F = model.apply(params, a)  # (3, nx)
	return F


class LearnedFlux(nn.Module):
	"""
	For a single set of Euler FV coefficients, applies a NN to produce
	the flux f_{j+1/2} of shape (nx, 3)
	"""

	features: Sequence[int]
	kernel_size: int = 5
	kernel_out: int = 4
	boundary_conditions: BoundaryCondition = BoundaryCondition.GHOST

	def setup(self):
		if self.boundary_conditions == BoundaryCondition.PERIODIC:
			self.conv = CNNPeriodic1D(
				self.features,
				kernel_size=self.kernel_size,
				kernel_out=self.kernel_out,
				N_out=3,
			)
		elif self.boundary_conditions == BoundaryCondition.GHOST:
			self.conv = CNNGhost1D(self.features,
				kernel_size=self.kernel_size,
				kernel_out=self.kernel_out,
				N_out=3,
			)

	def __call__(self, inputs):
		x = jnp.transpose(inputs, (1, 0)) # (nx, 3)
		x = self.conv(x)  # x is (nx, 3) if periodic or (nx+1, 3) if non-periodic
		x = jnp.transpose(x, (1, 0)) # (3, nx) or (3, nx+1)
		return x


class LearnedStencil(nn.Module):
	"""
	For a single set of Euler FV coefficients, applies a NN to produce
	the stencil s_{j+\frac{1}{2}, l, k}

	Inputs
	features: A sequence of ints, which gives the number of features in each convolution. The
					  length of features gives the number of convolutions (until the last convolution)
	kernel_size: The size of the convolutional kernel, should be even
	S: The number of outputs for each position, which is the size of the stencil applied to the network.
			   Should be even.
	inputs: (3, nx) array of FV Euler coefficients

	Outputs
	learned_stencil: A (3, nx, S) array of the finite-difference coefficients
									 to compute F, i.e. the learned stencil
	"""

	features: Sequence[int]
	kernel_size: int = 5
	kernel_out: int = 4
	stencil_width: int = 4  # S

	def setup(self):
		assert self.stencil_width % 2 == 0 and self.stencil_width > 0
		self.conv = CNNPeriodic1D(
			self.features,
			kernel_size=self.kernel_size,
			kernel_out=self.kernel_out,
			N_out=self.stencil_width * 3,
		)

	def __call__(self, inputs): # inputs is (3, nx)
		x = jnp.transpose(inputs, (1, 0)) # (nx, 3)
		x = self.conv(x)  # (nx, 3 * S)
		x = x.reshape(*x.shape[:-1], self.stencil_width, 3)  # (nx, S, 3)
		x = jnp.transpose(x, (2, 0, 1)) # (3, nx, S)
		x = x - jnp.mean(x, axis=-1)[..., None]
		return x


class CNNPeriodic1D(nn.Module):
	"""
	1D convolutional neural network which takes an array in (3, nx)
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
			kernel_init=kernel_init,
			bias_init=bias_init,
		)

	def __call__(self, inputs):
		"""
		inputs: (nx, 3)
		outputs: (nx, n_out)
		"""
		assert inputs.shape[1] == 3
		# TODO : DO ALL THE PADDING AT ONCE
		x = inputs
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



class CNNGhost1D(nn.Module):
	"""
	1D convolutional neural network which takes an array in (3, nx)
	and returns an array of size (nx, N_out).

	The convolutional network has num_layers = len(features), with
	len(features) + 1 total convolutions. The last convolution outputs an
	array of size (nx+1, N_out).
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
			kernel_init=kernel_init,
			bias_init=bias_init,
		)

	def __call__(self, inputs):
		"""
		inputs: (nx, 3)
		outputs: (nx, n_out)
		"""
		assert inputs.shape[1] == 3

		x = inputs
		
		# TODO : DO ALL THE PADDING AT ONCE
		for lyr in self.layers:
			x = jnp.pad(
				x,
				(((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2), (0, 0)),
				"edge",
			)
			x = lyr(x)
			x = nn.relu(x)
		x = jnp.pad(
			x, ((self.kernel_out // 2 - 1, self.kernel_out // 2 + 1), (0, 0)), "edge"
		)
		x = self.output(x)
		return x
