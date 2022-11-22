from jax import config

config.update("jax_enable_x64", True)
from flax import linen as nn
from flax import serialization
import jax
import jax.numpy as np
import numpy as numpy
from flux import Flux
from typing import Sequence


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
    stencil_size: int = 6  # S
    p: int = 1

    def setup(self):
        assert self.stencil_size % 2 == 0 and self.stencil_size > 0
        negonetok = (np.ones(self.p) * -1) ** np.arange(self.p)
        self.base_stencil = np.zeros((self.stencil_size, self.p))
        self.base_stencil = self.base_stencil.at[self.stencil_size // 2 - 1, :].set(0.5)
        self.base_stencil = self.base_stencil.at[self.stencil_size // 2, :].set(
            0.5 * negonetok
        )
        self.conv = CNNPeriodic1D(
            self.features,
            kernel_size=self.kernel_size,
            kernel_out=self.kernel_out,
            N_out=self.stencil_size * self.p,
        )

    def __call__(self, inputs):
        x = self.conv(inputs)  # x is (nx, S * p)
        x = x.reshape(*x.shape[:-1], self.stencil_size, self.p)  # (nx, S, p)
        x = x.at[:, :, 0].add(-np.mean(x[..., 0], axis=-1)[..., None])
        x = x + self.base_stencil[None, :, :]
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
        dtype = np.float64
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
        inputs: (nx, p)
        outputs: (nx, n_out)
        """
        x = inputs
        # TODO : DO ALL THE PADDING AT ONCE
        for lyr in self.layers:
            x = np.pad(
                x,
                (((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2), (0, 0)),
                "wrap",
            )
            x = lyr(x)
            x = nn.relu(x)
        x = np.pad(
            x, ((self.kernel_out // 2 - 1, self.kernel_out // 2), (0, 0)), "wrap"
        )
        x = self.output(x)
        return x
