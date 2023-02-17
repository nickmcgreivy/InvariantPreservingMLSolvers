import jax
import jax.numpy as jnp
from flax import linen as nn

from mlparams import ModelParams


def output_flux(zeta, alpha_R, alpha_T, model, params):
    return model.apply(params, zeta, alpha_R, alpha_T)


class LearnedFlux2D(nn.Module):

    ml_params: ModelParams

    def setup(self):
        assert self.ml_params.kernel_size % 2 == 1
        assert self.ml_params.depth >= 1
        assert self.ml_params.width >= 1
        self.network = CNNNetwork(self.ml_params)

    def __call__(self, zeta, alpha_R, alpha_T):
        output = self.network(zeta, alpha_R, alpha_T)
        flux_R = output[...,0]
        flux_T = output[...,1]

        return flux_R, flux_T

class CNNNetwork(nn.Module):

    ml_params: ModelParams

    def setup(self):
        dtype = jnp.float64
        kernel_init = nn.initializers.lecun_normal(dtype=dtype)
        zeros_init = nn.initializers.zeros

        widths = [self.ml_params.width for _ in range(self.ml_params.depth-1)]

        kernel_inits = [kernel_init for _ in range(self.ml_params.depth-1)]

        self.layers = [
            nn.Conv(
                features=width,
                kernel_size=(self.ml_params.kernel_size, self.ml_params.kernel_size),
                padding="VALID",
                kernel_init=kernel_init,
                bias_init=zeros_init,
            )
            for width in widths
        ]
        self.final_layer = nn.Conv(
            features = 2,
            kernel_size=(self.ml_params.kernel_size, self.ml_params.kernel_size),
            padding="VALID",
            kernel_init = zeros_init,
            bias_init = zeros_init,
        )

        self.pad_width = sum([(self.ml_params.kernel_size-1)//2 for _ in range(self.ml_params.depth)])


    def pad(self, x, width):
        return jnp.pad(x, ((width, width), (width, width), (0,0)), "wrap")


    def __call__(self, zeta, alpha_R, alpha_T):
        x = jnp.concatenate((zeta[...,None], alpha_R[..., None], alpha_T[..., None]), axis=-1)

        x = self.pad(x, self.pad_width)

        for lyr in self.layers:
            x = lyr(x)
            x = nn.relu(x)

        x = self.final_layer(x)

        return x