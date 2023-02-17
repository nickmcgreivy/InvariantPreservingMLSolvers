import torch
import math
import jax.numpy as jnp
import jax
from jax import config
import jax_cfd.base as jax_cfd
import jax_cfd.base.grids as grids
config.update("jax_enable_x64", True)


class GaussianRF(object):

    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff

        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real



def init_fn_FNO():
    GRF = GaussianRF(2, 256, alpha=2.5, tau=7)
    return jnp.asarray(GRF.sample(1)[0])


def init_sum_sines(key):

    Lx = 2 * PI
    Ly = 2 * PI

    max_k = 5
    min_k = 1
    num_init_modes = 6
    amplitude_max = 4.0

    def sum_modes(x, y, amplitudes, ks_x, ks_y, phases_x, phases_y):
        return jnp.sum(
            amplitudes[None, :]
            * jnp.sin(
                ks_x[None, :] * 2 * PI / Lx * x[:, None] + phases_x[None, :]
            ) * jnp.sin(
                ks_y[None, :] * 2 * PI / Ly * y[:, None] + phases_y[None, :]
            ),
            axis=1,
        )

    key1, key2, key3, key4, key5 = jax.random.split(key, 5)
    phases_x = jax.random.uniform(key1, (num_init_modes,)) * 2 * PI
    phases_y = jax.random.uniform(key2, (num_init_modes,)) * 2 * PI
    ks_x = jax.random.randint(
        key3, (num_init_modes,), min_k, max_k
    )
    ks_y = jax.random.randint(
        key4, (num_init_modes,), min_k, max_k
    )
    amplitudes = jax.random.uniform(key5, (num_init_modes,)) * amplitude_max
    return lambda x, y, t: sum_modes(x, y, amplitudes, ks_x, ks_y, phases_x, phases_y)



def get_u0_jax_cfd(key, sim_params, max_velocity, ic_wavenumber):
    grid = grids.Grid((sim_params.nx, sim_params.ny), domain=((0, sim_params.Lx), (0, sim_params.Ly)))
    return jax_cfd.initial_conditions.filtered_velocity_field(key, grid, max_velocity, ic_wavenumber)


def vorticity(u):
    return jax_cfd.finite_differences.curl_2d(u).data


def init_fn_jax_cfd(*args):
    u0 = get_u0_jax_cfd(*args)
    return vorticity(u0)