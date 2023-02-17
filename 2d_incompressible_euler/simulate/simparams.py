from dataclasses import dataclass
import jax.numpy as jnp

from rungekutta import FUNCTION_MAP
from flux import Flux

@dataclass
class SimulationParams:
    name: str
    nx: int
    ny: int
    Lx: float
    Ly: float
    dx: float
    dy: float
    cfl_safety: float
    rk: str


class FiniteVolumeSimulationParams(SimulationParams):

    def __init__(self, name, basedir, readwritedir, nx, ny, Lx, Ly, cfl_safety, rk, flux, global_stabilization=False):
        super().__init__(name, nx, ny, Lx, Ly, Lx / nx, Ly / ny, cfl_safety, rk)
        self.basedir = basedir
        self.readwritedir = readwritedir
        self.flux = flux
        self.global_stabilization = global_stabilization
        self.rk_fn = FUNCTION_MAP[self.rk]