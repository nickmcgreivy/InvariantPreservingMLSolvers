from dataclasses import dataclass

from rungekutta import FUNCTION_MAP
from flux import Flux
from boundaryconditions import BoundaryCondition


@dataclass
class CoreParams:
    Lx: float
    gamma: float
    boundary_condition: str
    fluxstr: str

    def __post_init__(self):
        self.flux = Flux(self.fluxstr)
        self.bc = BoundaryCondition(self.boundary_condition)


@dataclass
class SimulationParams:
    name: str
    basedir: str
    readwritedir: str
    cfl_safety: float
    rk: str

    def __post_init__(self):
        self.rk_fn = FUNCTION_MAP[self.rk]
