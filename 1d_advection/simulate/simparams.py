from dataclasses import dataclass

from rungekutta import FUNCTION_MAP
from flux import Flux

@dataclass
class CoreParams:
	Lx: float
	fluxstr: str

	def __post_init__(self):
		self.flux = Flux(self.fluxstr)
		self.order = None


class CoreParamsDG(CoreParams):

	def __init__(self, Lx, flux, order):
		super().__init__(Lx, flux)
		self.order = order


@dataclass
class SimulationParams:
	name: str
	basedir: str
	readwritedir: str
	cfl_safety: float
	rk: str

	def __post_init__(self):
		self.rk_fn = FUNCTION_MAP[self.rk]
		