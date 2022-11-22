from dataclasses import dataclass

@dataclass
class TrainingParams:
	unique_id: str
	batch_size: int
	learning_rate: float
	num_epochs: float

@dataclass
class StencilParams(TrainingParams):
	stencil_width: int
	kernel_size: int
	depth: int
	width: int