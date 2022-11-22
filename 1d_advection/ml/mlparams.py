from dataclasses import dataclass

@dataclass
class TrainingParams:
	n_data: int
	num_epochs: int
	unique_id: str
	batch_size: int
	learning_rate: float
	

@dataclass
class StencilParams:
	kernel_size: int
	kernel_out: int
	stencil_width: int
	depth: int
	width: int