from dataclasses import dataclass

@dataclass
class TrainingParams:
	n_data: int
	num_epochs: int
	train_id: str
	batch_size: int
	learning_rate: float
	optimizer: str

	def __post_init__(self):
		self.num_training_iterations = (self.n_data // self.batch_size) * self.num_epochs

class TrainingParamsUnroll(TrainingParams):

	def __init__(self, n_unroll, *args):
		super().__init__(*args)
		self.n_unroll = n_unroll
	

@dataclass
class StencilParams:
	kernel_size: int
	kernel_out: int
	stencil_width: int
	depth: int
	width: int