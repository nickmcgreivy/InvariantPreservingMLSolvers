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
        self.num_training_iterations = (
            self.n_data // self.batch_size
        ) * self.num_epochs


@dataclass
class ModelParams:
    kernel_size: int
    kernel_out: int
    depth: int
    width: int
