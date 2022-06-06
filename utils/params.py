from typing import Tuple, List
from original_code.synflow.Utils import load


class ModelParams:
    model: str
    model_class: str
    lr: float
    lr_drop_rate: float 
    lr_drops: List
    weight_decay: float
    dense_classifier: bool
    pretrained: bool
    optimizer: str
    init_strategy: str
 
    def __init__(self, **kwargs):
        # Defaults
        self.model_class="lottery"
        self.model="lenet_300_100"
        self.lr = 0.1
        self.lr_drop_rate= 0.0
        self.lr_drops= []
        self.weight_decay = 0.0
        self.dense_classifier = False
        self.pretrained = False
        self.optimizer = "sgd" # SGD was used but could adam be better?
        self.init_strategy = "standard"

        self.__dict__.update(**kwargs)

class DataParams:
    dataset: str
    num_classes: int
    input_shape: Tuple[int, int, int]
    train_batch_size: int
    test_batch_size: int
    prune_batch_size: int
    workers: int
    prune_dataset_ratio: int

    def __init__(self, **kwargs):
        # Defaults
        self.dataset="mnist"
        self.train_batch_size = 64
        self.test_batch_size = 256
        self.prune_batch_size = 256
        self.workers = 4
        self.prune_dataset_ratio = 10

        self.__dict__.update(**kwargs)
        self.input_shape, self.num_classes = load.dimension(self.dataset) 

class TrainingPhase:
    train_epochs: int
    def __init__(self, train_epochs):
        self.train_epochs = train_epochs

class PruningPhase:
    strategy: str
    sparsity: float
    prune_epochs: int # for eg synflow, it is beneficial to iterate towards the desired sparsity

    # TODO: necessary?
    prune_bias: bool
    prune_batchnorm: bool
    prune_residual: bool
    compression_schedule: str
    mask_scope: str
    reinitialize: bool
    prune_train_mode: bool
    shuffle: bool
    invert: bool

    def __init__(
            self, 
            strategy = "mag", 
            sparsity = 0.2, 
            prune_epochs = 1,
            prune_bias = False,
            prune_batchnorm = False,
            prune_residual = False,
            compression_schedule = "exponential",
            mask_scope = "global",
            reinitialize = False,
            prune_train_mode = False,
            shuffle = False,
            invert = False,
            ):
        self.strategy = strategy 
        self.sparsity = sparsity 
        self.prune_epochs = prune_epochs
        self.prune_bias = prune_bias
        self.prune_batchnorm = prune_batchnorm
        self.prune_residual = prune_residual
        self.compression_schedule = compression_schedule
        self.mask_scope = mask_scope
        self.reinitialize = reinitialize
        self.prune_train_mode = prune_train_mode
        self.shuffle = shuffle
        self.invert = invert

class IterativePruningPhase:
    train_params: TrainingPhase
    prune_params: PruningPhase
    iterations: int
    rewind: bool
    rewind_iteration: int

    def __init__(self, train_params: dict = {}, prune_params: dict = {}, iterations: int = 10, rewind_iteration: int = 0, rewind: bool = True):
        self.train_params = train_params if isinstance(train_params, TrainingPhase) else TrainingPhase(**train_params)
        self.prune_params = prune_params if isinstance(prune_params, PruningPhase) else PruningPhase(**prune_params)
        self.iterations = iterations
        self.rewind_iteration = rewind_iteration
        self.rewind = rewind

def create_phase_params(**kwargs):
    if not "kind" in kwargs.keys():
        print("specify what kind of phase it should be!")
    kind = kwargs.pop("kind")
    if kind == "train":
        return TrainingPhase(**kwargs)
    elif kind == "prune":
        return PruningPhase(**kwargs)
    elif kind == "iterative":
        return IterativePruningPhase(**kwargs)
    else:
        print("invalid phase action: " + kind)