from collections import defaultdict
from typing import Callable, Dict, Union
import torch
from utils.init import initialize
from utils.params import DataParams, IterativePruningPhase, ModelParams, PruningPhase, TrainingPhase
from original_code.synflow.Pruners.pruners import Pruner
from original_code.synflow.Utils import load
from original_code.synflow.Utils import generator
from utils.environment import ENV
import dill
import os
from uuid import uuid1

TORCH_SUFFIX = "pt"
PICKLE_SUFFIX = "pkl"
DEFAULT_SAVE_LOCATION = "./saves"
DEFAULT_BASE_MODEL_PATH = f"base_models"

import dill
PICKLE_MODULE = dill

class State:
    config: Dict
    prune_loader = None
    train_loader = None
    test_loader = None
    model: torch.nn.Module
    loss: torch.nn.modules.loss._Loss
    scheduler: torch.optim.lr_scheduler.MultiStepLR 
    optimizer = torch.optim.Optimizer 
    pruner: Pruner
    initial_state: 'State' = None
    state_to_rewind: 'State' = None

    total_iterations = 0
    total_epochs = 0
    current_iterations = 0
    current_epochs = 0
    best_accuracy = 0

    callbacks: Dict

    base_model_id: str = None
    initialization_id: str = None

    def __init__(
        self, 
        model_params: ModelParams = None, 
        data_params: DataParams = None,
        config = None,
        init_id = None
        ):
        if data_params: prepare_data(self, data_params)
        if model_params and data_params: prepare_model(self, model_params, data_params)

        self.config = config
        self.callbacks = {k: defaultdict(lambda: []) for k in ["total_iterations", "total_epochs", "current_iterations", "current_epochs"]} 

    def set_pruner(self, prune_params: Union[TrainingPhase, PruningPhase, IterativePruningPhase]):
        if isinstance(prune_params, TrainingPhase):
            print("no need to prepare pruner for train phase")
        else:
            prepare_pruner(self, prune_params)

    def bake_initial_state(self):
        self.initial_state = self.copy()

    def bake_rewind_state(self):
        if not self.state_to_rewind:
            self.state_to_rewind = self.copy()
        else:
            print("bake rewind state called but has already been set")

    def copy(self):
        """no recursive states! no data!"""
        relevant_fields = {k:v for k, v in self.__dict__.items() if (
            not isinstance(v, State) 
            and not isinstance(v, torch.utils.data.dataloader.DataLoader)
            and k != "callbacks"
            )}
        tmp_state = State()
        tmp_state.__dict__.update(relevant_fields)
        return dill.copy(tmp_state)

    def save(self, name:str):
        suffix = "" if name.endswith(".pt") else ".pt"
        file_path = f"{DEFAULT_SAVE_LOCATION}/{name}{suffix}"
        relevant_fields = {k:v for k, v in self.__dict__.items() if (
            not isinstance(v, torch.utils.data.dataloader.DataLoader)
            and k != "callbacks"
            )}
        torch.save(relevant_fields, file_path, pickle_module=dill)

    def load(self, name:str, load_data = True):
        suffix = "" if name.endswith(".pt") else ".pt"
        file_path = f"{DEFAULT_SAVE_LOCATION}/{name}{suffix}"
        self.__dict__.update(**torch.load(file_path, pickle_module=dill))
        if load_data:
            prepare_data(self, self.config.data_params)
        
        # torch 1.7 to 1.11 compatibility:
        self.loss.label_smoothing = 0.0


    def reset(self, new_state:'State'):
        original_weights = {k:v for k, v in new_state.model.state_dict().items() if k.endswith(('.weight', '.bias'))}
        model_dict = self.model.state_dict()
        model_dict.update(original_weights)
        self.model.load_state_dict(model_dict)

        self.optimizer.load_state_dict(new_state.optimizer.state_dict())
        self.scheduler.load_state_dict(new_state.scheduler.state_dict())
        self.current_epochs = 0
        self.current_iterations = 0

    def register_callback(self, type: str, iteration: int, func: Callable[[None], None]):
        self.callbacks[type][iteration].append(func)
        

    def trigger_callbacks(self):
        for counter in ["total_iterations", "total_epochs", "current_iterations", "current_epochs"]:
            for action in self.callbacks[counter][getattr(self, counter)]:
                print(f"callback {action.__name__} at iteration {getattr(self, counter)}")
                action()

    def inc_iterations(self):
        self.current_iterations += 1
        self.total_iterations += 1
        self.trigger_callbacks()

    def inc_epochs(self):
        self.current_epochs += 1
        self.total_epochs += 1
        self.trigger_callbacks()

    def set_best_accuracy(self, accuracy):
        # if accuracy > self.best_accuracy:
        #     self.best_accuracy = accuracy
        self.best_accuracy = accuracy

def get_all_saved_states():
    return [file for file in os.listdir(DEFAULT_SAVE_LOCATION) if not os.path.isdir(f"{DEFAULT_SAVE_LOCATION}/{file}")]

def prepare_data(state: State, params: DataParams):
    print(f'Loading {params.dataset} dataset.')

    state.prune_loader = load.dataloader(params.dataset, params.prune_batch_size, True, params.workers, params.prune_dataset_ratio * params.num_classes)
    state.train_loader = load.dataloader(params.dataset, params.train_batch_size, True, params.workers)
    state.test_loader = load.dataloader(params.dataset, params.test_batch_size, False, params.workers)

def prepare_model(state: State, model_params: ModelParams, data_params: DataParams):
    print(f'Creating {model_params.model_class}-{model_params.model} model.')

    state.model = load.model(
        model_params.model, 
        model_params.model_class
        )(
            data_params.input_shape, 
            data_params.num_classes, 
            model_params.dense_classifier, 
            model_params.pretrained,
            model_params.init_strategy
        ).to(ENV.device)
    state.loss = torch.nn.CrossEntropyLoss()
    initialize(state)
    opt_class, opt_kwargs = load.optimizer(model_params.optimizer)
    state.optimizer = opt_class(generator.parameters(state.model), lr=model_params.lr, weight_decay=model_params.weight_decay, **opt_kwargs)
    state.scheduler = torch.optim.lr_scheduler.MultiStepLR(state.optimizer, milestones=model_params.lr_drops, gamma=model_params.lr_drop_rate)


def prepare_pruner(state:State, prune_params: Union[PruningPhase, IterativePruningPhase]):
    if isinstance(prune_params, IterativePruningPhase):
        prune_params = prune_params.prune_params
    state.pruner = load.pruner(prune_params.strategy)(generator.masked_parameters(state.model, prune_params.prune_bias, prune_params.prune_batchnorm, prune_params.prune_residual))

def create_base_model(model_params: ModelParams, data_params: DataParams):
    s = State(model_params, data_params)
    prepare_model(s, model_params, data_params)

    base_model_id = str(uuid1())

    torch.save(s.model, f"{DEFAULT_SAVE_LOCATION}/{DEFAULT_BASE_MODEL_PATH}/{base_model_id}.pt", pickle_module=dill)
    return base_model_id



def load_base_model_weights(base_model_id, model):
    base_model = torch.load(f"{DEFAULT_SAVE_LOCATION}/{DEFAULT_BASE_MODEL_PATH}/{base_model_id}.pt", pickle_module=dill)
    model.load_state_dict(base_model.state_dict())

def get_base_model_strategy(base_model_id):
    try:
        return torch.load(f"{DEFAULT_SAVE_LOCATION}/{DEFAULT_BASE_MODEL_PATH}/{base_model_id}.pt", pickle_module=dill).init_strategy
    except:
        # default
        return "standard"

def get_all_existing_basemodel_ids():
    base_model_path = f"{DEFAULT_SAVE_LOCATION}/{DEFAULT_BASE_MODEL_PATH}/"
    for file in os.listdir(base_model_path):
        if not os.path.isdir(file):
            yield file.split(".")[0]
