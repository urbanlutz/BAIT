import os
import re
from datetime import datetime
from functools import partialmethod
from itertools import product
from typing import Dict, List, Union

import dill
from tqdm import tqdm
from utils.environment import ENV
from utils.monitor import MONITOR
from utils.params import (DataParams, IterativePruningPhase, ModelParams,
                          PruningPhase, TrainingPhase)
from utils.state import State, create_base_model, load_base_model_weights

from src.phase_actions import run_phase
from src.train import get_metrics
import logging
from itertools import product
from analysis.results_utils import load_csv

import traceback


REDO_ALL = False

LOGS_DIR = "./logs"
RESULTS_DIR = "./results"

NUM_INSTANCES = 3
REPEATS = 1 # repeats per instance  
INIT_STRATEGIES = ["standard", "snip", "synflow", "grasp", "bi-modal"]
# INIT_STRATEGIES = ["standard"]



baselines = ["snip", "synflow", "grasp", "mag", "rand"]
ensembles = ['PyTorchModel', 'RandomForest', "DecisionTree", "SGD", "LinearRegression"]

ensembles_plus = [f"{e}Plus" for e in ensembles]

stacked_basis = ["SynFlow", "SNIP", "GraSP", "Mag", "Rand"]
stackeds = [f"{a}{b}Stacked" for a, b in product(stacked_basis, stacked_basis)]

ALGOS = stackeds


# SPARSITIES = [0.004, 0.005, 0.007, 0.01, 0.015, 0.2, 0.5]
SPARSITIES = [0.1]

# PRUNE_EPOCHS = [1, 10]
PRUNE_EPOCHS = [10]
PRUNING_STRATEGIES = ["o"] #, "i"] #["o", "i", "c"] -> c = 90/10 combination of oneshot & iterative

MODEL = "lenet_300_100"
DATASET = "mnist"


def get_instance_id():
    instance = os.environ.get('INSTANCE_ID', "dev")
    if instance == "dev":
        return 0, True
    else:
        return int(instance), False
INSTANCE_NR, IS_DEV = get_instance_id()

# tidy up outputs if running in prod:
tqdm.__init__ = partialmethod(tqdm.__init__, disable=not IS_DEV)

logging.basicConfig(filename=f'{LOGS_DIR}/experiment.log', level=logging.WARNING)


class Experiment:
    name: str
    pruning_strategy: str
    model_params: ModelParams
    data_params: DataParams
    pruning_phases: List[Union[TrainingPhase, PruningPhase, IterativePruningPhase]]

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

class ExperimentExecution:
    experiment: Experiment
    state: State
    init_id: str
    repeat_nr: int
    instance: int

    total_iterations: int
    accuracy: float
    total_epochs: int
    time: str
    file_name: str
    
    def __init__(self, experiment, init_id, repeat_nr):
        self.experiment = experiment
        self.init_id = init_id
        self.repeat_nr = NUM_INSTANCES * INSTANCE_NR + repeat_nr
        self.time = re.sub("[-_\. :]", "", str(datetime.now()))
        self.file_name = f"{experiment.name}_{str(NUM_INSTANCES * INSTANCE_NR + repeat_nr)}_{self.time}"


def get_init(experiment: Experiment, repeat):
    name = experiment.name
    init_strategy = experiment.model_params.init_strategy
    repeat_nr = repeat

    try:
        df = load_csv()
        df = df.pivot_table(index = "base_model_id", columns=["name", "init_strategy", "repeat_nr"], values ="file", aggfunc="count").melt(ignore_index = False).dropna().reset_index()
        df = df.query(f"init_strategy == '{init_strategy}' and repeat_nr == {repeat_nr}")[["base_model_id", "name"]]
        df = df.groupby("base_model_id")["name"].apply(list).reset_index()
        df["count"] = df["name"].apply(lambda x: len(x))
        init_id = df[df["name"].apply(lambda x: name not in x)].sort_values("count", ascending = False).iloc[0]["base_model_id"]
        print(f"found suitable init! {init_id}" )
        return init_id

    except Exception as e:
        print(f"no suitable init found! {e}")
        # traceback.print_exc()
        return create_base_model(experiment.model_params, experiment.data_params)

def run_experiments(states = None):
    for repeat in range(REPEATS):

        for init_strategy in INIT_STRATEGIES:
            experiments = define_experiments(init_strategy)
            

            for experiment in experiments:
                init_id = get_init(experiment, repeat)
                if not REDO_ALL and alread_done(experiment.name, repeat):
                    print(f"skipping {experiment.name}, already done it before.")
                    continue
                try:
                    completed_experiment = run(experiment, init_id, repeat)
                    if not states is None: # ability to get states out of this function directly without having to collect them in a list in every case, optimizing memory usage..
                        states.append(completed_experiment)
                    save(completed_experiment)
                except Exception as e:
                    if IS_DEV:
                        raise e
                    else:
                        logging.exception(f"experiment {experiment.name} failed!")
    

def run(expr: Experiment, init_id: str, repeat_nr: int) -> ExperimentExecution:
    expr = ExperimentExecution(expr, init_id, repeat_nr)
    prepare_state(expr)
    run_phases(expr)
    return expr


def define_experiments(init_strategy: str) -> List[Experiment]:
    phase_definitions = {
        "o": lambda algo, sparsity, prune_epochs: [
                PruningPhase(
                    strategy=algo,
                    sparsity=sparsity,
                    prune_epochs=prune_epochs
                ),
                TrainingPhase(
                    train_epochs=100
                )
            ],
        "i": lambda algo, sparsity, prune_epochs: [
            IterativePruningPhase(
                prune_params = PruningPhase(
                    strategy=algo,
                    sparsity=sparsity,
                    prune_epochs=prune_epochs
                ),
                train_params = TrainingPhase(
                    train_epochs=100
                ),
                iterations = 10,
                rewind = True
                )
            ],
        "c": lambda algo, sparsity, prune_epochs: [
            PruningPhase(
                    strategy=algo,
                    sparsity= (1 - sparsity) * 0.1,
                    prune_epochs=prune_epochs
                ),
            IterativePruningPhase(
                prune_params = PruningPhase(
                    strategy=algo,
                    sparsity=sparsity,
                    prune_epochs=prune_epochs
                ),
                train_params = TrainingPhase(
                    train_epochs=100
                ),
                iterations = 10,
                rewind = True
                )
            ],

    }

    experiments = []
    for sparsity, algo, prune_epoch, pruning_strategy in product(SPARSITIES, ALGOS, PRUNE_EPOCHS, PRUNING_STRATEGIES):
        experiments.append(
            Experiment(
                name = f"{init_strategy}_{pruning_strategy}{prune_epoch}_{algo}_{sparsity}",
                pruning_strategy = pruning_strategy,
                prune_epochs = prune_epoch,
                model_params = ModelParams(model=MODEL, init_strategy=init_strategy),
                data_params = DataParams(dataset=DATASET),
                pruning_phases = phase_definitions[pruning_strategy](algo, sparsity, prune_epoch)
            )
        )
    return experiments

    
def prepare_state(expr: ExperimentExecution):
    s = State(
        expr.experiment.model_params,
        expr.experiment.data_params,
        expr.experiment
        )
    load_base_model_weights(expr.init_id, s.model)
    s.base_model_id = expr.init_id
    s.bake_initial_state()
    expr.state = s

def run_phases(expr: ExperimentExecution):
    MONITOR.start(expr.file_name)
    print(f"running {expr.file_name}")
    for index, pruning_phase in enumerate(expr.experiment.pruning_phases):
        print(f"phase {index + 1}, doing {pruning_phase.__class__.__name__}")
        run_phase(expr.state, pruning_phase)
    log_hparams(expr)


def save(expr: ExperimentExecution):
    expr.state.save(expr.file_name)
    with open(f"{RESULTS_DIR}/{expr.file_name}_dict.dill", "wb") as f:
        dill.dump(convert_to_result(expr), f)


def log_hparams(expr: ExperimentExecution):
    # Log hyperparams
    metrics = get_metrics(expr.state, ENV.device)
    
    hparams_phases = {}
    for j, phase in enumerate(expr.experiment.pruning_phases):
        for k, v in phase.__dict__.items():
            if hasattr(v, "__dict__"):
                hparams_phases.update({f"phase_{j}-{k}-{vk}":vv for vk, vv in v.__dict__.items()})
                hparams_phases.update({f"phase_{j}-type": str(v.__class__.__name__)})
            else:
                hparams_phases[f"phase_{j}-{k}"] = v
    hparams = {
        "name": expr.file_name, 
        "repeat": NUM_INSTANCES * INSTANCE_NR + expr.repeat_nr, 
        **expr.experiment.model_params.__dict__, 
        **expr.experiment.data_params.__dict__, 
        **hparams_phases
        }
    MONITOR.track_params(hparams, metrics, expr.file_name)

def get_sparsity(state:State) -> float:
    remaining_params, total_params = state.pruner.stats()
    return remaining_params / total_params

def convert_to_result(expr: ExperimentExecution) -> Dict:
    s = expr.state

    is_baseline = len([phase for phase in s.config.pruning_phases if not isinstance(phase, TrainingPhase)]) == 1

    phase0kind = s.config.pruning_phases[0].__class__.__name__
    r = {
        "file": expr.file_name,
        "name": expr.experiment.name,
        "repeat_nr": expr.repeat_nr,
        "model": expr.experiment.model_params.model,
        "init_strategy": expr.experiment.model_params.init_strategy,
        "dataset": expr.experiment.data_params.dataset,
        "base_model_id": s.base_model_id,
        "final_sparsity": get_sparsity(s),
        "accuracy": s.best_accuracy,
        "is_baseline": is_baseline,
        "baseline_type": None if not is_baseline else phase0kind,
        "total_iterations": s.total_iterations,
        "total_epochs": s.total_epochs,
        "type_short": f"{expr.experiment.pruning_strategy}{expr.experiment.prune_epochs}"
    }

    for i, phase in enumerate(s.config.pruning_phases):
        d = {}
        if isinstance(phase, PruningPhase):
            p = phase
            t = None
        elif isinstance(phase, IterativePruningPhase):
            p = phase.prune_params
            t = phase.train_params
            d.update({
                f"iterations": phase.iterations,
                f"rewind_iteration": phase.rewind_iteration,
                f"rewind": phase.rewind,
            })
        elif isinstance(phase, TrainingPhase):
            t = phase
            p = None

        if p:
            d.update({
                f"strategy": p.strategy,
                f"sparsity": p.sparsity,
                f"prune_epochs": p.prune_epochs,
            })
        if t:
            d.update({
                f"train_epochs": t.train_epochs,
            })
        d.update({f"type": f"{phase.__class__.__name__}",})

        r.update({f"phase_{i}_{k}": v for k, v in d.items()})
    return r

def alread_done(experiment_name, repeat_nr):

    return f"{experiment_name}_{str(INSTANCE_NR * NUM_INSTANCES + repeat_nr)}" in completed_experiemnts()

def completed_experiemnts():
    return [file.rsplit("_", 2)[0] for file in os.listdir(RESULTS_DIR)]
