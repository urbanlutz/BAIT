"""
Implementations for the 3 possible pruning phases: no pruning (regular training), one_shot pruning and iterative pruning
"""

import numpy as np

from tqdm import tqdm
from utils.environment import ENV
from utils.monitor import MONITOR
from utils.params import IterativePruningPhase, PruningPhase, TrainingPhase
from utils.state import State

from src.train import train_eval_loop

def run_phase(state, phase_params):
    mapping = {
        TrainingPhase: regular_train,
        PruningPhase: one_shot_pruning,
        IterativePruningPhase: iterative_pruning
    }
    return mapping[type(phase_params)](state, phase_params)


def regular_train(state: State, train_params: TrainingPhase) -> None:
    """no pruning, only training for train_epochs number of epochs or until early stopping"""
    print(f"regular training for {train_params.train_epochs} epochs")
    train_eval_loop(state, ENV.device, train_params.train_epochs)

def one_shot_pruning(state: State, prune_params: PruningPhase) -> None:
    """only pruning, no training"""
    print(f"one-shot pruning with {prune_params.strategy} until {prune_params.sparsity} sparsity")

    state.set_pruner(prune_params)
    current_sparsity = _current_sparsity(state)
    
    if current_sparsity < prune_params.sparsity:
        print(f"nothing to prune: currently at {current_sparsity}, target is {prune_params.sparsity}")
        return
    
    prune_loop(state, ENV.device, prune_params.sparsity, prune_params.compression_schedule, prune_params.mask_scope, epochs=prune_params.prune_epochs)
    # prune_loop(state.model, state.loss, state.pruner, state.prune_loader, ENV.device, prune_params.sparsity, 
    #            prune_params.compression_schedule, prune_params.mask_scope, epochs=1)
    current_sparsity = _current_sparsity(state)
    MONITOR.track("Sparsity/Current Sparsity", current_sparsity, state.total_iterations)


def iterative_pruning(state:State, iterative_prune_params: IterativePruningPhase) -> None:
    """training and pruning"""
    prune_params = iterative_prune_params.prune_params
    train_params = iterative_prune_params.train_params
    print(f"iterative pruning with {prune_params.strategy} until {prune_params.sparsity} sparsity in {iterative_prune_params.iterations} steps with max {train_params.train_epochs} epochs of training")

    state.set_pruner(prune_params)
    current_sparsity = _current_sparsity(state)
    print(f"starting with {current_sparsity * 100}% weights remaining")

    sparsity_targets = list(np.linspace(current_sparsity, prune_params.sparsity, iterative_prune_params.iterations + 1))[1:]


    state.register_callback("total_iterations", iterative_prune_params.rewind_iteration + state.total_iterations, state.bake_rewind_state)
    state.trigger_callbacks()
    for i, sparsity in enumerate(sparsity_targets):


        # train
        regular_train(state, train_params)

        # prune
        current_sparsity = _current_sparsity(state)
        MONITOR.track("Sparsity/Current Sparsity", current_sparsity, state.total_iterations)
        
        if sparsity < 1:
            prune_params.sparsity = sparsity
            # prune_result = run.prune(state, prune_params, data_params)
            prune_loop(state, ENV.device, prune_params.sparsity, 
               prune_params.compression_schedule, prune_params.mask_scope, prune_params.prune_epochs, prune_params.reinitialize, prune_params.prune_train_mode, prune_params.shuffle, prune_params.invert)

        # TODO: onvert to callback
        # rewind
        if (
            iterative_prune_params.rewind 
            and state.total_iterations >= iterative_prune_params.rewind_iteration
            and i < len(sparsity_targets)-1):
            print(f"rewinding state at iteration {state.total_iterations }")
            _rewind_weights(state)

        # display
        print(f"pruned to {_current_sparsity(state) * 100}%")
        
    regular_train(state, train_params)

def _rewind_weights(state: State):
    state.reset(state.state_to_rewind)


def _current_sparsity(state:State) -> float:
    remaining_params, total_params = state.pruner.stats()
    return remaining_params / total_params

def prune_loop(state: State, device, sparsity, schedule, scope, epochs,
               reinitialize=False, train_mode=False, shuffle=False, invert=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    # Set model to train or eval mode
    state.model.train()
    if not train_mode:
        state.model.eval()

    # Prune model
    for epoch in tqdm(range(epochs)):
        state.pruner.score(state.model, state.loss, state.prune_loader, device)
        if schedule == 'exponential':
            sparse = sparsity**((epoch + 1) / epochs)
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs)
        # Invert scores
        if invert:
            state.pruner.invert()
        state.pruner.mask(sparse, scope)
    
    # Reainitialize weights
    if reinitialize:
        state.model._initialize_weights()

    # Shuffle masks
    if shuffle:
        state.pruner.shuffle()

    # Confirm sparsity level
    remaining_params, total_params = state.pruner.stats()
    if np.abs(remaining_params - total_params*sparsity) >= 5:
        raise Exception("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params*sparsity))
