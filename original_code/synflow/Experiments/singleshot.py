import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from original_code.synflow.Utils import generator
from original_code.synflow.Utils import load
from original_code.synflow.Utils import metrics
from original_code.synflow.train import *
from original_code.synflow.prune import *


def prepare_data(state, args):
    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    _, num_classes = load.dimension(args.dataset) 
    state.prune_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.prune_dataset_ratio * num_classes)
    state.train_loader = load.dataloader(args.dataset, args.train_batch_size, True, args.workers)
    state.test_loader = load.dataloader(args.dataset, args.test_batch_size, False, args.workers)

def prepare_platform(state, args):
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    state.device = load.device(args.gpu)

def prepare_model(state, args):
    state.input_shape, state.num_classes = load.dimension(args.dataset) 
     ## Model, Loss, Optimizer ##
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    state.model = load.model(args.model, args.model_class)(state.input_shape, 
                                                     state.num_classes, 
                                                     args.dense_classifier, 
                                                     args.pretrained).to(state.device)
    state.loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    state.optimizer = opt_class(generator.parameters(state.model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    state.scheduler = torch.optim.lr_scheduler.MultiStepLR(state.optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)


def prepare(state, args):
    # prepare
    prepare_platform(state, args)
    prepare_data(state, args)
    prepare_model(state, args)


def train(state, args, epochs):
    print('Train for {} epochs.'.format(epochs))
    result = train_eval_loop(state.model, state.loss, state.optimizer, state.scheduler, state.train_loader, state.test_loader, state.device, epochs, args.verbose)
    return result

# def pretrain(state, args):
#     ## Pre-Train ##
#     print('Pre-Train for {} epochs.'.format(args.pre_epochs))
#     pre_result = train_eval_loop(state.model, state.loss, state.optimizer, state.scheduler, state.train_loader, state.test_loader, state.device, args.pre_epochs, args.verbose)
#     return pre_result


def prune(state: State, args, strategy, sparsity):
     ## Prune ##
    print('Pruning with {} for {} epochs.'.format(strategy, args.prune_epochs))
    pruner = load.pruner(strategy)(generator.masked_parameters(state.model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
    
    prune_loop(state.model, state.loss, pruner, state.prune_loader, state.device, sparsity, 
               args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize, args.prune_train_mode, args.shuffle, args.invert)
    prune_result = metrics.summary(state.model, 
                                pruner.scores,
                                metrics.flop(state.model, state.input_shape, state.device),
                                lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))
    return prune_result


# def posttrain(state, args):
#     ## Post-Train ##
#     print('Post-Training for {} epochs.'.format(args.post_epochs))
#     post_result = train_eval_loop(state.model, state.loss, state.optimizer, state.scheduler, state.train_loader, state.test_loader, state.device, args.post_epochs, args.verbose) 
#     return post_result

def display(pre_result, prune_result, post_result):
    ## Display Results ##
    frames = [pre_result.head(1), pre_result.tail(1), post_result.head(1), post_result.tail(1)]
    train_result = pd.concat(frames, keys=['Init.', 'Pre-Prune', 'Post-Prune', 'Final'])
    total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
    possible_params = prune_result['size'].sum()
    total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
    possible_flops = prune_result['flops'].sum()
    print("Train results:\n", train_result)
    print("Prune results:\n", prune_result)
    print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
    print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))

def save(state, args, pre_result, prune_result, post_result):
    ## Save Results and Model ##
    print('Saving results.')
    pre_result.to_pickle("{}/pre-train.pkl".format(args.result_dir))
    post_result.to_pickle("{}/post-train.pkl".format(args.result_dir))
    prune_result.to_pickle("{}/compression.pkl".format(args.result_dir))
    torch.save(state.model.state_dict(),"{}/model.pt".format(args.result_dir))
    torch.save(state.optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
    torch.save(state.scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))

def run(state, args):
    prepare(state, args)

    pre_result = train(state, args, args.pre_epochs)
    prune_result = prune(state, args)
    post_result = train(state, args,args.post_epochs)
    display(pre_result, prune_result, post_result)
    if args.save:
        save(state, args)
