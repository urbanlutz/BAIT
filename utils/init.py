import torch
import torch.nn as nn
from original_code.synflow.Layers import layers
import torch.nn.functional as F
from original_code.synflow.Utils import load
from original_code.synflow.Utils import generator
from utils.environment import ENV
from utils.params import PruningPhase

def standard_init(s):
    modules = s.model.modules()
    for m in modules:
        if isinstance(m, (layers.Linear, nn.Linear, layers.Conv2d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, layers.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def bimodal_init(s):
    modules = s.model.modules()
    import dill
    with open("./artefacts/kde.dill", "rb") as f:
        kde = dill.load(f)

    for m in modules:
        if isinstance(m, (layers.Linear, nn.Linear, layers.Conv2d)):
            m.weight.data.copy_(torch.from_numpy(kde.sample(m.weight.numel())).float().reshape(m.weight.size()))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, layers.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)



def snip_init_multiply(s):
    pruningscore_init_multiply(s, "snip")

def synflow_init_multiply(s):
    pruningscore_init_multiply(s, "synflow")
    
def grasp_init_multiply(s):
    pruningscore_init_multiply(s, "grasp")

def pruningscore_init_multiply(s, algo):
    standard_init(s) # necessary in order to score something
    prune_params = PruningPhase(strategy=algo)
    pruner = load.pruner(
        prune_params.strategy
        )(
            generator.masked_parameters(
                s.model, 
                prune_params.prune_bias, 
                prune_params.prune_batchnorm, 
                prune_params.prune_residual
            )
        )
    pruner.score(s.model, s.loss, s.train_loader, ENV.device)
    for (_, score),(_, p) in zip(pruner.scores.items(), pruner.masked_parameters):
        score -= score.min()
        score /= score.max()
        p.data.copy_(score * p)

def snip_init_keepsign(s):
    pruning_score_keepsign(s, "snip")

def grasp_init_keepsign(s):
    pruning_score_keepsign(s, "grasp")
def synflow_init_keepsign(s):
    pruning_score_keepsign(s, "synflow")

def pruning_score_keepsign(s, algo):
    standard_init(s) # necessary in order to score something
    prune_params = PruningPhase(strategy=algo)
    pruner = load.pruner(
        prune_params.strategy
        )(
            generator.masked_parameters(
                s.model, 
                prune_params.prune_bias, 
                prune_params.prune_batchnorm, 
                prune_params.prune_residual
            )
        )
    pruner.score(s.model, s.loss, s.train_loader, ENV.device)
    for (_, score),(_, p) in zip(pruner.scores.items(), pruner.masked_parameters):
        signs = torch.sign(p)
        score -= score.min()
        score /= score.max()
        score_negative = score.abs_() * signs
        p.data.copy_(score_negative)



inits = {
    "standard": standard_init,
    "bi-modal": bimodal_init,
    "synflow": synflow_init_keepsign,
    "snip": snip_init_keepsign,
    "grasp": grasp_init_keepsign,
    "synflow_multiply": synflow_init_multiply,
    "snip_multiply": snip_init_multiply,
    "grasp_multiply": grasp_init_multiply,
    }

def initialize(s):
    inits[s.model.init_strategy](s)
    