from shutil import ExecError
from typing import OrderedDict
import torch
import numpy as np
import dill

class Pruner:
    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally.
        """
        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        global_layer_i = torch.cat([torch.tensor(i).repeat(v.numel()) for i, v in enumerate(self.scores.values())])

        k = global_scores.numel() - int(sparsity * global_scores.numel()) # number of elements

        _, global_topk_i = global_scores.topk(k, 0, largest=False)

        global_topk_layer_i = global_layer_i.cuda().gather(0, global_topk_i.cuda())
        index_offset = 0
        if not k < 1:
            for i, (mask, param) in enumerate(self.masked_parameters):
                index = torch.masked_select(global_topk_i, global_topk_layer_i == i)
                index = index - index_offset
                new_mask = torch.ones_like(param).flatten().index_fill(0, index, 0.).reshape(param.size()).to(mask.device)
                mask.copy_(new_mask)
                index_offset += param.numel()
                
        remaining_params, total_params = self.stats()
        expected = int(total_params*sparsity)
        if expected != remaining_params:
            raise Exception(f"expected {expected} but {remaining_params} params remain!")
    
    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise.
        """
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        if scope == 'global':
            self._global_mask(sparsity)
        if scope == 'local':
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, param in self.masked_parameters:
            shape = mask.shape
            perm = torch.randperm(mask.nelement())
            mask = mask.reshape(-1)[perm].reshape(shape)

    def invert(self):
        for v in self.scores.values():
            v.div_(v**2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        remaining_params, total_params = 0, 0 
        for mask, _ in self.masked_parameters:
             remaining_params += mask.detach().cpu().numpy().sum()
             total_params += mask.numel()
        return remaining_params, total_params


class Rand(Pruner):
    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.randn_like(p)


class Mag(Pruner):
    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)
    
    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, masked_parameters):
        super(GraSP, self).__init__(masked_parameters)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, loss, dataloader, device):
        from copy import deepcopy

        loss = deepcopy(loss)
        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=False)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

        # second gradient vector with computational graph
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=True)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            
            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()
        
        # calculate score Hg * theta (negate to remove top percent)
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


class SynFlow(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
      
        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        
        signs = linearize(model)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0,:].shape)
        input = torch.ones([1] + input_dim).to(device)#, dtype=torch.float64).to(device)
        output = model(input)
        torch.sum(output).backward()
        
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)

class Ensemble(Pruner):
    def __init__(self, masked_parameters):
        super(Ensemble, self).__init__(masked_parameters)


    def score(self, model, loss, dataloader, device):

        scorers = [cls(self.masked_parameters) for cls in [SynFlow, GraSP, SNIP, Mag]]
        [s.score(model, loss, dataloader, device) for s in scorers]
        for _, p in self.masked_parameters:
            all_relevant_scores = [score for s in scorers for id_p, score in s.scores.items() if id_p == id(p)]
            # normalize
            normalized_scores = [(score- score.mean())/score.std() for score in all_relevant_scores]
            self.scores[id(p)] = sum(normalized_scores)/len(scorers)

class PyTorchModelPruner(Pruner):
    def __init__(self, masked_parameters):
        super(PyTorchModelPruner, self).__init__(masked_parameters)
        self.model = torch.load("artefacts/pytorch-mlp.pt")
        self.model.eval()

        self.scorers = [cls(self.masked_parameters) for cls in [Mag, SNIP, GraSP, SynFlow]] #['mag', 'snip', 'grasp', 'synflow']:

    def score(self, model, loss, dataloader, device):
        [s.score(model, loss, dataloader, device) for s in self.scorers]
        for _, p in self.masked_parameters:
            all_features = []
            for s in self.scorers:
                score = s.scores[id(p)]
                score = score.flatten()
                score -= score.min()
                score /= score.max()
                all_features.append(score)
            
            # inits
            all_features.append(torch.clone(p).flatten())
            # norm
            all_features.append(torch.torch.norm(p, "fro").repeat(p.numel()).reshape(p.size()).flatten().to(device))
            # maxpool
            l = torch.nn.MaxPool1d(p.size()[1])
            t2 = torch.clone(p)
            all_features.append(l(t2.detach().cpu()).repeat(1,p.size()[1]).flatten().to(device))

            l2 = torch.nn.MaxPool1d(p.size()[0])
            t2 = torch.clone(p)
            all_features.append(l2(t2.detach().cpu().transpose_(0,1)).repeat(1, p.size()[0]).transpose_(0,1).flatten().to(device))

            # avg pool
            l = torch.nn.AvgPool1d(p.size()[1])
            t2 = torch.clone(p)
            all_features.append(l(t2.detach().cpu()).repeat(1,p.size()[1]).flatten().to(device))

            l2 = torch.nn.AvgPool1d(p.size()[0])
            t2 = torch.clone(p)
            all_features.append(l2(t2.detach().cpu().transpose_(0,1)).repeat(1, p.size()[0]).transpose_(0,1).flatten().to(device))



            scores_stack = torch.stack(all_features).transpose(0,1)
            scores_stack = np.nan_to_num(scores_stack.detach().cpu())
            y_pred = self.model(torch.from_numpy(scores_stack))
            score = y_pred.reshape(p.size()).to(device)
            self.scores[id(p)] = score
class PurePyTorchModelPruner(Pruner):
    def __init__(self, masked_parameters):
        super(PurePyTorchModelPruner, self).__init__(masked_parameters)
        self.model = torch.load("artefacts/PurePytorch-mlp.pt")
        self.model.eval()

        self.scorers = [cls(self.masked_parameters) for cls in [Mag, SNIP, GraSP, SynFlow]] #['mag', 'snip', 'grasp', 'synflow']:

    def score(self, model, loss, dataloader, device):
        [s.score(model, loss, dataloader, device) for s in self.scorers]
        for _, p in self.masked_parameters:
            all_features = []
            for s in self.scorers:
                score = s.scores[id(p)]
                score = score.flatten()
                score -= score.min()
                score /= score.max()
                all_features.append(score)
            
            scores_stack = torch.stack(all_features).transpose(0,1)
            scores_stack = np.nan_to_num(scores_stack.detach().cpu())
            y_pred = self.model(torch.from_numpy(scores_stack))
            score = y_pred.reshape(p.size()).to(device)
            self.scores[id(p)] = score

class SklearnModelPruner(Pruner):
    def __init__(self, masked_parameters, model_file):
        super(SklearnModelPruner, self).__init__(masked_parameters)
        with open(model_file, "rb") as f: # random forest works best
            self.rfr = dill.load(f)
        self.scorers = [cls(self.masked_parameters) for cls in [Mag, SNIP, GraSP, SynFlow]] #['mag', 'snip', 'grasp', 'synflow']:

    def score(self, model, loss, dataloader, device):
        [s.score(model, loss, dataloader, device) for s in self.scorers]
        for _, p in self.masked_parameters:
            all_features = []
            for s in self.scorers:
                score = s.scores[id(p)]
                score = score.flatten()
                score -= score.min()
                score /= score.max()
                all_features.append(score)
            
            # inits
            all_features.append(torch.clone(p).flatten())
            # norm
            all_features.append(torch.torch.norm(p, "fro").repeat(p.numel()).reshape(p.size()).flatten().to(device))
            # maxpool
            l = torch.nn.MaxPool1d(p.size()[1])
            t2 = torch.clone(p)
            all_features.append(l(t2.detach().cpu()).repeat(1,p.size()[1]).flatten().to(device))

            l2 = torch.nn.MaxPool1d(p.size()[0])
            t2 = torch.clone(p)
            all_features.append(l2(t2.detach().cpu().transpose_(0,1)).repeat(1, p.size()[0]).transpose_(0,1).flatten().to(device))

            # avg pool
            l = torch.nn.AvgPool1d(p.size()[1])
            t2 = torch.clone(p)
            all_features.append(l(t2.detach().cpu()).repeat(1,p.size()[1]).flatten().to(device))

            l2 = torch.nn.AvgPool1d(p.size()[0])
            t2 = torch.clone(p)
            all_features.append(l2(t2.detach().cpu().transpose_(0,1)).repeat(1, p.size()[0]).transpose_(0,1).flatten().to(device))



            scores_stack = torch.stack(all_features).transpose(0,1)
            scores_stack = np.nan_to_num(scores_stack.detach().cpu())
            y_pred = torch.from_numpy(self.rfr.predict(scores_stack))
            score = y_pred.reshape(p.size()).to(device)
            self.scores[id(p)] = score
class PureSklearnModelPruner(Pruner):
    def __init__(self, masked_parameters, model_file):
        super(PureSklearnModelPruner, self).__init__(masked_parameters)
        with open(model_file, "rb") as f: # random forest works best
            self.rfr = dill.load(f)
        self.scorers = [cls(self.masked_parameters) for cls in [Mag, SNIP, GraSP, SynFlow]] #['mag', 'snip', 'grasp', 'synflow']:

    def score(self, model, loss, dataloader, device):
        [s.score(model, loss, dataloader, device) for s in self.scorers]
        for _, p in self.masked_parameters:
            all_features = []
            for s in self.scorers:
                score = s.scores[id(p)]
                score = score.flatten()
                score -= score.min()
                score /= score.max()
                all_features.append(score)
            
        

            scores_stack = torch.stack(all_features).transpose(0,1)
            scores_stack = np.nan_to_num(scores_stack.detach().cpu())
            y_pred = torch.from_numpy(self.rfr.predict(scores_stack))
            score = y_pred.reshape(p.size()).to(device)
            self.scores[id(p)] = score

class PureRandomForestPruner(PureSklearnModelPruner):
    def __init__(self, masked_parameters):
         super(PureRandomForestPruner, self).__init__(masked_parameters, "artefacts/PureRandomForestRegressor.dill")
class PureSGDPruner(PureSklearnModelPruner):
    def __init__(self, masked_parameters):
         super(PureSGDPruner, self).__init__(masked_parameters, "artefacts/PureSGDRegressor.dill")
         
class PureLinearPruner(PureSklearnModelPruner):
    def __init__(self, masked_parameters):
         super(PureLinearPruner, self).__init__(masked_parameters, "artefacts/PureLinearRegression.dill")
class PureMLPPruner(PureSklearnModelPruner):
    def __init__(self, masked_parameters):
         super(PureMLPPruner, self).__init__(masked_parameters, "artefacts/PureMLPRegressor.dill")
class PureDecisionTreePruner(PureSklearnModelPruner):
    def __init__(self, masked_parameters):
         super(PureDecisionTreePruner, self).__init__(masked_parameters, "artefacts/PureDecisionTreeRegressor.dill")
class PureAdaBoostPruner(PureSklearnModelPruner):
    def __init__(self, masked_parameters):
         super(PureAdaBoostPruner, self).__init__(masked_parameters, "artefacts/PureAdaBoostRegressor.dill")
class PureGradientBoostingPruner(PureSklearnModelPruner):
    def __init__(self, masked_parameters):
         super(PureGradientBoostingPruner, self).__init__(masked_parameters, "artefacts/PureGradientBoostingRegressor.dill")

class RandomForestPruner(SklearnModelPruner):
    def __init__(self, masked_parameters):
         super(RandomForestPruner, self).__init__(masked_parameters, "artefacts/RandomForestRegressor.dill")
class SGDPruner(SklearnModelPruner):
    def __init__(self, masked_parameters):
         super(SGDPruner, self).__init__(masked_parameters, "artefacts/SGDRegressor.dill")
class MLPPruner(SklearnModelPruner):
    def __init__(self, masked_parameters):
         super(MLPPruner, self).__init__(masked_parameters, "artefacts/MLPRegressor.dill")
class DecisionTreePruner(SklearnModelPruner):
    def __init__(self, masked_parameters):
         super(DecisionTreePruner, self).__init__(masked_parameters, "artefacts/DecisionTreeRegressor.dill")
class AdaBoostPruner(SklearnModelPruner):
    def __init__(self, masked_parameters):
         super(AdaBoostPruner, self).__init__(masked_parameters, "artefacts/AdaBoostRegressor.dill")
class GradientBoostingPruner(SklearnModelPruner):
    def __init__(self, masked_parameters):
         super(GradientBoostingPruner, self).__init__(masked_parameters, "artefacts/GradientBoostingRegressor.dill")


class Stacked(Pruner):
    def __init__(self, masked_parameters, base_pruner_cls, final_pruner_cls):
        super(Stacked, self).__init__(list(masked_parameters))
        self.base_pruner = base_pruner_cls(list(masked_parameters))
        # self.base_pruner.masked_parameters = list(masked_parameters)
        self.final_pruner = final_pruner_cls(list(masked_parameters))
        # self.final_pruner.masked_parameters = list(masked_parameters)

    def score(self, model, loss, dataloader, device):
        # score the base pruner
        self.base_pruner.score(model, loss, dataloader, device)

        # we replace the weights with the scores, keeping the originals
        original_weights = OrderedDict()
        for (layer_name, p), (layer_id, score) in zip([(k, v) for k, v in model.state_dict().items() if k.endswith(".weight")], self.base_pruner.scores.items()):
            original_weights[layer_name] = torch.clone(p)
            new_state = model.state_dict()
            new_state.update({layer_name: score})
            model.load_state_dict(new_state)

        # we score with the final pruner
        self.final_pruner.score(model, loss, dataloader, device)

        # we reset the weights
        new_state = model.state_dict()
        new_state.update(original_weights)
        model.load_state_dict(new_state)

        self.scores = self.final_pruner.scores


# metaclasses ftw
def getStackedClass(A, B):
    return type(f"{A.__name__}{B.__name__}Stacked", (Stacked, ), {
        "__init__": lambda self, masked_parameters: super(self.__class__, self).__init__(list(masked_parameters), A, B)
    })


class UPrune(Pruner):
    def __init__(self, masked_parameters):
        super(UPrune, self).__init__(masked_parameters)
        
        with open("./artefacts/kde.dill", "rb") as f:
            self.kde = dill.load(f)
        self.min = -1
        self.max = 1
        self.bins = 100
        self.cols = np.linspace(self.min, self.max, self.bins)
        self.bounds = np.linspace(self.min, self.max, self.bins + 1)

        t_kde = torch.from_numpy(self.kde.sample(10000000))
        self.h = torch.histc(t_kde, bins=self.bins, min=self.min, max=self.max) / torch.numel(t_kde)


    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            kde_samples = torch.reshape(torch.tensor(self.kde.sample(p.flatten().numel())).squeeze(), p.size()).to(device)

            p_sort, p_sort_i = torch.sort(p.flatten())
            kde_sort, _ = torch.sort(kde_samples.flatten())
            # diff = torch.abs(p_sort - kde_sort)
            diff = p_sort - kde_sort
            diff = 1/torch.abs(diff)
            diff = torch.log(diff)
            scores = diff.gather(0, p_sort_i.argsort(0))
            scores = scores.reshape(p.size())
            self.scores[id(p)] = 1/scores

    # def score(self, model, loss, dataloader, device):

    #     lower_bounds = self.bounds[:-1]
    #     upper_bounds = self.bounds[1:]

    #     all_params = torch.cat([p.flatten() for _, p in self.masked_parameters])
    #     # global_numel = all_params.numel()
    #     global_kde = torch.histc(torch.from_numpy(self.kde.sample(1000000)), self.bins, self.min, self.max).to(device)

    #     # create all masks and set to 0
    #     for (_, p) in self.masked_parameters:
    #         self.scores[id(p)] = torch.zeros_like(p)

    #     def get_all_params_without_score():
    #         return torch.cat([torch.masked_select(p, scores == 0) for (_, p), (s_id, scores) in zip(self.masked_parameters, self.scores.items())])

    #     # calculate for all bins by how much they deviate from the target bin
    #     global_h = torch.histc(get_all_params_without_score(), self.bins, self.min, self.max).to(device)
    #     global_diff = (global_h / torch.sum(global_h)) - (global_kde / torch.sum(global_kde))
    #     global_diff = torch.where(global_diff > 0., global_diff, 0.) # remove entries for bins that have to few items in them
    #     global_diff_ratio = global_diff / torch.sum(global_diff)


    #     for (_, p), (s_id, scores) in zip(self.masked_parameters, self.scores.items()):
    #         layer_numel = p.numel()
    #         sorted, indices = torch.sort(global_diff_ratio, descending=True)
    #         for bin_rank, (bin_items, bin_i) in enumerate(zip(sorted, indices)):
    #             # print(f" processing bin ranked #{bin_rank}: bin {bin_i} {bin_items}")

    #             # per bin:
    #             lower_bound = lower_bounds[bin_i]
    #             upper_bound = upper_bounds[bin_i]

    #             # how many items are in bin?
    #             scope = (p >= lower_bound) & (p < upper_bound) & (scores == 0) # TODO: remove those which already have a score?
    #             numel_scope = torch.sum(scope).item()
    #             # print(f"{numel_scope} values in scope")
    #             # how much should i prune?
    #             ratio_to_prune = global_diff_ratio[bin_i].item()

    #             k = int(numel_scope * ratio_to_prune)
    #             # print(f"{ratio_to_prune} pct of which should be pruned -> k = {k}")
    #             if k > 0:
    #                 kth = torch.kthvalue(
    #                     torch.where(
    #                         scope,
    #                         p.double(),
    #                         1.
    #                     ).flatten(),
    #                     k
    #                 )
    #                 kth_upper_bound = kth.values.item()

    #                 scope = scope & (p < kth_upper_bound)
    #             # print(f"{torch.sum(scope).item()} values remain in scope")

    #             # scope now marks the values, where the score should be rank of bin + abs value, for the rest leave scores alone
    #             avg_score = torch.sum(torch.where(scope, bin_rank + torch.abs(p).double(), 0.)) / torch.sum(scope).double()
    #             # print(f"average score given to items in scope: {avg_score.item()}")
    #             scores = torch.where(scope, bin_rank + torch.abs(p), scores)
    #         self.scores[s_id] = scores
