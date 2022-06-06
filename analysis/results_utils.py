from typing import Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import torch
# from convert_states_to_results import load_state
from utils.params import PruningPhase
from utils.state import State, prepare_data, get_all_saved_states

import os
import dill
df_CSV = False
# FILE_PATH = "./analysis/results/results_combined.csv"
RESULTS_DIR = "./results"
def load_csv(results_dir = RESULTS_DIR):
    results = []
    for file in os.listdir(results_dir):
        if not os.path.isdir(f"{results_dir}/{file}"):
            with open(f"{results_dir}/{file}", "rb") as f:
                d = dill.load(f)
                d["date"] = int(d["file"].rsplit("_", 2)[-1][:8])
                results.append(d)

    return pd.DataFrame(results)


def convert_state(s) -> Tuple[str, Dict]:
    """Tuple of layer_name & all tensors per kind for that layer"""

    params = [(name, param) for name, param in s.model.named_parameters() if name.endswith(".weight")]
    init_params = [(name, param) for name, param in s.initial_state.model.named_parameters() if name.endswith(".weight")]
    
    results = []
    for (layer_name, init_param), (_, param), (_, score), (mask, _) in zip(init_params, params, s.pruner.scores.items(), s.pruner.masked_parameters):
        per_kind = {
            "initial_weight": init_param.T,
            "final_weight": param.T,
            "mask": (mask > 0).T,
            "score": torch.transpose(score, 0, 1),
            "movement": param.T - init_param.T
        }
        results.append((layer_name, per_kind))
    return results

def filter_experiments(df, **kwargs):
    df_f = df.copy()
    for col, value in kwargs.items():
        value = value if isinstance(value, list) else [value]
        df_f = df_f[df_f[col].isin(value)]
    return df_f

def get_file_names(**kwargs):
    df = df_CSV
    if not df:
        df = load_csv()
    df_f = filter_experiments(df, **kwargs)
    files = df_f["file"].tolist()
    return files

def filter_file_names(files, **kwargs):
    df = df_CSV
    if not df:
        df = load_csv()
    df_f = df[df["file"].isin(files)]
    df_f = filter_experiments(df_f, **kwargs)
    files = df_f["file"].tolist()
    return files

def get_file_name(init_id, sparsity, experiment_type, init_strategy, algo):
    files = get_file_names(base_model_id = init_id, phase_0_strategy = algo, final_sparsity = sparsity, init_strategy = init_strategy, type_short = experiment_type)

    if len(files) != 1:
        print(f"warning, trying to get single tensor but there are {len(files)} matches.")
    return files[0]

def get_tensor(init_id, sparsity, experiment_type, init_strategy, algo, layer_name, kind, masked = False):
    file = get_file_name(init_id, sparsity, experiment_type, init_strategy, algo)
    converted_state = load_and_convert_cached(file)
    for loaded_layer_name, per_kind in converted_state:
        if loaded_layer_name in layer_name:
            t = per_kind[kind]
            if masked:
                t = apply_mask(t, per_kind["mask"])
            return t


def get_tensors(init_id = None, sparsity = None, algo = None, experiment_type = None, layer_name = None, kind = None, masked = False, files=None, init_strategy=None) -> pd.DataFrame:
    col_mapping = {"base_model_id": init_id, "phase_0_strategy": algo, "final_sparsity": sparsity, "type_short": experiment_type, "init_strategy": init_strategy}
    criterias = {k: v if isinstance(v, list) else [v] for k, v in col_mapping.items() if v}

    results = []
    if files:
        files = filter_file_names(files, **criterias)
    else:
        files = get_file_names(**criterias)
    
    for file in files:
        cs = load_and_convert_cached(file)
        s = load_state_cached(file)
        for loaded_layer_name, per_kind in cs:
            mask = per_kind["mask"] if masked else None
            if layer_name and loaded_layer_name not in layer_name:
                continue
            for loaded_kind, t in per_kind.items():
                if kind and loaded_kind not in kind:
                    continue
                

                phase0 = s.config.pruning_phases[0]
                results.append({
                    "init_id": s.base_model_id,
                    "sparsity": phase0.sparsity if isinstance(phase0, PruningPhase) else phase0.prune_params.sparsity,
                    "algo": phase0.strategy if isinstance(phase0, PruningPhase) else phase0.prune_params.strategy,
                    "experiment_type": _experiment_shorthand_from_state(s),
                    "layer_name": loaded_layer_name,
                    "kind": loaded_kind,
                    "t": [apply_mask(t, mask)] if masked and kind != "mask" else t
               })
    print(f"get_tensors found {len(results)} tensors")
    return results

STATES_CACHE = {}
def load_state_cached(file):
    if file in STATES_CACHE:
        return STATES_CACHE[file]
    else:
        value = load_state(file)
        STATES_CACHE[file] = value
        return value

def _experiment_shorthand_from_state(s: State):
    phase0 = s.config.pruning_phases[0]
  
    if isinstance(phase0, PruningPhase):
        schedule = "o"
        epochs = phase0.prune_epochs
    else:
        schedule = "i"
        epochs = phase0.prune_params.prune_epochs

    return f"{schedule}{epochs}"

def _experiment_shorthand(row):
    pruning_schedule = "o" if row["phase_0_type"] == "PruningPhase" else "i"
    pruning_iterations = row["phase_0_prune_epochs"]
    return f"{pruning_schedule}{pruning_iterations}"




CONVERTED_STATE_CACHE = {}
def load_and_convert_cached(file):
    s = load_state_cached(file)
    if file in CONVERTED_STATE_CACHE:
        return CONVERTED_STATE_CACHE[file]
    else:
        res = convert_state(s)
        CONVERTED_STATE_CACHE[file] = res
        return res
        
def apply_mask(t, m):
    return torch.masked_select(t, m)


def to_numpy(t):
    return t.detach().cpu().numpy()

def get_score_corr_matrix(all_scores, init_id):
    def get_algo(algo):
        return get_tensor_from_df(all_scores, init_id=init_id, algo=algo)

    df_corr = pd.DataFrame(data={algo: to_numpy(get_algo(algo)).ravel() for (algo,) in get_unique(all_scores, algo=True) if not get_algo(algo) is None})
    df_corr = df_corr.corr().sort_values(by=list(df_corr.columns), axis=0, ascending=False).sort_values(by=list(df_corr.columns), axis=1, ascending=False)
    return df_corr



# TODO: get rid of this?
# Handle dataframe with tensors in it
def _create_tensors():
    return pd.DataFrame()

def _add_tensor(tensors: pd.DataFrame, **kwargs) -> pd.DataFrame:
    combined_df = pd.concat([tensors, pd.DataFrame.from_dict({k: v for k, v in kwargs.items()}, orient="index").T])
    return combined_df


def convert_states(df) -> pd.DataFrame:
    """pulls matrices out of the state object"""
    tensors = _create_tensors()
    for file, algo, init_id, sparsity in zip(df["file"], df["phase_0_strategy"], df["base_model_id"], df["final_sparsity"]): #10 rows per loop added
        for layer_name, per_kind in load_and_convert_cached(file):
            for kind, t in per_kind.items():
                tensors = _add_tensor(tensors, init_id=init_id, layer_name=layer_name, kind=kind, algo=algo, sparsity=sparsity, t=t)
    return tensors


def filter(tensors: pd.DataFrame, init_id = None, layer_name = None, kind = None, sparsity = None, algo = None):
    """dynamic filtery by kwargs"""
    filtered = tensors.copy()
    if init_id:
        filtered = filtered[filtered["init_id"] == init_id]
    if layer_name:
        filtered = filtered[filtered["layer_name"] == layer_name]
    if kind:
        filtered = filtered[filtered["kind"] == kind]
    if sparsity:
        filtered = filtered[filtered["sparsity"] == sparsity]
    if algo:
        filtered = filtered[filtered["algo"] == algo]
    
    return filtered
        
def _iter_tensors(tensors: pd.DataFrame):
    """shorthand to itereate through values"""
    for init_id, layer_name, kind, algo, sparsity, t in zip(tensors["init_id"], tensors["layer_name"], tensors["kind"], tensors["sparsity"], tensors["algo"], tensors["t"]):
        yield init_id, layer_name, kind, algo, sparsity, t



def get_tensor_from_df(tensors: pd.DataFrame, init_id = None, layer_name = None, kind = None, sparsity = None, algo = None):
    filtered = filter(tensors, init_id, layer_name, kind, sparsity, algo)
    if len(filtered) > 1:
        print(f"warning: trying to get 1, but there are {len(filtered)}")
    return filtered.loc[0]["t"]

def apply_masks(tensors: pd.DataFrame):
    masked_tensors = _create_tensors()
    for init_id, layer_name, kind, sparsity, algo, t in _iter_tensors(tensors):
        if kind != "mask":
            m = get_tensor_from_df(tensors, init_id, layer_name, "mask", sparsity, algo)
            masked_t = apply_mask(t, m)
            masked_tensors = _add_tensor(masked_tensors, init_id=init_id, layer_name=layer_name, kind=kind, sparsity=sparsity, algo=algo, t=masked_t)
    
    # add the masks for completeness:
    masked_tensors = pd.concat([masked_tensors, tensors[tensors["kind"] == "mask"]])
    return masked_tensors

def load_all(file, init_id = None, layer_name = None, kind = None, sparsity = None, algo = None, apply_mask = True):
    df = load_csv(file)
    all_tensors = convert_states(df)
    # apply masks before filtering to avoid having the masks filtered out before applied
    if apply_mask:
        all_tensors = apply_masks(all_tensors)

    all_tensors = filter(all_tensors, init_id, layer_name, kind, sparsity, algo)
    return all_tensors


def get_unique(tensors, init_id = False, layer_name = False, kind = False, sparsity = False, algo = False):
    """mark true the attributes that should be part of the tuples that are part of the set of combinations returned"""
    cols = [col_name for col_name, b in zip(["init_id", "layer_name", "kind", "sparsity", "algo"], [init_id, layer_name, kind, sparsity, algo]) if b]
    filtered = tensors[cols]

    return filtered.groupby(cols).mean().reset_index().to_records(index=False)




def print_histograms(df: pd.DataFrame, col="algo", row="kind", label=None, kde=True, sharex=True, sharey="row", palette="viridis", bins=100, **kwargs):
    df["t"] = df["t"].apply(lambda c: c[0].detach().cpu().numpy().ravel())
    df = df.set_index([c for c in df.columns if not c =="t"])
    df = df["t"].explode().reset_index()
    g = sns.FacetGrid(data=df, col=col, row=row, hue=label, sharey=sharey, sharex=sharex,  margin_titles=True, despine=False, legend_out=False, palette=palette)
    # g = sns.FacetGrid(data=df, col=col, row=row, hue=label, sharey=sharey, sharex=sharex, palette=palette)
    g.map(sns.histplot, "t",  kde=kde, bins=bins,**kwargs)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    return g

def print_relative_histograms(experiment):
    # only works for inits
    kind = "initial_weight" 
    experiment.pop("kind", None)
    
    bins = 100
    min = -0.3
    max = 0.3

    cols = list(np.linspace(min, max, bins))
    all_ts = {tuple(v for k, v in row.items() if k not in "t"): row["t"] for row in get_tensors(**experiment, masked=False)}
    
    all_hists = []
    for (init_id, sparsity, algo, experiment_type, layer_name, kind), t in all_ts.items():
        if kind == "initial_weight":
            mask = all_ts[(init_id, sparsity, algo, experiment_type, layer_name, "mask")][0]
            masked_inits = apply_mask(t[0], mask)

            hist_inits = torch.histc(t[0], bins=bins, min=min, max=max)
            hist_masked = torch.histc(masked_inits, bins=bins, min=min, max=max)

            hist_ratio = torch.div(hist_masked, hist_inits)
            res = {k: v.item() for k, v in zip(cols, hist_ratio)}
            res.update({
                "init_id": init_id,
                "sparsity": sparsity,
                "algo": algo,
                "experiment_type": experiment_type,
                "layer_name": layer_name,
            })
            all_hists.append(res)
    df = pd.DataFrame(all_hists)
    df = pd.melt(df, id_vars=["init_id", "sparsity", "algo", "experiment_type", "layer_name"], value_vars=cols)
    df["value"] = df["value"].fillna(0).astype(float)
    df["variable"] = df["variable"].fillna(0).astype(float)
    g = sns.FacetGrid(df, row="algo", col="sparsity", sharey="row", sharex=True,  margin_titles=True, despine=False)
    g.map_dataframe(sns.barplot, x="variable", y="value")


# data loader will be cached for performance
DATA_LOADERS = {}
def load_state(file) -> State:
    s = State()
    s.load(file, load_data=False)
    try:
        s.test_loader = DATA_LOADERS[s.config.data_params.dataset]
    except:
        prepare_data(s, s.config.data_params)
        DATA_LOADERS[s.config.data_params.dataset] = s.test_loader
    return s