FIG_PATH = "./figures"
def savefig(ax, name):
    try:
        fig = ax.get_figure()
    except:
        fig = ax.fig
    # size = fig.get_size_inches()*fig.dpi
    # size = [f"{s:.0f}".replace(".", "") for s in size]
    # fig.savefig(f"{FIG_PATH}/6_{size[0]}x{size[1]}_{name}")
    fig.savefig(f"{FIG_PATH}/{name}", dpi=200, transparent=True, bbox_inches='tight')


pretty_names = {
    "phase_0_strategy": "Scoring Criterion",
    "final_sparsity": "Sparsity",
    "accuracy": "Top-1 Accuracy",
    "type_short": "Pruning Schedule",
}

def prettify_names(df):
     

    return df.rename(columns=pretty_names)

def replace_typeshort(df):
    if "type_short" in df.columns:
        df["type_short"] = df["type_short"].apply(lambda x: transform_typeshort(x))
    return df

def replace_algo(df):
    if "phase_0_strategy" in df.columns:
        
        df["phase_0_strategy"] = df["phase_0_strategy"].apply(lambda x: transform_algos(x))
    return df

def transform_typeshort(s):
    return s.replace("10", "MS").replace("1", "SS").replace("i", "I").replace("o", "U")

def transform_algos(s):
    nice_names = {
            "snip": "SNIP",
            "synflow": "SynFlow", 
            "mag": "Magnitude",
            "grasp": "GraSP",
            "rand": "Random"
    }
    for k, v in nice_names.items():
        s = s.replace(k, v)
    return s