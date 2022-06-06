FIG_PATH = "./figures"
def savefig(ax, name):
    try:
        fig = ax.get_figure()
    except:
        fig = ax.fig
    size = fig.get_size_inches()*fig.dpi
    size = [f"{s:.0f}".replace(".", "") for s in size]
    # fig.savefig(f"{FIG_PATH}/6_{size[0]}x{size[1]}_{name}")
    fig.savefig(f"{FIG_PATH}/{name}", dpi=200)


pretty_names = {
    "phase_0_strategy": "Scoring Criterion",
    "final_sparsity": "Sparsity",
    "accuracy": "Top-1 Accuracy",
}

def prettify_names(df):
    return df.rename(columns=pretty_names)