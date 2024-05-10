import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.pyplot import Figure

from src.data_utils import split_path

CMAP = colormaps["Set2"]
XLABEL_FONTSIZE = 16
YLABEL_FONTSIZE = 16
FIGSIZE = (9, 7)
DPI = 900


def plot_matched_num_stats(fig: Figure, ax: Any, dataframe: pd.DataFrame, gap: int = 100) -> None:
    ax.barh(dataframe.matched_mol_num, np.log10(dataframe.items_num + 1), color=CMAP(0))
    ax.invert_yaxis()
    ax.set_ylim(dataframe.matched_mol_num.max() + gap, -5)
    ax.set_yticks(
        range(dataframe.matched_mol_num.min(), dataframe.matched_mol_num.max() + gap, gap)
    )
    ax.set_xlabel("Number of pairs (logarithm value)", fontsize=XLABEL_FONTSIZE)
    ax.set_ylabel("Number of matched compounds", fontsize=YLABEL_FONTSIZE)


if __name__ == "__main__":
    path = "../../data/1bond/dataset/stats/train_matched_num_dist.csv"
    data_dir, data_name = split_path(path)
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    plot_matched_num_stats(fig, ax, df, gap=1)
    plt.savefig(os.path.join(data_dir, f"{data_name}.svg"), dpi=DPI)
