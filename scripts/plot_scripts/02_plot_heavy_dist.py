import sys

sys.path.append("../..")

import matplotlib.pyplot as plt

import config
from src.data_utils import DatasetMetrics
from utils import DataReader

if __name__ == "__main__":
    cfg = {
        "dataset": "pretrain_test",
        "usecol": "frag_b_heavy",
        "save_path": "../../output/pretrain/pretrain_test_frag_b_heavy.png",
    }

    save_path = cfg["save_path"]
    df = DataReader.prepara_inp_df(cfg["dataset"])
    metrics = DatasetMetrics(df)
    data = metrics.metric_heavy(cfg["usecol"]).dropna()

    print(f"Max: {data.max()}, Min: {data.min()}")

    plt.ylim(0, 0.35)
    _, n_bins, _ = plt.hist(data, bins=100, density=True)
    plt.savefig(save_path)
