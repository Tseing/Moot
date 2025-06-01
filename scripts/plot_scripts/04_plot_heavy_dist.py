import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("../..")


from src.data_metrics import MolMetrics
from utils import DataReader

if __name__ == "__main__":
    save_path = "../../output/heavy_hist.png"
    test_data_path = "../../data/finetune/runtime/datasets_seed_0/finetune_dataset_test.csv"
    data = {}
    test_df = pd.read_csv(test_data_path)
    data["test"] = test_df["frag_b_heavy"].to_numpy().astype(int)

    df = DataReader.prepare_mol_df(
        test_data_path,
        "../../output/top1/train_optformer_selfies_top1.csv",
        topk=1,
    )
    metrics = MolMetrics(df, "SELFIES")
    data["optformer selfies"] = metrics.metric_heavy("gen_frag_b_heavy").to_numpy()

    df = DataReader.prepare_mol_df(
        test_data_path,
        "../../output/top1/train_optformer_smiles_top1.csv",
        topk=1,
    )
    metrics = MolMetrics(df, "SMILES")
    data["optformer smiles"] = metrics.metric_heavy("gen_frag_b_heavy").to_numpy()

    df = DataReader.prepare_mol_df(
        test_data_path,
        "../../output/top1/train_transformer_selfies_top1.csv",
        topk=1,
    )
    metrics = MolMetrics(df, "SELFIES")
    data["transformer selfies"] = metrics.metric_heavy("gen_frag_b_heavy").to_numpy()

    df = DataReader.prepare_mol_df(
        test_data_path,
        "../../output/top1/train_transformer_smiles_top1.csv",
        topk=1,
    )
    metrics = MolMetrics(df, "SMILES")
    data["transformer smiles"] = metrics.metric_heavy("gen_frag_b_heavy").to_numpy()

    _, n_bins, _ = plt.hist(data["test"], bins=75, density=True, alpha=0.75)
    _, n_bins, _ = plt.hist(data["optformer selfies"], bins=75, density=True, alpha=0.75)
    _, n_bins, _ = plt.hist(data["optformer smiles"], bins=75, density=True, alpha=0.75)
    _, n_bins, _ = plt.hist(data["transformer selfies"], bins=75, density=True, alpha=0.75)
    _, n_bins, _ = plt.hist(data["transformer smiles"], bins=75, density=True, alpha=0.75)

    plt.savefig(save_path)
