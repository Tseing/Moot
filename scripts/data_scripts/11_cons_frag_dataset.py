import sys

sys.path.append("../..")
import pandas as pd
from tqdm import tqdm

from src.data_utils import smiles2selfies


def get_frag_dataset(data_path: str, save_path: str):
    tqdm.pandas()
    df = pd.read_csv(data_path, usecols=["core", "frag_a", "frag_b"])
    df["core_selfies"] = df["core"].progress_apply(smiles2selfies)
    df["frag_a_selfies"] = df["frag_a"].progress_apply(smiles2selfies)
    df["frag_b_selfies"] = df["frag_b"].progress_apply(smiles2selfies)

    save_df = df[["core", "frag_a", "frag_b", "core_selfies", "frag_a_selfies", "frag_b_selfies"]]

    save_df.to_csv(save_path, index=False)
    save_df = save_df.dropna(how="any")
    assert save_df.shape[0] == df.shape[0], f"original: {df.shape} -> dropna: {save_df.shape} "


if __name__ == "__main__":
    get_frag_dataset(
        "../../data/finetune/runtime/datasets_seed_0/finetune_dataset_val.csv", "../../data/frag/frag_val.csv"
    )
    get_frag_dataset(
        "../../data/finetune/runtime/datasets_seed_0/finetune_dataset_test.csv", "../../data/frag/frag_test.csv"
    )
