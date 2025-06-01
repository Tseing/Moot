import sys

import pandas as pd
from tqdm import tqdm

sys.path.append("..")
from src.utils.molecule import get_marked_atoms


def generate_marked_dataset(data_path: str, save_path: str) -> None:
    tqdm.pandas()
    df = pd.read_csv(data_path)

    atoms = df.progress_apply(
        lambda df: get_marked_atoms(df["core"], df["frag_a"]), axis=1
    )
    nonan = atoms.dropna()
    assert nonan.shape[0] == df.shape[0], f"{df.shape} -> {nonan.shape}"

    atoms.name = "atoms"
    atoms.to_csv(save_path, index=False)


if __name__ == "__main__":
    generate_marked_dataset(
        "../../data/finetune/runtime/datasets_seed_0/finetune_dataset_train.csv",
        "../../data/atom/atom_train.csv",
    )
