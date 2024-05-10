import os
import pickle

import pandas as pd
from pandarallel import pandarallel

from src.data_utils import split_path

if __name__ == "__main__":
    path = "../../data/1bond/mmp_pair_1bond_raw.csv"
    data_dir, data_name = split_path(path)

    df = pd.read_csv(path)
    df.columns = ["col1", "col2"]
    original_size = df.shape[0]

    lookup_path = os.path.join(data_dir, "lookup_smiles.pkl")
    lookup_smiles = pickle.load(open(lookup_path, "br"))

    pandarallel.initialize(nb_workers=20)

    df["col1_smiles"] = df["col1"].parallel_apply(lambda id: lookup_smiles.get(id, None))
    df["col2_smiles"] = df["col2"].parallel_apply(lambda id: lookup_smiles.get(id, None))

    df = df.dropna(how="any")

    print(f"Num of removed nan items during lookup SMILES: {original_size - df.shape[0]}.")
    print(f"Num of fetched SMILES: {df.shape[0]}.")

    df.to_csv(os.path.join(data_dir, f"{data_name[:-len('_raw')]}.csv"), index=False)
