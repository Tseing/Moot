import os
import pickle

import pandas as pd
from pandarallel import pandarallel

from data_utils import canonicalize_smiles, split_path

if __name__ == "__main__":
    pandarallel.initialize(nb_workers=20)
    path = "../../data/1bond/chembl_id_lookup.csv"
    data_dir, _ = split_path(path)
    df = pd.read_csv(path)

    original_size = df.shape[0]
    print(f"Original size: {original_size}.")

    nan_items = df[pd.isna(df.canonical_smiles) == True]
    print("Nan items:", nan_items, sep="\n")
    print(f"Num of nan items: {nan_items.shape[0]}.")

    df = df.dropna(how="any")
    print(f"Non-nan size: {df.shape[0]}.")

    df["canonical_smiles"] = df["canonical_smiles"].parallel_apply(canonicalize_smiles)
    d = dict(zip(df["chembl_id"], df["canonical_smiles"]))
    pickle.dump(d, open(os.path.join(data_dir, "lookup_smiles.pkl"), "wb"))
