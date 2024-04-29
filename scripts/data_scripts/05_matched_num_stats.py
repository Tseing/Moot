import os
from collections import Counter

import pandas as pd

from data_utils import split_path, generate_unique_id

if __name__ == "__main__":
    path = "../../data/1bond/100k_dataset/val.csv"

    data_dir, data_name = split_path(path)
    save_dir = os.path.join(data_dir, "stats")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    df = pd.read_csv(path, usecols=["col1", "col2"])
    generate_unique_id(df)

    matched_num_df = df.groupby("col1").count()
    matched_num_df.columns = ["matched_mol_num"]
    matched_num_df.to_csv(
        os.path.join(save_dir, f"{data_name}_matched_num_per_mol.csv")
    )

    num_cnt = Counter(matched_num_df["matched_mol_num"].tolist())
    num_labels, num_values = zip(*sorted(num_cnt.items(), key=lambda x: x[0]))
    stats_df = pd.DataFrame({"matched_mol_num": num_labels, "items_num": num_values})
    stats_df.to_csv(
        os.path.join(save_dir, f"{data_name}_matched_num_dist.csv"),
        index=False,
    )
