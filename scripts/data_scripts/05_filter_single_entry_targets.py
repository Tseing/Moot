import json

import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    df_path = "../../data/targets/all_activities.csv"
    df = pd.read_csv(df_path)
    original_size = df.shape[0]

    grouped_df = df.groupby(["target_chembl_id", "assay_id"])
    drop_idx = []
    units_dict = {}
    for group in tqdm(grouped_df):
        if group[1].shape[0] == 1:
            drop_idx.append(group[1].index)

        units = [
            unit_group[0] for unit_group in group[1].groupby(["standard_type", "standard_units"])
        ]
        if len(units) > 1:
            units_dict["_".join([str(i) for i in group[0]])] = units

    drop_idx = pd.Index(np.concatenate(drop_idx, axis=0))
    df.drop(drop_idx, inplace=True)
    final_size = df.shape[0]

    # df.to_csv("../../data/targets/all_activities_drop_single_entry.csv", index=False)

    print(f"{original_size} -> {final_size}, {drop_idx.shape[0]} " f"entries have been dropped.")
    json.dump(units_dict, open("../../data/targets/activity_units.json", "w+"))