import copy
import os
from typing import Optional

import pandas as pd

from src.data_utils import split_path, generate_unique_id

if __name__ == "__main__":
    path = "../../data/1bond/mmp_pair_1bond_raw.csv"
    data_dir, data_name = split_path(path)
    df = pd.read_csv(path)
    df = generate_unique_id(df, return_df=True)
    df.to_csv(os.path.join(data_dir, f"{data_name}_unique_id.csv"), index=False)
