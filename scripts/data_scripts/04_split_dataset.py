import copy
import os
from typing import Callable, List, Sequence

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split

from data_utils import randomize_smiles, smiles2smarts, split_path

TRAIN_SIZE = 100000
VAL_SIZE = 10000
TEST_SIZE = 10000


class DatasetBuilder:
    def __init__(self, dataframe: pd.DataFrame, workers: int = 20) -> None:
        assert set(dataframe.columns) == set(
            ["col1", "col2", "col1_smiles", "col2_smiles"]
        ), f"Make sure labels of DataFrame columns are 'col1', 'col2', "
        f"'col1_smiles' and 'col2_smiles', but got '{dataframe.columns}'."

        self.df = dataframe
        pandarallel.initialize(nb_workers=workers)
        self._smiles_cols = ["col1_smiles", "col2_smiles"]
        self._rd_smiles_cols = ["col1_rd_smiles", "col2_rd_smiles"]
        self._smarts_cols = ["col1_smarts", "col2_smarts"]

    def create_dataset(self, idx: Sequence[int], save_path: str) -> None:
        df = copy.deepcopy(self.df.iloc[idx])
        df.to_csv(save_path, index=False)
        print(f"Dataset '{os.path.abspath(save_path)}' size: {df.shape}")

        save_dir, save_name = split_path(save_path)
        smiles_path = os.path.join(save_dir, f"{save_name}_smiles.csv")
        rd_smiles_path = os.path.join(save_dir, f"{save_name}_rd_smiles.csv")
        smarts_path = os.path.join(save_dir, f"{save_name}_smarts.csv")

        self.create_data_by_func(
            df, randomize_smiles, cols_in=self._smiles_cols, cols_out=self._rd_smiles_cols
        )
        self.create_data_by_func(
            df, smiles2smarts, cols_in=self._smiles_cols, cols_out=self._smarts_cols
        )

        self.dump_dataset(df, self._smiles_cols, smiles_path)
        self.dump_dataset(df, self._rd_smiles_cols, rd_smiles_path)
        self.dump_dataset(df, self._smarts_cols, smarts_path)

    @staticmethod
    def dump_dataset(dataset: pd.DataFrame, columns: List[str], save_path: str) -> None:
        assert 2 == len(
            columns
        ), f"'columns' should be a list only includes 2 columns labels, but got {columns}."
        df = dataset[columns]
        df.columns = ["col1", "col2"]
        df.to_csv(save_path, index=False)

        print(f"Dataset '{os.path.abspath(save_path)}' size: {df.shape}")

    @staticmethod
    def create_data_by_func(
        dataframe: pd.DataFrame, create_func: Callable, cols_in: List[str], cols_out: List[str]
    ) -> None:
        for col_in, col_out in zip(cols_in, cols_out):
            dataframe[col_out] = dataframe[col_in].parallel_apply(create_func)


if __name__ == "__main__":
    path = "../../data/1bond/mmp_pair_1bond.csv"
    data_dir, data_name = split_path(path)
    all_df = pd.read_csv(path)
    all_num = all_df.shape[0]

    all_idx = np.arange(all_num)
    train_idx, val_test_idx = train_test_split(
        all_idx, test_size=VAL_SIZE + TEST_SIZE, train_size=TRAIN_SIZE
    )
    val_idx, test_idx = train_test_split(val_test_idx, test_size=TEST_SIZE, train_size=VAL_SIZE)

    save_dir = os.path.join(data_dir, "100k_dataset")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_paths = (os.path.join(save_dir, f"{dataset}.csv") for dataset in ["train", "val", "test"])
    dataset_builder = DatasetBuilder(all_df)
    for idx, save_path in zip((train_idx, val_idx, test_idx), save_paths):
        dataset_builder.create_dataset(idx, save_path)
