import os.path as osp
import pickle
import sys
from typing import Tuple, Union, cast

sys.path.append("../..")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from terminaltables import AsciiTable
from tqdm import tqdm
from typing_extensions import TypeGuard

from src.data_utils import LookupDict, split_path


def fetch_item_by_chembl_id(df_path: str, save_path: str) -> None:
    smiles_dict = LookupDict("../../data/all/smiles_lookup.pkl")
    selfies_dict = LookupDict("../../data/all/selfies_lookup.pkl")
    target_dict = LookupDict(
        "../../data/all/all_activities.csv", ("assay_chembl_id", "target_chembl_id")
    )
    prot_dict = LookupDict("../../data/all/all_prot_seq_less1495.csv")

    print(smiles_dict["CHEMBL113419"])
    print(selfies_dict["CHEMBL113419"])
    print(target_dict["CHEMBL665886"])
    print(prot_dict["CHEMBL4662925"])

    tqdm.pandas()
    df = pd.read_csv(df_path)
    pre_size = df.shape[0]
    print(f"Original size: {pre_size}")

    df["mol_a_smiles"] = df["mol_a"].progress_apply(smiles_dict.lookup)
    df["mol_b_smiles"] = df["mol_b"].progress_apply(smiles_dict.lookup)
    df["mol_a_selfies"] = df["mol_a"].progress_apply(selfies_dict.lookup)
    df["mol_b_selfies"] = df["mol_b"].progress_apply(selfies_dict.lookup)
    df = df.dropna(how="any")
    print(
        f"Dataset size after fetching mol: {df.shape[0]}. "
        f"There are '{pre_size - df.shape[0]}' molecule cannot be found by ChEMBL ID."
    )
    pre_size = df.shape[0]

    df["target_chembl_id"] = df["assay_chembl_id"].progress_apply(target_dict.lookup)
    df = df.dropna(how="any")
    print(
        f"Dataset size after fetching targets: {df.shape[0]}. "
        f"There are '{pre_size - df.shape[0]}' targets cannot be found by assays ChEMBL ID."
    )
    pre_size = df.shape[0]

    df["sequence"] = df["target_chembl_id"].progress_apply(prot_dict.lookup)
    df = df.dropna(how="any")
    print(
        f"Final Dataset size: {df.shape[0]}. "
        f"There are '{pre_size - df.shape[0]}' sequence cannot be found by targets ChEMBL ID."
    )

    save_df = df[
        [
            "mol_a",
            "mol_b",
            "mol_a_smiles",
            "mol_b_smiles",
            "mol_a_selfies",
            "mol_b_selfies",
            "target_chembl_id",
            "sequence",
        ]
    ]
    save_df.columns = [
        "mol_a",
        "mol_b",
        "mol_a_smiles",
        "mol_b_smiles",
        "mol_a_selfies",
        "mol_b_selfies",
        "target",
        "sequence",
    ]
    save_df.to_csv(save_path, index=False)


def insert_mmp_info(df_path: str) -> None:
    mmp_path = f"{osp.splitext(df_path)[0]}_mmp.csv"
    df = pd.read_csv(df_path)
    mmp_df = pd.read_csv(mmp_path)
    assert (
        df.shape[0] == mmp_df.shape[0]
    ), f"Unmatched shapes between {df.shape} and {mmp_df.shape}."

    df = pd.concat([df, mmp_df], axis=1)
    df = df.dropna(how="any")
    df.to_csv(f"{osp.splitext(df_path)[0]}_dataset.csv", index=False)


def generate_finetune_dataset(df_path: str, save_path: str) -> None:
    df = pd.read_csv(df_path)
    drop_idx = (df["frag_a_heavy"] > df["core_heavy"]) & (df["frag_b_heavy"] > df["core_heavy"])
    df = df[~drop_idx]

    df.to_csv(save_path, index=False)


class DatasetSplit:
    def __init__(
        self,
        df_path: str,
    ) -> None:
        df = pd.read_csv(df_path)
        df_size = df.shape[0]
        _, file_name = split_path(df_path)

        self.file_name = file_name
        self.df = df
        self.df_size = df_size
        self.names = ["train", "val", "test"]

    @staticmethod
    def is_int_tuple(t: Tuple) -> TypeGuard[Tuple[int]]:
        if all(isinstance(i, int) for i in t):
            return True
        else:
            return False

    def get_size(
        self, df_size: int, props: Union[Tuple[float, float, float], Tuple[int, int, int]]
    ) -> Tuple[int, int, int]:
        assert len(props) == 3, "Attribute 'props' should include 3 values."
        if not self.is_int_tuple(props):
            assert sum(props) == 1, f"Invalid props: '{props}'."
            props = cast(Tuple[int, int, int], tuple(int(prop * df_size) for prop in props))

        else:
            props = cast(Tuple[int, int, int], tuple(int(prop) for prop in props))

        assert sum(props) <= df_size, f"Invalid size: '{props}', total size: '{df_size}'."

        return props

    def split(
        self, props: Union[Tuple[float, float, float], Tuple[int, int, int]], seed: int = 0
    ) -> None:
        sizes = dict(zip(self.names, self.get_size(self.df_size, props)))

        self.seed = seed

        all_idx = np.arange(self.df_size)
        train_idx, val_test_idx = train_test_split(
            all_idx,
            test_size=sizes["val"] + sizes["test"],
            train_size=sizes["train"],
            random_state=seed,
        )

        seed = 1120 + seed
        val_idx, test_idx = train_test_split(
            val_test_idx,
            test_size=sizes["test"],
            train_size=sizes["val"],
            random_state=seed,
        )

        idxes = dict(zip(self.names, (train_idx, val_idx, test_idx)))
        datasets = {name: self.df.iloc[idxes[name]] for name in self.names}
        self.datasets = datasets
        self.split_idxes = idxes
        self.summary()

    def dump(self, save_dir: str) -> None:
        save_paths = dict(
            zip(
                self.names,
                [osp.join(save_dir, f"{self.file_name}_{name}.csv") for name in self.names],
            )
        )
        for name in self.names:
            self.datasets[name].to_csv(save_paths[name], index=False)
            print(f"{name} {self.datasets[name].shape}: '{save_paths[name]}'.")
        pickle.dump(self.split_idxes, open(osp.join(save_dir, "split_idxes.pkl"), "wb"))

    def summary(self) -> None:
        table_data = [["Dataset", "Unique Items", "Nan Items", "Shape"]]
        table_data.extend(self._stats_df(self.df, "all"))
        for name in self.names:
            table_data.extend(self._stats_df(self.datasets[name], name))

        table = AsciiTable(table_data)
        table.title = f"Dataset Summary (Random Seed: {self.seed})"
        table.inner_row_border = True
        print(table.table)

    @staticmethod
    def _stats_df(df: pd.DataFrame, dataset_name: str) -> list:
        dataset_info = []
        for col_label in df.columns:
            col = df[col_label]
            col_info = [f"{dataset_name}.{col_label}"]
            col_info.append(col.drop_duplicates().shape[0])
            col_size = col.shape[0]
            col_info.append(col_size - col.dropna().shape[0])
            col_info.append(col.shape)
            dataset_info.append(col_info)

        return dataset_info


def split_dataset(total_csv_path: str, save_dir: str, seed: int = 0):
    dataset_split = DatasetSplit(total_csv_path)
    dataset_split.split((0.8, 0.1, 0.1), seed)
    dataset_split.dump(save_dir)


if __name__ == "__main__":
    # Generate pretrain dataset from original data
    # fetch_item_by_chembl_id("../../data/all/all_permed_mmp.csv", "../../data/pretrain/pretrain.csv")

    # Merge pretrain csv and mmp csv
    # insert_mmp_info("../../data/pretrain/pretrain.csv")

    # generate finetune dataset
    # generate_finetune_dataset(
    #     "../../data/pretrain/pretrain_dataset.csv", "../../data/finetune/finetune_dataset.csv"
    # )

    # split pretrain datasets
    # split_dataset(
    #     "../../data/pretrain/pretrain_dataset.csv",
    #     "../../data/pretrain/runtime/datasets_seed_0",
    #     seed=0,
    # )

    # split finetune datasets
    split_dataset(
        "../../data/finetune/finetune_dataset.csv",
        "../../data/finetune/runtime/datasets_seed_0",
        seed=0,
    )
