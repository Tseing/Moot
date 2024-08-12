import os
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
        "../../data/finetune/all_activities.csv", ("assay_chembl_id", "target_chembl_id")
    )
    prot_dict = LookupDict("../../data/finetune/all_prot_seq_less1495.csv")

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

    df[["mol_a", "mol_b", "target_chembl_id"]].to_csv(save_path, index=False)


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


def fetch_smiles(
    df_path: str, save_path: str, smiles_dict: LookupDict, prot_dict: LookupDict
) -> None:
    tqdm.pandas()
    df = pd.read_csv(df_path)
    df["mol_a"] = df["mol_a"].progress_apply(smiles_dict.lookup)
    df["mol_b"] = df["mol_b"].progress_apply(smiles_dict.lookup)
    df["target_chembl_id"] = df["target_chembl_id"].progress_apply(prot_dict.lookup)
    df.columns = ["mol_a", "mol_b", "sequence"]

    df.to_csv(save_path, index=False)


def fetch_selfies(
    df_path: str, save_path: str, selfies_dict: LookupDict, prot_dict: LookupDict
) -> None:
    tqdm.pandas()
    df = pd.read_csv(df_path)
    df["mol_a"] = df["mol_a"].progress_apply(selfies_dict.lookup)
    df["mol_b"] = df["mol_b"].progress_apply(selfies_dict.lookup)
    df["target_chembl_id"] = df["target_chembl_id"].progress_apply(prot_dict.lookup)
    df.columns = ["mol_a", "mol_b", "sequence"]

    df.to_csv(save_path, index=False)


def fetch_smiles_selfies_pipeline(dataset_dir: str, save_dir: str) -> None:
    smiles_dict = LookupDict("../../data/all/smiles_lookup.pkl")
    selfies_dict = LookupDict("../../data/all/selfies_lookup.pkl")
    prot_dict = LookupDict("../../data/dep_finetune/all_prot_seq_less1495.csv")
    file_names = list(filter(lambda s: s.endswith(".csv"), os.listdir(dataset_dir)))
    file_paths = [osp.join(dataset_dir, file_name) for file_name in file_names]
    smiles_save_paths = [
        osp.join(save_dir, f"{osp.splitext(file_name)[0]}_smiles.csv") for file_name in file_names
    ]
    selfies_save_paths = [
        osp.join(save_dir, f"{osp.splitext(file_name)[0]}_selfies.csv") for file_name in file_names
    ]

    for i, file in enumerate(file_paths):
        fetch_smiles(file, smiles_save_paths[i], smiles_dict, prot_dict)
        print(f"'{smiles_save_paths[i]}' is saved.")
        fetch_selfies(file, selfies_save_paths[i], selfies_dict, prot_dict)
        print(f"'{selfies_save_paths[i]}' is saved.")


if __name__ == "__main__":
    # fetch_item_by_chembl_id(
    #     "../../data/finetune/permed_mmp.csv", "../../data/pretrain/pretrain.csv"
    # )
    split_dataset(
        "../../data/pretrain/pretrain_nonan.csv",
        "../../data/pretrain/runtime/chembl_id_seed_0",
        seed=0,
    )
    fetch_smiles_selfies_pipeline(
        "../../data/pretrain/runtime/chembl_id_seed_0",
        "../../data/pretrain/runtime/datasets_seed_0",
    )
