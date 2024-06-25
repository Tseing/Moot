import os.path as osp
import pickle
from typing import List, Tuple
import sys
import pandas as pd
from pandarallel import pandarallel

sys.path.append("../..")
from src.data_utils import cal_mol_weight


def stats_unit(df_path: str) -> None:
    df_dir, _ = osp.split(df_path)
    df = pd.read_csv(df_path)
    grouped_df = df.groupby(["standard_type", "standard_units"])

    stats = [[":".join(group[0]), group[1].shape[0]] for group in grouped_df]
    stats_pd = pd.DataFrame(stats, columns=["unit", "number"])
    stats_pd.to_csv(osp.join(df_dir, "unit_stats.csv"), index=False)


def filter_unit(df_path: str, expected_unit: str) -> None:
    units = [tuple(unit.strip().split(":")) for unit in expected_unit.split(",")]

    pandarallel.initialize(nb_workers=30)
    df_dir, _ = osp.split(df_path)
    df = pd.read_csv(df_path)
    original_size = df.shape[0]

    drop_idx = df[["standard_type", "standard_units"]].parallel_apply(
        lambda df: False if (df["standard_type"], df["standard_units"]) in units else True, axis=1
    )

    df = df[~drop_idx]
    df.to_csv(osp.join(df_dir, "dropped_units.csv"), index=False)
    final_size = df.shape[0]
    print(
        f"Dataset size {original_size} -> {final_size}: {original_size - final_size} items have been dropped."
    )


def fetch_unstd_unit_value(df_path: str, unstd_units: List[Tuple[str, str]]) -> None:
    pandarallel.initialize(nb_workers=30)
    df_dir, _ = osp.split(df_path)
    df = pd.read_csv(df_path)
    unstd_idx = df[["standard_type", "standard_units"]].parallel_apply(
        lambda df: True if (df["standard_type"], df["standard_units"]) in unstd_units else False,
        axis=1,
    )

    df[~unstd_idx].to_csv(osp.join(df_dir, "std_units.csv"), index=False)
    unstd_df = df[unstd_idx]
    unstd_df.to_csv(osp.join(df_dir, "unstd_units.csv"), index=False)
    unstd_df["mol_chembl_id"].dropna().drop_duplicates().to_csv(
        osp.join(df_dir, "unstd_unique_mol_chembl_id.csv"), index=False
    )


class SMILESDict:
    def __init__(self, path: str) -> None:
        file_name, file_ext = osp.splitext(path)

        if file_ext == ".pkl":
            self.load_pkl(path)
        elif file_ext == ".csv":
            self.read_csv(path)
        else:
            assert False, f"Only support '.pkl' and '.csv' file, but got '{file_ext}'."

        self.dump_path = f"{file_name}.pkl"

    def __getitem__(self, key: str):
        return self.d[key]

    def read_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df.drop_duplicates(subset="chembl_id", inplace=True)
        self.d = dict(zip(df["chembl_id"], df["smiles"]))
        self._const_lookup_func()

    def load_pkl(self, path: str) -> None:
        self.d = pickle.load(open(path, "rb"))
        self._const_lookup_func()

    def dump(self):
        pickle.dump(self.d, open(self.dump_path, "wb"))

    def _const_lookup_func(self) -> None:
        self.lookup_smiles = lambda chembl_id: self[chembl_id]


def standardize_ug_mL(df_path: str) -> None:
    def convert_ugmL2mM(value: float, mw: float):
        return value / mw

    def stdize_pipeline(df: pd.DataFrame) -> float:
        smiles = smiles_dict.lookup_smiles(df["mol_chembl_id"])
        mw = cal_mol_weight(smiles)
        return convert_ugmL2mM(df["standard_value"], mw)

    pandarallel.initialize(nb_workers=30)
    df_dir, _ = osp.split(df_path)
    smiles_dict = SMILESDict(osp.join(df_dir, "unstd_unique_mol_smiles.csv"))
    df = pd.read_csv(df_path)

    unstd_df = df[df["standard_units"] == "ug.mL-1"]
    stdized_value = unstd_df.parallel_apply(stdize_pipeline, axis=1)
    unstd_df["standard_value"] = stdized_value
    unstd_df["standard_units"] = ["mM" for _ in range(stdized_value.shape[0])]
    unstd_df.to_csv(osp.join(df_dir, "stdized_unstd_units.csv"), index=False)


def merge_activities(df_paths: List[str], save_path: str) -> None:
    dfs = [pd.read_csv(df_path) for df_path in df_paths]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    finetune_expected_unit = r"""
    Potency:nM,
    IC50:nM,
    Inhibition:%,
    Percent Effect:%,
    Ki:nM,
    AC50:nM,
    EC50:nM,
    Kd:nM,
    Activity:%,
    Residual Activity:%,
    % Control:%,
    Emax:%,
    Efficacy:%,
    %Inhib (Mean):%,
    %Max (Mean):%,
    %max:%,
    IC50:ug.mL-1,
    Activity:uM,
    Intrinsic activity:%,
    Residual activity:%,
    Activity:nM,
    max activation:%,
    Control:%,
    % Ctrl:%,
    IC90:nM,
    EC90:nM,
    Activation:%
    """
    # stats_unit("../../data/finetune/all_activities.csv")
    # filter_unit("../../data/finetune/all_activities.csv", finetune_expected_unit)
    # fetch_unstd_unit_value("../../data/finetune/dropped_units.csv", [("IC50", "ug.mL-1")])
    # standardize_ug_mL("../../data/finetune/unstd_units.csv")
    merge_activities(
        ["../../data/finetune/std_units.csv", "../../data/finetune/unstd_units.csv"],
        "../../data/finetune/merged_activities.csv",
    )

    target_expected_unit = r"""
    Potency:nM,
    IC50:nM,
    Inhibition:%,
    Ki:nM,
    Percent Effect:%,
    AC50:nM,
    EC50:nM,
    Kd:nM,
    Activity:%,
    Residual Activity:%,
    Efficacy:%,
    %Inhib (Mean):%,
    IC50:ug.mL-1,
    Kb:nM,
    Intrinsic activity:%,
    Residual activity:%,
    Activity:nM,
    Activation:%,
    EC50:ug.mL-1,
    % inhibition:uM"""
    # fetch_unstd_unit_value("../../data/finetune/dropped_units.csv", [("IC50", "ug.mL-1"), ("EC50", "ug.mL-1")])
