import sys
from typing import Literal, Optional

import pandas as pd
from tqdm import tqdm

sys.path.append("../..")
from src.data_utils import cal_mol_weight, smiles2selfies

assign_activity = __import__("06_assign_activity_to_mmp")
compare_values = assign_activity.compare_values
tqdm.pandas()


def drop_unknown_target(df: pd.DataFrame) -> pd.DataFrame:
    original_size = df.shape[0]
    known_targets = set(
        pd.read_csv("../../data/all/all_prot_seq_less1495.csv")["target_chembl_id"].to_list()
    )

    df["known_target"] = df["target"].progress_apply(lambda s: 1 if s in known_targets else None)
    df = df.dropna()
    df = df.drop("known_target", axis=1)
    print(f"Dropped {original_size - df.shape[0]} items because of unknown target id.")
    return df


def standardize_ug_mL(df: pd.DataFrame):
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        activity, unit = row["unit"].split(":")
        if unit == "ug.mL-1":
            mw_a = cal_mol_weight(row["mol_a_smiles"])
            mw_b = cal_mol_weight(row["mol_b_smiles"])
            row["standard_value_a"] /= mw_a
            row["standard_value_b"] /= mw_b
            row["unit"] = f"{activity}:nM"

    return df


def perm_mmp(df: pd.DataFrame) -> pd.DataFrame:
    def compare(row: pd.DataFrame) -> Optional[Literal["keep", "revert"]]:
        _, unit = row["unit"].split(":")
        return compare_values(row["standard_value_a"], row["standard_value_b"], unit)

    def permutate(row: pd.DataFrame) -> None:
        row["mol_a"], row["mol_b"] = row["mol_b"], row["mol_a"]
        row["mol_a_smiles"], row["mol_b_smiles"] = row["mol_b_smiles"], row["mol_a_smiles"]
        row["standard_value_a"], row["standard_value_b"] = (
            row["standard_value_b"],
            row["standard_value_a"],
        )

    df["permutation"] = df.progress_apply(compare, axis=1)
    df = df.dropna()

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row["permutation"] == "keep":
            continue
        elif row["permutation"] == "revert":
            permutate(row)
        else:
            assert False, f"Invalid permutation: {row['permutation']}"

    df = df.drop("permutation", axis=1)
    return df


def filter_exist_record(df: pd.DataFrame) -> pd.DataFrame:
    all_df = pd.read_csv(
        "../../data/finetune/finetune_dataset.csv", usecols=["mol_a", "mol_b", "target"]
    )
    records = set(
        [f"{row['target']}:{row['mol_a']}>>{row['mol_b']}" for _, row in all_df.iterrows()]
    )
    del all_df

    df = df.drop_duplicates(["mol_a", "mol_b", "target"])
    df["kept"] = df.progress_apply(
        lambda row: None if f"{row['target']}:{row['mol_a']}>>{row['mol_b']}" in records else 1,
        axis=1,
    )
    df = df.dropna()
    df = df.drop("kept", axis=1)
    return df




if __name__ == "__main__":
    # 1. Process ex-data about target and activity
    # df = pd.read_csv(
    #     "../../data/exdata/exdata.csv",
    #     usecols=[
    #         "mol_a",
    #         "mol_b",
    #         "assay",
    #         "target",
    #         "standard_value_a",
    #         "standard_value_b",
    #         "unit",
    #         "mol_a_smiles",
    #         "mol_b_smiles",
    #     ],
    # )

    # df = drop_unknown_target(df)
    # df = standardize_ug_mL(df)
    # df = perm_mmp(df)
    # df.to_csv("../../data/exdata/exdata_proc_activity.csv", index=False)

    # 2. Filter record existing in training dataset
    # df = pd.read_csv("../../data/exdata/exdata_proc_activity.csv")
    # df = filter_exist_record(df)
    # df.to_csv("../../data/exdata/exdata_proc_duplicate.csv", index=False)

    # 3. Generate MMP information
    # run 08_check_mmps.py

    # 4. Generate selfies and add sequences
    # df = pd.read_csv("../../data/exdata/exdata_proc_duplicate.csv")
    # mmp_df = pd.read_csv("../../data/exdata/exdata_proc_duplicate_mmp.csv")
    # df = pd.concat([df, mmp_df], axis=1)
    # df = df.dropna()

    # df["mol_a_selfies"] = df["mol_a_smiles"].progress_apply(smiles2selfies)
    # df["mol_b_selfies"] = df["mol_b_smiles"].progress_apply(smiles2selfies)
    # df["core_selfies"] = df["core"].progress_apply(smiles2selfies)
    # df["frag_a_selfies"] = df["frag_a"].progress_apply(smiles2selfies)
    # df["frag_b_selfies"] = df["frag_b"].progress_apply(smiles2selfies)
    # df = df.dropna()

    # drop_idx = (df["frag_a_heavy"] > df["core_heavy"]) & (df["frag_b_heavy"] > df["core_heavy"])
    # df = df[~drop_idx]

    # targets_df = pd.read_csv("../../data/all/all_prot_seq_less1495.csv")
    # target_dict = dict(zip(targets_df["target_chembl_id"], targets_df["sequence"]))
    # df["sequence"] = df["target"].progress_apply(lambda s: target_dict.get(s, None))
    # df = df.dropna()

    # df.to_csv("../../data/exdata/exdata_dataset.csv", index=False)
