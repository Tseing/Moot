import os.path as osp
import pickle
from typing import Dict, Literal, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm


def cons_assay_dict(activity_df: pd.DataFrame) -> Dict[str, Set]:
    assay_dict = {}
    grouped_df = activity_df.groupby("mol_chembl_id")
    for mol_chembl_id, assays in tqdm(grouped_df):
        assay_dict[mol_chembl_id] = set([assay for assay in assays["assay_chembl_id"]])

    return assay_dict


def cons_value_dict(activity_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    value_dict: Dict[str, Dict[str, float]] = {}
    grouped_df = activity_df.groupby(["mol_chembl_id", "assay_chembl_id"])
    for chembl_id, chembl_id_group in tqdm(grouped_df):

        key = "_".join([str(s) for s in chembl_id])
        values: Dict[str, float] = {}
        for unit, unit_group in chembl_id_group.groupby(["standard_type", "standard_units"]):
            values["_".join([str(s) for s in unit])] = unit_group["standard_value"].mean()

        value_dict[key] = values

    return value_dict


def compare_values(
    value_a: float, value_b: float, unit: str
) -> Optional[Literal["revert", "keep"]]:

    if unit == "%":
        diff = value_b - value_a
        if abs(diff) < 50:
            return None
        if diff > 0:
            return "keep"
        else:
            return "revert"
    else:
        if value_a == 0:
            return None

        diff = value_b / value_a
        if diff <= 0.2:
            return "keep"
        elif diff >= 5:
            return "revert"
        else:
            return None


def assign_opt(
    mmp_path: str,
    activity_path: str,
    save_path: str,
    assay_dict_path: str,
    value_dict_path: str,
) -> None:
    mmp_df = pd.read_csv(mmp_path)
    activ_df = pd.read_csv(activity_path)

    if not osp.exists(assay_dict_path):
        assay_dict = cons_assay_dict(activ_df)
        pickle.dump(assay_dict, open(assay_dict_path, "wb"))
    else:
        assay_dict = pickle.load(open(assay_dict_path, "rb"))
    print("Assay dict has been constructed.")

    if not osp.exists(value_dict_path):
        value_dict = cons_value_dict(activ_df)
        pickle.dump(value_dict, open(value_dict_path, "wb"))
    else:
        value_dict = pickle.load(open(value_dict_path, "rb"))
    print("Value dict has been constructed.")

    def compare_pipeline(
        df: pd.DataFrame,
    ) -> Tuple[Optional[Literal["revert", "keep"]], Optional[str]]:
        mol_a, mol_b = df["mol_a"], df["mol_b"]
        assay_a, assay_b = assay_dict.get(mol_a, None), assay_dict.get(mol_b, None)

        if assay_a is None or assay_b is None:
            return None, None

        same_assays = assay_a.intersection(assay_b)
        if len(same_assays) == 0:
            return None, None

        for assay in same_assays:
            value_dict_a, value_dict_b = (
                value_dict[f"{mol_a}_{assay}"],
                value_dict[f"{mol_b}_{assay}"],
            )
            same_unit = set(value_dict_a.keys()).intersection(value_dict_b.keys())
            if len(same_unit) == 0:
                continue

            for unit in same_unit:
                _, std_unit = unit.split("_")
                value_a, value_b = value_dict_a[unit], value_dict_b[unit]
                compare_res = compare_values(value_a, value_b, std_unit)
                if compare_res is None:
                    continue
                else:
                    return compare_res, assay

        return None, None

    tqdm.pandas()
    revert = mmp_df.progress_apply(compare_pipeline, axis=1, result_type="expand")
    revert.columns = ["opt", "assay_chembl_id"]
    revert.to_csv(save_path, index=False)


def perm_mmp(mmp_path: str, assign_path: str) -> None:
    def perm_pipeline(df: pd.DataFrame) -> Tuple[str, str]:
        mol_a, mol_b = df["mol_a"], df["mol_b"]
        if df["opt"] == "keep":
            return mol_a, mol_b
        elif df["opt"] == "revert":
            return mol_b, mol_a
        else:
            assert False

    mmp_df = pd.read_csv(mmp_path)
    assign_df = pd.read_csv(assign_path)
    df_dir, _ = osp.split(assign_path)

    df = pd.concat([mmp_df, assign_df], axis=1, ignore_index=True)
    df.columns = ["mol_a", "mol_b", "opt", "assay_chembl_id"]
    df.dropna(how="any", axis=0, inplace=True)

    del mmp_df
    del assign_df

    tqdm.pandas()
    permed_df = df.progress_apply(perm_pipeline, axis=1, result_type="expand")
    permed_df = pd.concat([permed_df, df["assay_chembl_id"]], axis=1, ignore_index=True)
    permed_df.columns = ["mol_a", "mol_b", "assay_chembl_id"]
    permed_df.to_csv(osp.join(df_dir, "permed_mmp.csv"), index=False)


if __name__ == "__main__":
    # assign_opt(
    #     "../../data/1bond/mmp_pair_1bond_raw.csv",
    #     "../../data/finetune/merged_activities.csv",
    #     "../../data/finetune/assign_opt.csv",
    #     "../../data/finetune/assay_dict.pkl",
    #     "../../data/finetune/value_dict.pkl",
    # )
    perm_mmp("../../data/1bond/mmp_pair_1bond_raw.csv", "../../data/finetune/assign_opt.csv")
