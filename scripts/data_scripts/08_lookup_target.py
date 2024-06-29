import os.path as osp
from typing import Dict

import pandas as pd


def cons_assay_target_dict(df_path: str) -> Dict[str, str]:
    df = pd.read_csv(df_path)
    df = df[["assay_chembl_id", "target_chembl_id"]]
    df = df.drop_duplicates()

    return dict(zip(df["assay_chembl_id"], df["target_chembl_id"]))


def fetch_all_unique_target_id(target_dict: Dict[str, str], save_path: str) -> None:
    target_chembl_id = pd.DataFrame({"target_chembl_id": list(target_dict.values())})
    target_chembl_id = target_chembl_id.dropna(how="any").drop_duplicates()
    target_chembl_id.to_csv(save_path, index=False)


def fetch_unique_target_id(df_path: str, target_dict: Dict[str, str], save_path: str) -> None:
    df = pd.read_csv(df_path)
    target_chembl_id = df["assay_chembl_id"].apply(lambda s: target_dict.get(s, None))
    target_chembl_id.name = "target_chembl_id"
    target_chembl_id = target_chembl_id.dropna(how="any").drop_duplicates()
    target_chembl_id.to_csv(save_path, index=False)


def drop_long_prot(df_path: str, max_len: int = 1495) -> None:
    file_path, _ = osp.splitext(df_path)
    df = pd.read_csv(df_path, sep="\t")
    df = df[df["Length"] <= max_len][["From", "Sequence"]]
    df.columns = ["target_chembl_id", "sequence"]
    df.to_csv(f"{file_path}_less{max_len}.csv", index=False)

if __name__ == "__main__":
    # target_dict = cons_assay_target_dict("../../data/finetune/all_activities.csv")
    # fetch_unique_target_id(
    #     "../../data/finetune/assign_opt.csv",
    #     target_dict,
    #     "../../data/finetune/unique_target_id.csv",
    # )
    # fetch_all_unique_target_id(target_dict, "../../data/finetune/all_unique_target_id.csv")
    drop_long_prot("../../data/finetune/all_prot_seq.tsv")