import os.path as osp
import pickle
from typing import Dict

import pandas as pd
from numpy.typing import NDArray
from terminaltables import AsciiTable

cons_utils = __import__("09_cons_pretrain_datasets")


def comp_mmp(mmp_csv: str) -> pd.DataFrame:
    mmp_df = pd.read_csv(mmp_csv)
    # optimization of chain atoms should less than core atoms
    drop_idx = (mmp_df["frag_a_heavy"] > mmp_df["core_heavy"]) & (
        mmp_df["frag_b_heavy"] > mmp_df["core_heavy"]
    )
    return drop_idx


def gen_data_idxes(mmp_csv: str, pretrain_idxes: str, save_dir: str) -> Dict[str, NDArray]:
    split_idxes = pickle.load(open(pretrain_idxes, "rb"))
    drop_idx = comp_mmp(mmp_csv)
    data_idxes = {k: v[~drop_idx[v]] for k, v in split_idxes.items()}

    save_path = osp.join(save_dir, "split_idxes.pkl")
    pickle.dump(data_idxes, open(save_path, "wb"))

    return data_idxes


def summary(total_df: pd.DataFrame, datasets: Dict[str, pd.DataFrame]):
    table_data = [["Dataset", "Unique Items", "Nan Items", "Shape"]]
    table_data.extend(cons_utils.DatasetSplit._stats_df(total_df, "all"))
    for name in ["train", "val", "test"]:
        table_data.extend(cons_utils.DatasetSplit._stats_df(datasets[name], name))

    table = AsciiTable(table_data)
    table.title = f"Finetune Dataset Summary"
    table.inner_row_border = True
    print(table.table)


def dump_dataset(total_dataset: str, data_idxes: dict, save_dir: str, save_name: str):
    df = pd.read_csv(total_dataset)
    datasets = {dataset: df.iloc[data_idxes[dataset]] for dataset in data_idxes}
    summary(df, datasets)

    for dataset in datasets:
        save_path = osp.join(save_dir, f"{save_name}_{dataset}.csv")
        datasets[dataset].to_csv(save_path, index=False)


if __name__ == "__main__":
    # data_idxes = gen_data_idxes(
    #     "../../data/pretrain/pretrain_mmp.csv",
    #     "../../data/pretrain/runtime/chembl_id_seed_0/split_idxes.pkl",
    #     "../../data/finetune/runtime/chembl_id_seed_0"
    # )
    # data_idxes = pickle.load(
    #     open("../../data/finetune/runtime/chembl_id_seed_0/split_idxes.pkl", "rb")
    # )
    # dump_dataset(
    #     "../../data/pretrain/pretrain.csv",
    #     data_idxes,
    #     "../../data/finetune/runtime/chembl_id_seed_0",
    #     "finetune",
    # )

    cons_utils.fetch_smiles_selfies_pipeline(
        "../../data/finetune/runtime/chembl_id_seed_0",
        "../../data/finetune/runtime/datasets_seed_0",
    )
