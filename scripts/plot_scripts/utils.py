import pickle

import pandas as pd


class DataReader:
    INP_MMP_PATH = "../../data/pretrain/pretrain_mmp.csv"

    @staticmethod
    def __prepare_df(data_path, mmp_path, data_idx) -> pd.DataFrame:
        data_df = pd.read_csv(data_path)
        mmp_df = pd.read_csv(mmp_path)
        metric_df = data_df[["mol_a", "mol_b"]]
        metric_df.columns = ["src", "tgt"]

        assert (
            len(data_idx) == metric_df.shape[0]
        ), f"Unmatched shape between csv '{metric_df.shape}' and index '{len(data_idx)}'."
        mmp_info = (
            mmp_df[["core", "frag_a", "frag_b", "core_heavy", "frag_a_heavy", "frag_b_heavy"]]
            .iloc[data_idx]
            .set_index(metric_df.index)
        )
        metric_df = pd.concat([metric_df, mmp_info], axis=1)

        return metric_df

    @staticmethod
    def prepara_inp_df(dataset: str) -> pd.DataFrame:
        data_path, idx_pkl_path = CONFIGS[dataset]
        idx_key = dataset.split("_")[-1]
        idxes = pickle.load(open(idx_pkl_path, "rb"))
        return DataReader.__prepare_df(data_path, DataReader.INP_MMP_PATH, idxes[idx_key])


CONFIGS = {
    "pretrain_train": (
        "../../data/pretrain/runtime/datasets_seed_0/pretrain_train_smiles.csv",
        "../../data/pretrain/runtime/chembl_id_seed_0/split_idxes.pkl",
    ),
    "pretrain_val": (
        "../../data/pretrain/runtime/datasets_seed_0/pretrain_val_smiles.csv",
        "../../data/pretrain/runtime/chembl_id_seed_0/split_idxes.pkl",
    ),
    "pretrain_test": (
        "../../data/pretrain/runtime/datasets_seed_0/pretrain_test_smiles.csv",
        "../../data/pretrain/runtime/chembl_id_seed_0/split_idxes.pkl",
    ),
    "finetune_train": (
        "../../data/finetune/runtime/datasets_seed_0/finetune_train_smiles.csv",
        "../../data/finetune/runtime/chembl_id_seed_0/split_idxes.pkl",
    ),
    "finetune_val": (
        "../../data/finetune/runtime/datasets_seed_0/finetune_val_smiles.csv",
        "../../data/finetune/runtime/chembl_id_seed_0/split_idxes.pkl",
    ),
    "finetune_test": (
        "../../data/finetune/runtime/datasets_seed_0/finetune_test_smiles.csv",
        "../../data/finetune/runtime/chembl_id_seed_0/split_idxes.pkl",
    ),
}
