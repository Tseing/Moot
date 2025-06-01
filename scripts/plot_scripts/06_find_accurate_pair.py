import sys

sys.path.append("../..")

from src.data_metrics import MolMetrics
from utils import DataReader

if __name__ == "__main__":
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.*")

    data_format = "SMILES"
    file_name = "train_optformer_smiles_top1.csv"
    save_path = f"../../output/accurate/accurate_{file_name}"
    data_path = f"../../output/top1/{file_name}"
    test_path = "../../data/finetune/runtime/datasets_seed_0/finetune_dataset_test.csv"
    # test_path = "../../data/exdata/runtime/exdata_dataset.csv"

    gen_df = DataReader.prepare_mol_df(
        test_path,
        data_path,
        topk=1,
    )

    metrics = MolMetrics(gen_df, data_format=data_format, topk=1, worker=10)
    metrics.concat_tokens()
    gen_df["accurate"] = metrics.df["out"] == metrics.df["mol_b_smiles"]
    gen_df[["src", "out", "accurate"]].to_csv(save_path, index=False)
