import sys
from typing import Literal
import pandas as pd
from rdkit import rdBase

sys.path.append("../..")


from src.data_metrics import MolMetrics
from utils import DataReader

rdBase.DisableLog("rdApp.*")

TEST_PATH = "../../data/finetune/runtime/datasets_seed_0/finetune_dataset_test.csv"


def add_mol_chemblid(data_path: str, data_format: Literal["SMILES", "SELFIES"], save_path: str):
    target_df = pd.read_csv(TEST_PATH, usecols=["target"])

    df = DataReader.prepare_mol_df(
        TEST_PATH,
        data_path,
        topk=1,
    )

    metrics = MolMetrics(df, data_format=data_format, topk=1, worker=10)
    metrics.concat_tokens()
    metrics.cano_smiles("out")

    target_df["out"] = metrics.df["out"]
    target_df.dropna().to_csv(save_path, index=False)


if __name__ == "__main__":

    # end2end
    # for filename, data_format in [
    #     ("train_transformer_smiles_top1.csv", "SMILES"),
    #     ("train_transformer_selfies_top1.csv", "SELFIES"),
    #     ("train_optformer_smiles_top1.csv", "SMILES"),
    #     ("train_optformer_selfies_top1.csv", "SELFIES"),
    # ]:

    #     add_mol_chemblid(
    #         f"../../output/top1/{filename}",
    #         data_format,
    #         f"../../output/for-docking/{filename}",
    #     )

    # frag
    for filename in [
        "train_frag_transformer_smiles_top1.csv",
        "train_frag_transformer_selfies_top1.csv",
        "train_frag_optformer_smiles_top1.csv",
        "train_frag_optformer_selfies_top1.csv",
    ]:

        add_mol_chemblid(
            f"../../output/spliced/{filename}",
            data_format="SMILES",
            save_path=f"../../output/for-docking/{filename}",
        )