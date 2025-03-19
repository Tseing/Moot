import sys

sys.path.append("../..")


from src.data_metrics import MolMetrics
from utils import DataReader

if __name__ == "__main__":
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.*")

    topk = 10
    data_path = f"../../output/top{topk}/train_transformer_smiles_top{topk}.csv"
    df = DataReader.prepare_mol_df(
        "../../data/finetune/runtime/datasets_seed_0/finetune_dataset_test.csv",
        data_path,
        topk=topk,
    )

    metrics = MolMetrics(df, data_format="SMILES", topk=topk, worker=10)
    metrics.basic_metric()
    print(data_path)