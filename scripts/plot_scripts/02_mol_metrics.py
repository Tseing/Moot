import sys

sys.path.append("../..")


from src.data_metrics import MolMetrics
from utils import DataReader

if __name__ == "__main__":
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.*")

    topk = 1
    data_format = "SMILES"

    # data_path = f"../../output/top{topk}/train_optformer_{data_format.lower()}_top{topk}.csv"
    data_path = "../../output/pipeline_spliced/train_pipeline_transformer_smiles_extop1.csv"
    df = DataReader.prepare_mol_df(
        # "../../data/finetune/runtime/datasets_seed_0/finetune_dataset_test.csv",
        "../../data/exdata/runtime/exdata_dataset.csv",
        data_path,
        topk=topk,
    )

    metrics = MolMetrics(df, data_format=data_format, topk=topk, worker=10)
    metrics.basic_metric()
    print(data_path)
