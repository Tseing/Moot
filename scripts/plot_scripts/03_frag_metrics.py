import sys

sys.path.append("../..")


from src.data_metrics import FragMetrics
from utils import DataReader

if __name__ == "__main__":
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.*")

    topk = 50
    data_format = "SMILES"
    data_path = (
        f"../../output/top{topk}/train_frag_transformer_{data_format.lower()}_top{topk}.csv"
    )
    df = DataReader.prepare_frag_df(
        "../../data/frag/runtime/frag_test.csv",
        # "../../data/exdata/runtime/exdata_dataset.csv",
        data_path,
        topk=topk,
    )

    metrics = FragMetrics(df, data_format=data_format, topk=topk, worker=20)
    metrics.basic_metric()
    print(data_path)
