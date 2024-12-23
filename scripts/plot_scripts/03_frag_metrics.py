import sys

sys.path.append("../..")


from src.data_metrics import FragMetrics
from utils import DataReader

if __name__ == "__main__":
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.*")

    topk = 250
    data_path = f"../../output/top{topk}/train_frag_optformer_selfies_top{topk}.csv"
    df = DataReader.prepare_frag_df(
        "../../data/frag/runtime/frag_test.csv",
        data_path,
        topk=topk,
    )

    metrics = FragMetrics(df, data_format="SELFIES", topk=topk, worker=20)
    metrics.basic_metric()
    print(data_path)