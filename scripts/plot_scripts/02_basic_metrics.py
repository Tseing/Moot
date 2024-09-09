import sys

sys.path.append("../..")


from src.data_utils import ResultMetrics
from utils import DataReader

if __name__ == "__main__":
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.*")

    topk = 1
    df = DataReader.prepare_out_df(
        "../../output/finetune/0819finetune_epoch10_transformer_selfies_top1.csv",
        topk=topk
    )

    metrics = ResultMetrics(df, data_format="SELFIES", topk=topk)
    metrics.basic_metric()
