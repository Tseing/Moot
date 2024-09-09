import sys

sys.path.append("../..")


from src.data_utils import DatasetMetrics, ResultMetrics
from utils import DataReader

if __name__ == "__main__":
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.warning")

    save_path = "../../output/pretrain/0814pretrain_transformer_selfies_top10_mols.html"
    # df = DataReader.prepare_inp_df("pretrain_test")
    # metrics = DatasetMetrics(df)
    topk = 10
    df = DataReader.prepare_out_df(
        "../../output/pretrain/0814pretrain_transformer_selfies_top10.csv", topk=topk
    )
    # print(df.columns)
    # print(df.shape)
    # print(df.head())
    # for i, s in enumerate(df["src"]):
    #     try:
    #         src = "".join(s.strip().split(" "))
    #     except Exception as e:
    #         print(f"Exception: {s}")
    #         raise e
    # assert False
    metrics = ResultMetrics(df, data_format="SELFIES", topk=topk)
    metrics.visual(save_path, nrows=200)
