import sys

sys.path.append("../..")


from src.data_utils import DatasetMetrics
from utils import DataReader

if __name__ == "__main__":
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.warning")

    save_path = "../../output/pretrain/pretrain_test_mols.html"
    df = DataReader.prepara_inp_df("pretrain_test")
    metrics = DatasetMetrics(df)
    metrics.visual(save_path, nrows=200)
