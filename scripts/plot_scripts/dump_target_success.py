import pickle
import sys

from tqdm import tqdm

sys.path.append("../..")


from src.data_metrics import MolMetrics
from utils import DataReader

if __name__ == "__main__":
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.*")

    test_data_path = "../../data/finetune/runtime/datasets_seed_0/finetune_dataset_test.csv"
    data = {}

    df = DataReader.prepare_mol_df(
        test_data_path,
        "../../output/top1/train_transformer_smiles_top1.csv",
        topk=1,
    )
    metrics = MolMetrics(df, "SMILES")
    metrics.concat_tokens()
    metrics.cano_smiles("out")
    metrics.df = metrics.df.dropna()

    target_df = metrics.df.groupby("target")
    result = {}
    for target, df in tqdm(target_df):
        inp_df = df.groupby("mol_a_smiles")
        expected = {inp: set(expected_df["mol_b_smiles"].to_list()) for inp, expected_df in inp_df}

        cnt = 0
        for _, row in df.iterrows():
            if row["out"] in expected[row["mol_a_smiles"]]:
                cnt += 1

        # positive, total
        result[target] = (cnt, df.shape[0])

    pickle.dump(result, open("smiles_transformer_target_success.pkl", "wb"))
