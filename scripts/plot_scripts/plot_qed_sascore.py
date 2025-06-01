import pickle
import sys

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
        "../../output/spliced/train_frag_optformer_selfies_top1.csv",
        topk=1,
    )
    metrics = MolMetrics(df, "SMILES")
    metrics.concat_tokens()
    data["optformer selfies qed"] = metrics.metric_qed("out")
    data["optformer selfies sa"] = metrics.metric_sascore("out")

    data["test qed"] = metrics.metric_qed("mol_b_smiles")
    data["test sa"] = metrics.metric_sascore("mol_b_smiles")

    df = DataReader.prepare_mol_df(
        test_data_path,
        "../../output/spliced/train_frag_optformer_smiles_top1.csv",
        topk=1,
    )
    metrics = MolMetrics(df, "SMILES")
    metrics.concat_tokens()
    data["optformer smiles qed"] = metrics.metric_qed("out")
    data["optformer smiles sa"] = metrics.metric_sascore("out")

    df = DataReader.prepare_mol_df(
        test_data_path,
        "../../output/spliced/train_frag_transformer_selfies_top1.csv",
        topk=1,
    )
    metrics = MolMetrics(df, "SMILES")
    metrics.concat_tokens()
    data["transformer selfies qed"] = metrics.metric_qed("out")
    data["transformer selfies sa"] = metrics.metric_sascore("out")

    df = DataReader.prepare_mol_df(
        test_data_path,
        "../../output/spliced/train_frag_transformer_smiles_top1.csv",
        topk=1,
    )
    metrics = MolMetrics(df, "SMILES")
    metrics.concat_tokens()
    data["transformer smiles qed"] = metrics.metric_qed("out")
    data["transformer smiles sa"] = metrics.metric_sascore("out")

    pickle.dump(data, open("../../output/pkl/frag-qed-sascore-inp.pkl", "wb"))
