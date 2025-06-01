import sys

import pandas as pd
from pandarallel import pandarallel

sys.path.append("../..")


from src.data_utils import smiles2selfies
from utils import DataReader


def smiles2selfies_or_unk(s: str) -> str:
    selfies = smiles2selfies(s)
    if selfies is None:
        return "{unk}"
    else:
        return selfies


if __name__ == "__main__":
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.*")
    pandarallel.initialize(nb_workers=10, progress_bar=True)

    file_name = "train_optformer_smiles_extop1.csv"
    save_path = f"../../data/pipeline/{file_name}"
    data_path = f"../../output/top1/{file_name}"
    # test_path = "../../data/finetune/runtime/datasets_seed_0/finetune_dataset_test.csv"
    test_path = "../../data/exdata/runtime/exdata_dataset.csv"

    sequence_df = pd.read_csv(test_path, usecols=["sequence"])

    gen_df = DataReader.prepare_mol_df(
        test_path,
        data_path,
        topk=1,
    )

    assert gen_df.shape[0] == sequence_df.shape[0]
    df = pd.concat([gen_df[["gen_core", "gen_frag_a"]], sequence_df], axis=1)
    df = df.reindex(columns=["gen_core", "gen_frag_a", "sequence"])
    df.columns = ["core", "frag_a", "sequence"]

    df["core"] = df["core"].parallel_apply(lambda s: "{unk}" if pd.isna(s) else s)
    df["frag_a"] = df["frag_a"].parallel_apply(lambda s: "{unk}" if pd.isna(s) else s)

    df["core_selfies"] = df["core"].parallel_apply(
        lambda s: "{unk}" if s == "{unk}" else smiles2selfies_or_unk(s)
    )
    df["frag_a_selfies"] = df["frag_a"].parallel_apply(
        lambda s: "{unk}" if s == "{unk}" else smiles2selfies_or_unk(s)
    )

    # [core, core_selfies, frag_a, frag_a_selfies, sequence]
    df.to_csv(save_path, index=False)
