import sys

import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem

sys.path.append("../..")

from src.data_metrics import FragMetrics
from src.utils.molecule import splice_mol
from utils import DataReader


def splice_mol_or_eos(core: str, frag: str) -> str:
    eos = "{eos}"
    if None in (core, frag):
        return eos
    try:
        mol = splice_mol(core, frag)
    except:
        return eos

    if mol is None:
        return eos

    return Chem.MolToSmiles(mol)


if __name__ == "__main__":
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.*")
    pandarallel.initialize(nb_workers=10, progress_bar=True)

    data_format = "SELFIES"
    model = "transformer"
    suffix = "extop1"

    file_name = f"train_pipeline_{model}_{data_format.lower()}_{suffix}.csv"
    test_path = f"../../data/pipeline/train_{model}_{data_format.lower()}_{suffix}.csv"

    if suffix == "top1":
        input_path = "../../data/finetune/runtime/datasets_seed_0/finetune_dataset_test.csv"
    elif suffix == "extop1":
        input_path = "../../data/exdata/runtime/exdata_dataset.csv"
    else:
        assert False

    save_path = f"../../output/pipeline_spliced/{file_name}"
    data_path = f"../../output/pipeline/{file_name}"

    input_df = pd.read_csv(input_path, usecols=["mol_a_smiles"])

    gen_df = DataReader.prepare_frag_df(
        test_path,
        data_path,
        topk=1,
    )

    metrics = FragMetrics(gen_df, data_format=data_format, topk=1, worker=10)
    metrics.concat_tokens()

    assert gen_df.shape[0] == input_df.shape[0]
    gen_df["input"] = input_df["mol_a_smiles"]
    gen_df["output"] = metrics.df.parallel_apply(
        lambda row: splice_mol_or_eos(row["core"], row["out"]), axis=1
    )
    gen_df[["input", "output"]].to_csv(save_path, index=False)
