import os.path as osp
import sys
from typing import Literal

sys.path.append("../..")

import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

from src.data_utils import MMPChecker, selfies2smiles

tqdm.pandas()


# Training datasets
def check_inp_mmp(data_path: str):
    checker = MMPChecker()
    df = pd.read_csv(data_path, usecols=["mol_a_smiles", "mol_b_smiles"])

    smiles_list = (
        pd.concat([df["mol_a_smiles"], df["mol_b_smiles"]], axis=0).drop_duplicates().to_list()
    )

    checker.cache_records(smiles_list, "./cached_records.pkl", worker=128)

    check_results = df.progress_apply(
        lambda df: checker.check_mmp(df["mol_a_smiles"], df["mol_b_smiles"]), axis=1
    ).to_list()

    mmp_infos = [
        result.parse_list() if result is not None else [None] * 6 for result in check_results
    ]
    mmp_df = pd.DataFrame(
        mmp_infos,
        columns=["core", "core_heavy", "frag_a", "frag_a_heavy", "frag_b", "frag_b_heavy"],
    )

    mmp_df.to_csv(f"{osp.splitext(data_path)[0]}_mmp.csv", index=False)


# Generated molecule
def check_out_mmp(data_path: str, data_format: Literal["SMILES", "SELFIES"], worker: int = 10):
    # Process Output SMILES / SELFIES
    pandarallel.initialize(nb_workers=worker, progress_bar=True)
    gen_df = pd.read_csv(data_path, usecols=["input", "output"])
    inp_smi = gen_df["input"].parallel_apply(lambda s: "".join(s.strip().split(" ")))
    out_smi = gen_df["output"].parallel_apply(
        lambda s: "".join(s.strip("{eos}").strip().split(" "))
    )

    if data_format == "SELFIES":
        inp_smi = inp_smi.parallel_apply(selfies2smiles)
        out_smi = out_smi.parallel_apply(selfies2smiles)

    del gen_df
    df = pd.concat([inp_smi, out_smi], axis=1)
    df.columns = ["inp_smi", "out_smi"]

    smiles_list = pd.concat([inp_smi, out_smi], axis=0).drop_duplicates().to_list()

    checker = MMPChecker()
    checker.cache_records(smiles_list, "./cached_records.pkl", worker=128)

    # get stuck when use `parallel_apply` on large DataFrame
    check_results = df.progress_apply(
        lambda df: checker.check_mmp(df["inp_smi"], df["out_smi"]), axis=1
    ).to_list()

    mmp_infos = [
        result.parse_list() if result is not None else [None] * 6 for result in check_results
    ]
    mmp_df = pd.DataFrame(
        mmp_infos,
        columns=[
            "gen_core",
            "gen_core_heavy",
            "gen_frag_a",
            "gen_frag_a_heavy",
            "gen_frag_b",
            "gen_frag_b_heavy",
        ],
    )
    mmp_df.to_csv(f"{osp.splitext(data_path)[0]}_mmp.csv", index=False)


if __name__ == "__main__":
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.*")

    # check_inp_mmp("../../data/exdata/exdata_proc_duplicate.csv")

    check_out_mmp(
        data_path="../../output/top10/train_transformer_selfies_top10.csv",
        data_format="SELFIES",
    )
