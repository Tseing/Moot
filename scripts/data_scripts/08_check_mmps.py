import sys

sys.path.append("../..")

from typing import Literal

import pandas as pd
from tqdm import tqdm

from src.data_utils import LookupDict, MMPChecker, selfies2smiles


# Training datasets
def check_inp_mmp():
    smiles_dict = LookupDict("../../data/all/smiles_lookup.pkl")
    checker = MMPChecker()
    df = pd.read_csv("../../data/pretrain/pretrain.csv", usecols=["mol_a", "mol_b"])

    df["mol_a_smi"] = df["mol_a"].progress_apply(smiles_dict.lookup)
    df["mol_b_smi"] = df["mol_b"].progress_apply(smiles_dict.lookup)
    smiles_list = pd.concat([df["mol_a_smi"], df["mol_a_smi"]], axis=0).drop_duplicates().to_list()

    checker.cache_records(smiles_list, "./cached_records.pkl", worker=128)

    check_results = df.progress_apply(
        lambda df: checker.check_mmp(df["mol_a_smi"], df["mol_b_smi"]), axis=1
    ).to_list()

    mmp_infos = [
        result.parse_list() if result is not None else [None] * 6 for result in check_results
    ]
    mmp_df = pd.DataFrame(
        mmp_infos,
        columns=["core", "core_heavy", "frag_a", "frag_a_heavy", "frag_b", "frag_b_heavy"],
    )
    mmp_df.to_csv("../../data/pretrain/pretrain_mmp.csv", index=False)


# Generated molecule
def check_out_mmp(gen_path: str, gen_format: Literal["SMILES", "SELFIES"], save_path: str):
    # Process Output SMILES / SELFIES
    gen_df = pd.read_csv(gen_path, usecols=["input", "output"])
    inp_smi = gen_df["input"].progress_apply(lambda s: "".join(s.strip().split(" ")))
    out_smi = gen_df["output"].progress_apply(
        lambda s: "".join(s.strip("{eos}").strip().split(" "))
    )

    if gen_format == "SELFIES":
        inp_smi = inp_smi.progress_apply(selfies2smiles)
        out_smi = out_smi.progress_apply(selfies2smiles)

    del gen_df
    df = pd.concat([inp_smi, out_smi], axis=1)
    df.columns = ["inp_smi", "out_smi"]

    smiles_list = pd.concat([inp_smi, out_smi], axis=0).drop_duplicates().to_list()

    checker = MMPChecker()
    checker.cache_records(smiles_list, "./cached_records.pkl", worker=128)

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
    mmp_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.*")
    tqdm.pandas()
    check_out_mmp(
        gen_path="../../output/finetune/0819finetune_epoch10_transformer_selfies_top1.csv",
        gen_format="SELFIES",
        save_path="../../output/finetune/0819finetune_epoch10_transformer_selfies_top1_mmp.csv",
    )
