import sys

sys.path.append("../..")


import pandas as pd
from tqdm import tqdm

from src.data_utils import LookupDict, MMPChecker

if __name__ == "__main__":
    tqdm.pandas()
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
