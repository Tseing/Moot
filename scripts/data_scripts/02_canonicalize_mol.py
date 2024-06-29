import sys

import pandas as pd
from pandarallel import pandarallel

sys.path.append("../..")
from src.data_utils import canonicalize_smiles, smiles2selfies
from src.tokenizer import SmilesTokenizer, SelfiesTokenizer


def canonicalize_mol(df_path: str, save_path: str) -> None:
    pandarallel.initialize(nb_workers=20, progress_bar=True)
    df = pd.read_csv(df_path)
    pre_size = df.shape[0]
    print(f"Original size: {pre_size}.")

    df = df.dropna(how="any")
    print(
        f"Non-nan size: {df.shape[0]}. There are '{pre_size - df.shape[0]}' "
        f"molecules cannot be found by ChEMBL ID."
    )
    pre_size = df.shape[0]

    df["smiles"] = df["smiles"].parallel_apply(canonicalize_smiles)
    df = df.dropna(how="any")
    print(
        f"Canonical size: {df.shape[0]}. There are '{pre_size - df.shape[0]}' "
        f"molecules failed in canonicalization."
    )
    pre_size = df.shape[0]

    df["selfies"] = df["smiles"].parallel_apply(smiles2selfies)
    df = df.dropna(how="any")
    print(
        f"Full size: {df.shape[0]}. There are '{pre_size - df.shape[0]}' "
        f"molecules failed in SELFIES conversion."
    )

    df.to_csv(save_path, index=False)


def smiles_word_table(all_str: pd.Series) -> None:
    tokenizer = SmilesTokenizer()
    word_table = tokenizer.build_word_table(
        all_str, dump_path="../../data/all/smiles_word_table.yaml"
    )
    print(word_table[:30])


def selfies_word_table(all_str: pd.Series) -> None:
    all_str = all_str.parallel_apply(smiles2selfies)
    all_str = all_str.dropna()
    tokenizer = SelfiesTokenizer()
    word_table = tokenizer.build_word_table(
        all_str, dump_path="../../data/all/selfies_word_table.yaml"
    )
    print(word_table[:30])


def cons_vocab(df_path: str):
    df = pd.read_csv(df_path)
    smiles_tokenizer = SmilesTokenizer()
    selfies_tokenizer = SelfiesTokenizer()

    smiles_vocab = smiles_tokenizer.build_word_table(
        df["smiles"], "../../data/all/smiles_word_table.yaml"
    )
    print(f"SMILES vocab: {smiles_vocab[:30]} ...")
    selfies_vocab = selfies_tokenizer.build_word_table(
        df["selfies"], "../../data/all/selfies_word_table.yaml"
    )
    print(f"SELFIES vocab: {selfies_vocab[:30]} ...")


def cal_token_num(df_path: str, save_path: str) -> None:
    pandarallel.initialize(nb_workers=20, progress_bar=True)
    df = pd.read_csv(df_path)

    smiles_tokenizer = SmilesTokenizer()
    selfies_tokenizer = SelfiesTokenizer()
    smiles_tokenizer.load_word_table("../../data/all/smiles_word_table.yaml")
    selfies_tokenizer.load_word_table("../../data/all/selfies_word_table.yaml")

    df["smiles_token_num"] = df["smiles"].parallel_apply(
        lambda s: smiles_tokenizer.tokenize(s).shape[0]
    )
    df["selfies_token_num"] = df["selfies"].parallel_apply(
        lambda s: selfies_tokenizer.tokenize(s).shape[0]
    )

    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    # canonicalize_mol(
    #     "../../data/all/all_unique_mol_smiles.csv",
    #     "../../data/all/all_unique_mol.csv",
    # )
    # cons_vocab("../../data/all/all_unique_mol.csv")
    cal_token_num("../../data/all/all_unique_mol.csv", "../../data/all/all_unique_mol_wnum.csv")
