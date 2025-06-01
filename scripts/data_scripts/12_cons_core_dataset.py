import re

import pandas as pd
from pandarallel import pandarallel


def merge(df, atom_df):
    df = pd.read_csv(df)
    atom_df = pd.read_csv(atom_df)

    return pd.concat([df, atom_df], axis=1)


regex = re.compile("[A-Z][a-z]?|\*")


def atom2token(s):
    return "".join([f"[{atom}]" for atom in regex.findall(s)])


def gen_atom_tokens(df_path, save_path):
    pandarallel.initialize(nb_workers=50, progress_bar=True)
    df = pd.read_csv(df_path)
    df["atoms"] = df["atoms"].parallel_apply(atom2token)
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    df = merge("../../data/finetune/runtime/datasets_seed_0/finetune_dataset_train.csv",
               "../../data/atom/atom_token_train.csv")
    print(df.head())
    df.to_csv("../../data/atom/runtime/atom_dataset_train.csv", index=False)
    # gen_atom_tokens("../../data/atom/atom_train.csv", "../../data/atom/atom_token_train.csv")
