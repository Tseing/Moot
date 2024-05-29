import pickle
import sys

import pandas as pd
from pandarallel import pandarallel

sys.path.append("../..")
from src.data_utils import smiles2selfies
from src.tokenizer import SelfiesTokenizer, SmilesTokenizer


def smiles_word_table(all_str: pd.Series) -> None:
    tokenizer = SmilesTokenizer()
    word_table = tokenizer.build_word_table(all_str, dump_path="../../data/smiles_word_table.yaml")
    print(word_table[:30])


def selfies_word_table(all_str: pd.Series) -> None:
    all_str = all_str.parallel_apply(smiles2selfies)
    all_str = all_str.dropna()
    tokenizer = SelfiesTokenizer()
    word_table = tokenizer.build_word_table(all_str, dump_path="../../data/selfies_word_table.yaml")
    print(word_table[:30])


if __name__ == "__main__":
    data_path = "../../data/1bond/mmp_pair_1bond_unique_id.csv"
    lookup_smiles_path = "../../data/1bond/lookup_smiles.pkl"
    pandarallel.initialize(nb_workers=20, progress_bar=True)

    df = pd.read_csv(data_path)
    lookup_smiles = pickle.load(open(lookup_smiles_path, "rb"))

    all_str = df.iloc[:, 0].parallel_apply(lambda id: lookup_smiles.get(id, None))
    all_str = all_str.dropna()
    selfies_word_table(all_str)
