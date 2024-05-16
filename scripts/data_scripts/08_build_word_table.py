import sys

sys.path.append("../..")

import pandas as pd

from src.tokenizer import MMPTokenizer

if __name__ == "__main__":
    data_path = "../../data/1bond/mmp_pair_1bond.csv"
    tokenizer = MMPTokenizer()

    df = pd.read_csv(data_path)
    all_str = pd.concat([df["col1_smiles"], df["col2_smiles"]], axis=0, ignore_index=True)
    word_table = tokenizer.build_word_table(all_str, dump_path="../../data/word_table.yaml")
    print(word_table[:30])

