import sys

sys.path.append("..")

import numpy as np

from src.dataset import MMPDataset
from src.tokenizer import MMPTokenizer
from src.utils import trim_seqs

if __name__ == "__main__":
    smiles = [
        "Nc1ccc(C(=O)Nc2cc(CN3CCCC3)c(O)c(CN3CCCC3)c2)cc1",
        "CN1CCC(Oc2ccc(-c3ccc(NC(=O)c4ccc(-c5ccccc5)cc4)cc3)cc2)CC1",
        "NC(=O)c1cn(-c2ccc(O)c(F)c2)c2cc(-c3ccncc3)ccc2c1=O",
        "CC1(N)Cc2ccccc21",
    ]

    tokenizer = MMPTokenizer()
    tokenizer.load_word_table("../data/smiles_word_table.yaml")

    tokens = [tokenizer.tokenize(s) for s in smiles]
    array = np.array(
        [
            MMPDataset.pad_sequence(
                seq, max_len=150, pad_value=tokenizer.vocab2index[tokenizer.pad]
            )
            for seq in tokens
        ]
    )

    trimmed_seqs = trim_seqs(array, tokenizer)
    print(trimmed_seqs)
    for l in trimmed_seqs:
        print("".join(l))
