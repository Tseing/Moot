import pickle
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from setup import init_model
from torch import nn
from tqdm import tqdm

sys.path.append("..")
from src.tokenizer import StrTokenizer
from src.utils import pad_sequence


def extract_feat(
    model: nn.Module,
    data_path: str,
    mol_col: str,
    batch_size: int,
    mol_tokenizer: StrTokenizer,
    mol_max_len: int,
    prot_tokenizer: Optional[StrTokenizer],
    prot_max_len: Optional[int],
    device=torch.device("cpu"),
) -> np.ndarray:
    def _get_tensor(seq_list: List[str], tokenizer: StrTokenizer, max_len: int):
        seqs = []
        for seq in seq_list:
            tokens = tokenizer.tokenize(seq)
            if len(tokens) > max_len - 5:
                seqs.append(tokens[: max_len - 5])
            else:
                seqs.append(tokens)

        padded_seqs = (
            torch.Tensor(
                np.stack(
                    [pad_sequence(seq, max_len, tokenizer.pad_id, left_pad=False) for seq in seqs],
                )
            )
            .to(device)
            .int()
        )
        return padded_seqs

    chunks = pd.read_csv(data_path, chunksize=batch_size)

    feats = []
    for chunk in tqdm(chunks):
        with torch.no_grad():

            mol_list, protein_list = chunk[mol_col].to_list(), chunk["sequence"].to_list()
            mols = _get_tensor(mol_list, mol_tokenizer, mol_max_len)

            if prot_tokenizer:
                protein_list = [
                    "".join(f"-{token}" for token in protein) for protein in protein_list
                ]

                prots = _get_tensor(protein_list, prot_tokenizer, prot_max_len)

                feat, mask = model.enc_forward(mols, prots)

            else:
                feat, mask = model.enc_forward(mols)

            feat = feat.transpose(0, 1)
            feat[mask, :] = 0
            feats.append(feat.cpu())

    return torch.concatenate(feats, dim=0).numpy()


if __name__ == "__main__":
    label = "probe_optformer_selfies"
    data_path = "../data/bindingdb/train.csv"
    save_path = "../output/bindingdb_feat/optformer_selfies_train.pkl"

    model, mol_tokenizer, prot_tokenizer = init_model(label)
    feats = extract_feat(
        model,
        data_path,
        "selfies",
        32,
        mol_tokenizer,
        250,
        prot_tokenizer,
        1500,
        torch.device("cuda:0"),
    )
    pickle.dump(feats, open(save_path, "wb"))
