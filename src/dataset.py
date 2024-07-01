from typing import Tuple

import numpy as np
import pandas as pd
from jaxtyping import Int
from numpy import ndarray
from numpy.typing import DTypeLike
from torch.utils.data import Dataset

from .tokenizer import StrTokenizer


class MMPDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_len: int,
        tokenizer: StrTokenizer,
        left_pad: bool = False,
        dtype: DTypeLike = np.int32,
        **kwargs,
    ) -> None:
        super().__init__()
        df = pd.read_csv(data_path, **kwargs)
        src = df.iloc[:, 0].to_numpy()
        tgt = df.iloc[:, 1].to_numpy()
        assert len(src) == len(tgt), f"Unmatched shape: src '{src.shape}', tgt '{tgt.shape}'."
        self.len = src.shape[0]
        self.src = src
        self.tgt = tgt
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.left_pad = left_pad
        self.dtype = dtype

    def __len__(self):
        return self.len

    def __getitem__(self, index: int) -> Tuple[Int[ndarray, "max_len"], Int[ndarray, "max_len"]]:
        return self.transform(self.src[index]).astype(self.dtype), self.transform(
            self.tgt[index]
        ).astype(self.dtype)

    def transform(self, seq: str) -> Int[ndarray, "max_len"]:
        tokenized_seq = self.tokenizer.tokenize(seq)
        padded_seq = self.pad_sequence(
            tokenized_seq,
            self.max_len,
            self.tokenizer.vocab2index[self.tokenizer.pad],
            self.left_pad,
        )
        return padded_seq

    @staticmethod
    def pad_sequence(
        seq: Int[ndarray, "seq"],
        max_len: int,
        pad_value: int,
        left_pad: bool = False,
    ) -> Int[ndarray, "max_len"]:
        if left_pad:
            pad_pos = (max_len - seq.shape[0], 0)
        else:
            pad_pos = (0, max_len - seq.shape[0])
        padded_atoms = np.pad(seq, pad_pos, "constant", constant_values=pad_value)

        return padded_atoms


class MolProtDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        mol_max_len: int,
        prot_max_len: int,
        mol_tokenizer: StrTokenizer,
        prot_tokenizer: StrTokenizer,
        left_pad: bool = False,
        dtype: DTypeLike = np.int32,
        **kwargs,
    ) -> None:
        super().__init__()
        df = pd.read_csv(data_path, **kwargs)
        mol_a = df["mol_a"].to_numpy()
        mol_b = df["mol_b"].to_numpy()
        prot = df["sequence"].to_numpy()

        assert len(mol_a) == len(mol_b) and len(mol_a) == len(
            prot
        ), f"Unmatched shape: src mol '{mol_a.shape}' prot '{prot.shape}' and tgt mol '{mol_b.shape}'."
        self.len = mol_a.shape[0]

        self.src_a = mol_a
        self.src_b = prot
        self.tgt = mol_b

        self.mol_max_len = mol_max_len
        self.mol_tokenizer = mol_tokenizer
        self.prot_max_len = prot_max_len
        self.prot_tokenizer = prot_tokenizer
        self.left_pad = left_pad
        self.dtype = dtype

    def __len__(self):
        return self.len

    def __getitem__(
        self, index: int
    ) -> Tuple[
        Int[ndarray, "mol_max_len"], Int[ndarray, "prot_max_len"], Int[ndarray, "mol_max_len"]
    ]:
        return self.transform(self.src_a[index], self.src_b[index], self.tgt[index])

    def transform(
        self, mol: str, prot: str, tgt: str
    ) -> Tuple[
        Int[ndarray, "mol_max_len"], Int[ndarray, "prot_max_len"], Int[ndarray, "mol_max_len"]
    ]:
        tokenized_mol = self.mol_tokenizer.tokenize(mol)
        prot = "".join([f"-{letter}" for letter in prot])
        tokenized_prot = self.prot_tokenizer.tokenize(prot)
        tokenized_tgt = self.mol_tokenizer.tokenize(tgt)

        padded_mol = self.pad_sequence(
            tokenized_mol,
            self.mol_max_len,
            self.mol_tokenizer.vocab2index[self.mol_tokenizer.pad],
            self.left_pad,
        )
        padded_prot = self.pad_sequence(
            tokenized_prot,
            self.prot_max_len,
            self.prot_tokenizer.vocab2index[self.prot_tokenizer.pad],
            self.left_pad,
        )
        padded_tgt = self.pad_sequence(
            tokenized_tgt,
            self.mol_max_len,
            self.mol_tokenizer.vocab2index[self.mol_tokenizer.pad],
            self.left_pad,
        )

        return padded_mol, padded_prot, padded_tgt

    @staticmethod
    def pad_sequence(
        seq: Int[ndarray, "seq"],
        max_len: int,
        pad_value: int,
        left_pad: bool = False,
    ) -> Int[ndarray, "max_len"]:
        if left_pad:
            pad_pos = (max_len - seq.shape[0], 0)
        else:
            pad_pos = (0, max_len - seq.shape[0])
        padded_tokens = np.pad(seq, pad_pos, "constant", constant_values=pad_value)

        return padded_tokens
