from typing import List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
from jaxtyping import Int
from numpy.typing import DTypeLike, NDArray
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import TypeGuard

from .tokenizer import StrTokenizer
from .typing import Input, Target
from .utils import pad_sequence, pad_sequences


class PairDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        data_cols: Tuple[str, ...],
        pad_batch: bool,
        dtype: DTypeLike = np.int32,
        **kwargs,
    ) -> None:
        super().__init__()
        df = pd.read_csv(data_path, usecols=data_cols, **kwargs)
        self.data = df.to_numpy()
        self.len = self.data.shape[0]
        self.dtype = dtype
        self.pad_batch = pad_batch

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> Tuple[Input, Target]:
        transformed_data = self.transform(self.data[index])

        if self.pad_batch:
            return transformed_data
        else:
            return self.pad_transform(transformed_data)

    def transform(self, row: NDArray) -> Tuple[Input, Target]:
        raise NotImplementedError

    def pad_transform(self, data: Tuple[Input, Target]) -> Tuple[Input, Target]:
        raise NotImplementedError

    def pad_batch_fn(
        self, data: List[Tuple[Input, Target]]
    ) -> Tuple[Union[tuple, Int[Tensor, "bsz ..."]], Int[Tensor, "bsz ..."]]:
        raise NotImplementedError


class MolPairDataset(PairDataset):
    def __init__(
        self,
        data_path: str,
        data_cols: Tuple[str, str],
        tokenizer: StrTokenizer,
        max_len: Optional[int],
        left_pad: Optional[bool],
        pad_batch: bool,
        dtype: DTypeLike = np.int32,
        **kwargs,
    ) -> None:
        super().__init__(data_path, data_cols, pad_batch, dtype, **kwargs)
        self.tokenizer = tokenizer
        self.pad_batch = pad_batch

        if pad_batch:
            assert max_len is None and left_pad is None, (
                f"`max_len` is '{max_len}' and `left_pad` is '{left_pad}'"
                f"when `pad_batch` is '{pad_batch}'"
            )

        else:
            assert max_len is not None or left_pad is not None, (
                f"`max_len` is '{max_len}' and `left_pad` is '{left_pad}'"
                f"when `pad_batch` is '{pad_batch}'"
            )
            self.left_pad: TypeGuard[bool] = left_pad
            self.max_len: TypeGuard[int] = max_len

        self.pad_value = tokenizer.vocab2index[tokenizer.pad]

    def transform(self, row: NDArray) -> Tuple[Input, Target]:
        src, tgt = row
        tokenized_src = self.tokenizer.tokenize(src).astype(self.dtype)
        tokenized_tgt = self.tokenizer.tokenize(tgt).astype(self.dtype)

        return tokenized_src, tokenized_tgt

    def pad_transform(self, data: Tuple[Input, Target]) -> Tuple[Input, Target]:
        tokenized_src, tokenized_tgt = data
        tokenized_src = cast(NDArray, tokenized_src)
        padded_src = pad_sequence(tokenized_src, self.max_len, self.pad_value, self.left_pad)
        padded_tgt = pad_sequence(tokenized_tgt, self.max_len, self.pad_value, self.left_pad)
        return padded_src, padded_tgt

    def pad_batch_fn(
        self, data: List[Tuple[Input, Target]]
    ) -> Tuple[Int[Tensor, "bsz ..."], Int[Tensor, "bsz ..."]]:
        srcs, tgts = list(zip(*data))

        padded_srcs = torch.tensor(pad_sequences(srcs, self.pad_value, self.left_pad))
        padded_tgts = torch.tensor(pad_sequences(tgts, self.pad_value, self.left_pad))

        return padded_srcs, padded_tgts


class MolProtPairDataset(PairDataset):
    def __init__(
        self,
        data_path: str,
        data_cols: Tuple[str, str, str],
        mol_tokenizer: StrTokenizer,
        prot_tokenizer: StrTokenizer,
        mol_max_len: int,
        prot_max_len: int,
        left_pad: bool,
        pad_batch: bool,
        dtype: DTypeLike = np.int32,
        **kwargs,
    ) -> None:
        super().__init__(data_path, data_cols, False, dtype, **kwargs)
        self.mol_tokenizer = mol_tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.pad_batch = pad_batch

        if pad_batch:
            assert mol_max_len is None and prot_max_len is None and left_pad is None, (
                f"`mol_max_len` is '{mol_max_len}', `prot_max_len` is '{prot_max_len}' "
                f"and `left_pad` is '{left_pad}' when `pad_batch` is '{pad_batch}'"
            )

        else:
            assert mol_max_len is not None or prot_max_len is not None or left_pad is not None, (
                f"`mol_max_len` is '{mol_max_len}', `prot_max_len` is '{prot_max_len}' "
                f"and `left_pad` is '{left_pad}' when `pad_batch` is '{pad_batch}'"
            )
            self.left_pad: TypeGuard[bool] = left_pad
            self.mol_max_len = mol_max_len
            self.prot_max_len = prot_max_len

        self.mol_pad_value = mol_tokenizer.vocab2index[mol_tokenizer.pad]
        self.prot_pad_value = prot_tokenizer.vocab2index[prot_tokenizer.pad]

    def transform(self, row: NDArray) -> Tuple[Input, Target]:
        mol, tgt, prot = row

        tokenized_mol = self.mol_tokenizer.tokenize(mol).astype(self.dtype)
        prot = "".join([f"-{letter}" for letter in prot])
        tokenized_prot = self.prot_tokenizer.tokenize(prot).astype(self.dtype)
        tokenized_tgt = self.mol_tokenizer.tokenize(tgt).astype(self.dtype)

        return (tokenized_mol, tokenized_prot), tokenized_tgt

    def pad_transform(self, data: Tuple[Input, Target]) -> Tuple[Input, Target]:
        (tokenized_mol, tokenized_prot), tokenized_tgt = data
        padded_mol = pad_sequence(
            tokenized_mol, self.mol_max_len, self.mol_pad_value, self.left_pad
        )
        padded_prot = pad_sequence(
            tokenized_prot, self.prot_max_len, self.prot_pad_value, self.left_pad
        )
        padded_tgt = pad_sequence(
            tokenized_tgt, self.mol_max_len, self.mol_pad_value, self.left_pad
        )

        return (padded_mol, padded_prot), padded_tgt

    def pad_batch_fn(
        self, data: List[Tuple[Input, Target]]
    ) -> Tuple[Tuple[Int[Tensor, "bsz ..."], Int[Tensor, "bsz ..."]], Int[Tensor, "bsz ..."]]:

        srcs, tgts = tuple(zip(*data))
        mols, prots = tuple(zip(*srcs))

        padded_mols = torch.tensor(pad_sequences(mols, self.mol_pad_value, self.left_pad))
        padded_prots = torch.tensor(pad_sequences(prots, self.prot_pad_value, self.left_pad))
        padded_tgts = torch.tensor(pad_sequences(tgts, self.mol_pad_value, self.left_pad))

        return (padded_mols, padded_prots), padded_tgts
