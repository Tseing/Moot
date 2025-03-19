from typing import Callable, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
from jaxtyping import Int
from numpy.typing import DTypeLike, NDArray
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import TypeGuard

from .tokenizer import AtomTokenizer, MolTokenizer, ProteinTokenizer
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
        df = df.reindex(columns=data_cols)
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

    @property
    def collate_fn(self) -> Optional[Callable]:
        return self.pad_batch_fn if self.pad_batch else None


class MolPairDataset(PairDataset):
    def __init__(
        self,
        data_path: str,
        data_cols: Tuple[str, ...],
        tokenizer: MolTokenizer,
        max_len: Optional[int],
        left_pad: bool,
        pad_batch: bool,
        dtype: DTypeLike = np.int32,
        **kwargs,
    ) -> None:
        super().__init__(data_path, data_cols, pad_batch, dtype, **kwargs)
        self.tokenizer = tokenizer
        self.pad_batch = pad_batch

        if pad_batch:
            assert max_len is None, (
                f"`max_len` should be 'None' when `pad_batch` is '{pad_batch}', "
                f"but `max_len` is '{max_len}'."
            )

        else:
            assert max_len is not None, (
                f"`max_len` should not be 'None' when `pad_batch` is '{pad_batch}', "
                f"but `max_len` is '{max_len}'."
            )
            self.max_len: TypeGuard[int] = max_len

        self.left_pad = left_pad
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
        data_cols: Tuple[str, ...],
        mol_tokenizer: MolTokenizer,
        prot_tokenizer: ProteinTokenizer,
        mol_max_len: Optional[int],
        prot_max_len: Optional[int],
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
            assert mol_max_len is None and prot_max_len is None, (
                f"`max_len` should be 'None' when `pad_batch` is '{pad_batch}', but "
                f"`mol_max_len` is '{mol_max_len}', `prot_max_len is {prot_max_len}`."
            )

        else:
            assert mol_max_len is not None and prot_max_len is not None, (
                f"`max_len` should not be 'None' when `pad_batch` is '{pad_batch}', but "
                f"`mol_max_len` is '{mol_max_len}', `prot_max_len is {prot_max_len}`."
            )
            self.mol_max_len = mol_max_len
            self.prot_max_len = prot_max_len

        self.left_pad = left_pad
        self.pad_value = mol_tokenizer.vocab2index[mol_tokenizer.pad]

    def transform(self, row: NDArray) -> Tuple[Input, Target]:
        mol, tgt, prot = row

        tokenized_mol = self.mol_tokenizer.tokenize(mol).astype(self.dtype)
        prot = "".join([f"-{letter}" for letter in prot])
        tokenized_prot = self.prot_tokenizer.tokenize(prot).astype(self.dtype)
        tokenized_tgt = self.mol_tokenizer.tokenize(tgt).astype(self.dtype)

        return (tokenized_mol, tokenized_prot), tokenized_tgt

    def pad_transform(self, data: Tuple[Input, Target]) -> Tuple[Input, Target]:
        (tokenized_mol, tokenized_prot), tokenized_tgt = data
        padded_mol = pad_sequence(tokenized_mol, self.mol_max_len, self.pad_value, self.left_pad)
        padded_prot = pad_sequence(tokenized_prot, self.prot_max_len, self.pad_value, self.left_pad)
        padded_tgt = pad_sequence(tokenized_tgt, self.mol_max_len, self.pad_value, self.left_pad)

        return (padded_mol, padded_prot), padded_tgt

    def pad_batch_fn(
        self, data: List[Tuple[Input, Target]]
    ) -> Tuple[Tuple[Int[Tensor, "bsz ..."], Int[Tensor, "bsz ..."]], Int[Tensor, "bsz ..."]]:

        srcs, tgts = tuple(zip(*data))
        mols, prots = tuple(zip(*srcs))

        padded_mols = torch.tensor(pad_sequences(mols, self.pad_value, self.left_pad))
        padded_prots = torch.tensor(pad_sequences(prots, self.pad_value, self.left_pad))
        padded_tgts = torch.tensor(pad_sequences(tgts, self.pad_value, self.left_pad))

        return (padded_mols, padded_prots), padded_tgts


class MolAtomPairDataset(PairDataset):
    def __init__(
        self,
        data_path: str,
        data_cols: Tuple[str, ...],
        mol_tokenizer: MolTokenizer,
        atom_tokenizer: AtomTokenizer,
        mol_max_len: Optional[int],
        atom_max_len: Optional[int],
        left_pad: bool,
        pad_batch: bool,
        dtype: DTypeLike = np.int32,
        **kwargs,
    ) -> None:
        super().__init__(data_path, data_cols, False, dtype, **kwargs)
        self.mol_tokenizer = mol_tokenizer
        self.atom_tokenizer = atom_tokenizer
        self.pad_batch = pad_batch

        if pad_batch:
            assert mol_max_len is None and atom_max_len is None, (
                f"`max_len` should be 'None' when `pad_batch` is '{pad_batch}', but "
                f"`mol_max_len` is '{mol_max_len}', `atom_max_len is {atom_max_len}`."
            )

        else:
            assert mol_max_len is not None and atom_max_len is not None, (
                f"`max_len` should not be 'None' when `pad_batch` is '{pad_batch}', but "
                f"`mol_max_len` is '{mol_max_len}', `atom_max_len is {atom_max_len}`."
            )
            self.mol_max_len = mol_max_len
            self.atom_max_len = atom_max_len

        self.left_pad = left_pad
        self.pad_value = mol_tokenizer.vocab2index[mol_tokenizer.pad]

    def transform(self, row: NDArray) -> Tuple[Input, Target]:
        mol, atom = row

        tokenized_mol = self.mol_tokenizer.tokenize(mol).astype(self.dtype)
        tokenized_atom = self.atom_tokenizer.tokenize(atom).astype(self.dtype)

        return tokenized_mol, tokenized_atom

    def pad_transform(self, data: Tuple[Input, Target]) -> Tuple[Input, Target]:
        tokenized_mol, tokenized_atom = data
        tokenized_mol = cast(NDArray, tokenized_mol)
        padded_mol = pad_sequence(tokenized_mol, self.mol_max_len, self.pad_value, self.left_pad)
        padded_atom = pad_sequence(tokenized_atom, self.atom_max_len, self.pad_value, self.left_pad)
        return padded_mol, padded_atom

    def pad_batch_fn(
        self, data: List[Tuple[Input, Target]]
    ) -> Tuple[Int[Tensor, "bsz ..."], Int[Tensor, "bsz ..."]]:
        srcs, tgts = list(zip(*data))

        padded_srcs = torch.tensor(pad_sequences(srcs, self.pad_value, self.left_pad))
        padded_tgts = torch.tensor(pad_sequences(tgts, self.pad_value, self.left_pad))

        return padded_srcs, padded_tgts


class MolAtomProtPairDataset(PairDataset):
    def __init__(
        self,
        data_path: str,
        data_cols: Tuple[str, ...],
        mol_tokenizer: MolTokenizer,
        atom_tokenizer: AtomTokenizer,
        prot_tokenizer: ProteinTokenizer,
        mol_max_len: Optional[int],
        atom_max_len: Optional[int],
        prot_max_len: Optional[int],
        left_pad: bool,
        pad_batch: bool,
        dtype: DTypeLike = np.int32,
        **kwargs,
    ) -> None:
        super().__init__(data_path, data_cols, False, dtype, **kwargs)
        self.mol_tokenizer = mol_tokenizer
        self.atom_tokenizer = atom_tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.pad_batch = pad_batch

        if pad_batch:
            assert mol_max_len is None and atom_max_len is None and prot_max_len is None, (
                f"`max_len` should be 'None' when `pad_batch` is '{pad_batch}', but "
                f"`mol_max_len` is '{mol_max_len}', `atom_max_len is {atom_max_len}`, "
                f"`prot_max_len` is '{prot_max_len}'."
            )

        else:
            assert (
                mol_max_len is not None and atom_max_len is not None and prot_max_len is not None
            ), (
                f"`max_len` should not be 'None' when `pad_batch` is '{pad_batch}', but "
                f"`mol_max_len` is '{mol_max_len}', `atom_max_len is {atom_max_len}`, "
                f"`prot_max_len` is '{prot_max_len}'."
            )
            self.mol_max_len = mol_max_len
            self.atom_max_len = atom_max_len
            self.prot_max_len = prot_max_len

        self.left_pad = left_pad
        self.pad_value = mol_tokenizer.vocab2index[mol_tokenizer.pad]

    def transform(self, row: NDArray) -> Tuple[Input, Target]:
        mol, atom, prot = row

        tokenized_mol = self.mol_tokenizer.tokenize(mol).astype(self.dtype)
        prot = "".join([f"-{letter}" for letter in prot])
        tokenized_prot = self.prot_tokenizer.tokenize(prot).astype(self.dtype)
        tokenized_atom = self.atom_tokenizer.tokenize(atom).astype(self.dtype)

        return (tokenized_mol, tokenized_prot), tokenized_atom

    def pad_transform(self, data: Tuple[Input, Target]) -> Tuple[Input, Target]:
        (tokenized_mol, tokenized_prot), tokenized_tgt = data
        padded_mol = pad_sequence(tokenized_mol, self.mol_max_len, self.pad_value, self.left_pad)
        padded_prot = pad_sequence(tokenized_prot, self.prot_max_len, self.pad_value, self.left_pad)
        padded_tgt = pad_sequence(tokenized_tgt, self.atom_max_len, self.pad_value, self.left_pad)

        return (padded_mol, padded_prot), padded_tgt

    def pad_batch_fn(
        self, data: List[Tuple[Input, Target]]
    ) -> Tuple[Tuple[Int[Tensor, "bsz ..."], Int[Tensor, "bsz ..."]], Int[Tensor, "bsz ..."]]:

        srcs, tgts = tuple(zip(*data))
        mols, prots = tuple(zip(*srcs))

        padded_mols = torch.tensor(pad_sequences(mols, self.pad_value, self.left_pad))
        padded_prots = torch.tensor(pad_sequences(prots, self.pad_value, self.left_pad))
        padded_tgts = torch.tensor(pad_sequences(tgts, self.pad_value, self.left_pad))

        return (padded_mols, padded_prots), padded_tgts


class FragPairDataset(MolPairDataset):
    def __init__(
        self,
        data_path: str,
        data_cols: Tuple[str, str, str],
        tokenizer: MolTokenizer,
        max_len: Optional[int],
        left_pad: bool,
        pad_batch: bool,
        dtype: DTypeLike = np.int32,
        **kwargs,
    ) -> None:
        super().__init__(
            data_path, data_cols, tokenizer, max_len, left_pad, pad_batch, dtype, **kwargs
        )

    def transform(self, row: NDArray) -> Tuple[Input, Target]:
        core, frag, tgt = row
        src = "|".join([core, frag])
        tokenized_src = self.tokenizer.tokenize(src).astype(self.dtype)
        tokenized_tgt = self.tokenizer.tokenize(tgt).astype(self.dtype)

        return tokenized_src, tokenized_tgt


class FragProtPairDataset(MolProtPairDataset):
    def __init__(
        self,
        data_path: str,
        data_cols: Tuple[str, str, str, str],
        mol_tokenizer: MolTokenizer,
        prot_tokenizer: ProteinTokenizer,
        mol_max_len: Optional[int],
        prot_max_len: Optional[int],
        left_pad: bool,
        pad_batch: bool,
        dtype: DTypeLike = np.int32,
        **kwargs,
    ):
        super().__init__(
            data_path,
            data_cols,
            mol_tokenizer,
            prot_tokenizer,
            mol_max_len,
            prot_max_len,
            left_pad,
            pad_batch,
            dtype,
            **kwargs,
        )

    def transform(self, row: NDArray) -> Tuple[Input, Target]:
        core, frag, tgt, prot = row
        mol = "|".join([core, frag])

        tokenized_mol = self.mol_tokenizer.tokenize(mol).astype(self.dtype)
        prot = "".join([f"-{letter}" for letter in prot])
        tokenized_prot = self.prot_tokenizer.tokenize(prot).astype(self.dtype)
        tokenized_tgt = self.mol_tokenizer.tokenize(tgt).astype(self.dtype)

        return (tokenized_mol, tokenized_prot), tokenized_tgt


class MolClassifyDataset(Dataset):
    def __init__(
        self,
        data_path,
        data_cols,
        tokenizer: MolTokenizer,
        max_len: Optional[int],
        dtype: DTypeLike = np.int32,
    ):
        super().__init__()
        df = pd.read_csv(data_path, usecols=data_cols)
        df = df.reindex(columns=data_cols)
        self.data = df.to_numpy()
        self.len = self.data.shape[0]
        self.dtype = dtype
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_value = tokenizer.vocab2index[tokenizer.pad]

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> Tuple[Input, NDArray]:
        src, label = self.data[index]
        tokenized_src = self.tokenizer.tokenize(src).astype(self.dtype)

        if tokenized_src.shape[0] > self.max_len:
            padded_src = tokenized_src[: self.max_len]
        else:
            padded_src = pad_sequence(tokenized_src, self.max_len, self.pad_value, left_pad=False)

        return padded_src, np.array([label], dtype=np.float32)


class MolProtClassifyDataset(Dataset):
    def __init__(
        self,
        data_path,
        data_cols,
        mol_tokenizer: MolTokenizer,
        prot_tokenizer: ProteinTokenizer,
        mol_max_len: Optional[int],
        prot_max_len: Optional[int],
        dtype: DTypeLike = np.int32,
    ):
        super().__init__()
        df = pd.read_csv(data_path, usecols=data_cols)
        df = df.reindex(columns=data_cols)
        self.data = df.to_numpy()
        self.len = self.data.shape[0]
        self.dtype = dtype
        self.mol_tokenizer = mol_tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.mol_max_len = mol_max_len
        self.prot_max_len = prot_max_len
        self.pad_value = mol_tokenizer.vocab2index[mol_tokenizer.pad]

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int) -> Tuple[Input, NDArray]:
        mol, prot, label = self.data[index]
        tokenized_mol = self.mol_tokenizer.tokenize(mol).astype(self.dtype)
        prot = "".join([f"-{letter}" for letter in prot])
        tokenized_prot = self.prot_tokenizer.tokenize(prot).astype(self.dtype)

        if tokenized_mol.shape[0] > self.mol_max_len:
            padded_mol = tokenized_mol[: self.mol_max_len]
        else:
            padded_mol = pad_sequence(
                tokenized_mol, self.mol_max_len, self.pad_value, left_pad=False
            )

        if tokenized_prot.shape[0] > self.prot_max_len:
            padded_prot = tokenized_prot[: self.prot_max_len]
        else:
            padded_prot = pad_sequence(
                tokenized_prot, self.prot_max_len, self.pad_value, left_pad=False
            )

        return (padded_mol, padded_prot), np.array([label], dtype=np.float32)
