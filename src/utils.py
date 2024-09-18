import argparse
import logging
import os
import os.path as osp
import time
from typing import Any, Iterable, List, Sequence, Tuple, Union

import numpy as np
import selfies as sf
import yaml
from jaxtyping import Bool, Float, Int
from nltk.translate.chrf_score import sentence_chrf
from numpy import ndarray
from numpy.typing import NDArray
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch import Tensor, nn

from .tokenizer import StrTokenizer
from .typing import Device


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m) -> None:
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


def now_time() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def item(tensor: Tensor) -> Union[Tensor, float]:
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor


def move(
    data: Union[Tensor, Sequence[Tensor], Tuple[Tensor, ...]], device: Device
) -> Union[Tensor, Sequence[Tensor]]:
    if isinstance(data, Tensor):
        data = data.to(device)
    elif isinstance(data, (list, tuple)):
        data = tuple(d.to(device) for d in data)
    else:
        raise TypeError(f"'move()' not support type: {type(data)}.")

    return data


def maxlen(seqs: Iterable[NDArray]) -> int:
    return max([seq.shape[0] for seq in seqs])


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


def pad_sequences(
    seqs: Iterable[Int[ndarray, "seq"]], pad_value: int, left_pad: bool
) -> Int[ndarray, "bsz batch_max_len"]:
    max_len = maxlen(seqs)
    return np.stack([pad_sequence(seq, max_len, pad_value, left_pad) for seq in seqs])


def check_seq(
    src: Int[Tensor, "seq_len"],
    output: Float[Tensor, "seq_len vocab_size"],
    tokenizer: StrTokenizer,
) -> Tuple[str, str]:
    src_seq = src.cpu().tolist()
    output_seq = output.argmax(dim=-1).squeeze().cpu().tolist()

    src_seq_str = " ".join(tokenizer.convert_ids2tokens(src_seq))
    output_seq_str = " ".join(tokenizer.convert_ids2tokens(output_seq))

    return src_seq_str, output_seq_str


def trim_seqs(seqs: Int[ndarray, "bsz seq_len"], tokenizer: StrTokenizer) -> List[List[str]]:
    seq_len = seqs.shape[1]
    token_seqs = tokenizer.vec_ids2tokens(seqs)
    is_eos: Bool[ndarray, "bsz seq_len"] = token_seqs == tokenizer.eos
    eos_idxes = is_eos.argmax(axis=1)[:, None]

    # No <bos> token, do not clip sentences
    eos_idxes[eos_idxes < 1] = seq_len + 1

    idxes = np.zeros_like(token_seqs, dtype=np.int_)
    idxes[:] = np.arange(seq_len)

    # masked tokens judged by <eos> token
    eos_mask = idxes >= eos_idxes

    # simplest way to trim <bos> token is remove the first token
    bos_mask = idxes == 0

    mask = np.logical_or(bos_mask, eos_mask)

    trimmed_seqs = [seq[~mask[i]].tolist() for i, seq in enumerate(token_seqs)]

    return trimmed_seqs


def cal_chrf(hyp: List[str], ref: List[str]) -> float:
    chrf = sentence_chrf(ref, hyp, min_len=1, max_len=3, beta=2.0)
    return chrf


def cal_smiles_similarity(hyp: str, ref: str) -> float:
    try:
        hyp_mol = Chem.MolFromSmiles(hyp)
        ref_mol = Chem.MolFromSmiles(ref)
    except Exception:
        similarity = 0.0
        return similarity

    if hyp_mol is None or ref_mol is None:
        similarity = 0.0
    else:
        hyp_fp = AllChem.GetHashedMorganFingerprint(hyp_mol, 3, 2048)
        ref_fp = AllChem.GetHashedMorganFingerprint(ref_mol, 3, 2048)
        similarity = DataStructs.TanimotoSimilarity(hyp_fp, ref_fp)

    return similarity


def cal_selfies_similarity(hyp: str, ref: str) -> float:
    try:
        hyp_smi = sf.decoder(hyp)
        ref_smi = sf.decoder(ref)
    except Exception:
        similarity = 0.0
        return similarity

    return cal_smiles_similarity(hyp_smi, ref_smi)


def cal_smiles_validity(hyp: str) -> float:
    try:
        hyp_mol = Chem.MolFromSmiles(hyp)
    except Exception:
        validity = 0.0
        return validity

    if hyp_mol is None:
        validity = 0.0
    else:
        validity = 1.0

    return validity


def cal_selfies_validity(hyp: str) -> float:
    try:
        hyp_smi = sf.decoder(hyp)
    except Exception:
        validity = 0.0
        return validity

    return cal_smiles_validity(hyp_smi)


class Log(logging.Logger):
    def __init__(self, name: str, log_path: str) -> None:
        super().__init__(name)
        self.log_path = log_path

        self.setLevel(logging.DEBUG)
        self.file_handler = logging.FileHandler(log_path, mode="a+")
        self.console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.file_handler.setFormatter(formatter)
        self.console_handler.setFormatter(formatter)
        self.addHandler(self.file_handler)
        self.addHandler(self.console_handler)

    def console_off(self) -> None:
        self.removeHandler(self.console_handler)

    def console_on(self) -> None:
        self.addHandler(self.console_handler)


class Cfg:
    def __init__(self, scrip_dir: str = ".") -> None:
        self.__SCRIP_DIR = osp.abspath(scrip_dir)
        self.__TASK_DIR = osp.abspath(osp.join(self.__SCRIP_DIR, "tasks"))
        self.__BASE_DIR = osp.abspath(osp.join(self.__SCRIP_DIR, ".."))
        self.__DATA_DIR = osp.abspath(osp.join(self.__BASE_DIR, "data"))
        self.__LOG_DIR = osp.abspath(osp.join(self.__BASE_DIR, "log"))
        self.__OUTPUT_DIR = osp.abspath(osp.join(self.__BASE_DIR, "output"))
        self.__CKPT_DIR = osp.abspath(osp.join(self.__BASE_DIR, "checkpoints"))

        necessary_dirs = [
            self.__DATA_DIR,
            self.__LOG_DIR,
            self.__OUTPUT_DIR,
            self.__CKPT_DIR,
            self.__TASK_DIR,
        ]
        for folder in necessary_dirs:
            if not osp.exists(folder):
                os.mkdir(folder)

    def __getattr__(self, name: str) -> Any:
        return self._cfg[name]

    def __repr__(self) -> str:
        return "\n".join([": ".join([k, str(self._cfg[k])]) for k in sorted(self._cfg.keys())])

    @property
    def BASE_DIR(self):
        return self.__BASE_DIR

    @property
    def DATA_DIR(self):
        return self.__DATA_DIR

    @property
    def LOG_DIR(self):
        return self.__LOG_DIR

    @property
    def TASK_DIR(self):
        return self.__TASK_DIR

    @property
    def CKPT_DIR(self):
        return self.__CKPT_DIR

    @property
    def OUTPUT_DIR(self):
        return self.__OUTPUT_DIR

    def _load_cli(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("cfg_file", type=str, help="Config file.")
        return parser.parse_args()

    def _load_local(self, cfg_file: str) -> dict:
        cfg_path = osp.join(self.TASK_DIR, cfg_file)
        cfg = yaml.load(open(cfg_path, "r", encoding="utf-8"), yaml.FullLoader)
        assert isinstance(cfg, dict), f"YAML file '{cfg_path}' is not a valid config file."
        f"Config file should be 'dict' structure, but got {type(cfg)}."

        return cfg

    def parse(self):
        cli_cfg = self._load_cli()
        assert cli_cfg is not None, "Cannot fetch any command line config."
        cfg_file = cli_cfg.cfg_file
        self._cfg = self._load_local(cfg_file)

    def set(self, key: str, value: Any) -> None:
        if key in self._cfg.keys():
            raise AttributeError(f"Key '{key}' is in configs and its value is '{self._cfg[key]}'.")
        self._cfg[key] = value
