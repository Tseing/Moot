import logging
import os
import os.path as osp
import time
from typing import List, Optional, Tuple, Any
import argparse

import numpy as np
import yaml
from jaxtyping import Bool, Float, Int
from nltk.translate.chrf_score import sentence_chrf
from numpy import ndarray
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch import Tensor, nn

from .tokenizer import StrTokenizer


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m) -> None:
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


def now_time() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


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


def cal_similarity(hyp: str, ref: str) -> float:
    try:
        hyp_mol = Chem.MolFromSmiles(hyp)
        ref_mol = Chem.MolFromSmiles(ref)
    except Exception:
        similarity = 0.0

    if hyp_mol is None or ref_mol is None:
        similarity = 0.0
    else:
        hyp_fp = AllChem.GetHashedMorganFingerprint(hyp_mol, 3, 2048)
        ref_fp = AllChem.GetHashedMorganFingerprint(ref_mol, 3, 2048)
        similarity = DataStructs.TanimotoSimilarity(hyp_fp, ref_fp)

    return similarity


def cal_validity(hyp: str) -> float:
    try:
        hyp_mol = Chem.MolFromSmiles(hyp)
    except Exception:
        validity = 0.0
    if hyp_mol is None:
        validity = 0.0
    else:
        validity = 1.0

    return validity


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
        self._SCRIP_DIR = osp.abspath(scrip_dir)
        self._TASK_DIR = osp.abspath(osp.join(self._SCRIP_DIR, "tasks"))
        self._BASE_DIR = osp.abspath(osp.join(self._SCRIP_DIR, ".."))
        self._DATA_DIR = osp.abspath(osp.join(self._BASE_DIR, "data"))
        self._LOG_DIR = osp.abspath(osp.join(self._BASE_DIR, "log"))
        self._CKPT_DIR = osp.abspath(osp.join(self._BASE_DIR, "checkpoints"))

        necessary_dirs = [self._DATA_DIR, self._LOG_DIR, self._CKPT_DIR, self._TASK_DIR]
        for folder in necessary_dirs:
            if not osp.exists(folder):
                os.mkdir(folder)

    def __getattr__(self, name: str) -> Any:
        return self._cfg[name]

    def __repr__(self) -> str:
        return "\n".join([": ".join([k, str(self._cfg[k])]) for k in sorted(self._cfg.keys())])

    @property
    def BASE_DIR(self):
        return self._BASE_DIR

    @property
    def DATA_DIR(self):
        return self._DATA_DIR

    @property
    def LOG_DIR(self):
        return self._LOG_DIR

    @property
    def CKPT_DIR(self):
        return self._CKPT_DIR

    def _load_cli(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("cfg_file", type=str, help="Config file.")
        return parser.parse_args()

    def _load_local(self, cfg_file: str) -> dict:
        cfg_path = osp.join(self._TASK_DIR, cfg_file)
        cfg = yaml.load(open(cfg_path, "r", encoding="utf-8"), yaml.FullLoader)
        assert isinstance(cfg, dict), f"YAML file '{cfg_path}' is not a valid config file."
        f"Config file should be 'dict' structure, but got {type(cfg)}."

        return cfg

    def parse(self):
        cli_cfg = self._load_cli()
        assert cli_cfg is not None, "Cannot fetch any command line config."
        cfg_file = cli_cfg.cfg_file
        self._cfg = self._load_local(cfg_file)
