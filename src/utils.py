import time
from typing import List, Tuple

import numpy as np
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
    print(f"hyp: '{hyp}'")
    print(f"ref: '{ref}'")

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
