import time
from typing import Tuple, List

from jaxtyping import Float, Int
from torch import Tensor, nn

from .data_utils import read_smiles
from .tokenizer import StrTokenizer
from nltk.translate.chrf_score import sentence_chrf
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


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


def trim_seqs(seq_list: List[List[int]], tokenizer: StrTokenizer) -> List[List[str]]:
    trimmed_seqs: List[List[str]] = []

    for seq_ids in seq_list:
        seq_tokens = tokenizer.convert_ids2tokens(seq_ids)
        if tokenizer.bos == seq_tokens[0]:
            seq_tokens.pop(0)
        try:
            eos_idx = seq_tokens.index(tokenizer.eos)
        except ValueError:
            eos_idx = None

        seq = seq_tokens[:eos_idx]
        trimmed_seqs.append(seq)

    return trimmed_seqs


def cal_chrf(hyp: List[str], ref: List[str]) -> float:
    chrf = sentence_chrf(ref, hyp, min_len=1, max_len=3, beta=2.0)
    return chrf


def cal_similarity(hyp: List[str], ref: List[str]) -> float:
    hyp_str = "".join(hyp)
    ref_str = "".join(ref)

    print(f"hyp: '{hyp_str}'")
    print(f"ref: '{ref_str}'")

    try:
        hyp_mol = Chem.MolFromSmiles(hyp_str)
        ref_mol = Chem.MolFromSmiles(ref_str)
    except Exception:
        similarity = 0

    if hyp_mol is None or ref_mol is None:
        similarity = 0
    else:
        hyp_fp = AllChem.GetHashedMorganFingerprint(hyp_mol, 3, 2048)
        ref_fp = AllChem.GetHashedMorganFingerprint(ref_mol, 3, 2048)
        similarity = DataStructs.TanimotoSimilarity(hyp_fp, ref_fp)

    return similarity
