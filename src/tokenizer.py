import copy
import re
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import yaml
from numpy.typing import NDArray
from tqdm import tqdm
from typing_extensions import TypeAlias

Tokenizer: TypeAlias = Union[
    "StrTokenizer", "SmilesTokenizer", "SelfiesTokenizer", "ProteinTokenizer"
]


class BaseTokenizer(ABC):
    def __init__(self, special_tokens: Optional[List[str]] = None):
        self.pad = "{pad}"
        self.unk = "{unk}"
        self.cls = "{cls}"
        self.special_tokens = [self.pad, self.cls, self.unk]

        self.pattern: str

        if special_tokens is not None:
            self.special_tokens.extend(special_tokens)

    def load_word_table(self, word_table_path: str):
        with open(word_table_path, "r", encoding="utf-8") as f:
            word_table = yaml.safe_load(f)

        self.word_table: List[str] = self.special_tokens + word_table
        self.update_vocab()

    def update_vocab(self) -> None:
        self.vocab2index = {w: i for i, w in enumerate(self.word_table)}
        self.vocab2token = {i: w for i, w in enumerate(self.word_table)}
        self.vocab_size = len(self.word_table)
        self.special_tokens_id = [self.vocab2index[token] for token in self.special_tokens]

        self._vec_tokens2ids = np.vectorize(
            lambda token: self.vocab2index.get(token, self.vocab2index[self.unk])
        )
        self._vec_ids2tokens = np.vectorize(lambda id: self.vocab2token.get(id, self.unk))

    @abstractmethod
    def build_word_table(self, seqs: List[str], dump_path: str) -> List[str]:
        assert False, "Abstract method `tokenize` has not yet initialized."

    @abstractmethod
    def tokenize(self, seq: str) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        assert False, "Abstract method `tokenize` has not yet initialized."

    def convert_ids2tokens(self, ids: Iterable[int]) -> NDArray:
        return np.array([self.vocab2token.get(id, self.unk) for id in ids])

    def covert_tokens2ids(self, tokens: Iterable[str]) -> NDArray:
        return np.array(
            [self.vocab2index.get(token, self.vocab2index[self.unk]) for token in tokens],
            dtype=np.int32,
        )

    def vec_ids2tokens(self, ids: NDArray) -> NDArray:
        return self._vec_ids2tokens(ids)

    def vec_tokens2ids(self, tokens: NDArray) -> NDArray:
        return self._vec_tokens2ids(tokens)

    def find_bald_tokens(self, seq: str) -> List[str]:
        seq = seq.strip()
        regex = re.compile(self.pattern)
        tokens = regex.findall(seq)

        if "".join(tokens) != seq:
            print(f"UNK token Warning: '{seq}'\n-> {tokens}")
            tokens = self._replace_unk(seq, tokens)

        return tokens

    def _format_tokens(self, bald_tokens: List[str]) -> List[str]:
        return bald_tokens

    def _find_tokens(self, seq: str) -> List[str]:
        return self._format_tokens(self.find_bald_tokens(seq))

    def _replace_unk(self, seq: str, matched_tokens: List[str]) -> List[str]:
        tokens = copy.deepcopy(matched_tokens)
        unk_num = 0
        for i, matched_token in enumerate(matched_tokens):
            if seq.startswith(matched_token):
                seq = seq.replace(matched_token, "", 1)
            else:
                next_matched_pos = seq.index(matched_token)
                seq = seq.replace(seq[: next_matched_pos + len(matched_token)], "", 1)
                tokens.insert(i + unk_num, self.unk)
                unk_num += 1

        return tokens


class StrTokenizer(BaseTokenizer):
    def __init__(self, pattern: str):
        self.bos = "{bos}"
        self.eos = "{eos}"
        self.pattern = pattern
        special_tokens = [self.bos, self.eos]

        super().__init__(special_tokens)

    def build_word_table(self, seqs: Iterable[str], dump_path: Optional[str] = None) -> List[str]:
        unique_tokens = set()
        max_tokens_num = 0
        for seq in tqdm(seqs):
            tokens = self.find_bald_tokens(seq)
            unique_tokens.update(set(tokens))

            tokens_num = len(tokens)
            if tokens_num > max_tokens_num:
                max_tokens_num = tokens_num

        print(f"Max number of tokens in a sequence is '{max_tokens_num}'.")
        word_table = sorted(list(unique_tokens))
        if dump_path:
            with open(dump_path, "w", encoding="utf-8") as f:
                yaml.dump(word_table, f)

        return word_table

    def tokenize(self, seq: str) -> NDArray:
        tokens = np.array(self._find_tokens(seq))
        return self.vec_tokens2ids(tokens)

    def fast_tokenize(self, seq: str) -> NDArray:
        tokens = np.array(seq.split())
        return self.vec_tokens2ids(tokens)

    def bald_tokenize(self, bald_tokens: List[str]):
        tokens = np.array(self._format_tokens(bald_tokens))
        return self.vec_tokens2ids(tokens)

    def tokenize2str(self, seq: str) -> str:
        tokens = self._find_tokens(seq)
        return " ".join(tokens)


class SmilesTokenizer(StrTokenizer):
    def __init__(self):
        pattern = (
            "(\[[^\]]+]|{unk}|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|"
            "\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%"
            "[0-9]{2}|[0-9])"
        )

        super().__init__(pattern)

    def _format_tokens(self, bald_tokens: List[str]) -> List[str]:
        return [self.bos] + bald_tokens + [self.eos]


class SelfiesTokenizer(StrTokenizer):
    def __init__(self):
        pattern = "\[.*?\]|\.|{unk}"
        super().__init__(pattern)

    def _format_tokens(self, bald_tokens: List[str]) -> List[str]:
        return [self.bos] + bald_tokens + [self.eos]


class ProteinTokenizer(StrTokenizer):
    def __init__(self):
        pattern = "-[A|R|N|D|C|Q|E|G|H|I|L|K|M|F|P|S|T|W|Y|V|X|U|O]|{unk}"
        super().__init__(pattern)
        word_table = [
            "-A",
            "-R",
            "-N",
            "-D",
            "-C",
            "-Q",
            "-E",
            "-G",
            "-H",
            "-I",
            "-L",
            "-K",
            "-M",
            "-F",
            "-P",
            "-S",
            "-T",
            "-W",
            "-Y",
            "-V",
            "-X",
            "-U",   # selenocysteine (Sec)
            "-O"    # pyrrolysine (Pyl)
        ]
        self.word_table = self.special_tokens + word_table
        self.update_vocab()

    def _format_tokens(self, bald_tokens: List[str]) -> List[str]:
        return [self.bos] + bald_tokens + [self.eos]


def share_vocab(*args: Tokenizer) -> Tuple[Tokenizer, ...]:
    tokenizers = copy.deepcopy(args)
    special_tokens = tokenizers[0].special_tokens

    for tokenizer in tokenizers:
        assert special_tokens == tokenizer.special_tokens, (
            "Tokenizers with different special tokens cannot share vocab."
            f"'{tokenizers[0]}': '{special_tokens}'"
            f"'{tokenizer}': '{tokenizer.special_tokens}'"
        )

    merged_vocab = [token for tokenizer in tokenizers for token in tokenizer.word_table]
    shared_vocab = list(set(merged_vocab))
    shared_vocab.sort(key=merged_vocab.index)

    for tokenizer in tokenizers:
        tokenizer.word_table = shared_vocab
        tokenizer.update_vocab()

    return tokenizers
