import copy
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class Tokenizer(ABC):
    def __init__(
        self, word_table: List[str], special_tokens: Optional[List[str]] = None
    ):
        self.pad = "{pad}"
        self.unk = "{unk}"
        self.cls = "{cls}"
        self.special_tokens = [self.pad, self.cls, self.unk]

        self.pattern: str

        if special_tokens is not None:
            self.special_tokens.extend(special_tokens)

        self.word_table = self.special_tokens + word_table

        self.vocab2index = {w: i for i, w in enumerate(self.word_table)}
        self.vocab2token = {i: w for i, w in enumerate(self.word_table)}
        self.vocab_size = len(self.word_table)
        self.special_tokens_id = [
            self.vocab2index[token] for token in self.special_tokens
        ]

    @abstractmethod
    def tokenize(self, seq: str) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        assert False, "Abstract method `tokenize` has not yet initialized."

    def convert_ids2tokens(self, ids: List[int]) -> List[str]:
        return [self.vocab2token.get(id, self.unk) for id in ids]

    def covert_tokens2ids(self, tokens: List[str]) -> NDArray:
        return np.array(
            [
                self.vocab2index.get(token, self.vocab2index[self.unk])
                for token in tokens
            ],
            dtype=np.int32,
        )

    def find_bald_tokens(self, seq: str) -> List[str]:
        seq = seq.strip()
        regex = re.compile(self.pattern)
        tokens = regex.findall(seq)

        if "".join(tokens) != seq:
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


class StrTokenizer(Tokenizer):
    def __init__(self, word_table: List[str], pattern: str):
        self.bos = "{bos}"
        self.eos = "{eos}"
        self.pattern = pattern
        special_tokens = [self.bos, self.eos]

        super().__init__(word_table, special_tokens)

    def tokenize(self, seq: str) -> NDArray:
        tokens = self._find_tokens(seq)
        return self.covert_tokens2ids(tokens)

    def fast_tokenize(self, seq: str) -> NDArray:
        tokens = seq.split()
        return self.covert_tokens2ids(tokens)

    def bald_tokenize(self, bald_tokens: List[str]):
        tokens = self._format_tokens(bald_tokens)
        return self.covert_tokens2ids(tokens)

    def tokenize2str(self, seq: str) -> str:
        tokens = self._find_tokens(seq)
        return " ".join(tokens)


class SmilesTokenizer(StrTokenizer):
    def __init__(self):
        word_table = [
            "#",
            ".",
            "%10",
            "%11",
            "%12",
            "(",
            ")",
            "-",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "<",
            "=",
            "B",
            "Br",
            "C",
            "Cl",
            "F",
            "I",
            "N",
            "O",
            "P",
            "S",
            "[B-]",
            "[BH-]",
            "[BH2-]",
            "[BH3-]",
            "[B]",
            "[C+]",
            "[C-]",
            "[CH+]",
            "[CH-]",
            "[CH2+]",
            "[CH2]",
            "[CH]",
            "[F+]",
            "[H]",
            "[I+]",
            "[IH2]",
            "[IH]",
            "[N+]",
            "[N-]",
            "[NH+]",
            "[NH-]",
            "[NH2+]",
            "[NH3+]",
            "[N]",
            "[O+]",
            "[O-]",
            "[OH+]",
            "[O]",
            "[P+]",
            "[PH+]",
            "[PH2+]",
            "[PH]",
            "[S+]",
            "[S-]",
            "[SH+]",
            "[SH]",
            "[Se+]",
            "[SeH+]",
            "[SeH]",
            "[Se]",
            "[Si-]",
            "[SiH-]",
            "[SiH2]",
            "[SiH]",
            "[Si]",
            "[b-]",
            "[bH-]",
            "[c+]",
            "[c-]",
            "[cH+]",
            "[cH-]",
            "[n+]",
            "[n-]",
            "[nH+]",
            "[nH]",
            "[o+]",
            "[s+]",
            "[sH+]",
            "[se+]",
            "[se]",
            "b",
            "c",
            "n",
            "o",
            "p",
            "s",
        ]

        pattern = (
            "(\[[^\]]+]|{unk}|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|"
            "\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%"
            "[0-9]{2}|[0-9])"
        )

        super().__init__(word_table, pattern)


class MMPTokenizer(SmilesTokenizer):
    def __init__(self):
        super().__init__()

    def _format_tokens(self, bald_tokens: List[str]) -> List[str]:
        return [self.bos] + bald_tokens + [self.eos]
