from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from jaxtyping import Float
from numpy import ndarray
from tqdm import tqdm

from .tokenizer import MMPTokenizer
from .utils import cal_chrf, cal_similarity, cal_validity, trim_seqs


class ModelMetrics(ABC):
    def __init__(self, show_bar: bool = True) -> None:
        self._show_bar = show_bar
        self._hyps: List[Float[ndarray, "bsz ..."]] = []
        self._refs: List[Float[ndarray, "bsz ..."]] = []

    def update(
        self,
        hyp: Float[ndarray, "bsz ..."],
        ref: Float[ndarray, "bsz ..."],
    ) -> None:
        self._hyps.append(hyp)
        self._refs.append(ref)

    def metric(self) -> Dict[str, float]:
        hyps_array = np.concatenate(self._hyps, axis=0)
        refs_array = np.concatenate(self._refs, axis=0)

        total = hyps_array.shape[0]

        hyps = self._post_process(hyps_array)
        refs = self._post_process(refs_array)

        if self._show_bar:
            iterator = tqdm(zip(hyps, refs), total=total)
        else:
            iterator = zip(hyps, refs)

        self.results = self._metrics_pipeline(iterator, total)
        return self.results

    def clear(self) -> None:
        self._hyps = []
        self._refs = []

    @abstractmethod
    def _post_process(self, array: Float[ndarray, "total ..."]) -> Iterable:
        assert False, "Abstract method `_post_process` has not yet initialized."

    @abstractmethod
    def _metrics_pipeline(self, iterator: Iterable[Tuple], total: int) -> Dict[str, float]:
        assert False, "Abstract method `_metrics_pipeline` has not yet initialized."


class MMPMetrics(ModelMetrics):
    def __init__(
        self,
        tokenizer: MMPTokenizer,
    ) -> None:
        self.tokenizer = tokenizer
        super().__init__()

    def _post_process(self, array: ndarray[Any, np.dtype]) -> List[List[str]]:
        return trim_seqs(array, self.tokenizer)

    def _metrics_pipeline(
        self, iterator: Iterable[Tuple[List[str], List[str]]], total: int
    ) -> Dict[str, float]:
        validity = 0.0
        similarity = 0.0
        chrf = 0.0

        for hyp, ref in iterator:
            hyp_str = "".join(hyp)
            ref_str = "".join(ref)

            validity += cal_validity(hyp_str)
            similarity += cal_similarity(hyp_str, ref_str)
            chrf += cal_chrf(hyp, ref)

        return {
            "Validity": validity / total,
            "Similarity": similarity / total,
            "ChRF": chrf / total,
        }
