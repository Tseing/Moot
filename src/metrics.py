from abc import ABC, abstractmethod
from multiprocessing import Pool
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from jaxtyping import Float
from numpy import ndarray
from tqdm import tqdm

from .tokenizer import StrTokenizer
from .utils import (
    cal_chrf,
    cal_selfies_similarity,
    cal_selfies_validity,
    cal_smiles_similarity,
    cal_smiles_validity,
    trim_seqs,
)


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

        # array metric
        token_accuracy = (hyps_array == refs_array).sum() / hyps_array.size

        hyps = self._post_process(hyps_array)
        refs = self._post_process(refs_array)

        if self._show_bar:
            self._pbar = tqdm(total=total)

        iterator = zip(hyps, refs)
        self.results = self._metrics_pipeline(iterator, total)

        self.results["Token accuracy"] = token_accuracy

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


class SmilesMetrics(ModelMetrics):
    def __init__(self, tokenizer: StrTokenizer, worker: int = 10, show_bar: bool = True) -> None:
        self.tokenizer = tokenizer
        self.worker = worker
        super().__init__(show_bar)

    def _post_process(self, array: ndarray[Any, np.dtype]) -> List[List[str]]:
        return trim_seqs(array, self.tokenizer)

    @staticmethod
    def _metrics_proc(hyp: List[str], ref: List[str]) -> Tuple[float, ...]:
        hyp_str = "".join(hyp)
        ref_str = "".join(ref)

        validity = cal_smiles_validity(hyp_str)
        similarity = cal_smiles_similarity(hyp_str, ref_str)
        chrf = cal_chrf(hyp, ref)
        recovery = 1.0 if (hyp_str == ref_str) else 0.0

        return validity, similarity, chrf, recovery

    def _metrics_pipeline(
        self, iterator: Iterable[Tuple[List[str], List[str]]], total: int
    ) -> Dict[str, float]:
        with Pool(processes=self.worker) as pool:
            res = [
                pool.apply_async(
                    self._metrics_proc, args=(hyp, ref), callback=lambda *args: self._pbar.update()
                )
                for hyp, ref in iterator
            ]
            pool.close()
            pool.join()

        res_value = tuple(r.get() for r in res)
        validity, similarity, chrf, recovery = zip(*res_value)

        return {
            "Validity": sum(validity) / float(total),
            "Similarity": sum(similarity) / float(total),
            "ChRF": sum(chrf) / float(total),
            "Recovery": sum(recovery) / float(total),
        }


class SelfiesMetrics(SmilesMetrics):
    @staticmethod
    def _metrics_proc(hyp: List[str], ref: List[str]) -> Tuple[float, ...]:
        hyp_str = "".join(hyp)
        ref_str = "".join(ref)

        validity = cal_selfies_validity(hyp_str)
        similarity = cal_selfies_similarity(hyp_str, ref_str)
        chrf = cal_chrf(hyp, ref)
        recovery = 1.0 if (hyp_str == ref_str) else 0.0

        return validity, similarity, chrf, recovery
