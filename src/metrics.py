from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import numpy as np
from jaxtyping import Bool, Float, Int
from numpy import ndarray
from numpy.typing import NDArray

from .tokenizer import MMPTokenizer
from .utils import trim_seqs


class ModelMetrics(ABC):
    def __init__(
        self,
        metrics: OrderedDict[
            str, Callable[[Float[ndarray, "bsz ..."], Float[ndarray, "bsz ..."]], float]
        ],
    ) -> None:
        self._metrics = metrics.keys()
        for metric_name in self._metrics:
            setattr(self, f"__{metric_name}", metrics[metric_name])

        self._hyps: List[Float[ndarray, "bsz ..."]] = []
        self._refs: List[Float[ndarray, "bsz ..."]] = []

    def update(self, hyp: Float[ndarray, "bsz ..."], ref: Float[ndarray, "bsz ..."]) -> None:
        self._hyps.append(hyp)
        self._refs.append(ref)

    def metric(self) -> Dict[str, float]:
        hyps_array = np.concatenate(self._hyps, axis=0)
        refs_array = np.concatenate(self._refs, axis=0)
        self.results = self._metrics_pipeline(hyps_array, refs_array)
        return self.results

    def clear(self) -> None:
        self._hyps = []
        self._refs = []

    @abstractmethod
    def _metrics_pipeline(
        self, hyp: Float[ndarray, "bsz ..."], ref: Float[ndarray, "bsz ..."]
    ) -> Dict[str, float]:
        assert False, "Abstract method `tokenize` has not yet initialized."
        # return {
        #     metric: getattr(self, f"__{metric}")(hyps_array, refs_array) for metric in self._metrics
        # }


class MMPMetrics(ModelMetrics):
    def __init__(
        self,
        tokenizer: MMPTokenizer,
    ) -> None:
        self.tokenizer = tokenizer
        metrics = OrderedDict({"": lambda x: x,})
        super().__init__(metrics)

    def _metrics_pipeline(self, hyp: ndarray, ref: ndarray) -> Dict[str, float]:
        hyp_seqs = trim_seqs(hyp, self.tokenizer)
        ref_seqs = trim_seqs(ref, self.tokenizer)
        return {}
