from collections import namedtuple
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .generator import SequenceGenerator
from .model.crafted_transformer import Model
from .tokenizer import StrTokenizer
from .typing import Device
from .utils import item, move

Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


class Inferencer:
    def __init__(
        self,
        models: Sequence[Model],
        data_loader: DataLoader,
        tokenizer: StrTokenizer,
        max_len: int,
        n_best: int,
        beam_size: int = 1,
        min_len: int = 1,
        stop_early: bool = False,
        normalize_scores: bool = True,
        len_penalty=1,
        unk_penalty=0,
        sampling: bool = False,
        sampling_topk: int = -1,
        sampling_temperature=1,
        print_alignment: bool = True,
        device: Optional[Device] = None,
    ) -> None:
        assert (
            not sampling or n_best == beam_size
        ), "--sampling requires --nbest to be equal to --beam"

        # Optimize ensemble for generation
        for model in models:
            model.make_generation_fast_(need_attn=print_alignment)

        pad_idx = tokenizer.vocab2index[tokenizer.pad]
        bos_idx = tokenizer.vocab2index[tokenizer.bos]
        eos_idx = tokenizer.vocab2index[tokenizer.eos]
        unk_idx = tokenizer.vocab2index[tokenizer.unk]
        vocab_size = tokenizer.vocab_size

        if device is None:
            device = torch.device(torch._C._get_default_device())
        self.device = device

        # Initialize generator
        self.translator = SequenceGenerator(
            models,
            pad_idx=pad_idx,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
            unk_idx=unk_idx,
            vocab_size=vocab_size,
            maxlen=max_len,
            minlen=min_len,
            beam_size=beam_size,
            stop_early=stop_early,
            normalize_scores=normalize_scores,
            len_penalty=len_penalty,
            unk_penalty=unk_penalty,
            sampling=sampling,
            sampling_topk=sampling_topk,
            sampling_temperature=sampling_temperature,
            use_amp=True,
            device=device,
        )

        self.data_dl = data_loader
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.print_alignment = print_alignment
        self.nbest = n_best

    def make_result(self, src_str, hypos):
        # TODO: what are the types of src_str and hypos ?
        result = Translation(
            src_str=src_str,
            hypos=[],
            pos_scores=[],
            alignments=[],
        )

        # Process top predictions
        for hypo in hypos[: min(len(hypos), self.nbest)]:
            hypo_tokens = hypo["tokens"].int().cpu()
            alignment = hypo["alignment"].int().cpu() if hypo["alignment"] is not None else None
            hypo_str = " ".join(self.tokenizer.vec_ids2tokens(hypo_tokens)).strip()

            result.hypos.append((hypo["score"], hypo_str))
            result.pos_scores.append(
                "P\t" + " ".join(f"{x:.4f}" for x in hypo["positional_scores"].tolist())
            )
            result.alignments.append(
                "A\t" + " ".join(str(item(x)) for x in alignment) if self.print_alignment else None
            )

        return result

    def process_batch(self, batch: Tuple[Tuple[Tensor, ...], Tensor]):
        inp, _ = batch
        inp = move(inp, self.device)

        translations = self.translator.generate(
            inp,
            maxlen=self.max_len,
        )

        # TODO: Handle multi input situation
        if isinstance(inp, Tensor):
            src = inp
        else:
            src = inp[0]

        return [self.make_result(src[i], t) for i, t in enumerate(translations)]

    def inference(self, show: bool = True, save_path: Optional[str] = None) -> None:
        # for inputs in buffered_read(args.buffer_size, data_descriptor):
        indices = []
        results = []
        for batch_indices, batch in tqdm(enumerate(self.data_dl), desc="Inference", total=len(self.data_dl)):
            _, tgt = batch
            indices.extend([batch_indices] * tgt.shape[0])
            results += self.process_batch(batch)

        if show:
            self._show_result(indices, results)
        else:
            assert save_path is not None
            self._write_result(indices, results, save_path)

    def _show_result(self, indices: list, results: list) -> None:
        for i in np.argsort(indices):
            result = results[i]
            inp_array = result.src_str.cpu().numpy()
            # print(f"Input {inp_array}")
            print(f"Src\t{' '.join(self.tokenizer.vec_ids2tokens(inp_array))}")
            for hypo, pos_scores, align in zip(result.hypos, result.pos_scores, result.alignments):
                print(f"Score\t{hypo[0]}")
                print(f"Result\t{hypo[1]}")
                if align is not None:
                    print(f"Align\t{align}")
            print("----------------------------------------------------------")

    def _write_result(self, indices: list, results: list, save_path: str) -> None:
        inps = []
        outps = []

        for i in np.argsort(indices):
            result = results[i]
            inp_array = result.src_str.cpu().numpy()
            inp = " ".join(self.tokenizer.vec_ids2tokens(self.tokenizer.trim(inp_array)))
            inps.extend([inp] * self.nbest)
            outps.extend([hypo[1] for hypo in result.hypos])

        pd.DataFrame({"input": inps, "output": outps}).to_csv(save_path, index=False)
        print(f"Inference results are saved in '{save_path}'.")
