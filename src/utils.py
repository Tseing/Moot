import os.path as osp
import time
from typing import List, Tuple, Union

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from nltk.translate.chrf_score import sentence_chrf
from numpy import ndarray
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .metrics import ModelMetrics
from .tokenizer import StrTokenizer
from .typing import Device


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
    is_eos: Bool[ndarray, "bsz seq_len"] = seqs == tokenizer.eos
    eos_idxes = is_eos.argmax(axis=1)

    # No <bos> token, do not clip sentences
    eos_idxes[eos_idxes < 1] = seq_len + 1

    idxes = np.zeros_like(seqs)
    idxes[:] = np.arange(seq_len)

    # masked tokens judged by <eos> token
    eos_mask = idxes >= eos_idxes

    # simplest way to trim <bos> token is remove the first token
    bos_mask = idxes == 0

    mask = np.logical_and(bos_mask, eos_mask)

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


class ModelSaver:
    def __init__(
        self,
        save_dir: str,
        steps: int,
        model_name: str = "model",
        epoch_gap: int = 1,
        model_num_per_epoch: int = 1,
        save_in_step_zero: bool = False,
    ) -> None:
        self.save_dir = save_dir
        self.model_name = model_name
        self.epoch_gap = epoch_gap
        step_gap = steps // model_num_per_epoch
        if save_in_step_zero:
            save_steps = [step_gap * i for i in range(model_num_per_epoch)]
        else:
            save_steps = [step_gap * (1 + i) for i in range(model_num_per_epoch)]

        self.save_steps = set(save_steps)

    def monitor(self, model: nn.Module, epoch: int, step: int) -> None:
        if step in self.save_steps:
            file_name = f"{self.model_name}_epoch{epoch}_step{step}.pt"
            torch.save(
                model.state_dict(),
                osp.join(self.save_dir, file_name),
            )
            print(f"Model '{file_name}' is saved in {self.save_dir}.")


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: _Loss,
        scheduler: LRScheduler,
        device: Device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

    def setup_strategy(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epoch_num: int,
        metrics: ModelMetrics,
        saver: ModelSaver,
        tokenizer: StrTokenizer,
    ) -> None:
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epoch_num = epoch_num
        self.metrics = metrics
        self.saver = saver
        self.tokenizer = tokenizer

    def train(self, dataloader: DataLoader, saver: ModelSaver):
        self.model.train()
        epoch = self.check_now_epoch()
        epoch_loss = 0
        for step, (src, tgt) in enumerate(dataloader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(src, tgt)

            loss = self.criterion(output.transpose(1, 2), tgt.long())
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            print(
                f"[{now_time()}] Epoch: {epoch} Step: {step} ({round((step / len(dataloader)) * 100, 2)}%), "
                f"Train Loss: {loss.item()}."
            )

            saver.monitor(self.model, epoch, step)

    def evaluate(self, data_loader: DataLoader, metrics: ModelMetrics, tokenizer: StrTokenizer):
        self.model.eval()
        # epoch_loss = 0
        # epoch_chrf = 0.0
        # epoch_similarity = 0.0

        with torch.no_grad():
            for step, (src, tgt) in enumerate(data_loader):
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                output = self.model(src, tgt)
                # bsz = output.shape[0]
                # chrf = 0.0
                # similarity = 0.0

                loss = self.criterion(output.transpose(1, 2), tgt.long())
                # epoch_loss += loss.item()

                # output_seq = output.argmax(dim=-1).squeeze().cpu().tolist()
                # tgt_seq = tgt.cpu().tolist()
                # output_seq = trim_seqs(output_seq, self.tokenizer)
                # tgt_seq = trim_seqs(tgt_seq, self.tokenizer)

                # for hyp, ref in zip(output_seq, tgt_seq):
                #     chrf += cal_chrf(hyp, ref)
                #     similarity += cal_similarity(hyp, ref)

        #         epoch_chrf += chrf / bsz
        #         epoch_similarity += similarity / bsz

        # print(
        #     f"[{now_time()}] Average Val Loss: {loss.item()}. "
        #     f"Average ChrF: {chrf/len(dataloader)}, Similarity: {similarity/len(dataloader)}."
        # )

    def run(self) -> None:
        for epoch in range(self.epoch_num):
            self._now_epoch = epoch
            self.train(self.train_dataloader, self.saver)

        delattr(self, "_now_epoch")

    @property
    def now_epoch(self):
        return self._now_epoch

    def check_now_epoch(self) -> int:
        try:
            now_epoch = self.now_epoch
        except AttributeError:
            now_epoch = 0
        return now_epoch
