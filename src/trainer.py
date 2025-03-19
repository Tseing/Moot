import os.path as osp
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import auc, roc_curve, f1_score, average_precision_score, confusion_matrix
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .metrics import ModelMetrics
from .typing import Device
from .utils import Log


class ModelSaver:
    def __init__(
        self,
        save_dir: str,
        steps: int,
        model_name: str = "model",
        epoch_gap: int = 1,
        model_num_per_epoch: int = 1,
        save_in_step_zero: bool = True,
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

    def monitor(self, save_params: Dict[str, Any], epoch: int, step: int) -> None:
        if step in self.save_steps:
            file_name = f"{self.model_name}_epoch{epoch}_step{step}.pt"
            torch.save(
                save_params,
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
        warming_step: int,
        metrics: ModelMetrics,
        saver: ModelSaver,
        bos_value: int,
        device: Device,
        logger: Optional[Log] = None,
        log_interval: int = 10,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.warming_step = warming_step
        self.metrics = metrics
        self.saver = saver
        self.bos_value = bos_value
        self.device = device
        self.logger = logger
        self.log_interval = log_interval

        if self.logger:
            self.info = lambda stdout: self.logger.info(stdout)
        else:
            self.info = lambda stdout: print(stdout)

    @staticmethod
    def __process_multi_inp(
        inp: Union[Tuple[torch.Tensor, ...], List[torch.Tensor], torch.Tensor], device: Device
    ) -> Tuple[torch.Tensor, ...]:
        if isinstance(inp, (tuple, list)):
            inps = tuple(tensor.to(device) for tensor in inp)
        elif isinstance(inp, torch.Tensor):
            inps = tuple([inp.to(device)])
        else:
            raise TypeError(f"Cannot process 'inp' with type '{type(inp)}'.")

        return inps

    def train(self, dataloader: DataLoader, saver: ModelSaver):
        self.model.train()
        epoch = self.check_now_epoch()
        epoch_loss = 0
        for step, (src, tgt) in enumerate(dataloader):
            src = self.__process_multi_inp(src, self.device)
            tgt = tgt.to(self.device)

            self.optimizer.zero_grad()
            output, _ = self.model(*src, tgt[:, :-1])

            loss = self.criterion(output.transpose(1, 2), tgt[:, 1:].long())
            loss.backward()

            self.optimizer.step()

            if step + epoch * len(dataloader) > self.warming_step:
                self.scheduler.step()

            epoch_loss += loss.item()

            if step % self.log_interval == 0:
                step_stdout = (
                    f"Epoch: {epoch} Step: {step} "
                    f"({(step + 1) / len(dataloader) * 100:.2f}%), "
                    f"Train Loss: {loss.item()}, Learning Rate: {self.scheduler.get_last_lr()[0]}."
                )

                self.info(step_stdout)

            saver.monitor(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                epoch,
                step,
            )

    def evaluate(self, dataloader: DataLoader, metrics: ModelMetrics):
        self.model.eval()
        epoch = self.check_now_epoch()
        epoch_loss = 0.0

        with torch.no_grad():
            for step, (src, tgt) in enumerate(dataloader):
                src = self.__process_multi_inp(src, self.device)
                tgt = tgt.to(self.device)
                output, _ = self.model(*src, tgt[:, :-1])

                loss = self.criterion(output.transpose(1, 2), tgt[:, 1:].long())
                epoch_loss += loss.item()

                output_seq = output.argmax(dim=-1).cpu().numpy()
                # Insert <bos> token in the first position.
                output_seq = np.pad(
                    output_seq,
                    ([0, 0], [1, 0]),
                    "constant",
                    constant_values=self.bos_value,
                )

                tgt_seq = tgt.cpu().numpy()
                metrics.update(output_seq, tgt_seq)

                if step % self.log_interval == 0:
                    step_stdout = (
                        f"Epoch: {epoch} Step: {step} "
                        f"({(step + 1) / len(dataloader) * 100:.2f}%), "
                        f"Val Loss: {loss.item()}."
                    )

                    self.info(step_stdout)

            result = metrics.metric()
            self.metrics.clear()

        result_str = ", ".join([f"{k}: {v}" for k, v in result.items()])
        epoch_stdout = f"Epoch: {epoch} Average Val Loss: {loss.item()}. {result_str}"
        self.info(epoch_stdout)

    def run(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epoch_num: int,
        initial_epoch: int = 0,
    ) -> None:
        for epoch in range(epoch_num):
            epoch += initial_epoch
            self._now_epoch = epoch
            self.train(train_dataloader, self.saver)
            self.evaluate(val_dataloader, self.metrics)

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


class ClassifierTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: _Loss,
        scheduler: LRScheduler,
        warming_step: int,
        saver: ModelSaver,
        device: Device,
        logger: Optional[Log] = None,
        log_interval: int = 10,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.warming_step = warming_step
        self.saver = saver
        self.device = device
        self.logger = logger
        self.log_interval = log_interval

        if self.logger:
            self.info = lambda stdout: self.logger.info(stdout)
        else:
            self.info = lambda stdout: print(stdout)

    @staticmethod
    def __process_multi_inp(
        inp: Union[Tuple[torch.Tensor, ...], List[torch.Tensor], torch.Tensor], device: Device
    ) -> Tuple[torch.Tensor, ...]:
        if isinstance(inp, (tuple, list)):
            inps = tuple(tensor.to(device) for tensor in inp)
        elif isinstance(inp, torch.Tensor):
            inps = tuple([inp.to(device)])
        else:
            raise TypeError(f"Cannot process 'inp' with type '{type(inp)}'.")

        return inps

    def train(self, dataloader: DataLoader, saver: ModelSaver):
        self.model.train()
        epoch = self.check_now_epoch()
        epoch_loss = 0
        for step, (src, y) in enumerate(dataloader):
            src = self.__process_multi_inp(src, self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(*src)

            loss = self.criterion(output, y)
            loss.backward()

            self.optimizer.step()

            if step + epoch * len(dataloader) > self.warming_step:
                self.scheduler.step()

            epoch_loss += loss.item()

            if step % self.log_interval == 0:
                step_stdout = (
                    f"Epoch: {epoch} Step: {step} "
                    f"({(step + 1) / len(dataloader) * 100:.2f}%), "
                    f"Train Loss: {loss.item()}, Learning Rate: {self.scheduler.get_last_lr()[0]}."
                )

                self.info(step_stdout)

            saver.monitor(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                epoch,
                step,
            )

    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        epoch = self.check_now_epoch()
        epoch_loss = 0.0
        predicts = []
        labels = []

        with torch.no_grad():
            for step, (src, y) in enumerate(dataloader):
                src = self.__process_multi_inp(src, self.device)
                y = y.to(self.device)
                output = self.model(*src)

                loss = self.criterion(output, y)
                epoch_loss += loss.item()

                predicts.append(output.cpu().numpy())
                labels.append(y.cpu().numpy())

                if step % self.log_interval == 0:
                    step_stdout = (
                        f"Epoch: {epoch} Step: {step} "
                        f"({(step + 1) / len(dataloader) * 100:.2f}%), "
                        f"Val Loss: {loss.item()}."
                    )

                    self.info(step_stdout)

            predicts = np.concatenate(predicts, axis=0)
            labels = np.concatenate(labels, axis=0)
            fpr, tpr, _ = roc_curve(labels, predicts)
            roc_auc = auc(fpr, tpr)
            acc = ((predicts > 0.5) == labels).sum() / labels.shape[0]

        epoch_stdout = (
            f"Epoch: {epoch} Average Val Loss: {loss.item()}. Accuracy: {acc}, AUC-ROC: {roc_auc}."
        )
        self.info(epoch_stdout)

    def test(self, dataloader: DataLoader):
        self.model.eval()
        predicts = []
        labels = []

        with torch.no_grad():
            for _, (src, y) in enumerate(dataloader):
                src = self.__process_multi_inp(src, self.device)
                y = y.to(self.device)
                output = self.model(*src)

                predicts.append(output.cpu().numpy())
                labels.append(y.cpu().numpy())

            predicts = np.concatenate(predicts, axis=0)
            labels = np.concatenate(labels, axis=0)
            result = self.metric(labels, predicts)

        epoch_stdout = ", ".join([f"{k}: {v}" for k, v in result.items()])
        self.info(epoch_stdout)

    @staticmethod
    def metric(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
        metrics = {}
        y_pred = (y_scores > 0.5).astype(np.int32)
        metrics["PRC"] = average_precision_score(y_true, y_scores)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        metrics["AUC"] = auc(fpr, tpr)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["Sensitivity"] = tp / (tp + fn)
        metrics["Specificity"] = tn / (tn + fp)
        metrics["F1"] = f1_score(y_true, y_pred)
        return metrics

    def run(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        epoch_num: int,
        initial_epoch: int = 0,
    ) -> None:
        for epoch in range(epoch_num):
            epoch += initial_epoch
            self._now_epoch = epoch
            self.train(train_dataloader, self.saver)
            self.evaluate(val_dataloader)
            self.test(test_dataloader)

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
