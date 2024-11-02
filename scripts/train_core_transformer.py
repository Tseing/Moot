import sys

sys.path.append("..")
import os
import os.path as osp

import torch
from rdkit import RDLogger
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from src.dataset import MolAtomPairDataset
from src.launcher import ModelLauncher
from src.metrics import AtomMetrics
from src.tokenizer import AtomTokenizer, SmilesTokenizer, share_vocab
from src.trainer import ModelSaver, ModelTrainer
from src.utils import Cfg, Log

if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")

    cfg = Cfg()
    cfg.parse()

    task_name = cfg.task_name
    log_path = osp.join(cfg.LOG_DIR, f"{task_name}.log")
    save_dir = osp.join(cfg.CKPT_DIR, task_name)
    if not osp.exists(save_dir):
        os.mkdir(save_dir)

    logger = Log("train", log_path)
    logger.info(f"Config:\n{repr(cfg)}")

    device = torch.device(cfg.device)

    atom_tokenizer = AtomTokenizer()
    mol_tokenizer = SmilesTokenizer()
    mol_tokenizer.load_word_table(osp.join(cfg.DATA_DIR, cfg.word_table_path))
    atom_tokenizer, mol_tokenizer = share_vocab(atom_tokenizer, mol_tokenizer)

    train_dataset = MolAtomPairDataset(
        osp.join(cfg.DATA_DIR, cfg.train_data_path),
        cfg.data_cols,
        mol_tokenizer=mol_tokenizer,
        atom_tokenizer=atom_tokenizer,
        mol_max_len=cfg.max_len,
        atom_max_len=cfg.max_len,
        left_pad=cfg.left_pad,
        pad_batch=False,
    )
    val_dataset = MolAtomPairDataset(
        osp.join(cfg.DATA_DIR, cfg.val_data_path),
        cfg.data_cols,
        mol_tokenizer=mol_tokenizer,
        atom_tokenizer=atom_tokenizer,
        mol_max_len=cfg.max_len,
        atom_max_len=cfg.max_len,
        left_pad=cfg.left_pad,
        pad_batch=False,
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=20,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=20,
    )

    logger.info(f"Train steps per epoch: {len(train_dl)}. Val steps per epoch {len(val_dataset)}.")

    cfg.set("vocab_size", mol_tokenizer.vocab_size)
    cfg.set("pad_value", mol_tokenizer.vocab2index[mol_tokenizer.pad])

    launcher = ModelLauncher("Transformer", cfg, logger, "train", device)
    model = launcher.get_model()

    optimizer = Adam(params=model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=750, T_mult=2, eta_min=cfg.min_learning_rate
    )
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_value)
    metrics = AtomMetrics(atom_tokenizer)
    saver = ModelSaver(save_dir, len(train_dl), model_num_per_epoch=cfg.save_interval)

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        warming_step=cfg.warming_step,
        criterion=criterion,
        metrics=metrics,
        saver=saver,
        tokenizer=mol_tokenizer,
        device=device,
        logger=logger,
        log_interval=cfg.log_interval,
    )

    trainer.run(train_dl, val_dl, epoch_num=cfg.epoch_num)
