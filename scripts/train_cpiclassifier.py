import os
import os.path as osp
import sys
from typing import Optional

import torch
from rdkit import RDLogger
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

sys.path.append("..")
from src.dataset import MolClassifyDataset, MolProtClassifyDataset
from src.launcher import ModelLauncher
from src.model.classifier import CPIClassifier
from src.tokenizer import (
    ProteinTokenizer,
    SelfiesTokenizer,
    SmilesTokenizer,
    StrTokenizer,
    share_vocab,
)
from src.trainer import ClassifierTrainer, ModelSaver
from src.utils import Cfg, Log, initialize_weights

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

    mol_tokenizer: Optional[StrTokenizer] = None
    if cfg.data_format == "SMILES":
        mol_tokenizer = SmilesTokenizer()
    elif cfg.data_format == "SELFIES":
        mol_tokenizer = SelfiesTokenizer()
    else:
        assert False, (
            f"Config 'data_format' should be 'SMILES' or 'SELFIES', "
            f"but got '{cfg.data_format}'."
        )
    mol_tokenizer.load_word_table(osp.join(cfg.DATA_DIR, cfg.word_table_path))

    prot_tokenizer: Optional[StrTokenizer] = None
    if cfg.model == "Transformer":
        train_dataset = MolClassifyDataset(
            osp.join(cfg.DATA_DIR, cfg.train_data_path),
            cfg.data_cols,
            tokenizer=mol_tokenizer,
            max_len=cfg.max_len,
        )
        val_dataset = MolClassifyDataset(
            osp.join(cfg.DATA_DIR, cfg.val_data_path),
            cfg.data_cols,
            tokenizer=mol_tokenizer,
            max_len=cfg.max_len,
        )
        test_dataset = MolClassifyDataset(
            osp.join(cfg.DATA_DIR, cfg.test_data_path),
            cfg.data_cols,
            tokenizer=mol_tokenizer,
            max_len=cfg.max_len,
        )
        cfg.set("seq_len", cfg.max_len)

    elif cfg.model == "Optformer":
        prot_tokenizer = ProteinTokenizer()
        prot_tokenizer, mol_tokenizer = share_vocab(prot_tokenizer, mol_tokenizer)

        train_dataset = MolProtClassifyDataset(
            osp.join(cfg.DATA_DIR, cfg.train_data_path),
            cfg.data_cols,
            mol_tokenizer=mol_tokenizer,
            mol_max_len=cfg.mol_max_len,
            prot_tokenizer=prot_tokenizer,
            prot_max_len=cfg.prot_max_len,
        )
        val_dataset = MolProtClassifyDataset(
            osp.join(cfg.DATA_DIR, cfg.val_data_path),
            cfg.data_cols,
            mol_tokenizer=mol_tokenizer,
            mol_max_len=cfg.mol_max_len,
            prot_tokenizer=prot_tokenizer,
            prot_max_len=cfg.prot_max_len,
        )
        test_dataset = MolProtClassifyDataset(
            osp.join(cfg.DATA_DIR, cfg.test_data_path),
            cfg.data_cols,
            mol_tokenizer=mol_tokenizer,
            mol_max_len=cfg.mol_max_len,
            prot_tokenizer=prot_tokenizer,
            prot_max_len=cfg.prot_max_len,
        )
        cfg.set("seq_len", cfg.mol_max_len + cfg.prot_max_len)
    else:
        assert False

    cfg.set("vocab_size", mol_tokenizer.vocab_size)
    cfg.set("pad_value", mol_tokenizer.vocab2index[mol_tokenizer.pad])

    launcher = ModelLauncher(cfg.model, cfg, logger, "inference", device)
    encoder = launcher.get_model()
    model = CPIClassifier(encoder, cfg.seq_len, cfg.d_model, cfg.d_classifier).to(device)
    model.fuse_layer.apply(initialize_weights)
    model.layer1.apply(initialize_weights)
    model.layer2.apply(initialize_weights)

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
    test_dl = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=20,
    )

    logger.info(f"Train steps per epoch: {len(train_dl)}. Val steps per epoch {len(val_dataset)}.")

    optimizer = Adam(params=model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=750, T_mult=2, eta_min=cfg.min_learning_rate
    )
    criterion = nn.BCELoss()

    saver = ModelSaver(save_dir, len(train_dl), model_num_per_epoch=cfg.save_interval)

    trainer = ClassifierTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        warming_step=cfg.warming_step,
        criterion=criterion,
        saver=saver,
        device=device,
        logger=logger,
        log_interval=cfg.log_interval,
    )

    trainer.run(train_dl, val_dl, test_dl, epoch_num=cfg.epoch_num)
