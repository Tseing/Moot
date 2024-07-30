import sys

sys.path.append("..")
import os
import os.path as osp
from typing import Optional

import torch
import torch_npu
from rdkit import RDLogger
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from src.dataset import MolProtDataset
from src.metrics import SelfiesMetrics, SmilesMetrics
from src.model.crafted_transformer import OptFormer
from src.tokenizer import (
    ProteinTokenizer,
    SelfiesTokenizer,
    SmilesTokenizer,
    StrTokenizer,
    share_vocab,
)
from src.trainer import ModelSaver, ModelTrainer
from src.utils import Cfg, Log, count_parameters, initialize_weights

if __name__ == "__main__":
    # os.environ['ASCEND_LAUNCH_BLOCKING'] = '1'
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
    torch.npu.set_device(device)

    mol_tokenizer: Optional[StrTokenizer] = None
    if cfg.data_format == "SMILES":
        mol_tokenizer = SmilesTokenizer()
    elif cfg.data_format == "SELFIES":
        mol_tokenizer = SelfiesTokenizer()
    else:
        assert False, (
            f"Config 'data_format' should be 'SMILES' or 'SELFIES', " f"but got '{cfg.tokenizer}'."
        )
    mol_tokenizer.load_word_table(osp.join(cfg.DATA_DIR, cfg.word_table_path))
    prot_tokenizer = ProteinTokenizer()
    prot_tokenizer, smiles_tokenizer = share_vocab(prot_tokenizer, mol_tokenizer)

    train_dataset = MolProtDataset(
        osp.join(cfg.DATA_DIR, cfg.train_data_path),
        mol_max_len=cfg.mol_max_len,
        prot_max_len=cfg.prot_max_len,
        mol_tokenizer=mol_tokenizer,
        prot_tokenizer=prot_tokenizer,
        left_pad=True,
    )
    val_dataset = MolProtDataset(
        osp.join(cfg.DATA_DIR, cfg.val_data_path),
        mol_max_len=cfg.mol_max_len,
        prot_max_len=cfg.prot_max_len,
        mol_tokenizer=mol_tokenizer,
        prot_tokenizer=prot_tokenizer,
        left_pad=True,
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

    vocab_size = mol_tokenizer.vocab_size
    pad_value = mol_tokenizer.vocab2index[mol_tokenizer.pad]

    model = OptFormer(
        d_model=cfg.d_model,
        n_head=cfg.n_head,
        enc_n_layer=cfg.enc_n_layer,
        dec_n_layer=cfg.dec_n_layer,
        enc_d_ffn=cfg.d_enc_ffn,
        dec_d_ffn=cfg.d_dec_ffn,
        fuse_d_ffn=cfg.d_fuse_ffn,
        enc_dropout=cfg.enc_dropout,
        dec_dropout=cfg.dec_dropout,
        enc_embed_dropout=cfg.enc_embed_dropout,
        dec_embed_dropout=cfg.dec_embed_dropout,
        enc_relu_dropout=cfg.enc_relu_dropout,
        dec_relu_dropout=cfg.dec_relu_dropout,
        enc_attn_dropout=cfg.enc_attn_dropout,
        dec_attn_dropout=cfg.dec_attn_dropout,
        vocab_size=vocab_size,
        padding_idx=pad_value,
        mol_max_len=cfg.mol_max_len,
        prot_max_len=cfg.prot_max_len,
        device=device,
        seed=cfg.seed,
    ).to(device)

    logger.info(f"The model has {count_parameters(model):,} trainable parameters")
    logger.info(model)
    model.apply(initialize_weights)

    optimizer = Adam(params=model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=750, T_mult=2, eta_min=cfg.min_learning_rate
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_value)

    if cfg.data_format == "SMILES":
        metrics = SmilesMetrics(mol_tokenizer)
    elif cfg.data_format == "SELFIES":
        metrics = SelfiesMetrics(mol_tokenizer)
    else:
        assert False, (
            f"Config 'data_format' should be 'SMILES' or 'SELFIES', " f"but got '{cfg.tokenizer}'."
        )

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
