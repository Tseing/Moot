import sys

sys.path.append("..")
import os
import os.path as osp

import torch
import torch_npu
from rdkit import RDLogger
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from src.dataset import MMPDataset
from src.metrics import MMPMetrics
from src.model.crafted_transformer import Transformer
from src.tokenizer import MMPTokenizer
from src.trainer import ModelSaver, ModelTrainer
from src.utils import Cfg, Log, count_parameters, initialize_weights

RDLogger.DisableLog("rdApp.*")

if __name__ == "__main__":
    cfg = Cfg()
    cfg.parse()

    task_name = cfg.task_name
    log_path = osp.join(cfg.LOG_DIR, f"{task_name}.log")
    save_dir = osp.join(cfg.CKPT_DIR, task_name)
    if not osp.exists(save_dir):
        os.mkdir(save_dir)

    logger = Log("train", log_path)
    logger.info(cfg)

    device = torch.device(cfg.device)
    tokenizer = MMPTokenizer()
    tokenizer.load_word_table(osp.join(cfg.DATA_DIR, cfg.word_table_path))

    train_dataset = MMPDataset(
        osp.join(cfg.DATA_DIR, cfg.train_data_path),
        max_len=cfg.max_len,
        tokenizer=tokenizer,
        left_pad=True,
    )
    val_dataset = MMPDataset(
        osp.join(cfg.DATA_DIR, cfg.val_data_path), max_len=cfg.max_len, tokenizer=tokenizer
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

    vocab_size = tokenizer.vocab_size
    pad_value = tokenizer.vocab2index[tokenizer.pad]

    model = Transformer(
        d_model=cfg.d_model,
        n_head=cfg.n_head,
        enc_n_layer=cfg.enc_n_layer,
        dec_n_layer=cfg.dec_n_layer,
        enc_d_ffn=cfg.d_enc_ffn,
        dec_d_ffn=cfg.d_dec_ffn,
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
        max_len=cfg.max_len,
        device=device,
        seed=cfg.seed,
    ).to(device)

    logger.info(f"The model has {count_parameters(model):,} trainable parameters")
    logger.info(model)
    model.apply(initialize_weights)

    optimizer = Adam(params=model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=200, T_mult=2, eta_min=cfg.min_learning_rate
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_value)

    metrics = MMPMetrics(tokenizer)
    saver = ModelSaver(save_dir, len(train_dl), model_num_per_epoch=2)

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        warming_step=cfg.warming_step,
        criterion=criterion,
        metrics=metrics,
        saver=saver,
        tokenizer=tokenizer,
        device=device,
        logger=logger,
        log_frequency=cfg.log_frequency,
    )

    trainer.run(train_dl, val_dl, epoch_num=cfg.epoch_num)
