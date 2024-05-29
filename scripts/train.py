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
from src.utils import Log, count_parameters, initialize_weights

RDLogger.DisableLog("rdApp.*")

if __name__ == "__main__":
    learning_rate = 1e-4
    min_learning_rate = 1e-5
    warming_step = 2000
    weight_decay = 0.0
    batch_size = 128
    epoch_num = 5

    max_len = 300
    d_model = 1024
    n_head = 8
    d_ffn = 2048
    # d_linear = 512
    dropout = 0.1
    enc_n_layer = 4
    dec_n_layer = 4

    work_name = f"pretrain_crafted_100k_smiles_bsz{batch_size}"
    log_path = f"../log/{work_name}.log"
    save_dir = f"../checkpoints/{work_name}"
    if not osp.exists(save_dir):
        os.mkdir(save_dir)

    train_data_path = "../data/1bond/100k_dataset/train_smiles.csv"
    val_data_path = "../data/1bond/100k_dataset/val_smiles.csv"
    word_table_path = "../data/smiles_word_table.yaml"

    logger = Log("train", log_path)
    device = torch.device("npu:0")
    tokenizer = MMPTokenizer()
    tokenizer.load_word_table(word_table_path)

    train_dataset = MMPDataset(train_data_path, max_len=max_len, tokenizer=tokenizer, left_pad=True)
    val_dataset = MMPDataset(val_data_path, max_len=max_len, tokenizer=tokenizer)
    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=20,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=20,
    )

    vocab_size = tokenizer.vocab_size
    pad_value = tokenizer.vocab2index[tokenizer.pad]

    model = Transformer(
        d_model=d_model,
        n_head=n_head,
        enc_n_layer=enc_n_layer,
        dec_n_layer=dec_n_layer,
        enc_d_ffn=d_ffn,
        dec_d_ffn=d_ffn,
        enc_dropout=dropout,
        dec_dropout=dropout,
        enc_embed_dropout=dropout,
        dec_embed_dropout=dropout,
        enc_relu_dropout=dropout,
        dec_relu_dropout=dropout,
        enc_attn_dropout=dropout,
        dec_attn_dropout=dropout,
        vocab_size=vocab_size,
        padding_idx=pad_value,
        max_len=max_len,
        device=device,
        seed=24,
    ).to(device)

    logger.info(f"The model has {count_parameters(model):,} trainable parameters")
    logger.info(model)
    model.apply(initialize_weights)

    optimizer = Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=200, T_mult=2, eta_min=min_learning_rate
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_value)

    metrics = MMPMetrics(tokenizer)
    saver = ModelSaver(save_dir, len(train_dl), model_num_per_epoch=2)

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        warming_step=warming_step,
        criterion=criterion,
        metrics=metrics,
        saver=saver,
        tokenizer=tokenizer,
        device=device,
        logger=logger,
        step_per_info=50,
    )

    trainer.run(train_dl, val_dl, epoch_num=epoch_num)
