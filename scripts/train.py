import sys

sys.path.append("..")
import torch
import torch_npu
from rdkit import RDLogger
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.dataset import MMPDataset
from src.metrics import MMPMetrics
from src.model.transformer import Transformer
from src.tokenizer import MMPTokenizer
from src.trainer import ModelSaver, ModelTrainer
from src.utils import count_parameters, initialize_weights

RDLogger.DisableLog("rdApp.*")

if __name__ == "__main__":
    learning_rate = 1e-4
    weight_decay = 0.001
    eps = 1e-8
    factor = 0.9
    patience = 25

    device = torch.device("npu")
    tokenizer = MMPTokenizer()
    tokenizer.load_word_table("../data/smiles_word_table.yaml")

    train_dataset = MMPDataset(
        "../data/1bond/100k_dataset/train_smiles.csv", max_len=300, tokenizer=tokenizer, nrows=2000
    )
    val_dataset = MMPDataset(
        "../data/1bond/100k_dataset/val_smiles.csv", max_len=300, tokenizer=tokenizer, nrows=100
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=20,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=20,
    )

    vocab_size = tokenizer.vocab_size
    pad_value = tokenizer.vocab2index[tokenizer.pad]

    model = Transformer(
        d_model=1024,
        n_head=8,
        d_ffn=2048,
        d_linear=512,
        dropout=0.1,
        enc_n_layers=1,
        dec_n_layers=1,
        vocab_size=vocab_size,
        pad_value=pad_value,
        max_len=300,
        device=device,
    ).to(device)

    print(f"The model has {count_parameters(model):,} trainable parameters")
    model.apply(initialize_weights)

    optimizer = Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=eps
    )
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=optimizer, verbose=True, factor=factor, patience=patience
    # )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_value)

    metrics = MMPMetrics(tokenizer)
    saver = ModelSaver("../checkpoints", len(train_dl), model_num_per_epoch=5)

    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        metrics=metrics,
        saver=saver,
        tokenizer=tokenizer,
        device=device,
    )
    # trainer.evaluate(val_dl, metrics)
    trainer.run(train_dl, val_dl, epoch_num=5)
