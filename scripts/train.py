import sys

sys.path.append("..")
import torch
import torch_npu
from rdkit import RDLogger
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from src.dataset import MMPDataset
from src.model.transformer import Transformer
from src.tokenizer import MMPTokenizer, StrTokenizer
from src.utils import (
    ModelSaver,
    cal_chrf,
    cal_similarity,
    count_parameters,
    initialize_weights,
    now_time,
    trim_seqs,
)

RDLogger.DisableLog("rdApp.*")


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: _Loss,
    epoch: int,
    model_saver: ModelSaver,
) -> float:
    model.train()
    epoch_loss = 0
    for step, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()
        output = model(src, tgt)

        loss = criterion(output.transpose(1, 2), tgt.long())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        print(
            f"[{now_time()}] Epoch: {epoch} Step: {step} ({round((step / len(dataloader)) * 100, 2)}%), "
            f"Train Loss: {loss.item()}."
        )

        model_saver.monitor(model, epoch, step)

    return epoch_loss / len(dataloader)


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: _Loss, tokenizer: StrTokenizer, epoch: int
) -> float:
    model.eval()
    epoch_loss = 0
    epoch_chrf = 0.0
    epoch_similarity = 0.0

    with torch.no_grad():
        for step, (src, tgt) in enumerate(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)
            output = model(src, tgt)
            bsz = output.shape[0]
            chrf = 0.0
            similarity = 0.0

            loss = criterion(output.transpose(1, 2), tgt.long())
            epoch_loss += loss.item()

            output_seq = output.argmax(dim=-1).squeeze().cpu().tolist()
            tgt_seq = tgt.cpu().tolist()
            output_seq = trim_seqs(output_seq, tokenizer)
            tgt_seq = trim_seqs(tgt_seq, tokenizer)

            for hyp, ref in zip(output_seq, tgt_seq):
                chrf += cal_chrf(hyp, ref)
                similarity += cal_similarity(hyp, ref)

            epoch_chrf += chrf / bsz
            epoch_similarity += similarity / bsz

    print(
        f"[{now_time()}] Average Val Loss: {loss.item()}. "
        f"Average ChrF: {chrf/len(dataloader)}, Similarity: {similarity/len(dataloader)}."
    )
    return epoch_loss / len(dataloader)


def run(epoch_num: int, train_dl: DataLoader, val_dl: DataLoader) -> None:
    model_saver = ModelSaver("../checkpoints", len(train_dl), model_num_per_epoch=5)
    for epoch in range(epoch_num):
        valid_loss = evaluate(model, val_dl, criterion, tokenizer, epoch)
        train_loss = train(model, train_dl, optimizer, criterion, epoch, model_saver)


if __name__ == "__main__":
    learning_rate = 1e-4
    weight_decay = 0.001
    eps = 1e-8
    factor = 0.9
    patience = 25

    device = torch.device("npu")
    tokenizer = MMPTokenizer()
    tokenizer.load_word_table("../data/word_table.yaml")

    train_dataset = MMPDataset(
        "../data/1bond/100k_dataset/train_smiles.csv", max_len=300, tokenizer=tokenizer
    )
    val_dataset = MMPDataset(
        "../data/1bond/100k_dataset/val_smiles.csv", max_len=300, tokenizer=tokenizer
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, verbose=True, factor=factor, patience=patience
    )

    criterion = nn.CrossEntropyLoss(ignore_index=pad_value)

    run(epoch_num=5, train_dl=train_dl, val_dl=val_dl)
    # train(model, train_dl, optimizer, criterion)
    # evaluate(model, val_dl, criterion, tokenizer)
