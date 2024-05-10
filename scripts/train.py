import sys

sys.path.append("..")
import torch
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from src.dataset import MMPDataset
from src.model.transformer import Transformer
from src.tokenizer import StrTokenizer
from src.utils import (
    cal_chrf,
    cal_similarity,
    count_parameters,
    initialize_weights,
    now_time,
    trim_seqs,
)


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: _Loss,
) -> float:
    model.train()
    epoch_loss = 0
    for i, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()
        output = model(src, tgt)

        loss = criterion(output.transpose(1, 2), tgt.long())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        print(
            f"[{now_time()}] Step: {i} ({round((i / len(dataloader)) * 100, 2)}%), "
            f"Train Loss: {loss.item()}."
        )

    return epoch_loss / len(dataloader)


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: _Loss, tokenizer: StrTokenizer
) -> float:
    model.eval()
    epoch_loss = 0
    chrf = 0.0
    similarity = 0.0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        output = model(src, tgt)

        loss = criterion(output.transpose(1, 2), tgt.long())
        epoch_loss += loss.item()

        output_seq = output.argmax(dim=-1).squeeze().cpu().tolist()
        tgt_seq = tgt.cpu().tolist()
        output_seq = trim_seqs(output_seq, tokenizer)
        tgt_seq = trim_seqs(tgt_seq, tokenizer)

        for hyp, ref in zip(output_seq, tgt_seq):
            chrf += cal_chrf(hyp, ref)
            similarity += cal_similarity(hyp, ref)

    print(
        f"[{now_time()}] Average Val Loss: {loss.item()}. "
        f"Average ChrF: {chrf}, Similarity: {similarity}."
    )
    return epoch_loss / len(dataloader)


if __name__ == "__main__":
    learning_rate = 1e-4
    weight_decay = 0.001
    eps = 1e-8
    factor = 0.9
    patience = 25

    device = torch.device("cpu")
    dataset = MMPDataset("../data/1bond/100k_dataset/test_smiles.csv", max_len=300)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=20,
    )

    vocab_size = dataset.tokenizer.vocab_size
    pad_value = dataset.tokenizer.vocab2index[dataset.tokenizer.pad]

    model = Transformer(
        d_model=512,
        n_head=4,
        d_ffn=128,
        d_linear=256,
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

    # train(model, dataloader, optimizer, criterion)
    evaluate(model, dataloader, criterion, dataset.tokenizer)
