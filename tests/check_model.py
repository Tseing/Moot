import sys

sys.path.append("..")
import torch
from torch.utils.data import DataLoader

from src.dataset import MMPDataset
from src.model.transformer import Transformer

if __name__ == "__main__":
    device = torch.device("cpu")
    dataset = MMPDataset("../data/1bond/100k_dataset/test_smiles.csv", max_len=300)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=20,
    )

    vocab_size = dataset.tokenizer.vocab_size
    pad_idx = dataset.tokenizer.vocab2index[dataset.tokenizer.pad]

    model = Transformer(
        d_model=512,
        n_head=4,
        d_ffn=128,
        d_linear=256,
        dropout=0.1,
        enc_n_layers=1,
        dec_n_layers=1,
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        max_len=500,
        device=device,
    ).to(device)

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        print(src.shape)
        print(tgt.shape)

        out = model(src, tgt)
        print(out.shape)
        print(out)
        break
