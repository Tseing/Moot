import sys

sys.path.append("..")
import torch
import torch_npu
from torch.utils.data import DataLoader

from src.dataset import MMPDataset
from src.model.crafted_transformer import Transformer
from src.tokenizer import MMPTokenizer

if __name__ == "__main__":
    device = torch.device("npu")
    tokenizer = MMPTokenizer()
    tokenizer.load_word_table("../data/smiles_word_table.yaml")
    dataset = MMPDataset(
        "../data/1bond/100k_dataset/test_smiles.csv", max_len=300, tokenizer=tokenizer
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=20,
    )

    vocab_size = dataset.tokenizer.vocab_size
    pad_idx = dataset.tokenizer.vocab2index[dataset.tokenizer.pad]

    model = Transformer(
        d_model=4,
        n_head=4,
        enc_n_layer=2,
        dec_n_layer=2,
        enc_d_ffn=256,
        dec_d_ffn=256,
        enc_dropout=0.1,
        dec_dropout=0.1,
        enc_embed_dropout=0.1,
        dec_embed_dropout=0.1,
        enc_relu_dropout=0.1,
        dec_relu_dropout=0.1,
        enc_attn_dropout=0.1,
        dec_attn_dropout=0.1,
        vocab_size=vocab_size,
        padding_idx=pad_idx,
        device=device,
        max_len=300,
    ).to(device)

    for src, tgt in dataloader:
        src = src.int().to(device)
        tgt = tgt.int().to(device)
        print(src.shape, src.dtype)
        print(tgt.shape, tgt.dtype)

        out = model(src, tgt)
        print(out)
        print(out[0].shape, out[1].shape)
        break
