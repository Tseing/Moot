import sys

sys.path.append("..")
import torch
import torch_npu
from torch.utils.data import DataLoader

from src.dataset import MMPDataset
from src.tokenizer import MMPTokenizer

if __name__ == "__main__":
    device = torch.device("npu:0")
    tokenizer = MMPTokenizer()
    tokenizer.load_word_table("../data/smiles_word_table.yaml")
    dataset = MMPDataset(
        "../data/1bond/100k_dataset/test_smiles.csv", max_len=300, tokenizer=tokenizer, left_pad=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=20,
    )

    for src, tgt in dataloader:
        # print(batch.shape)
        print(src.shape)
        print(tgt.shape)
        print(src.device)
        print(tgt.device)
        src = src.to(device)
        tgt = tgt.to(device)
        print(src.device)
        print(tgt.device)
        print(src)
        print(tgt)
        break
