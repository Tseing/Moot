import sys

sys.path.append("..")
import torch
from torch.utils.data import DataLoader

from src.dataset import MolProtPairDataset
from src.tokenizer import (
    ProteinTokenizer,
    SelfiesTokenizer,
    share_vocab,
)

if __name__ == "__main__":
    device = torch.device("cpu")

    mol_tokenizer = SelfiesTokenizer()
    mol_tokenizer.load_word_table("../data/all/selfies_word_table.yaml")
    prot_tokenizer = ProteinTokenizer()
    prot_tokenizer, mol_tokenizer = share_vocab(prot_tokenizer, mol_tokenizer)


    dataset = MolProtPairDataset(
        "../data/finetune/runtime/datasets_seed_0/finetune_test_selfies.csv",
        ("mol_a", "mol_b", "sequence"),
        mol_tokenizer=mol_tokenizer,
        prot_tokenizer=prot_tokenizer,
        mol_max_len=None,
        prot_max_len=None,
        left_pad=False,
        pad_batch=True,
        nrows=200,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=20,
        collate_fn=dataset.pad_batch_fn
    )

    for src, tgt in dataloader:
        # print(batch.shape)
        print(type(src))
        print(src)
        print(tgt)
        mol, prot = src
        print(mol.shape)
        print(prot.shape)
        # print(tgt.shape)
        # print(mol.device)
        # print(prot.device)
        # print(tgt.device)
        # mol = mol.to(device)
        # prot = prot.to(device)
        # tgt = tgt.to(device)
        # print(mol.device)
        # print(prot.device)
        # print(tgt.device)
        # print(mol)
        # print(prot)
        # print(tgt)
        break
