import sys

sys.path.append("..")
import torch
import torch_npu
from torch.utils.data import DataLoader

from src.dataset import MolProtInferDataset
from src.tokenizer import (
    ProteinTokenizer,
    SelfiesTokenizer,
    SmilesTokenizer,
    share_vocab,
)

if __name__ == "__main__":
    device = torch.device("npu:1")

    smiles_tokenizer = SmilesTokenizer()
    smiles_tokenizer.load_word_table("../data/all/smiles_word_table.yaml")
    protein_tokenizer = ProteinTokenizer()
    protein_tokenizer, smiles_tokenizer = share_vocab(protein_tokenizer, smiles_tokenizer)

    dataset = MolProtInferDataset(
        "../data/finetune/runtime/datasets_seed_0/finetune_test_smiles.csv",
        mol_tokenizer=smiles_tokenizer,
        prot_tokenizer=protein_tokenizer,
    )

    pad_idx = dataset.mol_tokenizer.vocab2index[dataset.mol_tokenizer.pad]
    pad_fn = lambda data: dataset.pad_batch(data, pad_idx, left_pad=True)

    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=20, collate_fn=pad_fn
    )

    for src, tgt in dataloader:
        # print(batch.shape)
        print(type(src))
        print(src)
        mol, prot = src
        print(mol.shape)
        print(prot.shape)
        print(tgt.shape)
        print(mol.device)
        print(prot.device)
        print(tgt.device)
        mol = mol.to(device)
        prot = prot.to(device)
        tgt = tgt.to(device)
        print(mol.device)
        print(prot.device)
        print(tgt.device)
        print(mol)
        print(prot)
        print(tgt)
        break
