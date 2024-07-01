import sys

sys.path.append("..")
import torch
import torch_npu
from torch.utils.data import DataLoader

from src.dataset import MolProtDataset
from src.tokenizer import (
    ProteinTokenizer,
    SelfiesTokenizer,
    SmilesTokenizer,
    share_vocab,
)

if __name__ == "__main__":
    device = torch.device("npu:3")

    smiles_tokenizer = SmilesTokenizer()
    smiles_tokenizer.load_word_table("../data/all/smiles_word_table.yaml")
    protein_tokenizer = ProteinTokenizer()
    protein_tokenizer, smiles_tokenizer = share_vocab(protein_tokenizer, smiles_tokenizer)

    dataset = MolProtDataset(
        "../data/finetune/runtime/datasets_seed_0/finetune_test_smiles.csv",
        mol_max_len=250,
        prot_max_len=1500,
        mol_tokenizer=smiles_tokenizer,
        prot_tokenizer=protein_tokenizer,
        left_pad=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=20,
    )

    for mol, prot, tgt in dataloader:
        # print(batch.shape)
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
