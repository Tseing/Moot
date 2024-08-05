import sys

sys.path.append("..")
import torch
from torch.utils.data import DataLoader

from src.dataset import MolProtDataset
from src.model.optformer import OptFormer
from src.tokenizer import (
    ProteinTokenizer,
    SelfiesTokenizer,
    SmilesTokenizer,
    share_vocab,
)

if __name__ == "__main__":
    device = torch.device("cpu")

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

    vocab_size = dataset.mol_tokenizer.vocab_size
    pad_idx = dataset.mol_tokenizer.vocab2index[dataset.mol_tokenizer.pad]

    model = OptFormer(
        d_model=256,
        n_head=4,
        enc_n_layer=2,
        dec_n_layer=2,
        enc_d_ffn=256,
        dec_d_ffn=256,
        fuse_d_ffn=256,
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
        mol_max_len=250,
        prot_max_len=1500,
        device=device,
    ).to(device)

    for mol, prot, tgt in dataloader:
        print(f"vocab_size: {vocab_size}")
        mol = mol.int().to(device)
        prot = prot.int().to(device)
        tgt = tgt.int().to(device)
        print(mol.shape, mol.dtype)
        print(prot.shape, prot.dtype)
        print(tgt.shape, tgt.dtype)

        out, _ = model(mol, prot, tgt)
        # print(out)
        print(out.shape)
        break
