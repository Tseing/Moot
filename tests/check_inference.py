import sys

sys.path.append("..")

import torch
# import torch_npu
from rdkit import RDLogger
from torch.utils.data import DataLoader

from src.dataset import MolInferDataset
from src.inferencer import Inferencer
from src.model.optformer import Transformer
from src.tokenizer import SmilesTokenizer

RDLogger.DisableLog("rdApp.*")

if __name__ == "__main__":
    max_len = 250
    left_pad = False
    d_model = 512
    n_head = 8
    enc_d_ffn = 1024
    dec_d_ffn = 1024
    enc_n_layer = 3
    dec_n_layer = 4
    enc_dropout = 0.2
    dec_dropout = 0.2
    enc_embed_dropout = 0.1
    dec_embed_dropout = 0.1
    enc_relu_dropout = 0.1
    dec_relu_dropout = 0.1
    enc_attn_dropout = 0.15
    dec_attn_dropout = 0.15

    device = torch.device("cpu")

    mol_tokenizer = SmilesTokenizer()
    mol_tokenizer.load_word_table("../data/all/smiles_word_table.yaml")
    # protein_tokenizer = ProteinTokenizer()
    # protein_tokenizer, smiles_tokenizer = share_vocab(protein_tokenizer, smiles_tokenizer)

    ckpt = torch.load(
        "../checkpoints/0801basic_transformer_smiles_medium/model_epoch4_step70395.pt", map_location=device
    )

    dataset = MolInferDataset(
        "../data/finetune/runtime/datasets_seed_0/finetune_test_smiles.csv",
        ("mol_a", "mol_b"),
        tokenizer=mol_tokenizer,
        nrows=16,
    )

    print(f"Tokenizer vocab size: {mol_tokenizer.vocab_size}.")
    vocab_size = mol_tokenizer.vocab_size
    pad_value = mol_tokenizer.vocab2index[mol_tokenizer.pad]
    pad_fn = lambda data: dataset.pad_batch(data, pad_value, left_pad=left_pad)

    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=20, collate_fn=pad_fn
    )

    model = Transformer(
        d_model=d_model,
        n_head=n_head,
        enc_n_layer=enc_n_layer,
        dec_n_layer=dec_n_layer,
        enc_d_ffn=enc_d_ffn,
        dec_d_ffn=dec_d_ffn,
        enc_dropout=enc_dropout,
        dec_dropout=dec_dropout,
        enc_embed_dropout=enc_embed_dropout,
        dec_embed_dropout=dec_embed_dropout,
        enc_relu_dropout=enc_relu_dropout,
        dec_relu_dropout=dec_relu_dropout,
        enc_attn_dropout=enc_attn_dropout,
        dec_attn_dropout=dec_attn_dropout,
        vocab_size=vocab_size,
        padding_idx=pad_value,
        left_pad=left_pad,
        max_len=max_len,
        device=device,
    ).to(device)

    # model.apply(initialize_weights)
    model.load_state_dict(ckpt["model"])

    inferencer = Inferencer(
        [model],
        dataloader,
        tokenizer=mol_tokenizer,
        max_len=100,
        n_best=2,
        beam_size=2,
        min_len=1,
        stop_early=False,
        normalize_scores=True,
        len_penalty=1,
        unk_penalty=0,
        sampling=False,
        sampling_topk=-1,
        sampling_temperature=1,
        device=device,
    )

    inferencer.inference()
