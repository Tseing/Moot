import sys

sys.path.append("..")

import torch
from rdkit import RDLogger
from torch.utils.data import DataLoader

from src.dataset import MolProtPairDataset
from src.inferencer import Inferencer
from src.model.optformer import OptFormer
from src.tokenizer import ProteinTokenizer, SelfiesTokenizer, share_vocab

RDLogger.DisableLog("rdApp.*")

if __name__ == "__main__":
    mol_max_len = 250
    prot_max_len = 1500
    left_pad = False
    d_model = 512
    n_head = 4
    d_enc_ffn = 1024
    d_dec_ffn = 1024
    d_fuse_ffn = 1024
    enc_n_layer = 2
    dec_n_layer = 4
    enc_dropout = 0.2
    dec_dropout = 0.2
    enc_embed_dropout = 0.15
    dec_embed_dropout = 0.15
    enc_relu_dropout = 0.1
    dec_relu_dropout = 0.1
    enc_attn_dropout = 0.15
    dec_attn_dropout = 0.15

    device = torch.device("cpu")

    mol_tokenizer = SelfiesTokenizer()
    mol_tokenizer.load_word_table("../data/all/selfies_word_table.yaml")
    prot_tokenizer = ProteinTokenizer()
    prot_tokenizer, mol_tokenizer = share_vocab(prot_tokenizer, mol_tokenizer)

    ckpt = torch.load(
        "../checkpoints/pretrain_optformer_selfies/model_epoch2_step0.pt",
        map_location=device,
    )

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

    print(f"Tokenizer vocab size: {mol_tokenizer.vocab_size}.")
    vocab_size = mol_tokenizer.vocab_size
    pad_value = mol_tokenizer.vocab2index[mol_tokenizer.pad]

    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=20, collate_fn=dataset.collate_fn
    )
    model = OptFormer(
        d_model=d_model,
        n_head=n_head,
        enc_n_layer=enc_n_layer,
        dec_n_layer=dec_n_layer,
        enc_d_ffn=d_enc_ffn,
        dec_d_ffn=d_dec_ffn,
        fuse_d_ffn=d_fuse_ffn,
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
        mol_max_len=mol_max_len,
        prot_max_len=prot_max_len,
        left_pad=left_pad,
        device=device,
    ).to(device)

    model.load_state_dict(ckpt["model"])

    inferencer = Inferencer(
        [model],
        dataloader,
        tokenizer=mol_tokenizer,
        max_len=100,
        n_best=2,
        beam_size=5,
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

    inferencer.inference(show=True, save_path=None)
