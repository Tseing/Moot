import sys

sys.path.append("..")
import os.path as osp
from typing import Optional

import torch
from rdkit import RDLogger
from torch.utils.data import DataLoader

from src.dataset import MolInferDataset
from src.inferencer import Inferencer
from src.model.optformer import Transformer
from src.tokenizer import SelfiesTokenizer, SmilesTokenizer, StrTokenizer
from src.utils import Cfg

if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")

    cfg = Cfg()
    cfg.parse()

    print(f"Config:\n{repr(cfg)}")

    mol_tokenizer: Optional[StrTokenizer] = None
    if cfg.data_format == "SMILES":
        mol_tokenizer = SmilesTokenizer()
    elif cfg.data_format == "SELFIES":
        mol_tokenizer = SelfiesTokenizer()
    else:
        assert False, (
            f"Config 'data_format' should be 'SMILES' or 'SELFIES', " f"but got '{cfg.tokenizer}'."
        )
    mol_tokenizer.load_word_table(osp.join(cfg.DATA_DIR, cfg.word_table_path))
    device = torch.device(cfg.device)

    ckpt = torch.load(
        osp.join(cfg.CKPT_DIR, cfg.ckpt_path),
        map_location=device,
    )

    dataset = MolInferDataset(
        osp.join(cfg.DATA_DIR, cfg.test_data_path),
        ("mol_a", "mol_b"),
        tokenizer=mol_tokenizer,
    )

    print(f"Tokenizer vocab size: {mol_tokenizer.vocab_size}.")
    vocab_size = mol_tokenizer.vocab_size
    pad_value = mol_tokenizer.vocab2index[mol_tokenizer.pad]
    pad_fn = lambda data: dataset.pad_batch(data, pad_value, left_pad=cfg.infer_left_pad)

    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=20, collate_fn=pad_fn
    )

    model = Transformer(
        d_model=cfg.d_model,
        n_head=cfg.n_head,
        enc_n_layer=cfg.enc_n_layer,
        dec_n_layer=cfg.dec_n_layer,
        enc_d_ffn=cfg.d_enc_ffn,
        dec_d_ffn=cfg.d_dec_ffn,
        enc_dropout=cfg.enc_dropout,
        dec_dropout=cfg.dec_dropout,
        enc_embed_dropout=cfg.enc_embed_dropout,
        dec_embed_dropout=cfg.dec_embed_dropout,
        enc_relu_dropout=cfg.enc_relu_dropout,
        dec_relu_dropout=cfg.dec_relu_dropout,
        enc_attn_dropout=cfg.enc_attn_dropout,
        dec_attn_dropout=cfg.dec_attn_dropout,
        vocab_size=vocab_size,
        padding_idx=pad_value,
        left_pad=cfg.left_pad,
        max_len=cfg.max_len,
        device=device,
        seed=cfg.seed,
    ).to(device)

    model.load_state_dict(ckpt["model"])

    inferencer = Inferencer(
        [model],
        dataloader,
        tokenizer=mol_tokenizer,
        max_len=cfg.infer_max_len,
        n_best=cfg.n_best,
        beam_size=cfg.beam_size,
        min_len=cfg.infer_min_len,
        stop_early=cfg.stop_early,
        normalize_scores=cfg.normalize_scores,
        len_penalty=cfg.len_penalty,
        unk_penalty=cfg.unk_penalty,
        sampling=cfg.sampling,
        sampling_topk=cfg.sampling_topk,
        sampling_temperature=cfg.sampling_temperature,
        device=device,
    )

    inferencer.inference(show=False, save_path=osp.join(cfg.OUTPUT_DIR, cfg.save_path))
