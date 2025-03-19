import os.path as osp
import sys
from typing import Optional, Tuple

import torch
from torch import nn

sys.path.append("..")
from src.launcher import ModelLauncher
from src.tokenizer import (
    FragSelfiesTokenizer,
    FragSmilesTokenizer,
    ProteinTokenizer,
    SelfiesTokenizer,
    SmilesTokenizer,
    StrTokenizer,
    share_vocab,
)
from src.utils import Cfg, Log

CONFIGS = {
    "probe_transformer_smiles": {
        "task_name": "probe_transformer_smiles",
        "device": "cuda:0",
        "ckpt_path": "train_transformer_smiles/model_epoch9_step0.pt",
        "word_table_path": "all/smiles_word_table.yaml",
        "data_format": "SMILES",
        "model": "Transformer",
        "max_len": 250,
        "left_pad": False,
        "d_model": 512,
        "n_head": 4,
        "d_enc_ffn": 1024,
        "d_dec_ffn": 1024,
        "enc_n_layer": 2,
        "dec_n_layer": 4,
        "enc_dropout": 0.2,
        "dec_dropout": 0.2,
        "enc_embed_dropout": 0.15,
        "dec_embed_dropout": 0.15,
        "enc_relu_dropout": 0.1,
        "dec_relu_dropout": 0.1,
        "enc_attn_dropout": 0.15,
        "dec_attn_dropout": 0.15,
        "seed": 42,
    },
    "probe_transformer_selfies": {
        "task_name": "probe_transformer_selfies",
        "device": "cuda:0",
        "ckpt_path": "train_transformer_smiles/model_epoch9_step0.pt",
        "word_table_path": "all/selfies_word_table.yaml",
        "data_format": "SELFIES",
        "model": "Transformer",
        "max_len": 250,
        "left_pad": False,
        "d_model": 512,
        "n_head": 4,
        "d_enc_ffn": 1024,
        "d_dec_ffn": 1024,
        "enc_n_layer": 2,
        "dec_n_layer": 4,
        "enc_dropout": 0.2,
        "dec_dropout": 0.2,
        "enc_embed_dropout": 0.15,
        "dec_embed_dropout": 0.15,
        "enc_relu_dropout": 0.1,
        "dec_relu_dropout": 0.1,
        "enc_attn_dropout": 0.15,
        "dec_attn_dropout": 0.15,
        "seed": 42,
    },
    "probe_optformer_smiles": {
        "task_name": "probe_optformer_smiles",
        "device": "cuda:0",
        "ckpt_path": "train_optformer_smiles/model_epoch14_step0.pt",
        "word_table_path": "all/smiles_word_table.yaml",
        "data_format": "SMILES",
        "model": "Optformer",
        "mol_max_len": 250,
        "prot_max_len": 1500,
        "left_pad": False,
        "d_model": 512,
        "n_head": 4,
        "d_enc_ffn": 1024,
        "d_dec_ffn": 1024,
        "d_fuse_ffn": 1024,
        "enc_n_layer": 2,
        "dec_n_layer": 4,
        "enc_dropout": 0.2,
        "dec_dropout": 0.2,
        "enc_embed_dropout": 0.15,
        "dec_embed_dropout": 0.15,
        "enc_relu_dropout": 0.1,
        "dec_relu_dropout": 0.1,
        "enc_attn_dropout": 0.15,
        "dec_attn_dropout": 0.15,
        "seed": 42,
    },
    "probe_optformer_selfies": {
        "task_name": "probe_optformer_selfies",
        "device": "cuda:0",
        "ckpt_path": "train_optformer_selfies/model_epoch13_step0.pt",
        "word_table_path": "all/selfies_word_table.yaml",
        "data_format": "SELFIES",
        "model": "Optformer",
        "mol_max_len": 250,
        "prot_max_len": 1500,
        "left_pad": False,
        "d_model": 512,
        "n_head": 4,
        "d_enc_ffn": 1024,
        "d_dec_ffn": 1024,
        "d_fuse_ffn": 1024,
        "enc_n_layer": 2,
        "dec_n_layer": 4,
        "enc_dropout": 0.2,
        "dec_dropout": 0.2,
        "enc_embed_dropout": 0.15,
        "dec_embed_dropout": 0.15,
        "enc_relu_dropout": 0.1,
        "dec_relu_dropout": 0.1,
        "enc_attn_dropout": 0.15,
        "dec_attn_dropout": 0.15,
        "seed": 42,
    },
    "probe_frag_optformer_selfies": {
        "task_name": "probe_frag_optformer_smiles",
        "device": "cuda:0",
        "ckpt_path": "train_frag_optformer_smiles/model_epoch15_step0.pt",
        "word_table_path": "frag/smiles_word_table.yaml",
        "data_format": "FragSMILES",
        "model": "Optformer",
        "mol_max_len": 250,
        "prot_max_len": 1500,
        "left_pad": False,
        "d_model": 512,
        "n_head": 4,
        "d_enc_ffn": 1024,
        "d_dec_ffn": 1024,
        "d_fuse_ffn": 1024,
        "enc_n_layer": 2,
        "dec_n_layer": 4,
        "enc_dropout": 0.2,
        "dec_dropout": 0.2,
        "enc_embed_dropout": 0.15,
        "dec_embed_dropout": 0.15,
        "enc_relu_dropout": 0.1,
        "dec_relu_dropout": 0.1,
        "enc_attn_dropout": 0.15,
        "dec_attn_dropout": 0.15,
        "seed": 42,
    },
}


def init_model(label: str) -> Tuple[nn.Module, StrTokenizer, Optional[StrTokenizer]]:
    cfg = Cfg()
    cfg.load(CONFIGS[label])

    logger = Log("Inference", osp.join(cfg.LOG_DIR, f"{cfg.task_name}.log"))
    logger.info(f"Config:\n{repr(cfg)}")
    device = torch.device(cfg.device)

    mol_tokenizer: Optional[StrTokenizer] = None
    if cfg.data_format == "SMILES":
        mol_tokenizer = SmilesTokenizer()
    elif cfg.data_format == "SELFIES":
        mol_tokenizer = SelfiesTokenizer()
    elif cfg.data_format == "FragSMILES":
        mol_tokenizer = FragSmilesTokenizer()
    elif cfg.data_format == "FragSELFIES":
        mol_tokenizer = FragSelfiesTokenizer()
    else:
        assert False

    mol_tokenizer.load_word_table(osp.join(cfg.DATA_DIR, cfg.word_table_path))

    prot_tokenizer: Optional[StrTokenizer] = None
    if cfg.model == "Transformer":
        pass
    elif cfg.model == "Optformer":
        prot_tokenizer = ProteinTokenizer()
        prot_tokenizer, mol_tokenizer = share_vocab(prot_tokenizer, mol_tokenizer)
    else:
        assert False

    cfg.set("vocab_size", mol_tokenizer.vocab_size)
    cfg.set("pad_value", mol_tokenizer.vocab2index[mol_tokenizer.pad])

    launcher = ModelLauncher(cfg.model, cfg, logger, "inference", device)
    model = launcher.get_model()
    model.eval()

    return model, mol_tokenizer, prot_tokenizer