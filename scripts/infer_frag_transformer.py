import sys

sys.path.append("..")
import os.path as osp
from typing import Optional

import torch
from rdkit import RDLogger
from torch.utils.data import DataLoader

from src.dataset import FragPairDataset
from src.inferencer import Inferencer
from src.launcher import ModelLauncher
from src.tokenizer import FragSelfiesTokenizer, FragSmilesTokenizer, StrTokenizer
from src.utils import Cfg, Log

if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.*")

    cfg = Cfg()
    cfg.parse()

    logger = Log("train", osp.join(cfg.LOG_DIR, f"{cfg.task_name}.log"))
    logger.info(f"Config:\n{repr(cfg)}")

    mol_tokenizer: Optional[StrTokenizer] = None
    if cfg.data_format == "SMILES":
        mol_tokenizer = FragSmilesTokenizer()
    elif cfg.data_format == "SELFIES":
        mol_tokenizer = FragSelfiesTokenizer()
    else:
        assert False, (
            f"Config 'data_format' should be 'SMILES' or 'SELFIES', " f"but got '{cfg.tokenizer}'."
        )
    mol_tokenizer.load_word_table(osp.join(cfg.DATA_DIR, cfg.word_table_path))
    device = torch.device(cfg.device)

    dataset = FragPairDataset(
        osp.join(cfg.DATA_DIR, cfg.test_data_path),
        cfg.data_cols,
        tokenizer=mol_tokenizer,
        max_len=None,
        left_pad=cfg.left_pad,
        pad_batch=True
    )

    print(f"Tokenizer vocab size: {mol_tokenizer.vocab_size}.")
    cfg.set("vocab_size", mol_tokenizer.vocab_size)
    cfg.set("pad_value", mol_tokenizer.vocab2index[mol_tokenizer.pad])
    pad_fn = dataset.pad_batch_fn if dataset.pad_batch else None
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=20, collate_fn=pad_fn
    )

    launcher = ModelLauncher("Transformer", cfg, logger, "inference", device)
    model = launcher.get_model()

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
