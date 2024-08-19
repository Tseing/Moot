import os.path as osp

import torch

from .model.optformer import OptFormer, Transformer
from .typing import Device
from .utils import Cfg, Log, count_parameters, initialize_weights


class ModelLauncher:
    def __init__(self, cfg: Cfg, logger: Log, stage: str, device: Device) -> None:
        self.cfg = cfg
        self.logger = logger
        self.stage = stage
        self.device = device
        self.__init_model()
        self.__launch_model()

        self.logger.info(f"The model has {count_parameters(self.model):,} trainable parameters")
        self.logger.info(self.model)

    def __init_model(self) -> None:
        self.model = Transformer(
            d_model=self.cfg.d_model,
            n_head=self.cfg.n_head,
            enc_n_layer=self.cfg.enc_n_layer,
            dec_n_layer=self.cfg.dec_n_layer,
            enc_d_ffn=self.cfg.d_enc_ffn,
            dec_d_ffn=self.cfg.d_dec_ffn,
            enc_dropout=self.cfg.enc_dropout,
            dec_dropout=self.cfg.dec_dropout,
            enc_embed_dropout=self.cfg.enc_embed_dropout,
            dec_embed_dropout=self.cfg.dec_embed_dropout,
            enc_relu_dropout=self.cfg.enc_relu_dropout,
            dec_relu_dropout=self.cfg.dec_relu_dropout,
            enc_attn_dropout=self.cfg.enc_attn_dropout,
            dec_attn_dropout=self.cfg.dec_attn_dropout,
            vocab_size=self.cfg.vocab_size,
            padding_idx=self.cfg.pad_value,
            left_pad=self.cfg.left_pad,
            max_len=self.cfg.max_len,
            device=self.device,
            seed=self.cfg.seed,
        ).to(self.device)

    def __launch_model(self) -> None:
        if self.stage == "train":
            self.model.apply(initialize_weights)
        elif self.stage == "finetune":
            self.__load_state_dict()
        elif self.stage == "inference":
            self.__load_state_dict()
        else:
            assert False, f"'{self.stage}' is not a valid attribute of stage."

    def __load_state_dict(self) -> None:
        ckpt = torch.load(
            osp.join(self.cfg.CKPT_DIR, self.cfg.ckpt_path),
            map_location=self.device,
        )
        self.model.load_state_dict(ckpt["model"])
        self.logger.info(f"Loaded model from '{osp.join(self.cfg.CKPT_DIR, self.cfg.ckpt_path)}'.")

    def get_model(self):
        return self.model
