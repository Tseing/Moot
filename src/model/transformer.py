import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from jaxtyping import Bool, Float16, Int16
from torch import Tensor
from typing_extensions import TypeAlias

Device: TypeAlias = torch.device


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ffd: int,
        dropout: float,
        n_layers: int,
        device: Optional[Device] = None,
    ) -> None:
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_head, d_ffd, dropout, batch_first=True, device=device
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)

    def forward(
        self,
        src: Float16[Tensor, "bsz seq_len"],
        src_mask: Bool[Tensor, "seq_len seq_len"],
        src_key_padding_mask: Bool[Tensor, "seq_len seq_len"],
    ) -> Tensor:
        return self.encoder(src, src_mask, src_key_padding_mask)


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ffd: int,
        dropout: float,
        n_layers: int,
        device: Optional[Device] = None,
    ) -> None:
        super().__init__()

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_head, d_ffd, dropout, batch_first=True, device=device
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, n_layers)

    def forward(
        self,
        tgt: Float16[Tensor, "bsz seq_len"],
        memory: Tensor,
        tgt_mask: Bool[Tensor, "seq_len seq_len"],
        tgt_key_padding_mask: Bool[Tensor, "seq_len seq_len"],
        memory_key_padding_mask: Bool[Tensor, "seq_len seq_len"],
    ) -> Tensor:
        return self.decoder(tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)


class Transformer(nn.Module):
    def __init__(
        self,
        n_token: int,
        pad_idx: int,
        d_model: int,
        n_head: int,
        d_ffd: int,
        dropout: float,
        enc_n_layers: int,
        dec_n_layers: int,
        device: Optional[Device] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, device=device)
        self.embedding = nn.Embedding(n_token, d_model, device=device)
        self.encoder = Encoder(
            d_model=d_model,
            n_head=n_head,
            d_ffd=d_ffd,
            dropout=dropout,
            n_layers=enc_n_layers,
            device=device,
        )
        self.decoder = Decoder(
            d_model=d_model,
            n_head=n_head,
            d_ffd=d_ffd,
            dropout=dropout,
            n_layers=dec_n_layers,
            device=device,
        )
        self.linear = nn.Linear(d_model, n_token, device=device)

    def forward(
        self, src: Int16[Tensor, "bsz seq_len"], tgt: Int16[Tensor, "bsz seq_len"]
    ) -> Tensor:
        src_mask, src_key_padding_mask, tgt_mask, tgt_key_padding_mask = self.create_masks(
            src, tgt, self.pad_idx, self.device
        )

        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        src, tgt = self.pos_encoder(src), self.pos_encoder(tgt)

        memory = self.encoder(src, src_mask, src_key_padding_mask)
        out = self.decoder(tgt, memory, tgt_mask, tgt_key_padding_mask, src_key_padding_mask)

        return out

    @staticmethod
    def create_masks(
        src: Int16[Tensor, "bsz seq_len"],
        tgt: Int16[Tensor, "bsz seq_len"],
        pad_idx: int,
        device: Optional[Device] = None,
    ) -> Tuple[
        Bool[Tensor, "src_seq_len src_seq_len"],
        Bool[Tensor, "src_seq_len src_seq_len"],
        Bool[Tensor, "tgt_seq_len tgt_seq_len"],
        Bool[Tensor, "tgt_seq_len tgt_seq_len"],
    ]:
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

        src_padding_mask = (src == pad_idx).to(device)
        tgt_padding_mask = (tgt == pad_idx).to(device)

        return src_mask, src_padding_mask, tgt_mask, tgt_padding_mask


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        batch_first: bool = True,
        device: Optional[Device] = None,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        if device is None:
            device = torch.device(torch._C._get_default_device())

        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model, device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if self.batch_first:
            pe = pe.transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(
        self, x: Union[Float16[Tensor, "bsz seq_len"], Float16[Tensor, "seq_len bsz"]]
    ) -> Tensor:
        if self.batch_first:
            x = x + self.pe[:, : x.size(0), :]
        else:
            x = x + self.pe[: x.size(0), :, :]
        return self.dropout(x)
