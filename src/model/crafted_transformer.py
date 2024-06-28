import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ..typing import Device
from .modules import MultiheadAttention, SinusoidalPositionalEmbedding


class Transformer(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_head: int,
        enc_n_layer: int,
        dec_n_layer: int,
        enc_d_ffn: int,
        dec_d_ffn: int,
        enc_dropout: float,
        dec_dropout: float,
        enc_embed_dropout: float,
        dec_embed_dropout: float,
        enc_relu_dropout: float,
        dec_relu_dropout: float,
        enc_attn_dropout: float,
        dec_attn_dropout: float,
        vocab_size: int,
        padding_idx: int,
        max_len: int = 512,
        device: Optional[Device] = None,
        seed=0,
    ):
        super().__init__()
        self._is_generation_fast = False

        if device is None:
            device = torch.device(torch._C._get_default_device())

        enc_embedding = Embedding(vocab_size, d_model, padding_idx)
        dec_embedding = enc_embedding

        encoder = TransformerEncoder(
            enc_embedding,
            max_len=max_len,
            n_head=n_head,
            n_layer=enc_n_layer,
            d_ffn=enc_d_ffn,
            dropout=enc_dropout,
            embed_dropout=enc_embed_dropout,
            relu_dropout=enc_relu_dropout,
            attn_dropout=enc_attn_dropout,
            device=device,
            seed=seed,
        )
        decoder = TransformerDecoder(
            dec_embedding,
            max_len=max_len,
            n_head=n_head,
            n_layer=dec_n_layer,
            d_ffn=dec_d_ffn,
            dropout=dec_dropout,
            embed_dropout=dec_embed_dropout,
            relu_dropout=dec_relu_dropout,
            attn_dropout=dec_attn_dropout,
            device=device,
            seed=seed,
        )

        self.encoder = encoder
        self.decoder = decoder
        self.seed = seed

    def make_generation_fast_(self, **kwargs):
        """Optimize model for faster generation."""
        if self._is_generation_fast:
            return  # only apply once
        self._is_generation_fast = True

        # remove weight norm from all modules in the network
        def apply_remove_weight_norm(module):
            try:
                nn.utils.remove_weight_norm(module)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(apply_remove_weight_norm)

        def apply_make_generation_fast_(module):
            if module != self and hasattr(module, "make_generation_fast_"):
                module.make_generation_fast_(**kwargs)

        self.apply(apply_make_generation_fast_)

        def train(mode):
            if mode:
                raise RuntimeError("cannot train after make_generation_fast")

        # this model should no longer be used for training
        self.eval()
        self.train = train

    def forward(
        self, src_tokens: Int[Tensor, "bsz seq_len"], prev_output_tokens: Int[Tensor, "bsz seq_len"]
    ):

        encoder_out, padding_mask = self.encoder(src_tokens)
        decoder_out = self.decoder(prev_output_tokens, encoder_out, padding_mask)
        return decoder_out


class TransformerEncoder(nn.Module):
    """Transformer encoder."""

    def __init__(
        self,
        embedding: nn.Embedding,
        max_len: int,
        n_head: int,
        n_layer: int,
        d_ffn: int,
        dropout: float,
        embed_dropout: float,
        relu_dropout: float,
        attn_dropout: float,
        device=Device,
        left_pad: bool = True,
        seed=0,
    ) -> None:
        super().__init__()

        d_model = embedding.embedding_dim
        padding_idx = embedding.padding_idx
        assert padding_idx is not None, f"'padding_idx' of {embedding} is None."
        self.padding_idx = padding_idx

        self.embedding = embedding
        self.embed_scale = math.sqrt(d_model)
        self.embed_positions = PositionalEmbedding(
            max_len,
            d_model,
            padding_idx,
            left_pad=left_pad,
        )
        self.embed_dropout = embed_dropout

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerEncoderLayer(
                    d_model, n_head, d_ffn, dropout, attn_dropout, relu_dropout, device, seed
                )
                for _ in range(n_layer)
            ]
        )

    def forward(
        self,
        encoder_inp: Int[Tensor, "bsz seq_len"],
    ) -> Tuple[Float[Tensor, "seq_len bsz d_model"], Bool[Tensor, "bsz seq_len"]]:
        x = self.embed_scale * self.embedding(encoder_inp)
        if self.embed_positions is not None:
            x += self.embed_positions(encoder_inp)
        x = F.dropout(x, p=self.embed_dropout, training=self.training)

        # B:batch size ; T: seq length ; C: embedding dim 512
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = encoder_inp.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            _encoder_padding_mask = None
        else:
            _encoder_padding_mask = encoder_padding_mask
        # encoder layers
        for layer in self.layers:
            x = layer(x, _encoder_padding_mask)

        # x.shape == T x B x C, encoder_padding_mask.shape == B x T
        return x, encoder_padding_mask

    def reorder_encoder_out(self, encoder_out, encoder_padding_mask, new_order):
        if encoder_out is not None:
            encoder_out = encoder_out.index_select(1, new_order)
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.int()
            encoder_padding_mask = encoder_padding_mask.index_select(0, new_order)
            encoder_padding_mask = encoder_padding_mask.bool()
        return encoder_out, encoder_padding_mask


class IncrementalDecoder(nn.Module):
    """Base class for incremental decoders."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        prev_output_tokens: Tensor,
        encoder_out: Tensor,
        encoder_padding_mask: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ):
        raise NotImplementedError

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder incremental state.

        This should be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """

        def apply_reorder_incremental_state(module):
            if module != self and hasattr(module, "reorder_incremental_state"):
                module.reorder_incremental_state(
                    incremental_state,
                    new_order,
                )

        self.apply(apply_reorder_incremental_state)

    def set_beam_size(self, beam_size):
        """Sets the beam size in the decoder and all children."""
        if getattr(self, "_beam_size", -1) != beam_size:

            def apply_set_beam_size(module):
                if module != self and hasattr(module, "set_beam_size"):
                    module.set_beam_size(beam_size)

            self.apply(apply_set_beam_size)
            self._beam_size = beam_size


class TransformerDecoder(IncrementalDecoder):
    """Transformer decoder."""

    def __init__(
        self,
        embedding: nn.Embedding,
        max_len: int,
        n_head: int,
        n_layer: int,
        d_ffn: int,
        dropout: float,
        embed_dropout: float,
        relu_dropout: float,
        attn_dropout: float,
        device: Device,
        left_pad=False,
        no_token_positional_embeddings: bool = False,
        seed=0,
    ) -> None:
        super().__init__()

        d_model = embedding.embedding_dim
        padding_idx = embedding.padding_idx
        assert padding_idx is not None, f"'padding_idx' of {embedding} is None."

        self.embedding = embedding
        self.embed_scale = math.sqrt(d_model)
        self.embed_positions = (
            PositionalEmbedding(
                max_len,
                d_model,
                padding_idx,
                left_pad=left_pad,
            )
            if not no_token_positional_embeddings
            else None
        )
        self.embed_dropout = embed_dropout

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(
                    d_model, n_head, d_ffn, dropout, attn_dropout, relu_dropout, device, seed
                )
                for _ in range(n_layer)
            ]
        )

        self.embed_out = self.embedding.weight

    def forward(
        self,
        prev_output_tokens: Int[Tensor, "bsz seq_len"],
        encoder_out: Float[Tensor, "seq_len bsz d_model"],
        encoder_padding_mask: Bool[Tensor, "bsz seq_len"],
        incremental_state: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ) -> Tuple[Float[Tensor, "bsz seq_len d_model"], Optional[Any]]:
        positions = (
            self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embedding(prev_output_tokens)
        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.embed_dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out,
                encoder_padding_mask if encoder_padding_mask.any() else None,
                incremental_state,
            )

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        x = F.linear(x, self.embed_out)

        return x, attn


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ffn: int,
        dropout: float,
        attn_dropout: float,
        relu_dropout: float,
        device: Device,
        seed=0,
    ):
        super().__init__()

        self.self_attn = MultiheadAttention(
            d_model, n_head, dropout=attn_dropout, device=device, seed=seed
        )
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.fc1 = Linear(d_model, d_ffn)
        self.fc2 = Linear(d_ffn, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, encoder_padding_mask: Bool[Tensor, "bsz seq_len"]):
        residual = x

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask_future_timesteps=False,
            key_padding_mask=encoder_padding_mask,
            incremental_state=None,
            need_weights=False,
            static_kv=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.norm1(x)

        residual = x
        x = F.threshold(self.fc1(x), 0.0, 0.0)
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.norm2(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ffn: int,
        dropout: float,
        attn_dropout: float,
        relu_dropout: float,
        device: Device,
        seed=0,
    ):
        super().__init__()

        self.self_attn = MultiheadAttention(
            d_model, n_head, dropout=attn_dropout, device=device, seed=seed
        )
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.self_attn_layer_norm = nn.LayerNorm(d_model)

        self.encoder_attn = MultiheadAttention(
            d_model, n_head, dropout=attn_dropout, device=device, seed=seed
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)

        self.fc1 = Linear(d_model, d_ffn)
        self.fc2 = Linear(d_ffn, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        self.need_attn = True

    def forward(
        self,
        x: Float[Tensor, "seq_len bsz d_model"],
        encoder_out: Float[Tensor, "seq_len bsz d_model"],
        encoder_padding_mask: Bool[Tensor, "bsz seq_len"],
        incremental_state: Optional[Dict[str, Dict[str, Tensor]]],
    ):
        residual = x

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask_future_timesteps=True,
            key_padding_mask=None,
            incremental_state=incremental_state,
            need_weights=False,
            static_kv=False,
        )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        attn = None
        if self.encoder_attn is not None:
            residual = x

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                mask_future_timesteps=False,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x

            x = self.encoder_attn_layer_norm(x)

        residual = x
        x = F.threshold(self.fc1(x), 0.0, 0.0)
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.layer_norm(x)
        return x, attn

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


class OptFormerEncoder(nn.Module):
    def __init__(
        self,
        embedding: nn.Embedding,
        max_len: int,
        n_head: int,
        n_layer: int,
        d_ffn: int,
        dropout: float,
        embed_dropout: float,
        relu_dropout: float,
        attn_dropout: float,
        device=Device,
        left_pad: bool = True,
        seed=0,
    ) -> None:
        super().__init__()

        d_model = embedding.embedding_dim
        padding_idx = embedding.padding_idx
        assert padding_idx is not None, f"'padding_idx' of {embedding} is None."
        self.padding_idx = padding_idx

        self.embedding = embedding
        self.embed_scale = math.sqrt(d_model)
        self.embed_positions = PositionalEmbedding(
            max_len,
            d_model,
            padding_idx,
            left_pad=left_pad,
        )
        self.embed_dropout = embed_dropout

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                OptFormerEncoderLayer(
                    d_model, n_head, d_ffn, dropout, attn_dropout, relu_dropout, device, seed
                )
                for _ in range(n_layer)
            ]
        )

    def compute_padding_mask(self, inp: Tensor) -> Optional[Tensor]:
        padding_mask = inp.eq(self.padding_idx)
        if not padding_mask.any():
            _padding_mask = None
        else:
            _padding_mask = padding_mask

        return _padding_mask

    def forward(
        self,
        encoder_inp_a: Int[Tensor, "bsz seq_len_a"],
        encoder_inp_b: Int[Tensor, "bsz seq_len_b"],
    ) -> Tuple[
        Float[Tensor, "seq_len_a bsz d_model"],
        Float[Tensor, "seq_len_b bsz d_model"],
        Optional[Bool[Tensor, "bsz seq_len_a"]],
        Optional[Bool[Tensor, "bsz seq_len_b"]]
    ]:
        x_a = self.embed_scale * self.embedding(encoder_inp_a)
        if self.embed_positions is not None:
            x_a += self.embed_positions(encoder_inp_a)

        x_b = encoder_inp_b

        x_a = F.dropout(x_a, p=self.embed_dropout, training=self.training)
        x_b = F.dropout(x_b, p=self.embed_dropout, training=self.training)

        # B:batch size ; T: seq length ; C: embedding dim 512
        # B x T x C -> T x B x C
        x_a = x_a.transpose(0, 1)
        x_b = x_b.transpose(0, 1)

        # compute padding mask
        x_a_padding_mask = self.compute_padding_mask(x_a)
        x_b_padding_mask = self.compute_padding_mask(x_b)

        # encoder layers
        for layer in self.layers:
            x_a, x_b = layer(x_a, x_a_padding_mask, x_b, x_b_padding_mask)

        # x.shape == T x B x C, encoder_padding_mask.shape == B x T
        return x_a, x_b, x_a_padding_mask, x_b_padding_mask

    def reorder_encoder_out(self, encoder_out, encoder_padding_mask, new_order):
        if encoder_out is not None:
            encoder_out = encoder_out.index_select(1, new_order)
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.int()
            encoder_padding_mask = encoder_padding_mask.index_select(0, new_order)
            encoder_padding_mask = encoder_padding_mask.bool()
        return encoder_out, encoder_padding_mask


class OptFormerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ffn: int,
        dropout: float,
        attn_dropout: float,
        relu_dropout: float,
        device: Device,
        seed=0,
    ):
        super().__init__()
        self.dropout = dropout
        self.relu_dropout = relu_dropout

        self.self_attn_a = MultiheadAttention(
            d_model, n_head, dropout=attn_dropout, device=device, seed=seed
        )
        self.self_attn_b = MultiheadAttention(
            d_model, n_head, dropout=attn_dropout, device=device, seed=seed
        )

        self.norm_a1 = nn.LayerNorm(d_model)
        self.norm_b1 = nn.LayerNorm(d_model)

        self.cross_attn = CrossAttnLayer(
            d_model, n_head, attn_dropout=attn_dropout, device=device, seed=seed
        )

        self.norm_a2 = nn.LayerNorm(d_model)
        self.norm_b2 = nn.LayerNorm(d_model)

        self.ffn_a1 = Linear(d_model, d_ffn)
        self.ffn_a2 = Linear(d_ffn, d_model)
        self.ffn_b1 = Linear(d_model, d_ffn)
        self.ffn_b2 = Linear(d_ffn, d_model)

        self.norm_a3 = nn.LayerNorm(d_model)
        self.norm_b3 = nn.LayerNorm(d_model)

    def __forward_self_attn(
        self,
        x: Tensor,
        key_padding_mask: Bool[Tensor, "bsz seq_len"],
        self_attn_layer: MultiheadAttention,
        norm_layer: nn.LayerNorm,
    ) -> Float[Tensor, "bsz seq_len d"]:
        residual = x
        x, _ = self_attn_layer(
            query=x,
            key=x,
            value=x,
            mask_future_timesteps=False,
            key_padding_mask=key_padding_mask,
            incremental_state=None,
            need_weights=False,
            static_kv=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = norm_layer(x)
        return x

    def __forward_post_cross_attn(
        self,
        x: Float[Tensor, "bsz seq_len d"],
        residual: Float[Tensor, "bsz seq_len d"],
        norm_layer_pre_ffn: nn.LayerNorm,
        ffn_1: nn.Linear,
        ffn_2: nn.Linear,
        norm_layer_post_ffn: nn.LayerNorm,
    ) -> Float[Tensor, "bsz seq_len d"]:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = norm_layer_pre_ffn(x)

        residual = x
        x = F.threshold(ffn_1(x), 0.0, 0.0)
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = ffn_2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = norm_layer_post_ffn(x)
        return x

    def forward(
        self,
        x_a: Tensor,
        x_a_padding_mask: Bool[Tensor, "bsz seq_len_a"],
        x_b: Tensor,
        x_b_padding_mask: Bool[Tensor, "bsz seq_len_b"],
    ) -> Tuple[Float[Tensor, "bsz seq_len_a d"], Float[Tensor, "bsz seq_len_b d"]]:
        x_a = self.__forward_self_attn(
            x_a,
            x_a_padding_mask,
            self.self_attn_a,
            self.norm_a1,
        )
        x_b = self.__forward_self_attn(
            x_b,
            x_b_padding_mask,
            self.self_attn_b,
            self.norm_b1,
        )
        residual_a, residual_b = x_a, x_b
        x_a, x_b = self.cross_attn(x_a, x_a_padding_mask, x_b, x_b_padding_mask)
        x_a = self.__forward_post_cross_attn(
            x_a, residual_a, self.norm_a2, self.ffn_a1, self.ffn_a2, self.norm_a3
        )
        x_b = self.__forward_post_cross_attn(
            x_b, residual_b, self.norm_b2, self.ffn_b1, self.ffn_b2, self.norm_b3
        )

        return x_a, x_b


class CrossAttnLayer(nn.Module):
    def __init__(
        self, d_model: int, n_head: int, attn_dropout: float, device: Device, seed: int = 0
    ) -> None:
        super().__init__()

        self.cross_attn_a = MultiheadAttention(
            d_model, n_head, dropout=attn_dropout, device=device, seed=seed
        )
        self.cross_attn_b = MultiheadAttention(
            d_model, n_head, dropout=attn_dropout, device=device, seed=seed
        )

    def forward(
        self,
        x_a: Tensor,
        x_a_padding_mask: Bool[Tensor, "bsz seq_len1"],
        x_b: Tensor,
        x_b_padding_mask: Bool[Tensor, "bsz seq_len2"],
    ) -> Tuple[Float[Tensor, "seq_len1 d"], Float[Tensor, "seq_len2 d"]]:
        attn_a, _ = self.cross_attn1(
            query=x_a,
            key=x_b,
            value=x_b,
            mask_future_timesteps=False,
            key_padding_mask=x_b_padding_mask,
            incremental_state=None,
            need_weights=False,
            static_kv=False,
        )

        attn_b, _ = self.cross_attn2(
            query=x_b,
            key=x_a,
            value=x_a,
            mask_future_timesteps=False,
            key_padding_mask=x_a_padding_mask,
            incremental_state=None,
            need_weights=False,
            static_kv=False,
        )

        return attn_a, attn_b


def Embedding(num_embeddings: int, embedding_dim: int, padding_idx: int) -> nn.Embedding:
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features: int, out_features: int, bias=True) -> nn.Linear:
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.0)
    return m


def PositionalEmbedding(
    num_embeddings: int, embedding_dim: int, padding_idx: int, left_pad: bool
) -> SinusoidalPositionalEmbedding:
    m = SinusoidalPositionalEmbedding(
        embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1
    )
    return m
