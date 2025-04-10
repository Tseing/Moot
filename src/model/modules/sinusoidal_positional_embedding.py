import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx, left_pad, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.register_buffer(
            "positions_buffer", torch.arange(padding_idx + 1, init_size + padding_idx + 1).int()
        )  # JIT compliance

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: int):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        emb[padding_idx] = torch.zeros(emb.shape[1])  # emb[padding_idx, :] = 0
        return emb

    def forward(
        self, input: Tensor, incremental_state: Optional[Dict[str, Dict[str, Tensor]]] = None
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        # recompute/expand embeddings if needed
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.type_as(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            return self.weights[self.padding_idx + seq_len, :].expand(bsz, 1, -1)

        mask = input.ne(self.padding_idx)
        positions = self.positions_buffer

        positions = positions[: input.size(1)]
        positions = positions.expand_as(input)

        if self.left_pad:
            positions = positions - mask.size(1) + mask.float().sum(dim=1).unsqueeze(1).int()

        positions = input.clone().masked_scatter_(mask, torch.masked_select(positions, mask))
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()
