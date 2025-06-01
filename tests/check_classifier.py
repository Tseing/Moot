import sys

import numpy as np
import torch
from torch import nn

sys.path.append("..")
from scripts.setup import init_model


class TransformerCPI(nn.Module):
    def __init__(self, encoder: nn.Module, seq_len: int, d_model: int, d_hidden: int):
        super().__init__()
        self.encoder = encoder
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False

        self.fuse_layer = nn.Linear(seq_len, 1)
        self.layer1 = nn.Linear(d_model, d_hidden)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(d_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mol):
        feat, mask = self.encoder.enc_forward(mol)
        feat = feat.transpose(0, 1)  # bsz, seq_len, d_model
        feat[mask, :] = 0
        feat = feat.transpose(1, 2)

        x = self.fuse_layer(feat)
        x = x.squeeze()
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))

        return x


if __name__ == "__main__":
    model, _, _ = init_model("probe_transformer_selfies")
    cpi = TransformerCPI(model, 250, 512, 1024)
    x = cpi(torch.Tensor(np.random.randint(0, 15, (2, 250))).int())
    print(x.shape)