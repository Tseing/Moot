from torch import nn
import torch

class CPIClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, seq_len: int, d_model: int, d_hidden: int):
        super().__init__()
        self.encoder = encoder
        self.fuse_layer = nn.Linear(seq_len, 1)
        self.layer1 = nn.Linear(d_model, d_hidden)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(d_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, *inp):
        with torch.no_grad():
            feat, mask = self.encoder.enc_forward(*inp)

        feat = feat.transpose(0, 1)  # bsz, seq_len, d_model
        feat[mask, :] = 0
        feat = feat.transpose(1, 2)

        x = self.fuse_layer(feat)
        x = x.squeeze()
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))

        return x
