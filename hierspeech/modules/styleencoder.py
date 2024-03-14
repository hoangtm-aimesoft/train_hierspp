from attentions.attentions import MultiHeadAttention
from activations.activations import Mish
from .conv_modules import Conv1dGLU
from torch import nn
import torch


class StyleEncoder(torch.nn.Module):
    def __init__(self, in_dim=513, hidden_dim=128, out_dim=256):

        super().__init__()

        self.in_dim = in_dim  # Linear 513 wav2vec 2.0 1024
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.kernel_size = 5
        self.n_head = 2
        self.dropout = 0.1

        self.spectral = nn.Sequential(
            nn.Conv1d(self.in_dim, self.hidden_dim, 1),
            Mish(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
            Mish(),
            nn.Dropout(self.dropout)
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = MultiHeadAttention(self.hidden_dim, self.hidden_dim, self.n_head,
                                           p_dropout=self.dropout, proximal_bias=False, proximal_init=True)
        self.atten_drop = nn.Dropout(self.dropout)
        self.fc = nn.Conv1d(self.hidden_dim, self.out_dim, 1)

    def forward(self, x, mask=None):

        # spectral
        x = self.spectral(x) * mask
        # temporal
        x = self.temporal(x) * mask

        # self-attention
        attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
        y = self.slf_attn(x, x, attn_mask=attn_mask)
        x = x + self.atten_drop(y)

        # fc
        x = self.fc(x)

        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)

        return w

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=2)
        else:
            len_ = mask.sum(dim=2)
            x = x.sum(dim=2)

            out = torch.div(x, len_)
        return out
