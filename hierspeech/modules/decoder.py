from attentions.attentions import Encoder
import torch.nn as nn


class MelDecoder(nn.Module):
    def __init__(self,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 mel_size=20,
                 gin_channels=0):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_pre = nn.Conv1d(hidden_channels, hidden_channels, 3, 1, padding=1)

        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

        self.proj = nn.Conv1d(hidden_channels, mel_size, 1, bias=False)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)

    def forward(self, x, x_mask, g=None):

        x = self.conv_pre(x * x_mask)
        if g is not None:
            x = x + self.cond(g)

        x = self.encoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask

        return x
