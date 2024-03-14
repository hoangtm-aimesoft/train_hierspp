from torch.nn.utils import weight_norm
from ..alias_free_torch import Activation1d
from .amp import AMPBlock1
from .modules import WN
from activations.activations import SnakeBeta
import torch
import torch.nn as nn


class PosteriorAudioEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.down_pre = nn.Conv1d(1, 16, 7, 1, padding=3)
        self.resblocks = nn.ModuleList()
        downsample_rates = [8, 5, 4, 2]
        downsample_kernel_sizes = [17, 10, 8, 4]
        ch = [16, 32, 64, 128, 192]

        resblock = AMPBlock1
        resblock_kernel_sizes = [3, 7, 11]
        resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        self.num_kernels = 3
        self.downs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(downsample_rates, downsample_kernel_sizes)):
            self.downs.append(weight_norm(
                nn.Conv1d(ch[i], ch[i + 1], k, u, padding=(k - 1) // 2)))
        for i in range(4):
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch[i + 1], k, d, activation="snakebeta"))

        activation_post = SnakeBeta(ch[i + 1], alpha_logscale=True)
        self.activation_post = Activation1d(activation=activation_post)

        self.conv_post = nn.Conv1d(ch[i + 1], hidden_channels, 7, 1, padding=3)

        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels * 2, out_channels * 2, 1)

    def forward(self, x, x_audio, x_mask, g=None):

        x_audio = self.down_pre(x_audio)

        for i in range(4):

            x_audio = self.downs[i](x_audio)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x_audio)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x_audio)
            x_audio = xs / self.num_kernels

        x_audio = self.activation_post(x_audio)
        x_audio = self.conv_post(x_audio)

        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)

        x_audio = x_audio * x_mask

        x = torch.cat([x, x_audio], dim=1)

        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs


class PosteriorSFEncoder(nn.Module):
    def __init__(self,
                 src_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre_source = nn.Conv1d(src_channels, hidden_channels, 1)
        self.pre_filter = nn.Conv1d(1, hidden_channels, kernel_size=9, stride=4, padding=4)
        self.source_enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers // 2,
                             gin_channels=gin_channels)
        self.filter_enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers // 2,
                             gin_channels=gin_channels)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers // 2, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x_src, x_ftr, x_mask, g=None):
        x_src = self.pre_source(x_src) * x_mask
        x_ftr = self.pre_filter(x_ftr) * x_mask
        x_src = self.source_enc(x_src, x_mask, g=g)
        x_ftr = self.filter_enc(x_ftr, x_mask, g=g)
        x = self.enc(x_src + x_ftr, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs
