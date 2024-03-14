from torch.nn.utils import weight_norm
from .amp import AMPBlock1
from activations.activations import SnakeBeta
from hierspeech.alias_free_torch import Activation1d
from commons.commons import init_weights
import torch.nn as nn


class SourceNetwork(nn.Module):
    def __init__(self, upsample_initial_channel=256):
        super().__init__()

        resblock_kernel_sizes = [3, 5, 7]
        upsample_rates = [2, 2]
        initial_channel = 192
        upsample_initial_channel = upsample_initial_channel
        upsample_kernel_sizes = [4, 4]
        resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = weight_norm(nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        resblock = AMPBlock1

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
                                   k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, activation="snakebeta"))

        activation_post = SnakeBeta(ch, alpha_logscale=True)
        self.activation_post = Activation1d(activation=activation_post)

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)

        self.cond = nn.Conv1d(256, upsample_initial_channel, 1)

        self.ups.apply(init_weights)

    def forward(self, x, g):

        x = self.conv_pre(x) + self.cond(g)

        for i in range(self.num_upsamples):

            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        ## Predictor
        x_ = self.conv_post(x)
        # return p_h, f0
        return x, x_
