import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm
from ..modules import modules, styleencoder
from commons import commons


class PitchPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        resblock_kernel_sizes = [3, 5, 7]
        upsample_rates = [2, 2]
        initial_channel = 1024
        upsample_initial_channel = 256
        upsample_kernel_sizes = [4, 4]
        resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        resblock = modules.ResBlock1

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(commons.init_weights)

        self.cond = Conv1d(256, upsample_initial_channel, 1)

    def forward(self, x, g):
        x = self.conv_pre(x) + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        ## Predictor
        x = self.conv_post(x)

        return x


class TTV_Pitch_Predictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pp = PitchPredictor()
        self.emb_g = styleencoder.StyleEncoder(in_dim=80, hidden_dim=256, out_dim=256)

    @torch.no_grad()
    def infer_noise_control(self, w2v, src_mel, src_length, denoise_ratio=0):
        src_mask = torch.unsqueeze(commons.sequence_mask(src_length, src_mel.size(2)), 1).to(src_mel.dtype)
        g = self.emb_g(src_mel, src_mask).unsqueeze(-1)
        g_org, g_denoise = g[:1, :, :], g[1:, :, :]
        g = (1 - denoise_ratio) * g_org + denoise_ratio * g_denoise
        pitch = self.pp(w2v, g)
        return pitch
