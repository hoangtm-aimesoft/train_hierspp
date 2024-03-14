import torch
import torch.nn as nn
from .modules import Flip, DiTConVBlock


class ResidualCouplingLayer_Transformer_simple(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 p_dropout=0.1,
                 mean_only=False):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)

        self.enc_block = torch.nn.ModuleList([
            DiTConVBlock(hidden_channels, 2, mlp_ratio=4.0, kernel=5, p_dropout=p_dropout) for _ in range(n_layers)
        ])

        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)

        self.initialize_weights()

        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.enc_block:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask

        # h = self.enc(h, x_mask, g=g)
        h = h.transpose(1, 2)
        x_mask = x_mask.transpose(1, 2)

        for blk in self.enc_block:
            h = blk(h, g, x_mask)

        x_mask = x_mask.transpose(1, 2)
        h = h.transpose(1, 2)

        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class ResidualCouplingBlock_Transformer(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers=3,
                 n_flows=4,
                 gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.cond_block = torch.nn.Sequential(torch.nn.Linear(gin_channels, 4 * hidden_channels),
                                              nn.SiLU(), torch.nn.Linear(4 * hidden_channels, hidden_channels))

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer_Transformer_simple(channels, hidden_channels, kernel_size, dilation_rate,
                                                         n_layers, mean_only=True))
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):

        g = self.cond_block(g.squeeze(2))

        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x
