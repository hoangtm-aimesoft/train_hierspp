from torch.nn.utils import weight_norm, remove_weight_norm
from commons.commons import init_weights
import torch.nn as nn
import torch.nn.functional as F

LRELU_SLOPE = 0.1


class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor):
        super().__init__()
        self.factor = factor
        self.residual_dense = weight_norm(nn.Conv1d(input_size, hidden_size, 1))
        self.conv = nn.ModuleList([
            weight_norm(nn.Conv1d(input_size, hidden_size, 3, dilation=1, padding=1)),
            weight_norm(nn.Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2)),
            weight_norm(nn.Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4)),
        ])
        self.conv.apply(init_weights)

    def forward(self, x):
        size = x.shape[-1] // self.factor

        residual = self.residual_dense(x)
        residual = F.interpolate(residual, size=size)

        x = F.interpolate(x, size=size)
        for layer in self.conv:
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = layer(x)

        return x + residual

    def remove_weight_norm(self):
        for l in self.conv:
            remove_weight_norm(l)
