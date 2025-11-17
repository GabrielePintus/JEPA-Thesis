import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
import math
from rotary_embedding_torch import RotaryEmbedding

#
#   VLAD's Models
#

class MeNet6(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.activation = nn.ReLU()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=1),
            nn.GroupNorm(4, 16, eps=1e-05, affine=True),
            self.activation,

            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.GroupNorm(8, 32, eps=1e-05, affine=True),
            self.activation,

            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.GroupNorm(8, 32, eps=1e-05, affine=True),
            self.activation,

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32, eps=1e-05, affine=True),
            self.activation,

            nn.Conv2d(32, 16, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return self.net(x)

class Expander2D(nn.Module):
    """Expands a low-dimensional proprioceptive vector into a 2D feature map and applies BatchNorm1d before expansion."""
    def __init__(self, target_shape=(16, 16), out_channels=4, use_batchnorm=True):
        super().__init__()
        self.target_shape = target_shape
        self.out_channels = out_channels
        self.use_batchnorm = use_batchnorm
        # apply 1D batch norm on the channel vector before reshaping/expanding
        self.bn = nn.BatchNorm1d(out_channels) if use_batchnorm else None

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1)                 # expect shape (B, out_channels)
        if self.bn is not None:
            x = self.bn(x)                # BN over (B, C)
        out = x.view(B, self.out_channels, 1, 1)
        out = out.expand(-1, -1, *self.target_shape).contiguous()
        return out




