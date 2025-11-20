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
        # self.activation = nn.GELU()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=1), # (16, 60, 60)
            nn.GroupNorm(4, 16, eps=1e-05, affine=True),
            self.activation,

            nn.Conv2d(16, 32, kernel_size=5, stride=2), # (32, 28, 28)
            nn.GroupNorm(8, 32, eps=1e-05, affine=True),
            self.activation,

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), # (32, 28, 28)
            nn.GroupNorm(8, 32, eps=1e-05, affine=True),
            self.activation,

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0), # (64, 26, 26)
        )

    def forward(self, x):
        return self.net(x)


class CNNEncoder(nn.Module):
    def __init__(self, input_channels=3, base_channels=32):
        super().__init__()
        self.activation = nn.GELU()  # Better than ReLU for smooth gradients
        
        # Stem: preserve more spatial information early
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, base_channels),
            self.activation,
        )
        
        # Residual blocks with gradual downsampling
        self.block1 = self._make_residual_block(base_channels, base_channels*2, stride=2)
        # (B, 64, 32, 32)
        
        self.block2 = self._make_residual_block(base_channels*2, base_channels*3, stride=2)
        # (B, 128, 16, 16)
        
        self.block3 = self._make_residual_block(base_channels*3, base_channels*3, stride=2)
        # (B, 128, 8, 8)
        
        # Don't pool too aggressively - keep 8×8 spatial structure
        
    def _make_residual_block(self, in_channels, out_channels, stride):
        """Residual block for better gradient flow"""
        return nn.Sequential(
            # Main path
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1),
            nn.GroupNorm(min(32, out_channels//4), out_channels),
            self.activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1),
            nn.GroupNorm(min(32, out_channels//4), out_channels),
        )
    
    def forward(self, x):
        x = self.stem(x)
        
        # Residual connections
        x1 = self.block1(x)
        x2 = self.block2(x1) 
        x3 = self.block3(x2)
        
        return x3  # (B, 128, 8, 8)

class MLPHead(nn.Module):
    def __init__(self, spatial_features=96*8*8, emb_dim=256):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(spatial_features),
            nn.Dropout(0.1),
            nn.Linear(spatial_features, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
    
    def forward(self, x):
        h = self.head(x)
        # Unit norm embeddings make distances more interpretable
        return F.normalize(h, p=2, dim=-1)


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




