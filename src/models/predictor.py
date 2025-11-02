import torch
import torch.nn as nn
from flash_attn import flash_attn_func


class ConvPredictor(nn.Module):
    def __init__(self, in_channels=20, hidden_channels=32, out_channels=18):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.layers(x)


class Expander2D(nn.Module):
    """Same definition as in the encoder."""
    def __init__(self, target_shape=(16, 16), out_channels=4):
        super().__init__()
        self.target_shape = target_shape
        self.out_channels = out_channels

    def forward(self, x):
        B, D = x.shape
        out = x.view(B, self.out_channels, 1, 1)
        out = out.expand(-1, -1, *self.target_shape)
        return out


class JEPAPredictor(nn.Module):
    def __init__(self, in_channels=22, hidden_channels=32, out_channels=20):
        super().__init__()
        self.action_encoder = Expander2D()
        self.predictor = ConvPredictor(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels
        )

    def forward(self, z, action=None):
        if action is not None:
            a_enc = self.action_encoder(action)
            x = torch.cat([z, a_enc], dim=1)
        else:
            x = z
        return self.predictor(x)



from src.models.encoder import Transformer
class FlashAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * heads * dim_head, bias=False)
        self.proj_out = nn.Linear(heads * dim_head, dim)

    def forward(self, x):
        """
        x: (B, N, D)
        returns: (B, N, D)
        """
        b, n, _ = x.shape
        h, d = self.heads, self.dim_head

        x = self.norm(x)

        # project to Q, K, V
        qkv = self.qkv(x).view(b, n, 3, h, d)
        q, k, v = qkv.unbind(dim=2)
        # FlashAttention expects (B*H, N, 3, D)
        qkv = torch.stack([q, k, v], dim=2).permute(0, 3, 1, 2, 4).contiguous()
        qkv = qkv.view(b * h, n, 3, d)

        # efficient fused kernel
        out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)

        out = out.view(b, h, n, d).permute(0, 2, 1, 3).reshape(b, n, h * d)
        return self.proj_out(out)


# --------------------------------------------------
# Transformer using FlashAttention
# --------------------------------------------------
class TransformerPredictor(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                FlashAttentionBlock(dim, heads=heads, dim_head=dim_head),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Linear(mlp_dim, dim),
                )
            ])
            for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return self.norm(x)

