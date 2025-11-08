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
    """Expands a low-dimensional proprioceptive vector into a 2D feature map."""
    def __init__(self, target_shape=(16, 16), out_channels=4):
        super().__init__()
        self.target_shape = target_shape
        self.out_channels = out_channels

    def forward(self, x):
        # Flatten any extra dims except batch
        B = x.shape[0]
        x = x.view(B, -1)
        out = x.view(B, self.out_channels, 1, 1)
        out = out.expand(-1, -1, *self.target_shape)

        return out



#
#   CUSTOM
#



# ===========================
# Visual Encoder: Conv Stem → ViT
# ===========================
class ConvStem(nn.Module):
    """
    Light conv stem WITHOUT downsampling (keeps green dot sharp).
    64x64x3 → 64x64x64
    """
    def __init__(self, in_ch=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, hidden, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)  # (B, 64, 64, 64)


class PatchEmbed(nn.Module):
    """
    Patch embedding directly on full resolution (no pre-downsampling).
    patch_size can be 4, 8, or any divisor of 64.

    64×64 → (64/patch_size)^2 tokens
    """
    def __init__(self, in_ch, patch_size, emb_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_ch, emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, feat):     # feat: (B, 64, 64, 64)
        x = self.proj(feat)      # (B, D, H, W)
        B, D, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, D)
        return tokens, (H, W)


class PositionalEncoding2D(nn.Module):
    """
    Learnable positional embeddings for patches + CLS.
    Works with varying grid sizes (depends on patch_size).
    """
    def __init__(self, emb_dim, grid_hw, add_cls=True):
        super().__init__()
        H, W = grid_hw
        self.add_cls = add_cls
        self.pos = nn.Parameter(torch.randn(1, H * W, emb_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, emb_dim)) if add_cls else None

    def forward(self, patches, cls_token=None):
        patches = patches + self.pos
        if self.add_cls and cls_token is not None:
            cls_token = cls_token + self.cls_pos
            patches = torch.cat([cls_token, patches], dim=1)
        return patches





class SinusoidalPositionalEncoding2D(nn.Module):
    """
    Non-learnable 2D sinusoidal positional encoding (fixed).
    Generates positional encodings based on (row, col) grid index.
    """
    def __init__(self, emb_dim, add_cls=True):
        super().__init__()
        assert emb_dim % 4 == 0, "Embedding dimension must be divisible by 4"
        self.add_cls = add_cls
        self.emb_dim = emb_dim

    def forward(self, patches, cls_token=None):
        """
        patches: (B, N, D)
        cls_token: (B, 1, D) or None
        """
        B, N, D = patches.shape
        grid_size = int(N ** 0.5)
        H, W = grid_size, grid_size

        device = patches.device

        # Generate coordinate grid
        y = torch.arange(H, device=device).repeat(W)          # (N,)
        x = torch.arange(W, device=device).repeat_interleave(H)  # (N,)

        # Frequencies for sin/cos
        dim_quarter = D // 4
        omega = torch.exp(
            torch.arange(dim_quarter, device=device) * -(math.log(10000.0) / dim_quarter)
        )

        # (N, dim_quarter)
        pos_x = x[:, None] * omega[None, :]
        pos_y = y[:, None] * omega[None, :]

        # Allocate (1, N, D)
        out = torch.zeros(1, N, D, device=device)
        out[..., 0:dim_quarter] = torch.sin(pos_x)
        out[..., dim_quarter:dim_quarter * 2] = torch.cos(pos_x)
        out[..., dim_quarter * 2:dim_quarter * 3] = torch.sin(pos_y)
        out[..., dim_quarter * 3:] = torch.cos(pos_y)

        patches = patches + out  # broadcast to (B, N, D)

        if self.add_cls and cls_token is not None:
            return torch.cat([cls_token, patches], dim=1)

        return patches




class SwiGLU(nn.Module):
    """SwiGLU MLP (weight-efficient, more expressive than GELU MLP)"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.out(F.silu(self.w1(x)) * self.w2(x))


class TransformerEncoder(nn.Module):
    def __init__(self, dim=128, depth=4, heads=4, mlp_dim=256, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

        # RoPE (using xpos=True for rotary_embedding_torch compatibility)
        from rotary_embedding_torch import RotaryEmbedding
        self.rope = RotaryEmbedding(dim=self.head_dim, use_xpos=True)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),                 # attn LN
                nn.Linear(dim, dim * 3),           # QKV projection
                nn.Linear(dim, dim),               # out projection
                nn.LayerNorm(dim),                 # ff LN
                SwiGLU(dim, mlp_dim),              # SwiGLU MLP
            ])
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, S, D = x.shape

        for attn_ln, qkv_proj, out_proj, ff_ln, ff in self.layers:

            # ----- Self-Attention -----
            normed = attn_ln(x)
            qkv = qkv_proj(normed)
            qkv = qkv.view(B, S, 3, self.heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)        # (B, S, H, Dh)

            # ✅ RoPE applied here
            q, k = self.rope.rotate_queries_and_keys(q, k)

            # Standard attention: (B, H, S, Dh)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)

            if self.training and self.dropout > 0:
                attn = F.dropout(attn, p=self.dropout)

            y = attn @ v                          # (B, H, S, Dh)
            y = y.transpose(1, 2).reshape(B, S, D)
            x = x + out_proj(y)                   # residual

            # ----- Feedforward -----
            x = x + ff(ff_ln(x))                  # residual

        return self.final_norm(x)



class VisualEncoder(nn.Module):
    """
    Visual Encoder used in JEPA:
    Outputs:
        cls:     (B, D)
        patches: (B, N, D)
        grid_hw: (H, W)
    Where N = (64 / patch_size)^2
    """
    def __init__(
        self,
        emb_dim=128,
        depth=3,
        heads=4,
        mlp_dim=256,
        patch_size=8
    ):
        super().__init__()

        self.patch_size = patch_size
        self.stem = ConvStem(in_ch=3, hidden=64)

        self.patch = PatchEmbed(
            in_ch=64,
            patch_size=patch_size,
            emb_dim=emb_dim
        )

        # will shape positional embedding dynamically in forward()
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos = SinusoidalPositionalEncoding2D(emb_dim, add_cls=True)

        self.tr = TransformerEncoder(dim=emb_dim, depth=depth,
                                     heads=heads, mlp_dim=mlp_dim)

    def forward(self, x):
        feat = self.stem(x)
        tokens, grid_hw = self.patch(feat)  # (B, N, D), (H, W)

        B, N, D = tokens.shape
        cls = self.cls_token.expand(B, -1, -1)
        x_all = self.pos(tokens, cls)
        x_all = self.tr(x_all)

        return x_all[:, 0], x_all[:, 1:], grid_hw


# ===========================
# Proprio encoder & heads
# ===========================
class ProprioEncoder(nn.Module):
    """Encodes proprio state (x, y, vx, vy) → latent token (D)."""
    def __init__(self, input_dim = 4, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(),
            nn.Linear(64, emb_dim)
        )

    def forward(self, xy):  # (B, 2)
        return self.net(xy)  # (B, D)


class MLPProjection(nn.Module):
    def __init__(self, in_dim, proj_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            # nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, proj_dim),
            # nn.GELU(),
            # nn.Linear(proj_dim, proj_dim),
            # nn.LayerNorm(proj_dim),
        )

    def forward(self, x):  # (..., in_dim)
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])  # safe for non-contiguous tensors
        y = self.net(x)
        return y.reshape(*orig_shape[:-1], -1)



class LocalizationHead(nn.Module):
    """
    Predicts heatmap over 8x8 patches from patch tokens.
    Input:  (B, N=64, D)
    Output: (B, 64) logits
    """
    def __init__(self, dim=128, grid_hw=(8, 8)):
        super().__init__()
        H, W = grid_hw
        self.H, self.W = H, W
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, patch_tokens):  # (B, 64, D)
        logits = self.net(patch_tokens).squeeze(-1)  # (B, 64)
        return logits







#
#   Simple ViT
#




