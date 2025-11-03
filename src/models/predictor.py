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

# --------------------------------------------------
# Full standard FlashAttention (self-attention)
# --------------------------------------------------
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
        qkv = self.qkv(x).view(b, n, 3, h, d)
        q, k, v = qkv.unbind(dim=2)
        out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
        out = out.view(b, n, h * d)
        return self.proj_out(out)


# --------------------------------------------------
# Explicit Cross-Attention (full-rank)
# --------------------------------------------------
class CrossAttention(nn.Module):
    def __init__(self, dim=32, heads=4, dim_head=8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, heads * dim_head, bias=False)
        self.k_proj = nn.Linear(dim, heads * dim_head, bias=False)
        self.v_proj = nn.Linear(dim, heads * dim_head, bias=False)
        self.proj_out = nn.Linear(heads * dim_head, dim)

    def forward(self, q_src, kv_src):
        """
        q_src: (B, Nq, D)
        kv_src: (B, Nk, D)
        returns: (B, Nq, D)
        """
        b, nq, _ = q_src.shape
        nk = kv_src.size(1)
        h, d = self.heads, self.dim_head

        q = self.q_proj(self.norm_q(q_src)).view(b, nq, h, d)
        k = self.k_proj(self.norm_kv(kv_src)).view(b, nk, h, d)
        v = self.v_proj(self.norm_kv(kv_src)).view(b, nk, h, d)

        out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=self.scale, causal=False)
        out = out.view(b, nq, h * d)
        return self.proj_out(out)


# --------------------------------------------------
# Full Transformer Predictor (no low-rank approximation)
# --------------------------------------------------
class TransformerPredictor(nn.Module):
    def __init__(self, dim=32, depth=3, heads=4, dim_head=8, mlp_dim=128, num_latents=8):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 66, dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                # 1. condition state on action
                "cross_state_action": CrossAttention(dim, heads, dim_head),

                # 2. encode: latents ← tokens
                "encode": CrossAttention(dim, heads, dim_head),

                # 3. latent self-mixer
                "latent_mixer": FlashAttentionBlock(dim, heads, dim_head),

                # 4. decode: tokens ← latents
                "decode": CrossAttention(dim, heads, dim_head),

                # 5. feedforward MLP
                "ff": nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Linear(mlp_dim, dim),
                )
            })
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, z_visual_prop, z_action):
        """
        z_visual_prop: (B, N, D)
        z_action: (B, M, D)
        """
        B, N, D = z_visual_prop.shape
        x = z_visual_prop + self.pos_embedding[:, :N, :]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            # 1️⃣ condition tokens on action
            x = x + layer["cross_state_action"](x, z_action)

            # 2️⃣ encode: latents attend to tokens
            latents = latents + layer["encode"](latents, x)

            # 3️⃣ latent self-attention
            latents = latents + layer["latent_mixer"](latents)

            # 4️⃣ decode: tokens attend to latents
            x = x + layer["decode"](x, latents)

            # 5️⃣ feedforward
            x = x + layer["ff"](x)

        return self.norm(x)
