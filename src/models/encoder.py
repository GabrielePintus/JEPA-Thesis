import torch
import torch.nn as nn


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



class Expander1D(nn.Module):
    """Expands a low-dimensional proprioceptive vector into a higher-dimensional vector."""
    def __init__(self, out_dim=32):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x):
        # Repeat along feature dimension
        B = x.shape[0]
        x = x.view(B, -1)
        out = x.unsqueeze(2).expand(-1, -1, self.out_dim // x.shape[1])
        out = out.contiguous().view(B, self.out_dim)
        return out



class JEPAEncoder(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.backbone = MeNet6(input_channels=input_channels)
        self.propio_encoder = Expander2D()

    def forward(self, img, proprio=None):
        # Encode image
        z_img = self.backbone(img)
        if proprio is not None:
            z_prop = self.propio_encoder(proprio)
            # Concatenate along channel dimension
            z = torch.cat([z_img, z_prop], dim=1)
        else:
            z = z_img
        return z




#
#   ViT
#
# --------------------------------------------------------
# helpers
# --------------------------------------------------------

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# --------------------------------------------------------
# building blocks
# --------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv

        # (b, n, h*d) → (b, h, n, d)
        def reshape_heads(t):
            b, n, hd = t.shape
            d = hd // h
            return t.view(b, n, h, d).permute(0, 2, 1, 3)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        # (b, h, n, d) → (b, n, h*d)
        out = out.permute(0, 2, 1, 3).reshape(b, n, h * v.shape[-1])

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# --------------------------------------------------------
# Simple ViT
# --------------------------------------------------------


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# --------------------------------------------------------
# building blocks
# --------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


from flash_attn import flash_attn_func
import torch
from torch import nn

class FlashAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, 3 * heads * dim_head, bias=False)
        self.to_out = nn.Linear(heads * dim_head, dim, bias=False)

    def forward(self, x):
        """
        x: (B, N, D)
        """
        b, n, _ = x.shape
        h, d = self.heads, self.dim_head

        x = self.norm(x)

        # --- QKV projection ---
        qkv = self.to_qkv(x)
        qkv = qkv.view(b, n, 3, h, d)
        q, k, v = qkv.unbind(dim=2)  # each (B, N, H, D)

        # --- FlashAttention requires sequence-first format ---
        # Convert to (B, N, H, D) -> (B, N, 3, H, D) equivalent usage
        # Here we concatenate along last dim for efficiency
        qkv = torch.stack([q, k, v], dim=2)  # (B, N, 3, H, D)
        qkv = qkv.permute(0, 3, 1, 2, 4).contiguous()  # (B, H, N, 3, D)

        # flatten batch and heads for FlashAttn (B*H, N, 3, D)
        qkv = qkv.view(b * h, n, 3, d)

        # --- FlashAttention forward ---
        # out = flash_attn_func(qkv, causal=False)  # (B*H, N, D)
        out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)

        # reshape back
        out = out.view(b, h, n, d).permute(0, 2, 1, 3).contiguous().view(b, n, h * d)

        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv

        # (b, n, h*d) → (b, h, n, d)
        def reshape_heads(t):
            b, n, hd = t.shape
            d = hd // h
            return t.view(b, n, h, d).permute(0, 2, 1, 3)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        # (b, h, n, d) → (b, n, h*d)
        out = out.permute(0, 2, 1, 3).reshape(b, n, h * v.shape[-1])

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                FlashAttention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# --------------------------------------------------------
# Simple ViT
# --------------------------------------------------------

class SimpleViT(nn.Module):
    def __init__(
        self, *,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            "Image dimensions must be divisible by the patch size."

        self.num_patches_h = image_height // patch_height
        self.num_patches_w = image_width // patch_width
        patch_dim = channels * patch_height * patch_width

        # Patch embedding (manual rearrange)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.channels = channels

        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=self.num_patches_h,
            w=self.num_patches_w,
            dim=dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

    def _to_patch_embedding(self, img):
        # img: (b, c, h, w)
        b, c, h, w = img.shape
        ph, pw = self.patch_height, self.patch_width
        nh, nw = self.num_patches_h, self.num_patches_w

        # Divide into patches
        img = img.view(b, c, nh, ph, nw, pw)
        # Move patch axes next to batch
        img = img.permute(0, 2, 4, 1, 3, 5).contiguous()
        # Flatten patches
        patches = img.view(b, nh * nw, c * ph * pw)
        return self.patch_embed(patches)

    def forward(self, img):
        device = img.device

        x = self._to_patch_embedding(img)
        x = x + self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        return x


class LinearEncoder(nn.Module):
    def __init__(self, input_dim=2, output_dim=32):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.net(x)

