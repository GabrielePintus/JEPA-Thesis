import torch
import torch.nn as nn

class MeNet6Decoder128(nn.Module):
    """
    Symmetric decoder mirroring MeNet6_128.
    Input:  (B, 16, 28, 28)
    Output: (B, 3, 128, 128)
    """
    def __init__(self, latent_channels=18, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            # 7×7 → 14×14
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),

            # 14×14 → 28×28
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            # 28×28 → 56×56
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),

            # Refinement (keep 56×56)
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),

            # Final RGB reconstruction to 56×56
            nn.ConvTranspose2d(16, out_channels, kernel_size=3, stride=1, padding=1),

            # 56×56 → 64×64 (exact match)
            nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)




class InverseDynamicsModel(nn.Module):
    """
    Inverse Dynamics Model (IDM):
    Given current and next latent states, predict the action taken.
    """
    def __init__(self, action_dim=2, hidden_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(36, 64, kernel_size=5, stride=1, padding=2),  # 26x26 -> 26x26
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),  # 13x13 -> 11x11
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fcnn = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.fcnn = torch.compile(self.fcnn, mode="default")

    def forward(self, z_curr, z_next):
        # Concatenate current and next latent states
        z_concat = torch.cat([z_curr, z_next], dim=-1)
        z_concat = z_concat.view(z_concat.size(0), 36, 26, 26)

        z_cnn = self.cnn(z_concat)
        z_cnn_flat = z_cnn.view(z_cnn.size(0), -1)
        action_pred = self.fcnn(z_cnn_flat)
        return action_pred


from flash_attn import flash_attn_func
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
        # out = flash_attn_func(qkv, causal=False)  # (B*H, N, D)
        out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)

        out = out.view(b, h, n, d).permute(0, 2, 1, 3).reshape(b, n, h * d)
        return self.proj_out(out)


class InverseDynamicsTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, out_dim=2):
        super().__init__()
        self.dim = dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.temporal_embed = nn.Parameter(torch.randn(1, 2, dim))  # embeddings for (t, t+1)

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

        self.fc_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim)
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, z_t, z_tp1):
        """
        z_t, z_tp1: (B, N, D)
        returns: predicted action (B, out_dim)
        """
        b, n, d = z_t.shape

        # Concatenate states and add temporal embeddings
        z_t = z_t + self.temporal_embed[:, 0].unsqueeze(1)
        z_tp1 = z_tp1 + self.temporal_embed[:, 1].unsqueeze(1)
        x = torch.cat([z_t, z_tp1], dim=1)

        # prepend CLS
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # transformer layers
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)

        x = x[:, 0]  # CLS
        return self.fc_head(x)


class VisualDecoderV2(nn.Module):
    """
    Visual decoder that reconstructs a (B, C, H, W) image
    from ViT-style patch embeddings (B, N, D).
    """

    def __init__(self, image_size=64, patch_size=8, dim=32, out_channels=3):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.dim = dim

        # Derived grid sizes
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."
        self.num_patches_h = image_size // patch_size
        self.num_patches_w = image_size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Each patch corresponds to patch_size × patch_size pixels × C channels
        patch_dim = out_channels * patch_size * patch_size

        # --- Project tokens → pixel patches ---
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim)
        )

        # --- Convolutional refinement head ---
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # normalized reconstruction in [0,1]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, N, D) — patch tokens from ViT encoder.
        Returns:
            img: (B, C, H, W)
        """
        b, n, d = x.shape
        nh, nw = self.num_patches_h, self.num_patches_w
        ph, pw = self.patch_size, self.patch_size
        c = self.out_channels

        # --- Project tokens to pixel patches ---
        patches = self.proj(x)  # (B, N, C*ph*pw)
        assert n == nh * nw, f"Expected N={nh*nw}, got N={n}"
        patches = patches.view(b, nh, nw, c, ph, pw)

        # --- Reassemble into full image ---
        # Move patch axes into correct spatial positions:
        img = (
            patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, nh, ph, nw, pw)
            .contiguous()
            .view(b, c, nh * ph, nw * pw)      # (B, C, H, W)
        )

        # --- Optional convolutional refinement ---
        img = self.refine(img)
        return img
    


class VisualDecoderSimple(nn.Module):
    """
    Simple symmetric decoder that reconstructs (B, 3, 64, 64)
    from ViT patch tokens (B, N, D), where N = (H/ps)*(W/ps).

    Example:
        image_size=64, patch_size=8  →  N = 64
        Input:  (B, 64, D)
        Output: (B, 3, 64, 64)
    """

    def __init__(self, image_size=64, patch_size=8, dim=32, out_channels=3):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.out_channels = out_channels

        # Spatial shape of patch grid
        self.nh = image_size // patch_size
        self.nw = image_size // patch_size

        # --- Map tokens to base feature map ---
        self.token_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 64),  # expand token embedding to feature channels
        )

        # --- Upsampling path: (8→16→32→64) ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 8→16
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16→32
            nn.GroupNorm(4, 32),
            nn.GELU(),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 32→64
            nn.GroupNorm(4, 16),
            nn.GELU(),

            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # assumes target in [0,1]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tokens):
        """
        Args:
            tokens: (B, N, D) patch tokens
        Returns:
            img: (B, 3, 64, 64)
        """
        B, N, D = tokens.shape
        assert N == self.nh * self.nw or N == self.nh * self.nw + 1, (
            f"Expected {self.nh*self.nw} tokens (+CLS), got {N}"
        )

        # Drop CLS if present
        if N == self.nh * self.nw + 1:
            tokens = tokens[:, 1:, :]

        # Map each token → local feature
        x = self.token_proj(tokens)               # (B, N, 64)
        x = x.view(B, self.nh, self.nw, 64)       # (B, nh, nw, 64)
        x = x.permute(0, 3, 1, 2).contiguous()    # (B, 64, nh, nw) e.g. (B, 64, 8, 8)

        # Decode up to full resolution
        img = self.decoder(x)
        return img
