import torch
import torch.nn as nn



class ResUpBlock(nn.Module):
    """
    Residual upsampling block:
        x → upsample → conv → conv → residual add
    """
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super().__init__()
        self.scale = scale_factor
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.proj  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # Upsample
        x_up = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)

        # Residual block
        y = F.gelu(self.conv1(x_up))
        y = self.conv2(y)
        return F.gelu(y + self.proj(x_up))  # skip connection
        

class MeNet6DecoderStrong(nn.Module):
    """
    Stronger, smoother, high-capacity decoder for MeNet6.
    Input:  (B,16,26,26)
    Output: (B,3,64,64)
    """
    def __init__(self, out_channels=3):
        super().__init__()

        # First expand channels BEFORE upsampling
        self.pre = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
        )

        # Upsample 26 → 52 (approx original 28→60)
        self.up1 = ResUpBlock(32, 32, scale_factor=2)

        # Slight refine at 52×52
        self.refine1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
        )

        # Upsample 52 → 64
        self.up2 = ResUpBlock(32, 16, scale_factor=64/52)  # ≈1.23 upscale

        # Final refinement
        self.refine2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        """
        z: (B,16,26,26)
        """
        x = self.pre(z)                # → (B,32,26,26)
        x = self.up1(x)                # → (B,32,52,52)
        x = self.refine1(x)            # refine
        x = self.up2(x)                # → (B,16,64,64)
        x = self.refine2(x)            # → (B,3,64,64)
        return x



class MeNet6Decoder(nn.Module):
    """
    Approximate inverse of MeNet6.
    Takes latent feature map (B,16,26,26) and reconstructs (B,3,64,64).

    This is ONLY for visualization — it does not need to be a perfect inverse.
    """

    def __init__(self, out_channels=3):
        super().__init__()

        # 1) Reverse last Conv1x1 (16 → 32)
        self.up1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # 2) Reverse stride-1 Conv3x3 (32 → 32)
        self.up2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # 3) Reverse stride-2 Conv5x5 (32 → 16), using ConvTranspose2D
        # Original reduced 60→28; we restore 28→60
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1, output_padding=1),
            nn.GELU(),
        )

        # 4) Reverse first Conv5x5 (16 → out_channels), restore 60→64
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(16, out_channels, kernel_size=5, stride=1),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels*2, kernel_size=3, padding=1),
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        z: latent feature map of shape (B,16,26,26)
        returns: reconstructed frame (B,out_channels,64,64)
        """
        x = self.up1(z)             # (B,32,26,26)
        x = self.up2(x)             # (B,32,26,26)
        x = self.up3(x)             # (B,16,60,60)
        x = self.up4(x)             # (B,out,ch,64,64)
        x = self.refine(x)          # refine

        # Ensure output is exactly 64×64 (rare off-by-one)
        x = F.interpolate(x, size=(64,64), mode="bilinear", align_corners=False)

        return x



class ProprioDecoder(nn.Module):
    """
    Decoder for proprioceptive state from latent representation.
    Simple MLP since proprio is low-dimensional.
    """
    
    def __init__(self, emb_dim=128, output_dim=4, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, token, mean=None, std=None):
        """
        Args:
            token: (B, D) proprioceptive latent
        
        Returns:
            state: (B, state_dim) decoded state
        """
        out = self.net(token)
        if mean is not None and std is not None:
            out = out * std + mean
        return out




import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualDecoder(nn.Module):
    def __init__(self, emb_dim=64, patch_size=8, img_channels=3, img_size=64):
        super().__init__()

        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.img_channels = img_channels
        self.img_size = img_size

        self.num_patches = (img_size // patch_size) ** 2

        # 1) Projection head: patch embeddings → pixel patches
        # self.proj = nn.Linear(emb_dim, img_channels * patch_size * patch_size)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Linear(emb_dim * 2, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, img_channels * patch_size * patch_size)
        )

        # 2) Decoder: refine using ConvTranspose
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(img_channels, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, img_channels, kernel_size=5, padding=0),
            nn.Conv2d(img_channels, img_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, x):
        """
        x: (B, N, D) tokens from ViT (CLS token included)
        """

        B, N, D = x.shape

        # Project each token into a patch: (B, N-1, C * P * P)
        x = self.proj(x)

        # Reshape into (B, H/P, W/P, C, P, P)
        num_patches_per_dim = self.img_size // self.patch_size
        x = x.view(
            B,
            num_patches_per_dim,
            num_patches_per_dim,
            self.img_channels,
            self.patch_size,
            self.patch_size
        )

        # Rearrange to (B, C, H, W)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(
            B,
            self.img_channels,
            self.img_size,
            self.img_size
        )

        # Refine prediction with ConvTranspose decoder
        x = self.decoder(x)

        return x


# class VisualDecoder(nn.Module):
#     def __init__(self, emb_dim=64, patch_size=8, img_channels=3, img_size=64):
#         super().__init__()

#         self.emb_dim = emb_dim
#         self.patch_size = patch_size
#         self.img_channels = img_channels
#         self.img_size = img_size

#         self.num_patches = (img_size // patch_size) ** 2

#         # 1) Projection head: patch embeddings → pixel patches
#         # This MLP is all we need.
#         self.proj = nn.Sequential(
#             nn.Linear(emb_dim, emb_dim * 2),
#             nn.GELU(),
#             nn.Linear(emb_dim * 2, emb_dim * 4),
#             nn.GELU(),
#             nn.Linear(emb_dim * 4, img_channels * patch_size * patch_size)
#         )

#         # 2) REMOVED THE self.decoder (ConvTranspose2d, etc.)
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(img_channels, 64, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(64, img_channels, kernel_size=3, padding=0),
#         )
        
        

#     def forward(self, x):
#         """
#         x: (B, N, D) tokens from ViT
#         """

#         B, N, D = x.shape

#         # Project each token into a patch: (B, N, C * P * P)
#         x = self.proj(x)

#         # Reshape into (B, H/P, W/P, C, P, P)
#         num_patches_per_dim = self.img_size // self.patch_size
#         x = x.view(
#             B,
#             num_patches_per_dim,
#             num_patches_per_dim,
#             self.img_channels,
#             self.patch_size,
#             self.patch_size
#         )

#         # Rearrange to (B, C, H, W)
#         x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
#         x = x.view(
#             B,
#             self.img_channels,
#             self.img_size,
#             self.img_size
#         )

#         # Refinement decoder removed.
#         x = self.decoder(x)

#         # Apply sigmoid to the final reassembled image
#         # Assumes input images are [0, 1]
#         return torch.sigmoid(x)





class RegressionTransformerDecoder(nn.Module):
    """
    Transformer *decoder* that queries ViT patch tokens and outputs a regression vector.

    Inputs
    -------
    patch_tokens : Tensor, shape (B, N, C)
        Patch embeddings from a ViT (no need to include [CLS]; just patches).
    key_padding_mask : Optional[BoolTensor], shape (B, N)
        True for padded positions (if you use variable-length sequences).

    Output
    ------
    y : Tensor, shape (B, D)

    Notes
    -----
    - We use K learnable query tokens (default 1) as 'tgt' to the TransformerDecoder.
      They cross-attend to the patch tokens ('memory').
    - If your ViT already added positional encodings, you typically don't need extra PE here.
    - Works even if N varies across batch when you provide key_padding_mask.
    """

    def __init__(
        self,
        d_model: int,            # must match ViT embedding dim C
        d_out: int,              # regression output size D
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        num_queries: int = 1,    # number of query tokens
        head_mlp_hidden: int = 512,
        norm_first: bool = True,
    ):
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,    # PyTorch's Transformer expects (S, B, E) unless batch_first=True (only for EncoderLayer as of older versions)
            norm_first=norm_first,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # Learnable query tokens (K, C)
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.randn(num_queries, d_model) / (d_model ** 0.5))

        # Head: pool queries -> regression
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, head_mlp_hidden),
            nn.GELU(),
            nn.Linear(head_mlp_hidden, d_out),
        )

        # Optional final norm on memory if you want (usually not needed)
        self.mem_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        patch_tokens: torch.Tensor,                 # (B, N, C)
        key_padding_mask: torch.Tensor | None = None,  # (B, N) True for pad
    ) -> torch.Tensor:
        B = patch_tokens.shape[0]
        C = patch_tokens.shape[-1]
        assert C == self.query_tokens.shape[-1], "d_model mismatch with ViT token dim"

        # Memory: (N, B, C)
        memory = self.mem_norm(patch_tokens).transpose(0, 1)  # (N, B, C)

        # Tgt: expand learnable queries for each batch -> (K, B, C)
        tgt = self.query_tokens.unsqueeze(1).expand(self.num_queries, B, C)

        # Cross-attention: queries attend to patch tokens
        # key_padding_mask marks padded memory positions (B, N)
        dec_out = self.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=key_padding_mask,  # None if not using padding
        )  # (K, B, C)

        # Pool over query tokens (mean or first). Mean is robust if K>1.
        q_repr = dec_out.mean(dim=0)  # (B, C)

        # Regression head
        y = self.head(q_repr)  # (B, D)
        return y


