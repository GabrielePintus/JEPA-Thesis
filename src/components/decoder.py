import torch
import torch.nn as nn




class ProprioDecoder(nn.Module):
    """
    Decoder for proprioceptive state from latent representation.
    Simple MLP since proprio is low-dimensional.
    """
    
    def __init__(self, emb_dim=128, output_dim=4, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, token):
        """
        Args:
            token: (B, D) proprioceptive latent
        
        Returns:
            state: (B, state_dim) decoded state
        """
        return self.net(token)




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


