import torch
import torch.nn as nn




class ProprioDecoder(nn.Module):
    """
    Decoder for proprioceptive state from latent representation.
    Simple MLP since proprio is low-dimensional.
    """
    
    def __init__(self, emb_dim=128, state_dim=4, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, state_dim),
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