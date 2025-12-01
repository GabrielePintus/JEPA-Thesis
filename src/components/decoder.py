import torch
import torch.nn as nn
import torch.nn.functional as F



class MeNet6Decoder(nn.Module):
    """
    Approximate inverse of MeNet6.
    Takes latent feature map (B,16,26,26) and reconstructs (B,3,64,64).

    This is ONLY for visualization — it does not need to be a perfect inverse.
    """

    def __init__(self, out_channels=3, in_channels=16):
        super().__init__()

        # 1) Reverse last Conv1x1 (16 → 32)
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
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



    
class IDMDecoderConv(nn.Module):
    """
    Decoder for IDM state from latent representation.
    Project C, H, W to 1, H, W using Conv2d.
    """
    
    def __init__(self, input_channels=36, output_dim=2, hidden_dim=24):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=5, padding=2, stride=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, stride=1),
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(26 * 26),
            nn.Linear(26 * 26, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
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


class PositionDecoder(nn.Module):
    """
    Predict 2D coordinates in a learned continuous coordinate system.

    Architecture:
        f (B,C,H,W)
          → Conv2d (3x3)
          → GroupNorm
          → Activation
          → Conv2d (1x1) → heatmap
          → soft-argmax over learnable coordinate grid
    """
    def __init__(self, in_channels=16, H=26, W=26, hidden_channels=32):
        super().__init__()

        # ---------------------------
        # Extra Conv + Norm + Act
        # ---------------------------
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1)
        )

        # ---------------------------
        # Learnable coordinate grid
        # (2, H, W) → (x_grid, y_grid)
        # ---------------------------
        self.coord_grid = nn.Parameter(
            torch.randn(2, H, W) * 0.1
        )

    def forward(self, f):
        """
        Args:
            f: (B, C, H, W) feature map

        Returns:
            coords: (B, 2) learned continuous coordinates (x, y)
            heatmap: (B, 1, H, W)
        """
        B, _, H, W = f.shape

        # Extra conv block
        heatmap = self.conv(f)   # (B, hidden_channels, H, W)

        # Softmax over spatial locations
        prob = F.softmax(
            heatmap.view(B, -1), dim=1
        ).view(B, 1, H, W)

        # Learned coordinate system
        x_grid = self.coord_grid[0]  # (H, W)
        y_grid = self.coord_grid[1]

        # Expected value in learned coordinate basis
        x = (prob[:, 0] * x_grid).sum(dim=(1, 2))
        y = (prob[:, 0] * y_grid).sum(dim=(1, 2))

        coords = torch.stack([x, y], dim=1)
        return coords, heatmap


