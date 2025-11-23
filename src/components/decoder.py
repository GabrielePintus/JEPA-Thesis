import torch
import torch.nn as nn
import torch.nn.functional as F



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


class IDMDecoder(nn.Module):
    """
    Decoder for IDM state from latent representation.
    Project C, H, W to 1, H, W using Conv2d.
    """
    
    def __init__(self, input_channels=36, output_dim=2, hidden_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, input_channels*2, kernel_size=5, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(input_channels*2, input_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(input_channels, input_channels//2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(input_channels//2, input_channels//4, kernel_size=3, padding=1, stride=1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_channels//4 * 12 * 12, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        out = self.cnn(x)  # (B, C, H', W')
        out = out.flatten(1)  # (B, C*H'*W')
        out = self.mlp(out)
        return out
    
class IDMDecoderConv(nn.Module):
    """
    Decoder for IDM state from latent representation.
    Project C, H, W to 1, H, W using Conv2d.
    """
    
    def __init__(self, input_channels=36, output_dim=2, hidden_dim=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=5, padding=2, stride=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1, stride=1),
        )
    
    def forward(self, x):
        return self.net(x)


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





