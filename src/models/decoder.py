import torch
import torch.nn as nn

class MeNet6Decoder128(nn.Module):
    """
    Symmetric decoder mirroring MeNet6_128.
    Input:  (B, 16, 28, 28)
    Output: (B, 3, 128, 128)
    """
    def __init__(self, latent_channels=20, out_channels=3):
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
