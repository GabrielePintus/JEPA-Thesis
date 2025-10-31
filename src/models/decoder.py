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
            # Mirror of Conv2d(32,16,1,1)
            nn.ConvTranspose2d(latent_channels, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Mirror of Conv2d(64,32,3,1,p1)
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Mirror of Conv2d(64,64,3,1)
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Mirror of Conv2d(32,64,4,2,p1)  (upsample ×2)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 28 -> 56
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Mirror of Conv2d(16,32,5,2) (upsample ×2)
            nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2, padding=2),  # 56 -> 112
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Final refinement + resize to exact 96×96
            nn.ConvTranspose2d(16, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Upsample(size=(96, 96), mode="bilinear", align_corners=False),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)
