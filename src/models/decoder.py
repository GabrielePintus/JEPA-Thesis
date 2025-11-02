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
    def __init__(self, action_dim=2, hidden_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(36, 36, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(36),
            nn.ReLU(),
        )
        self.fcnn = nn.Sequential(
            nn.Linear(5184, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
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


