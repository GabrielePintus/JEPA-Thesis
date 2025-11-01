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