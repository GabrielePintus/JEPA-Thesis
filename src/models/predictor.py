import torch
import torch.nn as nn



class ConvPredictor(nn.Module):
    def __init__(self, in_channels=22, hidden_channels=64, out_channels=20):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.layers(x)


class Expander2D(nn.Module):
    """Same definition as in the encoder."""
    def __init__(self, target_shape=(16, 16), out_channels=4):
        super().__init__()
        self.target_shape = target_shape
        self.out_channels = out_channels

    def forward(self, x):
        B, D = x.shape
        out = x.view(B, self.out_channels, 1, 1)
        out = out.expand(-1, -1, *self.target_shape)
        return out


class JEPAPredictor(nn.Module):
    def __init__(self, in_channels=22, hidden_channels=32, out_channels=20):
        super().__init__()
        self.action_encoder = Expander2D()
        self.predictor = ConvPredictor(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels
        )

    def forward(self, z, action=None):
        if action is not None:
            a_enc = self.action_encoder(action)
            x = torch.cat([z, a_enc], dim=1)
        else:
            x = z
        return self.predictor(x)





class InverseDynamics(nn.Module):
    """
    Predicts the action vector given two latent states (z_t, z_{t+1}).
    Operates on concatenated flattened latents.
    """
    def __init__(self, z_channels=20, action_dim=2, hidden_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2 * z_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),   # collapse spatial dimensions
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, z_t, z_next):
        x = torch.cat([z_t, z_next], dim=1)
        h = self.conv(x)
        return self.fc(h)


