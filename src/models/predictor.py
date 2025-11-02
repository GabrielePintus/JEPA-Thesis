import torch
import torch.nn as nn



class ConvPredictor(nn.Module):
    def __init__(self, in_channels=20, hidden_channels=32, out_channels=18):
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








