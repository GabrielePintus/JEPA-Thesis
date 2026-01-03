import torch.nn as nn


class SmoothMeNet6(nn.Module):
    """
    Downsamples input through unpadded convolutions and pooling.
    Note: Each Conv2d layer reduces spatial dimensions due to padding=0.
    """
    def __init__(self, input_channels=3):
        super().__init__()
        self.activation = nn.ReLU()

        self.net = nn.Sequential(
            # Layer 1: 5x5 Kernel, No Padding
            # (H, W) -> (H-4, W-4)
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=0),
            nn.GroupNorm(4, 16),
            self.activation,

            # Layer 2: 5x5 Kernel, No Padding
            # (H-4, W-4) -> (H-8, W-8)
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.GroupNorm(8, 32),
            self.activation,
            
            # Subsampling: Halves spatial dimensions
            nn.AvgPool2d(kernel_size=2, stride=2),

            # Layer 3: 3x3 Kernel, No Padding
            # Shrinks H and W by 2
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.GroupNorm(8, 32),
            self.activation,

            # Layer 4: 3x3 Kernel, With Padding
            # Maintains current spatial dimensions
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class Expander2D(nn.Module):
    """
    Broadcasting module: Transforms a vector (B, C) into a 2D map (B, C, H, W).
    Useful for concatenating scalar/proprioceptive data with image features.
    """
    def __init__(self, target_shape=(16, 16), out_channels=4, use_batchnorm=True):
        super().__init__()
        self.target_shape = target_shape
        self.out_channels = out_channels
        
        # BN1d expects (Batch, Channels)
        self.bn = nn.BatchNorm1d(out_channels) if use_batchnorm else None

    def forward(self, x):
        B = x.shape[0]
        # Ensure input is flattened to (B, out_channels)
        x = x.view(B, -1)
        
        if self.bn is not None:
            x = self.bn(x)
            
        # Reshape to (B, C, 1, 1) then broadcast to target H, W
        out = x.view(B, self.out_channels, 1, 1)
        out = out.expand(-1, -1, *self.target_shape).contiguous()
        return out


class MaskHead(nn.Module):
    """
    Predicts a 1-channel soft subject mask from high-dimensional features.
    Input Shape: (B, C, H, W)
    Output Shape: (B, 1, H, W) - Logits
    """
    def __init__(self, in_channels=16, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            # Refine features while maintaining spatial resolution
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(),
            # Final projection to a single mask channel
            nn.Conv2d(hidden_channels, 1, kernel_size=1, padding=0),
        )

    def forward(self, z):
        return self.net(z)



