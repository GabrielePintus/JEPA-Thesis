import torch
import torch.nn as nn

class ConvPredictor(nn.Module):
    """
    A 3-layer Convolutional Predictor using Group Normalization and ReLU activation.
    Maintains spatial dimensions (Height x Width) throughout the forward pass.
    """
    def __init__(self, in_channels=20, hidden_channels=32, out_channels=18):
        super(ConvPredictor, self).__init__()

        # Define the network architecture
        self.net = nn.Sequential(
            # --- Layer 1: Initial Feature Extraction ---
            # Input: (Batch, in_channels, H, W)
            # Kernel 5x5 with Padding 2 maintains spatial size: (H+2*2-5)/1 + 1 = H
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(num_groups=4, num_channels=hidden_channels),
            nn.ReLU(inplace=True),

            # --- Layer 2: Intermediate Processing ---
            # Kernel 3x3 with Padding 1 maintains spatial size: (H+2*1-3)/1 + 1 = H
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=hidden_channels),
            nn.ReLU(inplace=True),

            # --- Layer 3: Output Projection ---
            # Maps hidden features to the desired output channel dimension
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, in_channels, H, W)
        Returns:
            torch.Tensor: Predicted output of shape (Batch, out_channels, H, W)
        """
        return self.net(x)