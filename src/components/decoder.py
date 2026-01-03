import torch.nn as nn

class IDMDecoderConv(nn.Module):
    """
    Decoder for IDM state from latent representations.
    Processes spatial features and reduces them to a low-dimensional state vector.
    
    Input:  (Batch, 36, 26, 26)
    Output: (Batch, output_dim)
    """
    
    def __init__(self, input_channels=36, output_dim=2, hidden_dim=24):
        super().__init__()
        
        # --- Spatial Processing Block ---
        # Refines spatial features and collapses channels to 1
        self.conv_refiner = nn.Sequential(
            # Input: (B, 36, 26, 26) -> Output: (B, hidden_dim, 26, 26)
            nn.Conv2d(input_channels, hidden_dim, kernel_size=5, padding=2, stride=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            
            # Input: (B, hidden_dim, 26, 26) -> Output: (B, 1, 26, 26)
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, stride=1),
        )
        
        # --- State Prediction Block ---
        # Maps the 2D feature map to the final output dimensions
        self.fc_head = nn.Sequential(
            # Normalization over the flattened spatial pixels (26*26 = 676)
            nn.LayerNorm(26 * 26),
            nn.Linear(26 * 26, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Latent feature map of shape (B, C, H, W)
        Returns:
            torch.Tensor: State prediction of shape (B, output_dim)
        """
        # 1. Spatial reduction: (B, 36, 26, 26) -> (B, 1, 26, 26)
        x = self.conv_refiner(x)
        
        # 2. Flatten for FC layers: (B, 1, 26, 26) -> (B, 676)
        x = x.flatten(1)
        
        # 3. Final projection: (B, 676) -> (B, output_dim)
        x = self.fc_head(x)
        
        return x