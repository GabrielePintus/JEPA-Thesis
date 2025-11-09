import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalReconstructionLoss(nn.Module):
    """Focuses on small but important features"""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, pred, target):
        # Compute pixel-wise difference
        diff = torch.abs(pred - target)
        
        # Focal weighting: emphasize hard-to-reconstruct pixels
        focal_weight = (1 + diff) ** self.gamma
        
        # Edge detection to identify agent
        target_gray = target.mean(dim=1, keepdim=True)
        edges = F.conv2d(
            target_gray,
            torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).float().view(1, 1, 3, 3).to(target.device),
            padding=1
        )
        edge_weight = torch.sigmoid(edges.abs() * 10)
        
        # Combined loss
        loss = focal_weight * diff * (1 + self.alpha * edge_weight)
        
        return loss.mean()

