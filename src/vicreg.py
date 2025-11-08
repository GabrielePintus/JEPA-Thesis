import torch
import torch.nn as nn
import torch.nn.functional as F

class VICRegLoss(nn.Module):
    def __init__(self, inv_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, gamma=1.0):
        super().__init__()
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma

    def forward(self, z1, z2):
        N, D = z1.shape
        
        # Invariance loss
        inv_loss = F.mse_loss(z1, z2)
        
        # Variance loss
        z1_std = torch.sqrt(z1.var(dim=0) + 1e-4)
        z2_std = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(self.gamma - z1_std))
        var_loss += torch.mean(F.relu(self.gamma - z2_std))
        
        # Covariance loss
        z1_norm = z1 - z1.mean(dim=0)
        z2_norm = z2 - z2.mean(dim=0)
        
        cov1 = (z1_norm.T @ z1_norm) / (N - 1)
        cov2 = (z2_norm.T @ z2_norm) / (N - 1)
        
        cov_loss = off_diagonal(cov1).pow(2).sum() / D
        cov_loss += off_diagonal(cov2).pow(2).sum() / D
        
        loss = self.inv_coeff * inv_loss + self.var_coeff * var_loss + self.cov_coeff * cov_loss
        
        return {
            'loss': loss,
            'inv_loss': inv_loss,
            'var_loss': var_loss,
            'cov_loss': cov_loss
        }

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()