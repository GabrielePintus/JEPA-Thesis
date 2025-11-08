import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim

class SSIMMSELoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0)  # output is 0..1

        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss
