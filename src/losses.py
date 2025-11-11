import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict



# Inspired by: https://github.com/jolibrain/vicreg-loss
class VCRegLoss(nn.Module):
    def __init__(
        self,
        var_coeff: float = 15.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes the VICReg loss.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, representation_size].
            y: Features map.
                Shape of [batch_size, representation_size].

        ---
        Returns:
            The VICReg loss.
                Dictionary where values are of shape of [1,].
        """
        metrics = dict()
        metrics["var-loss"] = (
            self.var_coeff
            * (self.variance_loss(x, self.gamma) + self.variance_loss(y, self.gamma))
            / 2
        )
        metrics["cov-loss"] = (
            self.cov_coeff * (self.covariance_loss(x) + self.covariance_loss(y)) / 2
        )
        metrics["loss"] = sum(metrics.values())
        return metrics

    @staticmethod
    def variance_loss(x: torch.Tensor, gamma: float) -> torch.Tensor:
        """Computes the variance loss.
        Push the representations across the batch
        to be different between each other.
        Avoid the model to collapse to a single point.

        The gamma parameter is used as a threshold so that
        the model is no longer penalized if its std is above
        that threshold.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, representation_size].

        ---
        Returns:
            The variance loss.
                Shape of [1,].
        """
        x = x - x.mean(dim=0)
        print(x.shape)
        std = x.std(dim=0)
        print(std.shape)
        var_loss = F.relu(gamma - std)
        print(var_loss.shape)
        var_loss = var_loss.mean()
        return var_loss

    @staticmethod
    def covariance_loss(x: torch.Tensor) -> torch.Tensor:
        """Computes the covariance loss.
        Decorrelates the embeddings' dimensions, which pushes
        the model to capture more information per dimension.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, representation_size].

        ---
        Returns:
            The covariance loss.
                Shape of [1,].
        """
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (x.shape[0] - 1)
        cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / x.shape[1]
        return cov_loss
    




class TemporalVCRegLoss(nn.Module):
    def __init__(
        self,
        var_coeff: float = 15.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes the Temporal VICReg loss.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, temporal_dim, representation_size].
        ---
        Returns:
            The Temporal VICReg loss.
                Dictionary where values are of shape of [1,].
        """
        metrics = dict()
        metrics["var-loss"] = self.var_coeff * self.variance_loss(x, self.gamma)
        metrics["cov-loss"] = self.cov_coeff * self.covariance_loss(x)
        metrics["loss"] = sum(metrics.values())
        return metrics

    @staticmethod
    def variance_loss(x: torch.Tensor, gamma: float) -> torch.Tensor:
        """Computes the variance loss.
        Push the representations across the batch
        to be different between each other.
        Avoid the model to collapse to a single point.

        The gamma parameter is used as a threshold so that
        the model is no longer penalized if its std is above
        that threshold.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, representation_size].

        ---
        Returns:
            The variance loss.
                Shape of [1,].
        """
        x = x - x.mean(dim=0)
        std = x.std(dim=0)
        var_loss = F.relu(gamma - std)
        var_loss = var_loss.mean()
        return var_loss
    
    @staticmethod
    def covariance_loss(x: torch.Tensor) -> torch.Tensor:
        """Computes the covariance loss.
        Decorrelates the embeddings' dimensions, which pushes
        the model to capture more information per dimension.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, representation_size] or
                [batch_size, temporal_dim, representation_size].

        ---
        Returns:
            The covariance loss.
                Shape of [1,].
        """
        # Center x
        x = x - x.mean(dim=0, keepdim=True)

        if x.ndim == 2:
            # Original case: (B, D)
            cov = (x.T @ x) / (x.shape[0] - 1)
            cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / x.shape[1]
        
        elif x.ndim == 3:
            # New case: (B, T, D)
            B, T, D = x.shape
            
            # Compute covariance for each timestep: (B, T, D)^T @ (B, T, D) -> (T, D, D)
            cov = torch.einsum('btd,bte->tde', x, x) / (B - 1)

            # Zero out diagonals for all T covariance matrices
            eye = torch.eye(D, device=x.device, dtype=x.dtype).unsqueeze(0)  # (1, D, D)
            cov_no_diag = cov * (1 - eye)  # (T, D, D)
            
            # Compute loss per timestep and average
            cov_loss = cov_no_diag.pow(2).sum(dim=(1, 2)) / D  # (T,)
            cov_loss = cov_loss.mean()  # Average over time
        
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {x.ndim}D")
        
        return cov_loss






class TemporalVCRegLossOptimized(nn.Module):
    def __init__(
        self,
        var_coeff: float = 15.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes the Temporal VICReg loss.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, temporal_dim, representation_size].
        ---
        Returns:
            The Temporal VICReg loss.
                Dictionary where values are of shape of [1,].
        """
        # Center x
        x = x - x.mean(dim=0, keepdim=True)

        # Compute variance loss
        std = x.std(dim=0)
        var_loss = F.relu(self.gamma - std)
        var_loss = var_loss.mean()

        # Compute covariance loss
        if x.ndim == 2:
            # Original case: (B, D)
            cov = (x.T @ x) / (x.shape[0] - 1)
            cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / x.shape[1]
        
        elif x.ndim == 3:
            # New case: (B, T, D)
            B, T, D = x.shape
            
            # Compute covariance for each timestep: (B, T, D)^T @ (B, T, D) -> (T, D, D)
            cov = torch.einsum('btd,bte->tde', x, x) / (B - 1)

            # Zero out diagonals for all T covariance matrices
            eye = torch.eye(D, device=x.device, dtype=x.dtype).unsqueeze(0)  # (1, D, D)
            cov_no_diag = cov * (1 - eye)  # (T, D, D)
            
            # Compute loss per timestep and average
            cov_loss = cov_no_diag.pow(2).sum(dim=(1, 2)) / D  # (T,)
            cov_loss = cov_loss.mean()  # Average over time
        
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {x.ndim}D")
        
        loss = self.var_coeff * var_loss + self.cov_coeff * cov_loss

        metrics = {
            "var-loss": self.var_coeff * var_loss,
            "cov-loss": self.cov_coeff * cov_loss,
            "loss": loss
        }
        return metrics

