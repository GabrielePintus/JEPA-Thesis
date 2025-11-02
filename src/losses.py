import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class TemporalVICRegLoss(nn.Module):
    """
    Temporal VICReg loss (variance + covariance terms) consistent with
    the official PLDM implementation.

    Args:
        std_coeff:  weight for variance (std) regularizer
        cov_coeff:  weight for covariance regularizer
        std_coeff_t: weight for temporal variance regularizer
        cov_coeff_t: weight for temporal covariance regularizer
        std_margin: target std deviation (gamma in paper)
        std_margin_t: same but for temporal term
        adjust_cov: whether to divide covariance loss by (D-1)
    """

    def __init__(
        self,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        std_coeff_t: float = 25.0,
        cov_coeff_t: float = 1.0,
        std_margin: float = 1.0,
        std_margin_t: float = 1.0,
        adjust_cov: bool = True,
    ):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.std_coeff_t = std_coeff_t
        self.cov_coeff_t = cov_coeff_t
        self.std_margin = std_margin
        self.std_margin_t = std_margin_t
        self.adjust_cov = adjust_cov

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        z: (B, T, D)
        """
        # compute in float32 for stability under bf16 AMP
        z32 = z.float()

        B, T, _ = z32.shape
        # ----- Batch-wise stats (need B >= 2) -----
        if B >= 2:
            std_b = self.std_loss(z32.permute(1, 0, 2), across_time=False)
            cov_b = self.cov_loss(z32.permute(1, 0, 2), across_time=False)
            std_b_term = self.std_coeff * std_b.mean()
            cov_b_term = self.cov_coeff * cov_b.mean()
        else:
            std_b_term = z32.new_zeros(())
            cov_b_term = z32.new_zeros(())

        # ----- Temporal stats (need T >= 2) -----
        if T >= 2:
            std_t = self.std_loss(z32, across_time=True)
            cov_t = self.cov_loss(z32, across_time=True)
            std_t_term = self.std_coeff_t * std_t.mean()
            cov_t_term = self.cov_coeff_t * cov_t.mean()
        else:
            std_t_term = z32.new_zeros(())
            cov_t_term = z32.new_zeros(())

        total_loss = std_b_term + cov_b_term + std_t_term + cov_t_term

        # cast back to original dtype
        return {
            "std":   std_b_term.to(z.dtype),
            "cov":   cov_b_term.to(z.dtype),
            "std_t": std_t_term.to(z.dtype),
            "cov_t": cov_t_term.to(z.dtype),
            "loss":  total_loss.to(z.dtype),
        }

    # ------------------------------------------------------------------
    def std_loss(self, x: torch.Tensor, across_time: bool = False) -> torch.Tensor:
        """
        Compute variance (std) regularization term.

        Args:
            x: tensor of shape (T, B, D) if across_time=False,
               or (B, T, D) if across_time=True
        """
        x = x - x.mean(dim=1, keepdim=True)  # zero-mean per sample
        std = torch.sqrt(x.var(dim=1, unbiased=False) + 1e-4)

        margin = self.std_margin_t if across_time else self.std_margin
        std_loss = torch.mean(F.relu(margin - std), dim=-1)
        return std_loss

    # ------------------------------------------------------------------
    def cov_loss(self, x: torch.Tensor, across_time: bool = False) -> torch.Tensor:
        """
        Memory-safe covariance penalty.
        x: (T,B,D) if across_time=False, (B,T,D) if across_time=True
        Returns shape (1,) (averaged).
        """
        # Need at least 2 samples along the sample axis (= dim=1)
        if x.size(1) < 2:
            return torch.zeros((1,), device=x.device, dtype=x.dtype)

        # center across the sample axis
        x = x - x.mean(dim=1, keepdim=True)

        # collapse leading axis to make one big sample set
        BT = x.shape[0] * x.shape[1]
        D  = x.shape[-1]
        x_flat = x.reshape(BT, D)

        n = x_flat.size(0)
        if n < 2:
            return torch.zeros((1,), device=x.device, dtype=x.dtype)
        denom = float(n - 1)

        # chunked over features to avoid (D x D) allocation
        chunk = 512
        total = x_flat.new_tensor(0.0)
        parts = 0

        for start in range(0, D, chunk):
            end = min(start + chunk, D)
            c = end - start
            x_chunk = x_flat[:, start:end]              # (n, c)
            cov_blk = (x_chunk.T @ x_chunk) / denom     # (c, c)
            diag = torch.diagonal(cov_blk, dim1=0, dim2=1)
            offdiag_sq_sum = cov_blk.pow(2).sum() - diag.pow(2).sum()

            block = offdiag_sq_sum / c
            if self.adjust_cov and c > 1:
                block = block / (c - 1)

            total = total + block
            parts += 1

        cov = total / max(parts, 1)
        return cov.unsqueeze(0)

