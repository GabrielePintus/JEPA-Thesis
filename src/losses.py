import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# class TemporalVICRegLoss(nn.Module):
#     """
#     Temporal VICReg loss (variance + covariance terms) consistent with
#     the official PLDM implementation.

#     Args:
#         std_coeff:  weight for variance (std) regularizer
#         cov_coeff:  weight for covariance regularizer
#         std_coeff_t: weight for temporal variance regularizer
#         cov_coeff_t: weight for temporal covariance regularizer
#         std_margin: target std deviation (gamma in paper)
#         std_margin_t: same but for temporal term
#         adjust_cov: whether to divide covariance loss by (D-1)
#     """

#     def __init__(
#         self,
#         std_coeff: float = 25.0,
#         cov_coeff: float = 1.0,
#         std_coeff_t: float = 25.0,
#         cov_coeff_t: float = 1.0,
#         std_margin: float = 1.0,
#         std_margin_t: float = 1.0,
#         adjust_cov: bool = True,
#     ):
#         super().__init__()
#         self.std_coeff = std_coeff
#         self.cov_coeff = cov_coeff
#         self.std_coeff_t = std_coeff_t
#         self.cov_coeff_t = cov_coeff_t
#         self.std_margin = std_margin
#         self.std_margin_t = std_margin_t
#         self.adjust_cov = adjust_cov

#     # ------------------------------------------------------------------
#     def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
#         """
#         z: (B, T, D)
#         """
#         # compute in float32 for stability under bf16 AMP
#         z32 = z.float()

#         B, T, _ = z32.shape
#         # ----- Batch-wise stats (need B >= 2) -----
#         if B >= 2:
#             std_b = self.std_loss(z32.permute(1, 0, 2), across_time=False)
#             cov_b = self.cov_loss(z32.permute(1, 0, 2), across_time=False)
#             std_b_term = self.std_coeff * std_b.mean()
#             cov_b_term = self.cov_coeff * cov_b.mean()
#         else:
#             std_b_term = z32.new_zeros(())
#             cov_b_term = z32.new_zeros(())

#         # ----- Temporal stats (need T >= 2) -----
#         if T >= 2:
#             std_t = self.std_loss(z32, across_time=True)
#             cov_t = self.cov_loss(z32, across_time=True)
#             std_t_term = self.std_coeff_t * std_t.mean()
#             cov_t_term = self.cov_coeff_t * cov_t.mean()
#         else:
#             std_t_term = z32.new_zeros(())
#             cov_t_term = z32.new_zeros(())

#         total_loss = std_b_term + cov_b_term + std_t_term + cov_t_term

#         # cast back to original dtype
#         return {
#             "std":   std_b_term.to(z.dtype),
#             "cov":   cov_b_term.to(z.dtype),
#             "std_t": std_t_term.to(z.dtype),
#             "cov_t": cov_t_term.to(z.dtype),
#             "loss":  total_loss.to(z.dtype),
#         }

#     # ------------------------------------------------------------------
#     def std_loss(self, x: torch.Tensor, across_time: bool = False) -> torch.Tensor:
#         """
#         Compute variance (std) regularization term.

#         Args:
#             x: tensor of shape (T, B, D) if across_time=False,
#                or (B, T, D) if across_time=True
#         """
#         x = x - x.mean(dim=1, keepdim=True)  # zero-mean per sample
#         std = torch.sqrt(x.var(dim=1, unbiased=False) + 1e-4)

#         margin = self.std_margin_t if across_time else self.std_margin
#         std_loss = torch.mean(F.relu(margin - std), dim=-1)
#         return std_loss

#     # ------------------------------------------------------------------
#     def cov_loss(self, x: torch.Tensor, across_time: bool = False) -> torch.Tensor:
#         """
#         Memory-safe covariance penalty.
#         x: (T,B,D) if across_time=False, (B,T,D) if across_time=True
#         Returns shape (1,) (averaged).
#         """
#         # Need at least 2 samples along the sample axis (= dim=1)
#         if x.size(1) < 2:
#             return torch.zeros((1,), device=x.device, dtype=x.dtype)

#         # center across the sample axis
#         x = x - x.mean(dim=1, keepdim=True)

#         # collapse leading axis to make one big sample set
#         BT = x.shape[0] * x.shape[1]
#         D  = x.shape[-1]
#         x_flat = x.reshape(BT, D)

#         n = x_flat.size(0)
#         if n < 2:
#             return torch.zeros((1,), device=x.device, dtype=x.dtype)
#         denom = float(n - 1)

#         # chunked over features to avoid (D x D) allocation
#         chunk = 512
#         total = x_flat.new_tensor(0.0)
#         parts = 0

#         for start in range(0, D, chunk):
#             end = min(start + chunk, D)
#             c = end - start
#             x_chunk = x_flat[:, start:end]              # (n, c)
#             cov_blk = (x_chunk.T @ x_chunk) / denom     # (c, c)
#             diag = torch.diagonal(cov_blk, dim1=0, dim2=1)
#             offdiag_sq_sum = cov_blk.pow(2).sum() - diag.pow(2).sum()

#             block = offdiag_sq_sum / c
#             if self.adjust_cov and c > 1:
#                 block = block / (c - 1)

#             total = total + block
#             parts += 1

#         cov = total / max(parts, 1)
#         return cov.unsqueeze(0)



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class TemporalVICRegLoss(nn.Module):
    """
    Temporal VICReg as in PLDM (without similarity terms):
      - std/cov across batch at a single time step (default t=0)
      - std_t/cov_t across time on steps t>=1

    Inputs:
        z: (B, T, D) sequence embeddings.
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
        eps: float = 1e-4,
        use_time_index_for_batch_terms: int = 0,  # which t to use for the batch VICReg terms
        projector: Optional[nn.Module] = None,    # e.g., nn.Identity() or an MLP
    ):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.std_coeff_t = std_coeff_t
        self.cov_coeff_t = cov_coeff_t
        self.std_margin = std_margin
        self.std_margin_t = std_margin_t
        self.adjust_cov = adjust_cov
        self.eps = eps
        self.t0 = use_time_index_for_batch_terms
        self.projector = projector if projector is not None else nn.Identity()

    # --------------------------------------------------------------
    @staticmethod
    def _offdiag_cov_loss(x_centered: torch.Tensor, denom_features: int, adjust_cov: bool) -> torch.Tensor:
        """
        Compute off-diagonal covariance penalty per group.
        x_centered: (G, K, D)
        Returns: (G,) — loss per group
        """
        K = x_centered.shape[1]
        cov = torch.einsum("gkd,gke->gde", x_centered, x_centered) / max(K - 1, 1)

        diag_sq_sum = (cov.diagonal(dim1=-2, dim2=-1) ** 2).sum(dim=-1)
        full_sq_sum = (cov ** 2).sum(dim=(-1, -2))
        offdiag = (full_sq_sum - diag_sq_sum) / denom_features

        if adjust_cov and denom_features > 1:
            offdiag = offdiag / (denom_features - 1)

        return offdiag  # (G,)

    # --------------------------------------------------------------
    def _std_hinge(self, x_centered: torch.Tensor, margin: float) -> torch.Tensor:
        """
        Hinge loss to enforce std >= margin.
        x_centered: (G, K, D)
        Returns: (G,)
        """
        std = torch.sqrt(x_centered.var(dim=1, unbiased=False) + self.eps)
        return F.relu(margin - std).mean(dim=-1)

    # --------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute VICReg-style regularization across batch and time.

        Args:
            z: (B, T, D)
        """
        assert z.ndim == 3, f"Expected (B, T, D), got {z.shape}"
        B, T, D = z.shape
        assert T >= 2, "Need at least two timesteps for temporal terms."

        # Optional projection (MLP or Identity)
        z_proj = self.projector(z)
        Dp = z_proj.shape[-1]

        # -----------------------------
        # Batch-level VICReg (across batch at time t0)
        # -----------------------------
        t0 = min(self.t0, T - 1)
        xb = z_proj[:, t0, :]  # (B, Dp)
        xb = xb - xb.mean(dim=0, keepdim=True)  # center across batch
        xb = xb.unsqueeze(0)  # (1, B, Dp)

        std_loss = torch.zeros((), device=z.device)
        cov_loss = torch.zeros((), device=z.device)
        if self.std_coeff:
            std_loss = self._std_hinge(xb, self.std_margin).mean()
        if self.cov_coeff:
            cov_loss = self._offdiag_cov_loss(xb, Dp, self.adjust_cov).mean()

        # -----------------------------
        # Temporal VICReg (across time within each sample)
        # -----------------------------
        xt = z_proj[:, 1:, :] - z_proj[:, 1:, :].mean(dim=1, keepdim=True)  # (B, T-1, Dp)
        std_loss_t = torch.zeros((), device=z.device)
        cov_loss_t = torch.zeros((), device=z.device)

        if self.std_coeff_t:
            std_loss_t = self._std_hinge(xt, self.std_margin_t).mean()
        if self.cov_coeff_t:
            cov_loss_t = self._offdiag_cov_loss(xt, Dp, self.adjust_cov).mean()

        # -----------------------------
        # Combine losses
        # -----------------------------
        total_loss = (
            self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
            + self.std_coeff_t * std_loss_t
            + self.cov_coeff_t * cov_loss_t
        )

        return {
            "loss": total_loss,
            "std_loss": std_loss.detach(),
            "cov_loss": cov_loss.detach(),
            "std_loss_t": std_loss_t.detach(),
            "cov_loss_t": cov_loss_t.detach(),
        }

