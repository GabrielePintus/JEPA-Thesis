# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def off_diagonal(m: torch.Tensor) -> torch.Tensor:
    n = m.size(0)
    return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class TemporalVICRegLoss(nn.Module):
    """
    Temporal VICReg (PLDM-style) regularizer:

      L = L_sim
        + alpha * L_var_time(Z_true)       # variance over time per (b, feature)
        + beta  * L_cov_batch_per_t(Z_true)# off-diag batch covariance per time step
        + delta * L_time_smooth(Z_true)    # ||Z_{t+1}-Z_t||^2

    Accepts sequences:
      Z_pred, Z_true: (T,B,C,H,W) or (T,B,D)
    """
    def __init__(self, alpha=25.0, beta=1.0, delta=1.0, gamma=1.0, eps=1e-4):
        super().__init__()
        self.alpha, self.beta, self.delta = alpha, beta, delta
        self.gamma, self.eps = gamma, eps

    def _flatten_features(self, Z: torch.Tensor) -> torch.Tensor:
        # (T,B,C,H,W) -> (T,B,D), or pass-through if already (T,B,D)
        if Z.dim() == 5:
            T,B,C,H,W = Z.shape
            return Z.view(T, B, C*H*W)
        if Z.dim() == 3:
            return Z
        raise ValueError(f"Expected (T,B,C,H,W) or (T,B,D), got {Z.shape}")

    def _L_sim(self, Zp: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        # MSE averaged over T,B,D
        return F.mse_loss(Zp, Z)

    def _L_var_time(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: (T,B,D). Var over TIME per (b,j) -> (B,D)
        var_t = Z.var(dim=0, unbiased=False)
        std_t = torch.sqrt(var_t + self.eps)
        return F.relu(self.gamma - std_t).mean()

    def _L_cov_batch_per_t(self, Z: torch.Tensor) -> torch.Tensor:
        # Off-diagonal covariance across B for each time t, averaged over t, normalized by D
        T,B,D = Z.shape
        acc = 0.0
        for t in range(T):
            zt = Z[t] - Z[t].mean(dim=0)       # (B,D)
            cov = (zt.T @ zt) / max(1, B-1)    # (D,D)
            acc += (off_diagonal(cov).pow(2).sum() / D)
        return acc / T

    def _L_time_smooth(self, Z: torch.Tensor) -> torch.Tensor:
        # ||Z_{t+1}-Z_t||^2 averaged over (t,b)
        if Z.size(0) < 2: 
            return Z.new_tensor(0.0)
        diffs = Z[1:] - Z[:-1]
        return diffs.pow(2).mean()

    def forward(self, Z_pred: torch.Tensor, Z_true: torch.Tensor) -> dict:
        Zp = self._flatten_features(Z_pred)   # (T,B,D)
        Zt = self._flatten_features(Z_true)   # (T,B,D)

        L_sim  = self._L_sim(Zp, Zt)
        L_var  = self._L_var_time(Zt)
        L_cov  = self._L_cov_batch_per_t(Zt)
        L_tsim = self._L_time_smooth(Zt)

        loss = L_sim + self.alpha*L_var + self.beta*L_cov + self.delta*L_tsim
        return {
            "loss": loss,
            "loss_sim": L_sim,
            "loss_var_time": L_var,
            "loss_cov_batch_t": L_cov,
            "loss_time_smooth": L_tsim,
        }
