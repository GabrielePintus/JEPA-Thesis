# planning_pldm.py
#
# Latent-space planning on top of PLDMEncoder (JEPA-style)
# - Works purely in latent space z ∈ R^{C×H×W}
# - Supports:
#   * Generic CostFunction base + LatentGoalCost implementation
#   * Gradient-based planner (Adam on action sequence)
#   * Path Integral Control (PIC) planner
#   * Rollout in latent space using PLDMEncoder.predictor
#   * MPC wrapper that does replanning every k steps
#
# Assumes a trained PLDMEncoder with:
#   - encode_state(state, frame) -> z (B, 16, 26, 26)
#   - action_expander(action)    -> (B, 2, 26, 26)
#   - predictor(latent+action)   -> next latent (B, 16, 26, 26)
#
# You can adapt imports / typing to your project structure.

from __future__ import annotations
import abc
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

def freeze_jepa(model: nn.Module) -> None:
    """Freeze all JEPA parameters (no gradients for planning)."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


@torch.no_grad()
def encode_latent(
    model: Any,  # PLDMEncoder-like
    state: torch.Tensor,
    frame: torch.Tensor,
) -> torch.Tensor:
    """
    Encode (state, frame) → latent z using PLDMEncoder.encode_state.

    Args:
        model: PLDMEncoder instance (or compatible).
        state: (B, D_state) tensor. For PLDMEncoder, this should already
               be the velocity-only vector (vx, vy) if you follow training.
        frame: (B, 3, H, W) tensor in [0,1].

    Returns:
        z: (B, C, H_z, W_z)
    """
    return model.encode_state(state, frame)


def rollout_latent(
    model: Any,                # PLDMEncoder-like
    z0: torch.Tensor,          # (B, C, H, W)
    actions: torch.Tensor,     # (B or S, T, A)
    detach_model: bool = False
) -> torch.Tensor:
    """
    Roll out the latent dynamics using ConvPredictor.

    Args:
        model: PLDMEncoder-like object with .action_expander and .predictor.
        z0: initial latent (B, C, H, W).
        actions: action sequence:
            - Shape (B, T, A): per-batch sequences
            - Or shape (S, T, A) with B=1: S samples sharing same z0
        detach_model: if True, disables gradients through the model
                      (useful for PIC, where we don't need backprop).

    Returns:
        latents: (B or S, T+1, C, H, W) latent trajectory, including z0.
    """
    # Handle the case of S samples sharing a single z0
    if actions.dim() != 3:
        raise ValueError("actions must have shape (B, T, A)")

    B_z = z0.shape[0]
    B_a, T, A = actions.shape

    if B_z == 1 and B_a > 1:
        # Tile z0 for each sample
        z = z0.expand(B_a, -1, -1, -1).contiguous()
    elif B_z == B_a:
        z = z0
    else:
        raise ValueError(f"Incompatible batch sizes: z0 {z0.shape}, actions {actions.shape}")

    device = z.device
    dtype = z.dtype

    latents = torch.empty(B_a, T + 1, *z.shape[1:], device=device, dtype=dtype)
    latents[:, 0] = z

    # Choose context manager for model grads
    ctx = torch.no_grad if detach_model else torch.enable_grad
    with ctx():
        for t in range(T):
            a_t = actions[:, t]  # (B_a, A)
            # Expand action over spatial map
            a_map = model.action_expander(a_t)  # (B_a, 2, H, W)
            inp = torch.cat([z, a_map], dim=1)  # (B_a, C+2, H, W)
            z = model.predictor(inp)           # (B_a, C, H, W)
            latents[:, t + 1] = z

    return latents


# -------------------------------------------------------------------
# Cost functions
# -------------------------------------------------------------------

class BaseCostFunction(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for latent-space trajectory cost."""

    @abc.abstractmethod
    def forward(
        self,
        latents: torch.Tensor,  # (B, T+1, C, H, W)
        actions: torch.Tensor,  # (B, T, A)
        z_goal: torch.Tensor,   # (B, C, H, W) or (1, C, H, W)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            total_cost: (B,) cost per batch element
            components: dict of component costs (B,) each
        """
        raise NotImplementedError()


@dataclass
class LatentGoalCostConfig:
    distance_type: str = "mse"   # "mse" or "l2"
    goal_weight: float = 1.0     # main terminal cost weight

    # Trajectory-wide optional costs
    soft_goal_weight: float = 0.0        # distance to goal at intermediate steps
    action_smooth_weight: float = 0.0    # ||a_{t+1} - a_t||^2
    latent_smooth_weight: float = 0.0    # ||z_{t+1} - z_t||^2
    soft_goal_discount: float = 1.0      # discount factor for soft goal along time (<=1)

    # Additional user-defined costs: dict[name] = (weight, fn)
    # fn signature: fn(latents, actions, z_goal) -> (B,)
    extra_costs: Optional[Dict[str, Tuple[float, Callable]]] = None


class LatentGoalCost(BaseCostFunction):
    """
    Cost for planning in latent space:
      - main term: distance between final latent z_T and z_goal
      - optional trajectory terms: soft goal distance, smoothness in action/latent
      - extensible via user-defined extra cost functions.
    """

    def __init__(self, cfg: LatentGoalCostConfig):
        super().__init__()
        self.cfg = cfg

    def _dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute per-batch distance between x and y:
            x, y: (B, ...) same shape
        Returns:
            (B,)
        """
        if self.cfg.distance_type == "mse":
            return F.mse_loss(x, y, reduction="none").flatten(1).mean(dim=1)
        elif self.cfg.distance_type == "l2":
            return ((x - y) ** 2).flatten(1).sum(dim=1).sqrt()
        else:
            raise ValueError(f"Unknown distance_type: {self.cfg.distance_type}")

    def forward(
        self,
        latents: torch.Tensor,  # (B, T+1, C, H, W)
        actions: torch.Tensor,  # (B, T, A)
        z_goal: torch.Tensor,   # (B, C, H, W) or (1, C, H, W)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        B, T1, C, H, W = latents.shape
        T = T1 - 1

        if z_goal.shape[0] == 1 and B > 1:
            z_goal = z_goal.expand(B, -1, -1, -1)

        components: Dict[str, torch.Tensor] = {}

        # Terminal goal cost (MAIN TERM)
        z_T = latents[:, -1]  # (B, C, H, W)
        goal_dist = self._dist(z_T[:, :16], z_goal[:, :16])  # (B,)
        components["goal"] = self.cfg.goal_weight * goal_dist

        # Soft trajectory-wide goal distance
        if self.cfg.soft_goal_weight > 0.0:
            # (B, T1, C, H, W) vs (B, 1, C,H,W) broadcast
            z_goal_t = z_goal.unsqueeze(1)  # (B,1,C,H,W)
            # Manual MSE, supports broadcasting correctly
            err = (latents[:, :, :16] - z_goal_t[:, :, :16]) ** 2         # (B, T1, C, H, W)
            soft_dist = err.flatten(2).mean(dim=2)  # (B, T1)


            if self.cfg.soft_goal_discount < 1.0:
                # geometric discount over time
                # weights: 1, gamma, gamma^2, ...
                gamma = self.cfg.soft_goal_discount
                t_idx = torch.arange(T1, device=latents.device, dtype=latents.dtype)
                disc = gamma ** t_idx  # (T1,)
                soft_dist = (soft_dist * disc.unsqueeze(0)).sum(dim=1) / disc.sum()
            else:
                soft_dist = soft_dist.mean(dim=1)  # (B,)

            components["soft_goal"] = self.cfg.soft_goal_weight * soft_dist

        # Action smoothness
        if self.cfg.action_smooth_weight > 0.0:
            # ||a_{t+1} - a_t||^2
            diff = actions[:, 1:] - actions[:, :-1]  # (B, T-1, A)
            action_smooth = (diff ** 2).flatten(1).mean(dim=1)  # (B,)
            components["action_smooth"] = self.cfg.action_smooth_weight * action_smooth

        # Latent smoothness
        if self.cfg.latent_smooth_weight > 0.0:
            # ||z_{t+1} - z_t||^2
            diff_z = latents[:, 1:] - latents[:, :-1]  # (B, T, C, H, W)
            latent_smooth = (diff_z ** 2).flatten(1).mean(dim=1)  # (B,)
            components["latent_smooth"] = self.cfg.latent_smooth_weight * latent_smooth

        # Extra user-defined costs
        if self.cfg.extra_costs is not None:
            for name, (weight, fn) in self.cfg.extra_costs.items():
                cost_val = fn(latents, actions, z_goal)  # (B,)
                components[name] = weight * cost_val

        # Total cost
        total = sum(components.values())  # (B,)
        return total, components


# -------------------------------------------------------------------
# Planner base
# -------------------------------------------------------------------

class BasePlanner(metaclass=abc.ABCMeta):
    """Abstract base class for latent-space planners."""

    @abc.abstractmethod
    def plan(
        self,
        z0: torch.Tensor,           # (1, C, H, W) initial latent
        z_goal: torch.Tensor,       # (1, C, H, W) goal latent
        num_iterations: int,
        init_actions: Optional[torch.Tensor] = None,
        verbose: bool = False,
        return_trajectory: bool = False,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
              "actions": (1, T, A),
              "cost": (1,),
              "latents": (1, T+1, C, H, W),  # if return_trajectory=True
              "history": list[dict],          # if verbose=True
            }
        """
        raise NotImplementedError()

# -------------------------------------------------------------------
# Gradient-based planner (Adam) with best-action tracking
# -------------------------------------------------------------------

class GradientPlanner(BasePlanner):
    """
    Optimizes an action sequence in latent space using gradient descent (Adam).
    Tracks and returns the best action sequence encountered during optimization.
    """

    def __init__(
        self,
        model: Any,
        cost_fn: BaseCostFunction,
        horizon: int,
        action_dim: int,
        action_min: float = -1.0,
        action_max: float = 1.0,
        lr: float = 1e-2,
        device: Optional[torch.device] = None,
        noise: float = 0.0,
    ):
        self.model = model
        self.cost_fn = cost_fn
        self.horizon = horizon
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.lr = lr
        self.device = device or next(model.parameters()).device
        self.noise = noise

        freeze_jepa(self.model)

    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.clamp(actions, self.action_min, self.action_max)

    def plan(
        self,
        z0: torch.Tensor,              # (1, C, H, W)
        z_goal: torch.Tensor,          # (1, C, H, W)
        num_iterations: int = 50,
        init_actions: Optional[torch.Tensor] = None,
        verbose: bool = False,
        return_trajectory: bool = False,
        grad_clip: float = None,
    ) -> Dict[str, Any]:

        T, A = self.horizon, self.action_dim

        # Initialize action sequence
        if init_actions is not None:
            actions = init_actions.clone().detach().to(self.device)
            # if actions.shape != (1, T, A):
            #     raise ValueError(f"init_actions must have shape (1,{T},{A}), got {actions.shape}")
        else:
            # actions = torch.zeros(1, T, A, device=self.device)
            actions = 0.01 * torch.randn(1, T, A, device=self.device)

        actions.requires_grad_(True)

        optimizer = torch.optim.Adam([actions], lr=self.lr, weight_decay=0.0)
        history = []

        # -----------------------------
        # Best-seen actions tracking
        # -----------------------------
        best_cost = float("inf")
        best_actions = None
        best_latents = None

        progressbar = tqdm(range(num_iterations), desc="GradientPlanner", disable=not verbose)

        for it in progressbar:
            optimizer.zero_grad()

            # Clip (ensure valid range)
            with torch.no_grad():
                actions.data = self._clip_actions(actions.data)

            # Rollout latents
            latents = rollout_latent(self.model, z0, actions, detach_model=False)

            # Compute cost
            cost, components = self.cost_fn(latents, actions, z_goal)
            total_cost = cost.mean()

            # Backward
            total_cost.backward()

            # Optional gradient noise
            if self.noise > 0.0:
                with torch.no_grad():
                    actions.grad += self.noise * torch.randn_like(actions.grad)

            # Optional gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([actions], max_norm=grad_clip)

            # Adam update
            optimizer.step()

            # -----------------------------
            # Update best solution
            # -----------------------------
            with torch.no_grad():
                current_cost = total_cost.item()
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_actions = actions.detach().clone()
                    best_latents = latents.detach().clone()  # if trajectory needed

            # Verbose logging
            if verbose:
                info = {
                    "iter": it,
                    "cost": float(total_cost.detach().cpu()),
                    "grad_norm": float(actions.grad.norm().cpu()),
                }
                history.append(info)
                progressbar.set_postfix(info)

        # -----------------------------
        # Final: compute best clipped solution
        # -----------------------------
        with torch.no_grad():
            best_actions = self._clip_actions(best_actions)
            best_latents_final = rollout_latent(self.model, z0, best_actions, detach_model=True)
            best_cost_final, _ = self.cost_fn(best_latents_final, best_actions, z_goal)

        out = {
            "actions": best_actions,             # best (1, T, A)
            "cost": best_cost_final.detach(),    # (1,)
        }

        if return_trajectory:
            out["latents"] = best_latents_final

        if verbose:
            out["history"] = history

        return out




class CMAESPlanner(BasePlanner):
    """
    CMA-ES planner for latent-space planning on action sequences.

    - Optimizes a flattened action vector x ∈ R^{T * A}
    - Uses population-based sampling around a Gaussian N(m, σ^2 C)
    - Supports different covariance structures:
        * cov_mode="full"      → full covariance matrix C ∈ R^{D×D}
        * cov_mode="diag"      → diagonal covariance (separable CMA-ES)
        * cov_mode="blockdiag" → block-diagonal covariance, with block_size
                                 (e.g., block_size=action_dim = per-timestep full cov)

    Notes:
    - Minimizes the provided cost_fn(latents, actions, z_goal)
    - Uses rollout_latent with the given PLDMEncoder model
    - No gradients are used; everything runs under torch.no_grad()
    """

    def __init__(
        self,
        model: Any,
        cost_fn: BaseCostFunction,
        horizon: int,
        action_dim: int,
        action_bounds: Optional[Tuple[float, float]] = None,
        device: Optional[torch.device] = None,
        cov_mode: str = "full",         # "full", "diag", "blockdiag"
        block_size: Optional[int] = None,
        pop_size: Optional[int] = None, # λ (population size)
        sigma_init: float = 0.5,
        sigma_min: float = 1e-4,
        sigma_max: float = 2.0,
        use_best_sample: bool = True,
        cov_epsilon: float = 1e-8,      # jitter for numerical stability
    ):
        self.model = model
        self.cost_fn = cost_fn
        self.horizon = horizon
        self.action_dim = action_dim
        self.device = device or next(model.parameters()).device
        self.cov_mode = cov_mode.lower()
        self.block_size = block_size
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.use_best_sample = use_best_sample
        self.cov_epsilon = cov_epsilon

        if action_bounds is not None:
            self.action_min, self.action_max = action_bounds
        else:
            self.action_min, self.action_max = -1.0, 1.0

        freeze_jepa(self.model)

        # Dimension of the optimization vector
        self.dim = self.horizon * self.action_dim
        D = self.dim

        # Population size and number of elites μ
        if pop_size is None:
            # Standard CMA-ES default
            self.pop_size = 4 + int(3 * math.log(D + 1))
        else:
            self.pop_size = pop_size
        self.pop_size = max(self.pop_size, 4)
        self.mu = self.pop_size // 2

        # Recombination weights (logarithmic)
        weights = torch.tensor(
            [math.log(self.mu + 0.5) - math.log(i) for i in range(1, self.mu + 1)],
            dtype=torch.float32,
        )
        weights = weights / weights.sum()
        self.weights = weights.to(self.device)                      # (μ,)
        self.mu_eff = 1.0 / float((self.weights ** 2).sum().item()) # effective μ

        # Strategy parameters (standard CMA-ES settings)
        self.c_sigma = (self.mu_eff + 2) / (D + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0.0, math.sqrt((self.mu_eff - 1) / (D + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / D) / (D + 4 + 2 * self.mu_eff / D)
        self.c1 = 2 / ((D + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((D + 2) ** 2 + self.mu_eff),
        )

        # Expected norm of N(0, I) in D dimensions (for step-size adaptation)
        self.expected_norm = math.sqrt(D) * (1.0 - 1.0 / (4 * D) + 1.0 / (21 * D * D))

        # Block-diagonal mask (for cov_mode="blockdiag")
        if self.cov_mode == "blockdiag":
            if self.block_size is None:
                # Natural choice: each time-step (A dims) is its own block
                self.block_size = self.action_dim
            mask = torch.zeros(D, D, dtype=torch.float32)
            for start in range(0, D, self.block_size):
                end = min(D, start + self.block_size)
                mask[start:end, start:end] = 1.0
            self.block_mask = mask.to(self.device)
        else:
            self.block_mask = None

        if self.cov_mode not in ("full", "diag", "blockdiag"):
            raise ValueError(f"Unknown cov_mode '{cov_mode}'. Use 'full', 'diag', or 'blockdiag'.")

    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.clamp(actions, self.action_min, self.action_max)

    def _sample_population(
        self,
        mean: torch.Tensor,          # (D,)
        sigma: float,
        C_full: Optional[torch.Tensor],
        C_diag: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample population from N(mean, sigma^2 * C).
        Returns:
            x : (λ, D) sampled population in parameter space
            y : (λ, D) zero-mean steps in transformed space
            z : (λ, D) zero-mean steps in isotropic space (N(0, I))
        """
        lam = self.pop_size
        D = self.dim
        device = self.device

        z = torch.randn(lam, D, device=device)  # isotropic

        if self.cov_mode in ("full", "blockdiag"):
            # Cholesky factor of C: C = A A^T
            # Add a small jitter to ensure positive definiteness
            A = torch.linalg.cholesky(
                C_full + self.cov_epsilon * torch.eye(D, device=device)
            )  # (D, D)
            y = z @ A.T  # (λ, D)
        elif self.cov_mode == "diag":
            # Diagonal covariance
            std = torch.sqrt(torch.clamp(C_diag, min=1e-12))  # (D,)
            y = z * std.unsqueeze(0)  # (λ, D)
        else:
            raise RuntimeError("Invalid cov_mode in _sample_population")

        x = mean.unsqueeze(0) + sigma * y  # (λ, D)
        return x, y, z

    def plan(
        self,
        z0: torch.Tensor,           # (1, C, H, W)
        z_goal: torch.Tensor,       # (1, C, H, W)
        num_iterations: int,
        init_actions: Optional[torch.Tensor] = None,  # (1, T, A)
        verbose: bool = False,
        return_trajectory: bool = False,
    ) -> Dict[str, Any]:

        z0 = z0.to(self.device)
        z_goal = z_goal.to(self.device)

        if z0.shape[0] != 1:
            raise ValueError("CMAESPlanner currently assumes batch size B=1 for z0.")

        T = self.horizon
        A = self.action_dim
        D = self.dim
        lam = self.pop_size
        mu = self.mu
        w = self.weights  # (μ,)

        # Initial mean (flattened actions)
        if init_actions is not None:
            init_actions = init_actions.to(self.device)
            if init_actions.shape != (1, T, A):
                raise ValueError(f"init_actions must have shape (1, {T}, {A}), got {init_actions.shape}")
            mean = init_actions.view(D).clone().detach()
        else:
            mean = torch.zeros(D, device=self.device)

        # Initial step-size
        sigma = float(self.sigma_init)

        # Initial covariance
        if self.cov_mode in ("full", "blockdiag"):
            C_full = torch.eye(D, device=self.device)
            C_diag = None
        elif self.cov_mode == "diag":
            C_full = None
            C_diag = torch.ones(D, device=self.device)
        else:
            raise RuntimeError("Invalid cov_mode in plan")

        # Evolution paths
        p_sigma = torch.zeros(D, device=self.device)
        p_c = torch.zeros(D, device=self.device)

        best_cost = None
        best_x = None
        best_latents = None

        history = []
        progressbar = tqdm(range(num_iterations), desc="CMAESPlanner", disable=not verbose)

        with torch.no_grad():
            for it in progressbar:
                # ----------------------------
                # 1) Sample population
                # ----------------------------
                x, y, z = self._sample_population(mean, sigma, C_full, C_diag)  # (λ, D) each

                # ----------------------------
                # 2) Evaluate population
                # ----------------------------
                actions = x.view(lam, T, A)
                actions_clipped = self._clip_actions(actions)
                latents = rollout_latent(self.model, z0, actions_clipped, detach_model=True)
                cost, _ = self.cost_fn(latents, actions_clipped, z_goal)  # (λ,)

                J = cost  # rename for CMA notation
                # Track global best
                gen_best_idx = torch.argmin(J)
                gen_best_cost = J[gen_best_idx]
                if best_cost is None or gen_best_cost < best_cost:
                    best_cost = gen_best_cost
                    best_x = x[gen_best_idx].clone()
                    best_latents = latents[gen_best_idx].clone()

                # ----------------------------
                # 3) Sort by fitness
                # ----------------------------
                idx_sorted = torch.argsort(J)  # ascending (minimization)
                idx_mu = idx_sorted[:mu]

                x_mu = x[idx_mu]  # (μ, D)
                y_mu = y[idx_mu]  # (μ, D)
                z_mu = z[idx_mu]  # (μ, D)
                J_mu = J[idx_mu]  # (μ,)

                # ----------------------------
                # 4) Compute weighted means
                # ----------------------------
                # Weighted means in different spaces
                w_row = w.view(mu, 1)  # (μ, 1)
                y_w = (w_row * y_mu).sum(dim=0)  # (D,)
                z_w = (w_row * z_mu).sum(dim=0)  # (D,)
                mean = (w_row * x_mu).sum(dim=0)  # new mean (D,)

                # ----------------------------
                # 5) Update evolution paths
                # ----------------------------
                # Step-size path p_sigma is in isotropic / whitened space
                coeff_ps = math.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff)
                p_sigma = (1.0 - self.c_sigma) * p_sigma + coeff_ps * z_w

                norm_p_sigma = p_sigma.norm().item()
                # Heaviside for covariance path
                threshold = (1.4 + 2.0 / (D + 1.0)) * self.expected_norm
                h_sigma = 1.0 if norm_p_sigma < threshold else 0.0

                coeff_pc = math.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff)
                p_c = (1.0 - self.c_c) * p_c + h_sigma * coeff_pc * y_w

                # ----------------------------
                # 6) Adapt step-size σ
                # ----------------------------
                sigma *= math.exp(
                    (self.c_sigma / self.d_sigma) * (norm_p_sigma / self.expected_norm - 1.0)
                )
                sigma = float(max(self.sigma_min, min(self.sigma_max, sigma)))

                # ----------------------------
                # 7) Adapt covariance C
                # ----------------------------
                if self.cov_mode in ("full", "blockdiag"):
                    # rank-μ update: Σ_i w_i y_i y_i^T
                    w_sqrt = torch.sqrt(w).view(mu, 1)      # (μ,1)
                    y_wm = w_sqrt * y_mu                    # (μ, D)
                    C_mu = y_wm.T @ y_wm                    # (D, D)

                    # rank-1 from path
                    pc_outer = p_c.unsqueeze(1) @ p_c.unsqueeze(0)  # (D, D)

                    C_full = (
                        (1.0 - self.c1 - self.c_mu) * C_full
                        + self.c1 * pc_outer
                        + self.c_mu * C_mu
                    )

                    if self.cov_mode == "blockdiag" and self.block_mask is not None:
                        # Enforce block-diagonal structure
                        C_full = C_full * self.block_mask
                        # Ensure symmetry
                        C_full = 0.5 * (C_full + C_full.T)

                elif self.cov_mode == "diag":
                    # Only keep diagonal elements
                    # rank-μ diag update: Σ_i w_i y_i^2
                    y_sq = y_mu ** 2  # (μ, D)
                    C_mu_diag = (w_row * y_sq).sum(dim=0)  # (D,)

                    C_diag = (
                        (1.0 - self.c1 - self.c_mu) * C_diag
                        + self.c1 * (p_c ** 2)
                        + self.c_mu * C_mu_diag
                    )

                # ----------------------------
                # 8) Logging
                # ----------------------------
                if verbose:
                    info = {
                        "iter": it,
                        "mean_cost": float(J.mean().cpu()),
                        "min_cost": float(J.min().cpu()),
                        "best_cost_so_far": float(best_cost.cpu()),
                        "sigma": float(sigma),
                    }
                    history.append(info)
                    progressbar.set_postfix(info)

        # ----------------------------
        # Final result
        # ----------------------------
        if self.use_best_sample and best_x is not None:
            final_x = best_x
            final_cost = best_cost
            final_actions = final_x.view(1, T, A)
            out_latents = best_latents.unsqueeze(0) if (best_latents is not None and return_trajectory) else None
        else:
            # Use mean as final solution; recompute cost for it
            final_x = mean
            final_actions = final_x.view(1, T, A)
            final_actions_clipped = self._clip_actions(final_actions)
            with torch.no_grad():
                latents_final = rollout_latent(self.model, z0, final_actions_clipped, detach_model=True)
                final_cost, _ = self.cost_fn(latents_final, final_actions_clipped, z_goal)
            out_latents = latents_final if return_trajectory else None

        out = {
            "actions": self._clip_actions(final_actions),    # (1, T, A)
            "cost": final_cost.view(1).detach(),             # (1,)
        }
        if return_trajectory and out_latents is not None:
            out["latents"] = out_latents
        if verbose:
            out["history"] = history
        return out




class PICPlanner(BasePlanner):
    """
    Path Integral Control (sampling-based):
      - Maintains mean action sequence μ
      - At each iteration:
          * Sample K noisy action sequences around μ
          * Roll out latents, compute costs
          * Compute weights w_i ∝ exp(-β_eff (J_i - J_min))
          * Update μ = Σ_i w_i A_i
      - No gradients, runs entirely under torch.no_grad().

    Practical tweaks:
      - Optional cost normalization so that β is robust to scale.
      - Option to return either mean μ or the best sampled trajectory.
    """

    def __init__(
        self,
        model: Any,
        cost_fn: BaseCostFunction,
        horizon: int,
        action_dim: int,
        action_bounds: Optional[Tuple[float, float]] = None,
        device: Optional[torch.device] = None,
        num_samples: int = 256,
        noise_std: float = 0.3,
        beta: float = 1.0,
        init_std: float = 0.5,
        normalize_cost: bool = True,
        use_best_sample: bool = False,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.beta = beta
        self.init_std = init_std
        self.normalize_cost = normalize_cost
        self.use_best_sample = use_best_sample
        self.model = model
        self.cost_fn = cost_fn
        self.horizon = horizon
        self.action_dim = action_dim
        self.device = device or next(model.parameters()).device
        if action_bounds is not None:
            self.action_min, self.action_max = action_bounds
        else:
            self.action_min, self.action_max = -1.0, 1.0

    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.clamp(actions, self.action_min, self.action_max)

    def plan(
        self,
        z0: torch.Tensor,
        z_goal: torch.Tensor,
        num_iterations: int,
        init_actions: Optional[torch.Tensor] = None,
        verbose: bool = False,
        return_trajectory: bool = False,
    ) -> Dict[str, Any]:

        z0 = z0.to(self.device)
        z_goal = z_goal.to(self.device)

        if z0.shape[0] != 1:
            raise ValueError("PICPlanner currently assumes batch size B=1 for z0.")
        B = 1
        T = self.horizon
        A = self.action_dim

        # Initialize mean action sequence μ
        if init_actions is None:
            # Start with small random actions instead of exact zeros
            mu = self.init_std * torch.randn(T, A, device=self.device)
        else:
            init_actions = init_actions.to(self.device)
            if init_actions.shape != (B, T, A):
                raise ValueError(f"init_actions must have shape (1, T, A), got {init_actions.shape}")
            mu = init_actions[0].clone().detach()

        mu = self._clip_actions(mu)

        history = []
        best_actions = None
        best_cost = None
        best_latents = None

        progressbar = tqdm(range(num_iterations), desc="PICPlanner", disable=not verbose)
        for it in progressbar:
            # Sample K trajectories around mean μ
            noise = self.noise_std * torch.randn(self.num_samples, T, A, device=self.device)
            actions = mu.unsqueeze(0) + noise  # (K, T, A)
            actions = self._clip_actions(actions)

            with torch.no_grad():
                # Rollout for each sample; z0 is broadcast inside rollout_latent
                latents = rollout_latent(self.model, z0, actions, detach_model=True)
                cost, _ = self.cost_fn(latents, actions, z_goal)  # (K,)

            # Track best sample so far
            J = cost.detach()  # (K,)
            min_idx = torch.argmin(J)
            min_cost = J[min_idx]
            if best_cost is None or min_cost < best_cost:
                best_cost = min_cost
                best_actions = actions[min_idx].clone()
                best_latents = latents[min_idx].clone()

            # Compute PIC weights
            J_min = J.min()
            deltaJ = J - J_min  # (K,)

            if self.normalize_cost:
                # Adaptive effective β
                scale = deltaJ.std() + 1e-6
                beta_eff = self.beta / scale
            else:
                beta_eff = self.beta

            # Softmax for numerical stability
            weights = torch.softmax(-beta_eff * deltaJ, dim=0)  # (K,)

            # Update mean μ
            mu = (weights.view(-1, 1, 1) * actions).sum(dim=0)  # (T, A)
            mu = self._clip_actions(mu)

            if verbose:
                info = {
                    "iter": it,
                    "mean_cost": float(J.mean().cpu()),
                    "min_cost": float(J_min.cpu()),
                    "best_cost_so_far": float(best_cost.cpu()),
                    "beta_eff": float(beta_eff),
                }
                history.append(info)
                progressbar.set_postfix(info)

        # Final outputs
        if self.use_best_sample and best_actions is not None:
            action_seq = best_actions
        else:
            action_seq = mu

        out = {
            "actions": action_seq.unsqueeze(0),       # (1, T, A)
            "cost": best_cost.view(1).detach(),       # (1,)
        }
        if return_trajectory and best_latents is not None:
            out["latents"] = best_latents.unsqueeze(0)
        if verbose:
            out["history"] = history
        return out



# -------------------------------------------------------------------
# MPC wrapper
# -------------------------------------------------------------------

class MPCController:
    """
    MPC for PointMaze + JEPA latent-planning.

    Supports:
        - Passing raw (state, frame)
        - OR passing z0 directly
        - Passing goal_frame
        - OR passing z_goal directly

    Environment interface expected:
        state, frame = env.reset()
        next_state, next_frame, done, info = env.step(action)

    Where:
        state: (vx, vy)
        frame: (3, 64, 64) uint8 image
    """

    def __init__(
        self,
        model: Any,
        planner: BasePlanner,
        horizon: int,
        replan_interval: int,
        action_dim: int,
        device: Optional[torch.device] = None,
        goal_tolerance: float = 0.0,
        planning_iterations: int = 10,
        use_warm_start: bool = True,
    ):
        self.model = model
        self.planner = planner
        self.horizon = horizon
        self.replan_interval = replan_interval
        self.action_dim = action_dim
        self.goal_tolerance = goal_tolerance
        self.planning_iterations = planning_iterations
        self.use_warm_start = use_warm_start

        self.device = device or next(model.parameters()).device

        freeze_jepa(self.model)


