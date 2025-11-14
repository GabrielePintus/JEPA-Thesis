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
        goal_dist = self._dist(z_T, z_goal)  # (B,)
        components["goal"] = self.cfg.goal_weight * goal_dist

        # Soft trajectory-wide goal distance
        if self.cfg.soft_goal_weight > 0.0:
            # (B, T1, C, H, W) vs (B, 1, C,H,W) broadcast
            z_goal_t = z_goal.unsqueeze(1)  # (B,1,C,H,W)
            # Manual MSE, supports broadcasting correctly
            err = (latents - z_goal_t) ** 2         # (B, T1, C, H, W)
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

        # Action smoothness: ||a_{t+1} - a_t||^2
        if self.cfg.action_smooth_weight > 0.0 and T > 1:
            da = actions[:, 1:] - actions[:, :-1]  # (B, T-1, A)
            action_smooth = (da ** 2).mean(dim=(1, 2))  # (B,)
            components["action_smooth"] = self.cfg.action_smooth_weight * action_smooth

        # Latent smoothness: ||z_{t+1} - z_t||^2
        if self.cfg.latent_smooth_weight > 0.0 and T1 > 1:
            dz = latents[:, 1:] - latents[:, :-1]  # (B, T, C, H, W)
            latent_smooth = (dz ** 2).mean(dim=(1, 2, 3, 4))  # (B,)
            components["latent_smooth"] = self.cfg.latent_smooth_weight * latent_smooth

        # Extra user-defined costs
        if self.cfg.extra_costs is not None:
            for name, (w, fn) in self.cfg.extra_costs.items():
                extra = fn(latents, actions, z_goal)  # (B,)
                components[name] = w * extra

        # Sum all components
        total = torch.zeros_like(components["goal"])
        for v in components.values():
            total = total + v

        return total, components


# -------------------------------------------------------------------
# Base Planner
# -------------------------------------------------------------------

class BasePlanner(abc.ABC):
    """
    Base interface for latent-space planners operating on PLDMEncoder.

    All planners optimize an action sequence A_{0:T-1} ∈ R^{T×A} to minimize
    a latent-space cost between rollout(z0, A) and z_goal.
    """

    def __init__(
        self,
        model: Any,               # PLDMEncoder-like
        cost_fn: BaseCostFunction,
        horizon: int,
        action_dim: int,
        action_bounds: Optional[Tuple[float, float]] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.cost_fn = cost_fn
        self.horizon = horizon
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.device = device or next(model.parameters()).device

        freeze_jepa(self.model)

    @abc.abstractmethod
    def plan(
        self,
        z0: torch.Tensor,            # (B, C, H, W)
        z_goal: torch.Tensor,        # (B, C, H, W) or (1, C, H, W)
        num_iterations: int,
        init_actions: Optional[torch.Tensor] = None,  # (B, T, A)
        verbose: bool = False,
        return_trajectory: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs the planning algorithm.

        Returns:
            {
              "actions": (B, T, A) optimal actions,
              "cost":    (B,) final costs,
              "latents": (optional) (B, T+1, C, H, W)
              "history": (optional) list of diagnostics per iteration
            }
        """
        raise NotImplementedError()

    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_bounds is None:
            return actions
        low, high = self.action_bounds
        return torch.clamp(actions, low, high)


# -------------------------------------------------------------------
# Gradient-based planner (Adam on actions)
# -------------------------------------------------------------------

class GradientPlanner(BasePlanner):
    """
    Gradient-based planner:
      - Optimizes actions with Adam
      - Backpropagates through PLDMEncoder.predictor and cost_fn
      - Model weights are frozen.
    """

    def __init__(
        self,
        model: Any,
        cost_fn: BaseCostFunction,
        horizon: int,
        action_dim: int,
        action_bounds: Optional[Tuple[float, float]] = None,
        device: Optional[torch.device] = None,
        lr: float = 0.05,
        adam_eps: float = 1e-8,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
    ):
        super().__init__(
            model=model,
            cost_fn=cost_fn,
            horizon=horizon,
            action_dim=action_dim,
            action_bounds=action_bounds,
            device=device,
        )
        self.lr = lr
        self.adam_eps = adam_eps
        self.adam_betas = adam_betas

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

        B = z0.shape[0]
        T = self.horizon
        A = self.action_dim

        if init_actions is None:
            actions = torch.zeros(B, T, A, device=self.device)
        else:
            actions = init_actions.to(self.device)
            if actions.shape != (B, T, A):
                raise ValueError(f"init_actions has wrong shape {actions.shape}, expected {(B, T, A)}")

        actions = actions.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam(
            [actions], lr=self.lr, betas=self.adam_betas, eps=self.adam_eps
        )

        history = []

        progressbar = tqdm(range(num_iterations), desc="Planning", disable=not verbose)
        for it in progressbar:
            optimizer.zero_grad(set_to_none=True)

            latents = rollout_latent(self.model, z0, actions, detach_model=False)
            cost, components = self.cost_fn(latents, actions, z_goal)
            loss = cost.mean()

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                actions.data = self._clip_actions(actions.data)

            if verbose:
                info = {
                    "iter": it,
                    "loss": float(loss.detach().cpu()),
                }
                # log some components if present
                for k, v in components.items():
                    info[k] = float(v.mean().detach().cpu())
                history.append(info)
                progressbar.set_postfix(info)

        # Final rollout
        with torch.no_grad():
            latents = rollout_latent(self.model, z0, actions.detach(), detach_model=False)
            final_cost, _ = self.cost_fn(latents, actions.detach(), z_goal)

        out = {
            "actions": actions.detach(),  # (B, T, A)
            "cost": final_cost.detach(),  # (B,)
        }
        if return_trajectory:
            out["latents"] = latents.detach()
        if verbose:
            out["history"] = history
        return out


# -------------------------------------------------------------------
# Path Integral Control (PIC) planner
# -------------------------------------------------------------------

class PICPlanner(BasePlanner):
    """
    Path Integral Control (sampling-based):
      - Maintains mean action sequence μ
      - At each iteration:
          * Sample K noisy action sequences around μ
          * Roll out latents, compute costs
          * Compute weights w_i ∝ exp(-β (J_i - J_min))
          * Update μ = Σ_i w_i A_i
      - No gradients, runs entirely under torch.no_grad().
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
    ):
        super().__init__(
            model=model,
            cost_fn=cost_fn,
            horizon=horizon,
            action_dim=action_dim,
            action_bounds=action_bounds,
            device=device,
        )
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.beta = beta
        self.init_std = init_std

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
            mu = torch.zeros(T, A, device=self.device)
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

        progressbar = tqdm(range(num_iterations), desc="PIC Planning", disable=not verbose)
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
            min_idx = torch.argmin(cost)
            min_cost = cost[min_idx]
            if best_cost is None or min_cost < best_cost:
                best_cost = min_cost
                best_actions = actions[min_idx].clone()
                best_latents = latents[min_idx].clone()

            # Compute PIC weights
            J = cost  # (K,)
            J_min = J.min()
            weights = torch.exp(-self.beta * (J - J_min))
            weights = weights / (weights.sum() + 1e-8)  # (K,)

            # Update mean μ
            mu = (weights.view(-1, 1, 1) * actions).sum(dim=0)  # (T, A)
            mu = self._clip_actions(mu)

            if verbose:
                info = {
                    "iter": it,
                    "mean_cost": float(J.mean().cpu()),
                    "min_cost": float(J_min.cpu()),
                    "best_cost_so_far": float(best_cost.cpu()),
                }
                history.append(info)
                progressbar.set_postfix(info)

        # Final outputs
        out = {
            "actions": mu.unsqueeze(0),          # (1, T, A)
            "cost": best_cost.view(1).detach(),  # (1,)
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
    Wrapper that uses a planner (GradientPlanner or PICPlanner)
    to perform Model Predictive Control (MPC) with replanning every k steps.

    Flow:
      1. Encode current state/frame → latent z_t
      2. Encode goal → latent z_goal (once)
      3. Run planner from z_t over full horizon H
      4. Execute only first k actions in the real env
      5. Repeat until episode ends or goal reached.

    The environment interface is assumed to be:
        next_state, next_frame, done, info = env.step(action)
    where:
        - next_state: (D_state,) tensor or numpy, including (x, y, vx, vy) normalized
        - next_frame: (3, H, W) tensor in [0,1]

    You can adapt this to your own environment API.
    """

    def __init__(
        self,
        model: Any,                 # PLDMEncoder-like
        planner: BasePlanner,
        horizon: int,
        replan_interval: int,       # k steps before replanning
        action_dim: int,
        state_to_tensor: Callable[[Any], torch.Tensor],   # env_state -> (1, D_state)
        frame_to_tensor: Callable[[Any], torch.Tensor],   # env_frame -> (1, 3, H, W)
        goal_encoder: Optional[
            Callable[[Any, Any], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        goal_tolerance: float = 0.0,   # in latent space (MSE distance)
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.planner = planner
        self.horizon = horizon
        self.replan_interval = replan_interval
        self.action_dim = action_dim
        self.state_to_tensor = state_to_tensor
        self.frame_to_tensor = frame_to_tensor
        self.goal_encoder = goal_encoder  # optional custom encoder for goal
        self.goal_tolerance = goal_tolerance
        self.device = device or next(model.parameters()).device

        freeze_jepa(self.model)

    @torch.no_grad()
    def _encode_goal(
        self,
        goal_state: Optional[Any],
        goal_frame: Any,
    ) -> torch.Tensor:
        """
        Encode goal into latent space.

        If goal_encoder is provided, use it; else:
          - goal_state is ignored and we pass zeros for vx,vy
            (you can change this to match your training distribution).
        """
        if self.goal_encoder is not None:
            z0_goal, z_goal = self.goal_encoder(goal_state, goal_frame)
            return z_goal

        # Default: only visual frame, zero velocity
        frame = self.frame_to_tensor(goal_frame).to(self.device)  # (1, 3, H, W)
        # vx,vy zeros
        state = torch.zeros(1, 2, device=self.device)  # (1, 2) for (vx, vy)
        z_goal = encode_latent(self.model, state, frame)  # (1, C, H_z, W_z)
        return z_goal

    def run(
        self,
        env: Any,
        initial_state: Any,
        initial_frame: Any,
        goal_frame: Any,
        max_steps: int,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run MPC loop on an environment.

        Args:
            env: environment with .step(action) method.
            initial_state: env-specific state (e.g. numpy array)
            initial_frame: env-specific frame/image
            goal_frame: target frame
            max_steps: maximum number of real env steps
        """
        device = self.device

        # Encode goal once
        z_goal = self._encode_goal(None, goal_frame).to(device)

        # Initialize
        state = initial_state
        frame = initial_frame

        all_actions = []
        goal_reached = False

        for t_env in range(0, max_steps, self.replan_interval):
            # Encode current state/frame → z_t
            s_tensor = self.state_to_tensor(state).to(device)       # (1, D_state)
            f_tensor = self.frame_to_tensor(frame).to(device)       # (1, 3, H, W)
            z_t = encode_latent(self.model, s_tensor, f_tensor)     # (1, C, H_z, W_z)

            # Check goal distance in latent space (optional stopping)
            if self.goal_tolerance > 0.0:
                dist = F.mse_loss(z_t, z_goal)
                if dist.item() <= self.goal_tolerance:
                    goal_reached = True
                    if verbose:
                        print(f"[MPC] Goal reached at step {t_env}, latent MSE={dist.item():.4f}")
                    break

            # Plan from current latent state
            plan_result = self.planner.plan(
                z0=z_t,
                z_goal=z_goal,
                num_iterations=10,   # can be parameterized
                init_actions=None,
                verbose=verbose,
                return_trajectory=False,
            )
            planned_actions = plan_result["actions"][0]  # (T, A)

            # Execute first k actions
            for k in range(self.replan_interval):
                if t_env + k >= max_steps:
                    break

                a = planned_actions[k].detach().cpu().numpy()
                next_state, next_frame, done, info = env.step(a)

                all_actions.append(a)

                state = next_state
                frame = next_frame

                # Re-encode to check goal early (optional)
                if self.goal_tolerance > 0.0:
                    s_tensor = self.state_to_tensor(state).to(device)
                    f_tensor = self.frame_to_tensor(frame).to(device)
                    z_t_now = encode_latent(self.model, s_tensor, f_tensor)
                    dist_now = F.mse_loss(z_t_now, z_goal)
                    if dist_now.item() <= self.goal_tolerance:
                        goal_reached = True
                        if verbose:
                            print(f"[MPC] Goal reached inside horizon at step {t_env + k}, "
                                  f"latent MSE={dist_now.item():.4f}")
                        break

                if done:
                    if verbose:
                        print(f"[MPC] Environment done at step {t_env + k}")
                    break

            if goal_reached or done:
                break

        return {
            "actions": all_actions,   # list of np.array actions
            "goal_reached": goal_reached,
        }
