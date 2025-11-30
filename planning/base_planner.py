"""
Base planner class with modular cost functions for latent space trajectory optimization.
Provides the interface and shared utilities for all planning algorithms.

Cost Function Design:
- goal_cost: Distance from final predicted state to goal (penalizes not reaching goal)
- trajectory_cost: Sum of distances along entire trajectory (encourages staying close to goal)
- smoothness_cost: Penalizes large action changes (encourages smooth trajectories)

Total cost = goal_weight * goal_cost + traj_weight * trajectory_cost + smooth_weight * smoothness_cost
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CostConfig:
    """Configuration for cost function computation."""
    # Cost weights
    goal_weight: float = 1.0           # Weight for final state distance to goal
    trajectory_weight: float = 0.0      # Weight for cumulative trajectory distance
    smoothness_weight: float = 0.0      # Weight for action smoothness penalty
    
    # Smoothness penalty type
    smoothness_order: int = 1           # 1 = velocity (action diff), 2 = acceleration (second diff)
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.goal_weight >= 0, "goal_weight must be non-negative"
        assert self.trajectory_weight >= 0, "trajectory_weight must be non-negative"
        assert self.smoothness_weight >= 0, "smoothness_weight must be non-negative"
        assert self.smoothness_order in [1, 2], "smoothness_order must be 1 or 2"
        assert self.goal_weight + self.trajectory_weight > 0, \
            "At least one of goal_weight or trajectory_weight must be positive"


class LatentRollout(nn.Module):
    """
    Wrapper for latent space dynamics rollout using JEPA predictor.
    Handles action encoding and state prediction with modular cost computation.
    """
    
    def __init__(
        self,
        jepa_model,
        action_dim: int,
        channel_mask: Optional[torch.Tensor] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            jepa_model: Trained JEPA model with predictor and action encoder
            action_dim: Dimensionality of action space
            channel_mask: Optional mask for cost computation (C,) - which channels to use
                         e.g., only visual channels (first 16) or all 18 channels
            device: Device to run on
        """
        super().__init__()
        self.jepa = jepa_model
        self.action_dim = action_dim
        self.device = device
        
        # Channel mask for cost computation (e.g., only visual channels)
        if channel_mask is not None:
            self.register_buffer('channel_mask', channel_mask.to(device))
        else:
            self.channel_mask = None
            
        # Move JEPA to device and eval mode
        self.jepa = self.jepa.to(device)
        self.jepa.eval()
        
    def predict_step(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        enable_grad: bool = False
    ) -> torch.Tensor:
        """
        Predict next latent state given current state and action.
        
        Args:
            z: Current latent state (B, C, H, W) or (C, H, W)
            action: Action to apply (B, action_dim) or (action_dim,)
            enable_grad: If True, allow gradients to flow (for gradient-based planning)
            
        Returns:
            z_next: Predicted next state (B, C, H, W) or (C, H, W)
        """
        # Handle single state (no batch dim)
        unsqueezed = False
        if z.dim() == 3:
            z = z.unsqueeze(0)
            unsqueezed = True
        
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        # Ensure action is on same device as z
        action = action.to(z.device)
            
        # JEPA predictor expects eval mode for BatchNorm
        was_training = self.jepa.training
        self.jepa.eval()
        
        if enable_grad:
            z_next = self.jepa.predict_state(z, action)
        else:
            with torch.no_grad():
                z_next = self.jepa.predict_state(z, action)
        
        if was_training:
            self.jepa.train()
            
        if unsqueezed:
            z_next = z_next.squeeze(0)
            
        return z_next
    
    def rollout(
        self,
        z_init: torch.Tensor,
        actions: torch.Tensor,
        enable_grad: bool = False
    ) -> torch.Tensor:
        """
        Rollout trajectory from initial state using action sequence.
        
        Args:
            z_init: Initial latent state (C, H, W)
            actions: Action sequence (T, action_dim) or (B, T, action_dim)
            enable_grad: If True, allow gradients to flow
            
        Returns:
            trajectory: Latent trajectory (T+1, C, H, W) or (B, T+1, C, H, W)
        """
        # Handle batched actions
        if actions.dim() == 3:
            B, T, _ = actions.shape
            z_init = z_init.unsqueeze(0).expand(B, -1, -1, -1)
            
            trajectory = [z_init]
            z = z_init
            
            for t in range(T):
                z = self.predict_step(z, actions[:, t], enable_grad=enable_grad)
                trajectory.append(z)
                
            return torch.stack(trajectory, dim=1)  # (B, T+1, C, H, W)
            
        else:
            # Single trajectory
            T, _ = actions.shape
            trajectory = [z_init]
            z = z_init
            
            for t in range(T):
                z = self.predict_step(z, actions[t], enable_grad=enable_grad)
                trajectory.append(z)
                
            return torch.stack(trajectory, dim=0)  # (T+1, C, H, W)
    
    def _apply_channel_mask(
        self,
        z: torch.Tensor,
        z_goal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply channel mask to states for cost computation."""
        if self.channel_mask is not None:
            if z.dim() == 4:
                z_masked = z[:, self.channel_mask]
            else:
                z_masked = z[self.channel_mask]
            
            if z_goal.dim() == 4:
                z_goal_masked = z_goal[:, self.channel_mask]
            else:
                z_goal_masked = z_goal[self.channel_mask]
        else:
            z_masked = z
            z_goal_masked = z_goal
            
        return z_masked, z_goal_masked
    
    def state_distance(
        self,
        z: torch.Tensor,
        z_goal: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Euclidean distance between states (with channel masking).
        
        Args:
            z: State(s) - (C, H, W), (B, C, H, W), or (T, C, H, W)
            z_goal: Goal state (C, H, W)
            
        Returns:
            distance: Scalar for single state, or (B,) / (T,) for batched
        """
        z_masked, z_goal_masked = self._apply_channel_mask(z, z_goal)
        
        squared_dist = (z_masked - z_goal_masked) ** 2

        # # Isometry
        # h_masked = self.jepa.isometry(z_masked)
        # h_goal_masked = self.jepa.isometry(z_goal_masked)
        # squared_dist = (h_masked - h_goal_masked) ** 2
        
        if z.dim() == 3:
            # Single state -> scalar
            dist = squared_dist.sum().sqrt()
        else:
            # Batched -> per-sample distance
            dims = tuple(range(1, squared_dist.dim()))
            dist = squared_dist.sum(dim=dims).sqrt()
    
        # Apply transform per provar
        # dist = 1 - torch.exp(-5 * dist)

        return dist

    def goal_cost(
        self,
        trajectory: torch.Tensor,
        z_goal: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cost based on final state distance to goal.
        
        Args:
            trajectory: (T+1, C, H, W) or (B, T+1, C, H, W)
            z_goal: Goal state (C, H, W)
            
        Returns:
            cost: Scalar or (B,)
        """
        if trajectory.dim() == 5:
            # Batched: (B, T+1, C, H, W) -> final states (B, C, H, W)
            final_states = trajectory[:, -1]
        else:
            # Single: (T+1, C, H, W) -> final state (C, H, W)
            final_states = trajectory[-1]
            
        return self.state_distance(final_states, z_goal)
    
    def trajectory_cost(
        self,
        trajectory: torch.Tensor,
        z_goal: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cumulative cost along entire trajectory (no discount).
        
        Args:
            trajectory: (T+1, C, H, W) or (B, T+1, C, H, W)
            z_goal: Goal state (C, H, W)
            
        Returns:
            cost: Scalar or (B,)
        """
        if trajectory.dim() == 5:
            # Batched: (B, T+1, C, H, W)
            B, T_plus_1, C, H, W = trajectory.shape
            costs = torch.zeros(B, device=trajectory.device, dtype=trajectory.dtype)
            
            for t in range(T_plus_1):
                costs = costs + self.state_distance(trajectory[:, t], z_goal)
                
        else:
            # Single: (T+1, C, H, W)
            T_plus_1 = trajectory.shape[0]
            costs = torch.tensor(0.0, device=trajectory.device, dtype=trajectory.dtype)
            
            for t in range(T_plus_1):
                costs = costs + self.state_distance(trajectory[t], z_goal)
                
        return costs
    
    def smoothness_cost(
        self,
        actions: torch.Tensor,
        order: int = 1,
        min_action_value: float = 1e-2,
    ) -> torch.Tensor:
        """
        Compute action smoothness penalty.
        
        Args:
            actions: Action sequence (T, action_dim) or (B, T, action_dim)
            order: 1 for velocity (first diff), 2 for acceleration (second diff)
            
        Returns:
            cost: Scalar or (B,)
        """
        # Actions should be at least min_action_value on average to avoid zero division
        action_magnitude_penalty = (min_action_value - actions.abs()).clamp(min=0).mean()

        if actions.dim() == 3:
            # Batched: (B, T, action_dim)
            if order == 1:
                # Penalize action changes (velocity)
                diffs = actions[:, 1:] - actions[:, :-1]
            else:
                # Penalize acceleration
                diffs = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
            
            smooth_loss = (diffs ** 2).sum(dim=(1, 2))
        else:
            # Single: (T, action_dim)
            if order == 1:
                diffs = actions[1:] - actions[:-1]
            else:
                diffs = actions[2:] - 2 * actions[1:-1] + actions[:-2]
            
            smooth_loss = (diffs ** 2).sum()

        return smooth_loss + action_magnitude_penalty
    
    def total_cost(
        self,
        trajectory: torch.Tensor,
        z_goal: torch.Tensor,
        actions: torch.Tensor,
        config: CostConfig
    ) -> torch.Tensor:
        """
        Compute total weighted cost.
        
        Args:
            trajectory: Latent trajectory (T+1, C, H, W) or (B, T+1, C, H, W)
            z_goal: Goal state (C, H, W)
            actions: Action sequence (T, action_dim) or (B, T, action_dim)
            config: Cost configuration with weights
            
        Returns:
            total_cost: Weighted sum of costs
        """
        cost = torch.tensor(0.0, device=trajectory.device, dtype=trajectory.dtype)
        
        if config.goal_weight > 0:
            cost = cost + config.goal_weight * self.goal_cost(trajectory, z_goal)
            
        if config.trajectory_weight > 0:
            cost = cost + config.trajectory_weight * self.trajectory_cost(trajectory, z_goal)
            
        if config.smoothness_weight > 0:
            cost = cost + config.smoothness_weight * self.smoothness_cost(
                actions, order=config.smoothness_order
            )
            
        return cost
    
    def forward(
        self,
        z_init: torch.Tensor,
        actions: torch.Tensor,
        z_goal: torch.Tensor,
        config: CostConfig,
        enable_grad: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete forward pass: rollout + cost computation.
        
        Args:
            z_init: Initial state (C, H, W)
            actions: Action sequence (T, action_dim) or (B, T, action_dim)
            z_goal: Goal state (C, H, W)
            config: Cost configuration
            enable_grad: Whether to enable gradient computation
            
        Returns:
            cost: Total trajectory cost
            trajectory: Full latent trajectory
        """
        trajectory = self.rollout(z_init, actions, enable_grad=enable_grad)
        cost = self.total_cost(trajectory, z_goal, actions, config)
        return cost, trajectory


class BasePlanner(ABC):
    """
    Abstract base class for trajectory optimization planners.
    Defines the common interface for all planning algorithms.
    """
    
    def __init__(
        self,
        jepa_model,
        action_dim: int,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        channel_mask: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        cost_config: Optional[CostConfig] = None,
    ):
        """
        Args:
            jepa_model: Trained JEPA model
            action_dim: Dimensionality of action space
            action_bounds: (min, max) bounds for actions
            channel_mask: Optional mask for cost computation
            device: Device to run on
            cost_config: Configuration for cost function (defaults to goal_cost only)
        """
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.device = device
        self.cost_config = cost_config or CostConfig()
        
        # Create rollout module
        self.rollout = LatentRollout(
            jepa_model=jepa_model,
            action_dim=action_dim,
            channel_mask=channel_mask,
            device=device
        )
        
    def clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Clip actions to valid bounds."""
        return torch.clamp(actions, self.action_bounds[0], self.action_bounds[1])
    
    @abstractmethod
    def optimize(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        horizon: int,
        initial_actions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize action sequence to reach goal from initial state.
        
        Args:
            z_init: Initial latent state (C, H, W)
            z_goal: Goal latent state (C, H, W)
            horizon: Planning horizon (number of actions)
            initial_actions: Optional warm-start actions (horizon, action_dim)
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Dictionary containing:
                - 'actions': Optimized action sequence (horizon, action_dim)
                - 'trajectory': Predicted latent trajectory (horizon+1, C, H, W)
                - 'cost': Final trajectory cost (scalar)
                - 'cost_history': List of costs during optimization
                - Additional algorithm-specific information
        """
        pass
    
    def evaluate_actions(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        actions: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Evaluate a given action sequence.
        
        Args:
            z_init: Initial state (C, H, W)
            z_goal: Goal state (C, H, W)
            actions: Action sequence to evaluate (T, action_dim)
            
        Returns:
            Dictionary with trajectory, cost breakdown, and per-step distances
        """
        actions = self.clip_actions(actions)
        
        with torch.no_grad():
            trajectory = self.rollout.rollout(z_init, actions)
            
            # Compute individual cost components
            goal_cost = self.rollout.goal_cost(trajectory, z_goal).item()
            traj_cost = self.rollout.trajectory_cost(trajectory, z_goal).item()
            smooth_cost = self.rollout.smoothness_cost(
                actions, order=self.cost_config.smoothness_order
            ).item()
            total_cost = self.rollout.total_cost(
                trajectory, z_goal, actions, self.cost_config
            ).item()
            
            # Per-step distances
            per_step_distances = []
            for t in range(len(trajectory)):
                dist = self.rollout.state_distance(trajectory[t], z_goal).item()
                per_step_distances.append(dist)
        
        return {
            'trajectory': trajectory,
            'total_cost': total_cost,
            'goal_cost': goal_cost,
            'trajectory_cost': traj_cost,
            'smoothness_cost': smooth_cost,
            'per_step_distances': per_step_distances,
            'final_distance': per_step_distances[-1],
        }