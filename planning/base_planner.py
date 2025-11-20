"""
Base planner class with common functionality for latent space trajectory optimization.
Provides the interface and shared utilities for all planning algorithms.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod


class LatentRollout(nn.Module):
    """
    Wrapper for latent space dynamics rollout using JEPA predictor.
    Handles action encoding and state prediction.
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
            # Allow gradients to flow through for gradient-based planning
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
    
    def compute_cost(
        self,
        z: torch.Tensor,
        z_goal: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute cost between states (Euclidean distance in latent space).
        
        Args:
            z: Current state(s) - (C, H, W), (B, C, H, W), or (T, C, H, W)
            z_goal: Goal state (C, H, W)
            reduction: How to aggregate spatial dimensions ('mean', 'sum', or 'none')
            
        Returns:
            cost: Scalar cost or per-batch costs
        """
        # Apply channel mask if specified
        if self.channel_mask is not None:
            # Ensure z_goal has batch dim for indexing
            z_goal_expanded = z_goal.unsqueeze(0) if z_goal.dim() == 3 else z_goal
            z_masked = z[:, self.channel_mask] if z.dim() == 4 else z[self.channel_mask]
            z_goal_masked = z_goal_expanded[:, self.channel_mask].squeeze(0)
        else:
            z_masked = z
            z_goal_masked = z_goal
            
        # Compute squared Euclidean distance
        diff = z_masked - z_goal_masked
        squared_dist = (diff ** 2)
        
        # Aggregate spatial dimensions
        if reduction == 'mean':
            # For single state (C, H, W), return scalar
            if z.dim() == 3:
                cost = squared_dist.mean()
            else:
                # For batch (B, C, H, W), return (B,)
                dims = tuple(range(1, squared_dist.dim()))
                cost = squared_dist.mean(dim=dims)
        elif reduction == 'sum':
            if z.dim() == 3:
                cost = squared_dist.sum()
            else:
                dims = tuple(range(1, squared_dist.dim()))
                cost = squared_dist.sum(dim=dims)
        else:  # 'none'
            cost = squared_dist
            
        return cost
    
    def trajectory_cost(
        self,
        trajectory: torch.Tensor,
        z_goal: torch.Tensor,
        discount: float = 1.0
    ) -> torch.Tensor:
        """
        Compute total discounted cost over trajectory.
        
        Args:
            trajectory: Latent trajectory (T+1, C, H, W) or (B, T+1, C, H, W)
            z_goal: Goal state (C, H, W)
            discount: Discount factor for future costs
            
        Returns:
            total_cost: Scalar tensor for single trajectory, or (B,) tensor for batched
        """
        if trajectory.dim() == 5:
            # Batched trajectories (B, T+1, C, H, W)
            B, T, C, H, W = trajectory.shape
            
            # Compute per-timestep costs
            costs = []
            for t in range(T):
                cost_t = self.compute_cost(trajectory[:, t], z_goal)
                costs.append(cost_t * (discount ** t))
                
            return torch.stack(costs, dim=0).sum(dim=0)  # (B,)
            
        else:
            # Single trajectory (T+1, C, H, W)
            T, C, H, W = trajectory.shape
            
            # Accumulate scalar costs
            total_cost = torch.tensor(0.0, device=trajectory.device, dtype=trajectory.dtype)
            for t in range(T):
                cost_t = self.compute_cost(trajectory[t], z_goal)  # scalar
                total_cost = total_cost + cost_t * (discount ** t)
                
            return total_cost  # scalar tensor
    
    def forward(
        self,
        z_init: torch.Tensor,
        actions: torch.Tensor,
        z_goal: torch.Tensor,
        discount: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete forward pass: rollout + cost computation.
        
        Args:
            z_init: Initial state (C, H, W)
            actions: Action sequence (T, action_dim) or (B, T, action_dim)
            z_goal: Goal state (C, H, W)
            discount: Discount factor
            
        Returns:
            cost: Total trajectory cost
            trajectory: Full latent trajectory
        """
        trajectory = self.rollout(z_init, actions)
        cost = self.trajectory_cost(trajectory, z_goal, discount)
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
        discount: float = 0.99,
    ):
        """
        Args:
            jepa_model: Trained JEPA model
            action_dim: Dimensionality of action space
            action_bounds: (min, max) bounds for actions
            channel_mask: Optional mask for cost computation
            device: Device to run on
            discount: Discount factor for trajectory cost
        """
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.device = device
        self.discount = discount
        
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
            Dictionary with trajectory, cost, and per-step costs
        """
        actions = self.clip_actions(actions)
        cost, trajectory = self.rollout(z_init, actions, z_goal, self.discount)
        
        # Compute per-step costs for analysis
        per_step_costs = []
        for t in range(len(trajectory)):
            step_cost = self.rollout.compute_cost(trajectory[t], z_goal)
            per_step_costs.append(step_cost.item())
            
        return {
            'trajectory': trajectory,
            'cost': cost.item(),
            'per_step_costs': per_step_costs,
            'final_distance': per_step_costs[-1],
        }