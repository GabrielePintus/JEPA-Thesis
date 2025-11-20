"""
Gradient-based trajectory optimization using Adam optimizer.
Optimizes action sequences by backpropagating through the latent dynamics model.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm
from .base_planner import BasePlanner


class GradientPlanner(BasePlanner):
    """
    Gradient-based planner using Adam optimizer.
    Optimizes actions by computing gradients through the JEPA predictor.
    """
    
    def __init__(
        self,
        jepa_model,
        action_dim: int,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        channel_mask: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        discount: float = 0.99,
        lr: float = 0.01,
        num_iterations: int = 100,
        grad_clip: Optional[float] = 1.0,
    ):
        """
        Args:
            jepa_model: Trained JEPA model
            action_dim: Dimensionality of action space
            action_bounds: (min, max) bounds for actions
            channel_mask: Optional mask for cost computation
            device: Device to run on
            discount: Discount factor for trajectory cost
            lr: Learning rate for Adam optimizer
            num_iterations: Number of optimization iterations
            grad_clip: Gradient clipping value (None to disable)
        """
        super().__init__(
            jepa_model=jepa_model,
            action_dim=action_dim,
            action_bounds=action_bounds,
            channel_mask=channel_mask,
            device=device,
            discount=discount,
        )
        
        self.lr = lr
        self.num_iterations = num_iterations
        self.grad_clip = grad_clip
        
    def _initialize_actions(
        self,
        horizon: int,
        initial_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Initialize action sequence."""
        if initial_actions is not None:
            # Warm-start from provided actions
            actions = initial_actions.clone().detach()
            if len(actions) < horizon:
                # Pad with zeros if too short
                padding = torch.zeros(
                    horizon - len(actions), self.action_dim,
                    device=self.device
                )
                actions = torch.cat([actions, padding], dim=0)
            elif len(actions) > horizon:
                # Truncate if too long
                actions = actions[:horizon]
        else:
            # Initialize with small random actions
            actions = torch.randn(
                horizon, self.action_dim,
                device=self.device
            ) * 0.1
            
        actions = self.clip_actions(actions)
        actions.requires_grad_(True)
        
        return actions
    
    def optimize(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        horizon: int,
        initial_actions: Optional[torch.Tensor] = None,
        verbose: bool = False,
        log_frequency: int = 10,
    ) -> Dict[str, Any]:
        """
        Optimize action sequence using gradient descent.
        
        Args:
            z_init: Initial latent state (C, H, W) - should be 18 channels for JEPA
            z_goal: Goal latent state (C, H, W) - should be 18 channels for JEPA
            horizon: Planning horizon
            initial_actions: Optional warm-start actions
            verbose: Print optimization progress
            log_frequency: Print every k iterations
            
        Returns:
            Dictionary with optimized actions, trajectory, and cost history
        """
        # Validate inputs
        if z_init.dim() == 4 and z_init.shape[0] == 1:
            z_init = z_init.squeeze(0)
            if verbose:
                print(f"Squeezed z_init from batch dimension, new shape: {z_init.shape}")
        
        if z_goal.dim() == 4 and z_goal.shape[0] == 1:
            z_goal = z_goal.squeeze(0)
            if verbose:
                print(f"Squeezed z_goal from batch dimension, new shape: {z_goal.shape}")
        
        # Check channel count
        if z_init.shape[0] != 18:
            raise ValueError(
                f"z_init has {z_init.shape[0]} channels, expected 18 (16 visual + 2 proprio). "
                f"Make sure you're using model.encode_state(state, frame), not just model.visual_encoder(frame). "
                f"Got shape: {z_init.shape}"
            )
        
        if z_goal.shape[0] != 18:
            raise ValueError(
                f"z_goal has {z_goal.shape[0]} channels, expected 18 (16 visual + 2 proprio). "
                f"Make sure you're using model.encode_state(state, frame), not just model.visual_encoder(frame). "
                f"Got shape: {z_goal.shape}"
            )
        
        # Move to device
        z_init = z_init.to(self.device)
        z_goal = z_goal.to(self.device)
        # Initialize actions
        actions = self._initialize_actions(horizon, initial_actions)
        
        # Create optimizer
        optimizer = torch.optim.Adam([actions], lr=self.lr)
        
        # Optimization loop
        cost_history = []
        
        progressbar = tqdm(range(self.num_iterations), disable=not verbose)
        for iteration in progressbar:
            optimizer.zero_grad()
            
            # Rollout WITH gradients enabled for optimization
            trajectory = self.rollout.rollout(z_init, actions, enable_grad=True)
            cost = self.rollout.trajectory_cost(trajectory, z_goal, self.discount)
            
            # Backward pass
            cost.backward()
            
            # Gradient clipping
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([actions], self.grad_clip)
            
            # Optimization step
            optimizer.step()
            
            # Project actions to valid bounds
            # with torch.no_grad():
            actions.data = self.clip_actions(actions.data)
            
            # Log progress
            cost_val = cost.item()
            cost_history.append(cost_val)
            
            if verbose and (iteration % log_frequency == 0 or iteration == self.num_iterations - 1):
                grad_norm = actions.grad.norm().item() if actions.grad is not None else 0.0
                # print(f"Iter {iteration:4d}/{self.num_iterations}: "
                #       f"Cost = {cost_val:.6f}, Grad norm = {grad_norm:.6f}")
                progressbar.set_postfix(cost=cost_val, grad_norm=grad_norm)
        
        # Final evaluation
        with torch.no_grad():
            final_trajectory = self.rollout.rollout(z_init, actions)
            final_cost = self.rollout.trajectory_cost(final_trajectory, z_goal, self.discount)
        
        return {
            'actions': actions.detach(),
            'trajectory': final_trajectory.detach(),
            'cost': final_cost.item(),
            'cost_history': cost_history,
            'num_iterations': self.num_iterations,
        }


class GradientPlannerWithMomentum(GradientPlanner):
    """
    Enhanced gradient planner with momentum and adaptive learning rate.
    Uses SGD with momentum instead of Adam for potentially better convergence.
    """
    
    def __init__(
        self,
        jepa_model,
        action_dim: int,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        channel_mask: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        discount: float = 0.99,
        lr: float = 0.01,
        momentum: float = 0.9,
        num_iterations: int = 100,
        grad_clip: Optional[float] = 1.0,
        lr_decay: float = 0.995,
    ):
        """
        Args:
            momentum: Momentum coefficient for SGD
            lr_decay: Learning rate decay per iteration
        """
        super().__init__(
            jepa_model=jepa_model,
            action_dim=action_dim,
            action_bounds=action_bounds,
            channel_mask=channel_mask,
            device=device,
            discount=discount,
            lr=lr,
            num_iterations=num_iterations,
            grad_clip=grad_clip,
        )
        
        self.momentum = momentum
        self.lr_decay = lr_decay
        
    def optimize(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        horizon: int,
        initial_actions: Optional[torch.Tensor] = None,
        verbose: bool = False,
        log_frequency: int = 10,
    ) -> Dict[str, Any]:
        """Optimize using SGD with momentum and learning rate decay."""
        # Initialize actions
        actions = self._initialize_actions(horizon, initial_actions)
        
        # Create optimizer with momentum
        optimizer = torch.optim.SGD(
            [actions],
            lr=self.lr,
            momentum=self.momentum
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.lr_decay
        )
        
        # Optimization loop
        cost_history = []
        
        progressbar = tqdm(range(self.num_iterations), disable=not verbose)
        for iteration in progressbar:
            optimizer.zero_grad()
            
            # Rollout WITH gradients enabled for optimization
            trajectory = self.rollout.rollout(z_init, actions, enable_grad=True)
            cost = self.rollout.trajectory_cost(trajectory, z_goal, self.discount)
            
            # Backward pass
            cost.backward()
            
            # Gradient clipping
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([actions], self.grad_clip)
            
            # Optimization step
            optimizer.step()
            scheduler.step()
            
            # Project actions to valid bounds
            # with torch.no_grad():
            actions.data = self.clip_actions(actions.data)
            
            # Log progress
            cost_val = cost.item()
            cost_history.append(cost_val)
            
            if verbose and (iteration % log_frequency == 0 or iteration == self.num_iterations - 1):
                current_lr = scheduler.get_last_lr()[0]
                grad_norm = actions.grad.norm().item() if actions.grad is not None else 0.0
                # print(f"Iter {iteration:4d}/{self.num_iterations}: "
                #       f"Cost = {cost_val:.6f}, LR = {current_lr:.6f}, "
                #       f"Grad norm = {grad_norm:.6f}")
                progressbar.set_description(
                    f"Iter {iteration:4d}/{self.num_iterations}: "
                    f"Cost = {cost_val:.6f}, LR = {current_lr:.6f}, "
                    f"Grad norm = {grad_norm:.6f}"
                )
        
        # Final evaluation
        with torch.no_grad():
            final_trajectory = self.rollout.rollout(z_init, actions)
            final_cost = self.rollout.trajectory_cost(final_trajectory, z_goal, self.discount)
        
        return {
            'actions': actions.detach(),
            'trajectory': final_trajectory.detach(),
            'cost': final_cost.item(),
            'cost_history': cost_history,
            'num_iterations': self.num_iterations,
            'final_lr': scheduler.get_last_lr()[0],
        }