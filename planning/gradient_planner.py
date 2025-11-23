"""
Gradient-based trajectory optimization using Adam optimizer.
Optimizes action sequences by backpropagating through the latent dynamics model.

Features:
- Adam optimizer with configurable learning rate
- Optional noise injection for escaping local minima (wider basins of attraction)
- Gradient clipping for stability
- Progress tracking with tqdm
"""

import torch
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm

from .base_planner import BasePlanner, CostConfig


class GradientPlanner(BasePlanner):
    """
    Gradient-based planner using Adam optimizer with optional noise regularization.
    
    The noise injection helps escape local minima and find wider basins of attraction
    in the cost landscape. This can be seen as a form of simulated annealing or
    stochastic gradient descent with explicit noise.
    """
    
    def __init__(
        self,
        jepa_model,
        action_dim: int,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        channel_mask: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        cost_config: Optional[CostConfig] = None,
        # Optimization parameters
        initial_lr: float = 0.1,
        final_lr: Optional[float] = None,
        num_iterations: int = 100,
        grad_clip: Optional[float] = 1.0,
        # Noise regularization
        noise_std: float = 0.0,
        noise_decay: float = 0.995,
        min_noise: float = 0.0,
    ):
        """
        Args:
            jepa_model: Trained JEPA model
            action_dim: Dimensionality of action space
            action_bounds: (min, max) bounds for actions
            channel_mask: Optional mask for cost computation
            device: Device to run on
            cost_config: Cost function configuration
            lr: Learning rate for Adam optimizer
            num_iterations: Number of optimization iterations
            grad_clip: Gradient clipping value (None to disable)
            noise_std: Initial standard deviation of noise added to actions
            noise_decay: Decay rate for noise per iteration
            min_noise: Minimum noise standard deviation
        """
        super().__init__(
            jepa_model=jepa_model,
            action_dim=action_dim,
            action_bounds=action_bounds,
            channel_mask=channel_mask,
            device=device,
            cost_config=cost_config,
        )
        
        self.initial_lr = initial_lr
        self.final_lr = final_lr if final_lr is not None else initial_lr
        self.num_iterations = num_iterations
        self.grad_clip = grad_clip
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        self.min_noise = min_noise
        
    def _initialize_actions(
        self,
        horizon: int,
        initial_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Initialize action sequence."""
        if initial_actions is not None:
            actions = initial_actions.clone().detach().to(self.device)
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
    
    def _validate_inputs(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate and prepare input latent states."""
        # Handle extra batch dimensions
        if z_init.dim() == 4 and z_init.shape[0] == 1:
            z_init = z_init.squeeze(0)
            
        if z_goal.dim() == 4 and z_goal.shape[0] == 1:
            z_goal = z_goal.squeeze(0)
        
        # Check channel count (expecting 18 = 16 visual + 2 proprio)
        expected_channels = 18
        if z_init.shape[0] != expected_channels:
            raise ValueError(
                f"z_init has {z_init.shape[0]} channels, expected {expected_channels} "
                f"(16 visual + 2 proprio). Use model.encode_state(state, frame). "
                f"Got shape: {z_init.shape}"
            )
        
        if z_goal.shape[0] != expected_channels:
            raise ValueError(
                f"z_goal has {z_goal.shape[0]} channels, expected {expected_channels}. "
                f"Got shape: {z_goal.shape}"
            )
        
        # Move to device
        z_init = z_init.to(self.device)
        z_goal = z_goal.to(self.device)
        
        return z_init, z_goal
    
    def optimize(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        horizon: int,
        initial_actions: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Optimize action sequence using gradient descent with Adam.
        
        Args:
            z_init: Initial latent state (C, H, W) - 18 channels for JEPA
            z_goal: Goal latent state (C, H, W) - 18 channels for JEPA
            horizon: Planning horizon
            initial_actions: Optional warm-start actions
            verbose: Show progress bar
            
        Returns:
            Dictionary with optimized actions, trajectory, and cost history
        """
        # Validate inputs
        z_init, z_goal = self._validate_inputs(z_init, z_goal, verbose)
        
        # Initialize actions
        actions = self._initialize_actions(horizon, initial_actions)
        
        # Create optimizer
        optimizer = torch.optim.Adam([actions], lr=self.initial_lr)

        # Learning rate scheduler
        if self.initial_lr != self.final_lr:
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=self.final_lr / self.initial_lr,
                total_iters=self.num_iterations
            )
        
        # Tracking
        cost_history = []
        best_cost = float('inf')
        best_actions = None
        current_noise = self.noise_std
        
        # Optimization loop
        pbar = tqdm(range(self.num_iterations), disable=not verbose, desc="Gradient Opt")
        
        for iteration in pbar:
            optimizer.zero_grad()
            
            # Add noise for regularization (if enabled)
            if current_noise > 0:
                noise = torch.randn_like(actions) * current_noise
                actions_noisy = self.clip_actions(actions + noise)
            else:
                actions_noisy = actions
            
            # Rollout WITH gradients enabled
            trajectory = self.rollout.rollout(z_init, actions_noisy, enable_grad=True)
            cost = self.rollout.total_cost(trajectory, z_goal, actions_noisy, self.cost_config)
            
            # Backward pass
            cost.backward()
            
            # Gradient clipping
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([actions], self.grad_clip)
            
            # Optimization step
            optimizer.step()
            
            # Project to valid bounds
            actions.data = self.clip_actions(actions.data)
            
            # Track best solution
            cost_val = cost.item()
            if cost_val < best_cost:
                best_cost = cost_val
                best_actions = actions.detach().clone()
            
            # Decay noise
            current_noise = max(self.min_noise, current_noise * self.noise_decay)
            
            # Log progress
            cost_history.append(cost_val)

            # Step learning rate scheduler
            if self.initial_lr != self.final_lr:
                lr_scheduler.step()
            
            if verbose:
                grad_norm = actions.grad.norm().item() if actions.grad is not None else 0.0
                pbar.set_postfix({
                    'cost': f'{cost_val:.4f}',
                    'best': f'{best_cost:.4f}',
                    'grad': f'{grad_norm:.4f}',
                    'noise': f'{current_noise:.4f}'
                })
        
        # Use best actions found
        if best_actions is None:
            best_actions = actions.detach()
        
        # Final evaluation
        with torch.no_grad():
            final_trajectory = self.rollout.rollout(z_init, best_actions)
            final_cost = self.rollout.total_cost(
                final_trajectory, z_goal, best_actions, self.cost_config
            )
            
            # Compute cost breakdown
            goal_cost = self.rollout.goal_cost(final_trajectory, z_goal)
            traj_cost = self.rollout.trajectory_cost(final_trajectory, z_goal)
            smooth_cost = self.rollout.smoothness_cost(
                best_actions, order=self.cost_config.smoothness_order
            )
        
        return {
            'actions': best_actions,
            'trajectory': final_trajectory.detach(),
            'cost': final_cost.item(),
            'cost_history': cost_history,
            'num_iterations': self.num_iterations,
            # Cost breakdown
            'goal_cost': goal_cost.item(),
            'trajectory_cost': traj_cost.item(),
            'smoothness_cost': smooth_cost.item(),
        }