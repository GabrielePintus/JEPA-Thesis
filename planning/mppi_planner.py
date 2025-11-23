"""
Model Predictive Path Integral (MPPI) Control planner.
Information-theoretic sampling-based approach with temperature-weighted cost averaging.

Features:
- Importance sampling with exponentially weighted costs
- Temperature parameter for exploration/exploitation trade-off
- Elite sample retention for faster convergence
- Adaptive temperature based on cost variance
"""

import torch
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm

from .base_planner import BasePlanner, CostConfig


class MPPIPlanner(BasePlanner):
    """
    MPPI planner for trajectory optimization.
    Uses importance sampling with exponentially weighted costs.
    
    This is useful when:
    - You want smooth, natural-looking trajectories
    - The cost landscape is relatively smooth
    - You want controllable exploration via temperature
    """
    
    def __init__(
        self,
        jepa_model,
        action_dim: int,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        channel_mask: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        cost_config: Optional[CostConfig] = None,
        # MPPI parameters
        num_samples: int = 100,
        num_iterations: int = 10,
        temperature: float = 1.0,
        noise_sigma: float = 0.5,
        noise_decay: float = 0.95,
        # Elite retention
        num_elites: int = 10,
        elite_weight: float = 0.3,
        adaptive_temp: bool = True,
    ):
        """
        Args:
            jepa_model: Trained JEPA model
            action_dim: Dimensionality of action space
            action_bounds: (min, max) bounds for actions
            channel_mask: Optional mask for cost computation
            device: Device to run on
            cost_config: Cost function configuration
            num_samples: Number of trajectory samples per iteration
            num_iterations: Number of MPPI iterations
            temperature: Temperature for exponential weighting (lower = more greedy)
            noise_sigma: Standard deviation for sampling noise
            noise_decay: Decay rate for noise per iteration
            num_elites: Number of elite samples to retain
            elite_weight: Weight for blending with best elite
            adaptive_temp: Adapt temperature based on cost variance
        """
        super().__init__(
            jepa_model=jepa_model,
            action_dim=action_dim,
            action_bounds=action_bounds,
            channel_mask=channel_mask,
            device=device,
            cost_config=cost_config,
        )
        
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.noise_decay = noise_decay
        self.num_elites = num_elites
        self.elite_weight = elite_weight
        self.adaptive_temp = adaptive_temp
        
    def _initialize_actions(
        self,
        horizon: int,
        initial_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Initialize nominal action sequence."""
        if initial_actions is not None:
            actions = initial_actions.clone().to(self.device)
            if len(actions) < horizon:
                padding = torch.zeros(
                    horizon - len(actions), self.action_dim,
                    device=self.device
                )
                actions = torch.cat([actions, padding], dim=0)
            elif len(actions) > horizon:
                actions = actions[:horizon]
        else:
            actions = torch.zeros(horizon, self.action_dim, device=self.device)
            
        return actions
    
    def _sample_trajectories(
        self,
        nominal_actions: torch.Tensor,
        noise_sigma: float,
        elite_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample perturbed action sequences around nominal trajectory.
        
        Args:
            nominal_actions: Nominal action sequence (horizon, action_dim)
            noise_sigma: Noise standard deviation
            elite_actions: Elite samples from previous iteration
            
        Returns:
            samples: Sampled action sequences (num_samples, horizon, action_dim)
        """
        horizon, action_dim = nominal_actions.shape
        
        if elite_actions is not None:
            num_random = self.num_samples - len(elite_actions)
        else:
            num_random = self.num_samples
            
        # Sample random trajectories
        noise = torch.randn(num_random, horizon, action_dim, device=self.device) * noise_sigma
        random_samples = nominal_actions.unsqueeze(0) + noise
        random_samples = self.clip_actions(random_samples)
        
        # Combine with elites
        if elite_actions is not None and len(elite_actions) > 0:
            samples = torch.cat([random_samples, elite_actions], dim=0)
        else:
            samples = random_samples
            
        return samples
    
    def _compute_weights(
        self,
        costs: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """
        Compute importance weights from costs using exponential weighting.
        
        Args:
            costs: Trajectory costs (num_samples,)
            temperature: Temperature parameter
            
        Returns:
            weights: Normalized importance weights (num_samples,)
        """
        # Subtract minimum for numerical stability
        costs_normalized = (costs - costs.min()) / max(temperature, 1e-6)
        weights = torch.exp(-costs_normalized)
        
        # Normalize
        weights = weights / (weights.sum() + 1e-8)
        
        return weights
    
    def optimize(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        horizon: int,
        initial_actions: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Optimize using MPPI.
        
        Args:
            z_init: Initial latent state (C, H, W)
            z_goal: Goal latent state (C, H, W)
            horizon: Planning horizon
            initial_actions: Optional warm-start actions
            verbose: Show progress bar
            
        Returns:
            Dictionary with optimized actions, trajectory, and cost history
        """
        # Move inputs to device
        z_init = z_init.to(self.device)
        z_goal = z_goal.to(self.device)
        
        # Initialize
        nominal_actions = self._initialize_actions(horizon, initial_actions)
        elite_actions = None
        
        # Tracking
        cost_history = []
        temp_history = []
        current_noise_sigma = self.noise_sigma
        current_temp = self.temperature
        best_cost = float('inf')
        best_actions = None
        
        # MPPI loop
        pbar = tqdm(range(self.num_iterations), disable=not verbose, desc="MPPI")
        
        for iteration in pbar:
            # Sample trajectories (including elites)
            sampled_actions = self._sample_trajectories(
                nominal_actions, current_noise_sigma, elite_actions
            )
            
            # Evaluate samples
            with torch.no_grad():
                trajectories = self.rollout.rollout(z_init, sampled_actions)
                costs = self.rollout.total_cost(
                    trajectories, z_goal, sampled_actions, self.cost_config
                )
            
            # Adapt temperature based on cost variance
            if self.adaptive_temp:
                cost_std = costs.std().item()
                current_temp = max(0.1, cost_std)
            
            # Compute importance weights
            weights = self._compute_weights(costs, current_temp)
            
            # Select elites for next iteration
            elite_indices = torch.argsort(costs)[:self.num_elites]
            elite_actions = sampled_actions[elite_indices].clone()
            
            # Update best
            if costs[elite_indices[0]] < best_cost:
                best_cost = costs[elite_indices[0]].item()
                best_actions = sampled_actions[elite_indices[0]].clone()
            
            # Update nominal as weighted average
            nominal_actions = (weights.view(-1, 1, 1) * sampled_actions).sum(dim=0)
            
            # Blend with best elite
            if self.elite_weight > 0:
                best_elite = sampled_actions[elite_indices[0]]
                nominal_actions = (
                    (1 - self.elite_weight) * nominal_actions +
                    self.elite_weight * best_elite
                )
            
            nominal_actions = self.clip_actions(nominal_actions)
            
            # Decay noise
            current_noise_sigma *= self.noise_decay
            
            # Log progress
            cost_history.append(best_cost)
            temp_history.append(current_temp)
            
            if verbose:
                mean_cost = costs.mean().item()
                pbar.set_postfix({
                    'best': f'{best_cost:.4f}',
                    'mean': f'{mean_cost:.4f}',
                    'temp': f'{current_temp:.3f}',
                    'sigma': f'{current_noise_sigma:.4f}'
                })
        
        # Use best actions found
        if best_actions is None:
            best_actions = nominal_actions
        
        # Final evaluation
        with torch.no_grad():
            final_trajectory = self.rollout.rollout(z_init, best_actions)
            final_cost = self.rollout.total_cost(
                final_trajectory, z_goal, best_actions, self.cost_config
            )
            
            goal_cost = self.rollout.goal_cost(final_trajectory, z_goal)
            traj_cost = self.rollout.trajectory_cost(final_trajectory, z_goal)
            smooth_cost = self.rollout.smoothness_cost(
                best_actions, order=self.cost_config.smoothness_order
            )
        
        return {
            'actions': best_actions,
            'trajectory': final_trajectory,
            'cost': final_cost.item(),
            'cost_history': cost_history,
            'temp_history': temp_history,
            'num_iterations': self.num_iterations,
            'final_temp': current_temp,
            # Cost breakdown
            'goal_cost': goal_cost.item(),
            'trajectory_cost': traj_cost.item(),
            'smoothness_cost': smooth_cost.item(),
        }