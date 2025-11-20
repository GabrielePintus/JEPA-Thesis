"""
Model Predictive Path Integral (MPPI) Control planner.
Information-theoretic sampling-based approach with temperature-weighted cost averaging.
"""

import torch
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm
from .base_planner import BasePlanner


class MPPIPlanner(BasePlanner):
    """
    MPPI planner for trajectory optimization.
    Uses importance sampling with exponentially weighted costs.
    """
    
    def __init__(
        self,
        jepa_model,
        action_dim: int,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        channel_mask: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        discount: float = 0.99,
        num_samples: int = 100,
        num_iterations: int = 10,
        temperature: float = 1.0,
        noise_sigma: float = 0.5,
        noise_decay: float = 0.95,
    ):
        """
        Args:
            jepa_model: Trained JEPA model
            action_dim: Dimensionality of action space
            action_bounds: (min, max) bounds for actions
            channel_mask: Optional mask for cost computation
            device: Device to run on
            discount: Discount factor for trajectory cost
            num_samples: Number of trajectory samples per iteration
            num_iterations: Number of MPPI iterations
            temperature: Temperature for exponential weighting (lower = more greedy)
            noise_sigma: Standard deviation for sampling noise
            noise_decay: Decay rate for noise per iteration
        """
        super().__init__(
            jepa_model=jepa_model,
            action_dim=action_dim,
            action_bounds=action_bounds,
            channel_mask=channel_mask,
            device=device,
            discount=discount,
        )
        
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.noise_decay = noise_decay
        
    def _initialize_actions(
        self,
        horizon: int,
        initial_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Initialize nominal action sequence."""
        if initial_actions is not None:
            actions = initial_actions.clone()
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
        noise_sigma: float
    ) -> torch.Tensor:
        """
        Sample perturbed action sequences around nominal trajectory.
        
        Args:
            nominal_actions: Nominal action sequence (horizon, action_dim)
            noise_sigma: Noise standard deviation
            
        Returns:
            samples: Sampled action sequences (num_samples, horizon, action_dim)
        """
        horizon, action_dim = nominal_actions.shape
        
        # Sample noise
        noise = torch.randn(
            self.num_samples, horizon, action_dim,
            device=self.device
        ) * noise_sigma
        
        # Add noise to nominal
        samples = nominal_actions.unsqueeze(0) + noise
        
        # Clip to bounds
        samples = self.clip_actions(samples)
        
        return samples
    
    def _compute_weights(
        self,
        costs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance weights from costs using exponential weighting.
        
        Args:
            costs: Trajectory costs (num_samples,)
            
        Returns:
            weights: Normalized importance weights (num_samples,)
        """
        # Exponential weighting: exp(-cost / temperature)
        # Subtract minimum for numerical stability
        costs_normalized = (costs - costs.min()) / self.temperature
        weights = torch.exp(-costs_normalized)
        
        # Normalize
        weights = weights / weights.sum()
        
        return weights
    
    def optimize(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        horizon: int,
        initial_actions: Optional[torch.Tensor] = None,
        verbose: bool = False,
        log_frequency: int = 1,
    ) -> Dict[str, Any]:
        """
        Optimize using MPPI.
        
        Args:
            z_init: Initial latent state (C, H, W)
            z_goal: Goal latent state (C, H, W)
            horizon: Planning horizon
            initial_actions: Optional warm-start actions
            verbose: Print optimization progress
            log_frequency: Print every k iterations
            
        Returns:
            Dictionary with optimized actions, trajectory, and cost history
        """
        # Initialize nominal trajectory
        nominal_actions = self._initialize_actions(horizon, initial_actions)
        
        # MPPI loop
        cost_history = []
        current_noise_sigma = self.noise_sigma
        
        progressbar = tqdm(range(self.num_iterations), disable=not verbose)
        for iteration in progressbar:
            # Sample trajectories
            sampled_actions = self._sample_trajectories(nominal_actions, current_noise_sigma)
            
            # Evaluate samples
            with torch.no_grad():
                trajectories = self.rollout.rollout(z_init, sampled_actions)
                costs = self.rollout.trajectory_cost(trajectories, z_goal, self.discount)
            
            # Compute importance weights
            weights = self._compute_weights(costs)
            
            # Update nominal trajectory as weighted average
            nominal_actions = (weights.view(-1, 1, 1) * sampled_actions).sum(dim=0)
            nominal_actions = self.clip_actions(nominal_actions)
            
            # Decay noise
            current_noise_sigma *= self.noise_decay
            
            # Log progress
            best_cost = costs.min().item()
            mean_cost = costs.mean().item()
            cost_history.append(best_cost)
            
            if verbose and (iteration % log_frequency == 0 or iteration == self.num_iterations - 1):
                # print(f"Iter {iteration:4d}/{self.num_iterations}: "
                #       f"Best = {best_cost:.6f}, Mean = {mean_cost:.6f}, "
                #       f"Sigma = {current_noise_sigma:.6f}")
                progressbar.set_description(
                    f"Iter {iteration:4d}/{self.num_iterations}: "
                    f"Best = {best_cost:.6f}, Mean = {mean_cost:.6f}, "
                    f"Sigma = {current_noise_sigma:.6f}"
                )
        
        # Final evaluation
        with torch.no_grad():
            final_trajectory = self.rollout.rollout(z_init, nominal_actions)
            final_cost = self.rollout.trajectory_cost(final_trajectory, z_goal, self.discount)
        
        return {
            'actions': nominal_actions,
            'trajectory': final_trajectory,
            'cost': final_cost.item(),
            'cost_history': cost_history,
            'num_iterations': self.num_iterations,
        }


class MPPIPlannerWithElites(MPPIPlanner):
    """
    Enhanced MPPI with elite sample retention and adaptive temperature.
    """
    
    def __init__(
        self,
        jepa_model,
        action_dim: int,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        channel_mask: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        discount: float = 0.99,
        num_samples: int = 100,
        num_iterations: int = 10,
        temperature: float = 1.0,
        noise_sigma: float = 0.5,
        noise_decay: float = 0.95,
        num_elites: int = 10,
        elite_weight: float = 0.3,
        adaptive_temp: bool = True,
    ):
        """
        Args:
            num_elites: Number of elite samples to retain
            elite_weight: Weight for elite samples in next iteration
            adaptive_temp: Adapt temperature based on cost variance
        """
        super().__init__(
            jepa_model=jepa_model,
            action_dim=action_dim,
            action_bounds=action_bounds,
            channel_mask=channel_mask,
            device=device,
            discount=discount,
            num_samples=num_samples,
            num_iterations=num_iterations,
            temperature=temperature,
            noise_sigma=noise_sigma,
            noise_decay=noise_decay,
        )
        
        self.num_elites = num_elites
        self.elite_weight = elite_weight
        self.adaptive_temp = adaptive_temp
        
    def _sample_trajectories_with_elites(
        self,
        nominal_actions: torch.Tensor,
        noise_sigma: float,
        elite_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample trajectories with elite retention.
        
        Args:
            nominal_actions: Nominal action sequence
            noise_sigma: Noise standard deviation
            elite_actions: Elite samples from previous iteration
            
        Returns:
            samples: Sampled action sequences
        """
        horizon, action_dim = nominal_actions.shape
        
        if elite_actions is not None:
            # Use fewer random samples to make room for elites
            num_random = self.num_samples - len(elite_actions)
        else:
            num_random = self.num_samples
            
        # Sample random trajectories
        noise = torch.randn(num_random, horizon, action_dim, device=self.device) * noise_sigma
        random_samples = nominal_actions.unsqueeze(0) + noise
        random_samples = self.clip_actions(random_samples)
        
        # Combine with elites
        if elite_actions is not None:
            samples = torch.cat([random_samples, elite_actions], dim=0)
        else:
            samples = random_samples
            
        return samples
    
    def optimize(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        horizon: int,
        initial_actions: Optional[torch.Tensor] = None,
        verbose: bool = False,
        log_frequency: int = 1,
    ) -> Dict[str, Any]:
        """Optimize using MPPI with elite retention."""
        # Initialize
        nominal_actions = self._initialize_actions(horizon, initial_actions)
        elite_actions = None
        
        # Track statistics
        cost_history = []
        temp_history = []
        current_noise_sigma = self.noise_sigma
        current_temp = self.temperature
        
        progressbar = tqdm(range(self.num_iterations), disable=not verbose)
        for iteration in progressbar:
            # Sample trajectories (including elites)
            sampled_actions = self._sample_trajectories_with_elites(
                nominal_actions, current_noise_sigma, elite_actions
            )
            
            # Evaluate samples
            with torch.no_grad():
                trajectories = self.rollout.rollout(z_init, sampled_actions)
                costs = self.rollout.trajectory_cost(trajectories, z_goal, self.discount)
            
            # Adapt temperature based on cost variance
            if self.adaptive_temp:
                cost_std = costs.std().item()
                current_temp = max(0.1, cost_std)
            
            # Compute importance weights
            weights = self._compute_weights(costs)
            
            # Select elites for next iteration
            elite_indices = torch.argsort(costs)[:self.num_elites]
            elite_actions = sampled_actions[elite_indices].clone()
            
            # Update nominal as weighted average
            nominal_actions = (weights.view(-1, 1, 1) * sampled_actions).sum(dim=0)
            
            # Optionally blend with best elite
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
            best_cost = costs.min().item()
            mean_cost = costs.mean().item()
            cost_history.append(best_cost)
            temp_history.append(current_temp)
            
            if verbose and (iteration % log_frequency == 0 or iteration == self.num_iterations - 1):
                # print(f"Iter {iteration:4d}/{self.num_iterations}: "
                #       f"Best = {best_cost:.6f}, Mean = {mean_cost:.6f}, "
                #       f"Temp = {current_temp:.3f}, Sigma = {current_noise_sigma:.6f}")
                progressbar.set_description(
                    f"Iter {iteration:4d}/{self.num_iterations}: "
                    f"Best = {best_cost:.6f}, Mean = {mean_cost:.6f}, "
                    f"Temp = {current_temp:.3f}, Sigma = {current_noise_sigma:.6f}"
                )
        
        # Final evaluation
        with torch.no_grad():
            final_trajectory = self.rollout.rollout(z_init, nominal_actions)
            final_cost = self.rollout.trajectory_cost(final_trajectory, z_goal, self.discount)
        
        return {
            'actions': nominal_actions,
            'trajectory': final_trajectory,
            'cost': final_cost.item(),
            'cost_history': cost_history,
            'temp_history': temp_history,
            'num_iterations': self.num_iterations,
            'final_temp': current_temp,
        }
