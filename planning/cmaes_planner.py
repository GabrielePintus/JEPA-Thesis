"""
Covariance Matrix Adaptation Evolution Strategy (CMA-ES) planner.
Derivative-free optimization using evolutionary algorithm with adaptive covariance.

Features:
- Population-based optimization without gradients
- Adaptive covariance matrix for efficient search
- Elite selection and retention
- Configurable sigma decay
- Configurable covariance structure: full, diagonal, or block
"""

import torch
from typing import Optional, Dict, Any, Tuple, Literal
from tqdm import tqdm

from .base_planner import BasePlanner, CostConfig


CovarianceType = Literal["full", "diagonal", "block"]


class CMAESPlanner(BasePlanner):
    """
    CMA-ES planner for trajectory optimization.
    Uses evolutionary strategy to optimize action sequences without gradients.
    
    This is useful when:
    - The cost landscape has many local minima
    - Gradients are unreliable or unavailable
    - Exploration is more important than exploitation
    
    Covariance structure options:
    - "full": Full covariance matrix (captures all correlations, O(d²) memory)
    - "diagonal": Diagonal covariance (independent dimensions, O(d) memory)
    - "block": Block-diagonal covariance (correlations within timesteps, O(T*a²) memory)
    """
    
    def __init__(
        self,
        jepa_model,
        action_dim: int,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        channel_mask: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        cost_config: Optional[CostConfig] = None,
        # CMA-ES parameters
        population_size: int = 50,
        num_generations: int = 100,
        sigma: float = 0.3,
        elite_fraction: float = 0.5,
        # Adaptive parameters
        sigma_decay: float = 0.995,
        min_sigma: float = 0.01,
        retain_best: bool = True,
        # Covariance structure
        covariance_type: CovarianceType = "diagonal",
    ):
        """
        Args:
            jepa_model: Trained JEPA model
            action_dim: Dimensionality of action space
            action_bounds: (min, max) bounds for actions
            channel_mask: Optional mask for cost computation
            device: Device to run on
            cost_config: Cost function configuration
            population_size: Number of candidate solutions per generation
            num_generations: Number of generations to evolve
            sigma: Initial standard deviation for sampling
            elite_fraction: Fraction of population to use as elites
            sigma_decay: Decay rate for sigma per generation
            min_sigma: Minimum sigma value
            retain_best: Keep best solution across generations
            covariance_type: Type of covariance matrix structure:
                - "full": Full covariance matrix (all correlations)
                - "diagonal": Diagonal only (independent dimensions)
                - "block": Block-diagonal (correlations within each timestep)
        """
        super().__init__(
            jepa_model=jepa_model,
            action_dim=action_dim,
            action_bounds=action_bounds,
            channel_mask=channel_mask,
            device=device,
            cost_config=cost_config,
        )
        
        self.population_size = population_size
        self.num_generations = num_generations
        self.sigma = sigma
        self.num_elites = max(1, int(population_size * elite_fraction))
        self.sigma_decay = sigma_decay
        self.min_sigma = min_sigma
        self.retain_best = retain_best
        self.covariance_type = covariance_type
        
    def _initialize_distribution(
        self,
        horizon: int,
        initial_actions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Any]:
        """
        Initialize mean and covariance for CMA-ES.
        
        Returns:
            mean: Initial mean (horizon * action_dim,)
            cov: Covariance representation (structure depends on covariance_type)
        """
        dim = horizon * self.action_dim
        
        if initial_actions is not None:
            mean = initial_actions.flatten().to(self.device)
            if len(mean) < dim:
                padding = torch.zeros(dim - len(mean), device=self.device)
                mean = torch.cat([mean, padding])
            elif len(mean) > dim:
                mean = mean[:dim]
        else:
            mean = torch.zeros(dim, device=self.device)
        
        # Initialize covariance based on type
        if self.covariance_type == "full":
            # Full covariance matrix
            cov = torch.eye(dim, device=self.device) * (self.sigma ** 2)
        elif self.covariance_type == "diagonal":
            # Just store diagonal as vector
            cov = torch.ones(dim, device=self.device) * (self.sigma ** 2)
        elif self.covariance_type == "block":
            # List of block matrices, one per timestep
            cov = [
                torch.eye(self.action_dim, device=self.device) * (self.sigma ** 2)
                for _ in range(horizon)
            ]
        else:
            raise ValueError(f"Unknown covariance_type: {self.covariance_type}")
        
        return mean, cov
    
    def _sample_population(
        self,
        mean: torch.Tensor,
        cov: Any,
        population_size: int,
        horizon: int
    ) -> torch.Tensor:
        """
        Sample population from multivariate Gaussian with specified covariance structure.
        """
        dim = len(mean)
        
        if self.covariance_type == "full":
            # Full covariance: use Cholesky decomposition
            try:
                L = torch.linalg.cholesky(cov)
            except RuntimeError:
                cov_reg = cov + torch.eye(dim, device=self.device) * 1e-6
                L = torch.linalg.cholesky(cov_reg)
            
            z = torch.randn(population_size, dim, device=self.device)
            samples = mean + (z @ L.T)
            
        elif self.covariance_type == "diagonal":
            # Diagonal: element-wise scaling
            std = torch.sqrt(cov)
            z = torch.randn(population_size, dim, device=self.device)
            samples = mean + z * std
            
        elif self.covariance_type == "block":
            # Block-diagonal: sample each timestep block independently
            samples = torch.zeros(population_size, dim, device=self.device)
            
            for t in range(horizon):
                start_idx = t * self.action_dim
                end_idx = start_idx + self.action_dim
                block_cov = cov[t]
                
                try:
                    L = torch.linalg.cholesky(block_cov)
                except RuntimeError:
                    block_reg = block_cov + torch.eye(self.action_dim, device=self.device) * 1e-6
                    L = torch.linalg.cholesky(block_reg)
                
                z = torch.randn(population_size, self.action_dim, device=self.device)
                samples[:, start_idx:end_idx] = mean[start_idx:end_idx] + (z @ L.T)
        else:
            raise ValueError(f"Unknown covariance_type: {self.covariance_type}")
        
        return samples
    
    def _update_covariance(
        self,
        elites: torch.Tensor,
        mean: torch.Tensor,
        current_sigma: float,
        horizon: int
    ) -> Any:
        """
        Update covariance estimate from elite samples.
        
        Args:
            elites: Elite samples (num_elites, dim)
            mean: Current mean (dim,)
            current_sigma: Current sigma for regularization
            horizon: Planning horizon
            
        Returns:
            Updated covariance (structure depends on covariance_type)
        """
        centered = elites - mean
        dim = len(mean)
        
        if self.covariance_type == "full":
            # Full empirical covariance + regularization
            cov = (centered.T @ centered) / self.num_elites
            cov = cov + torch.eye(dim, device=self.device) * (current_sigma ** 2)
            
        elif self.covariance_type == "diagonal":
            # Diagonal: just variances + regularization
            variances = (centered ** 2).mean(dim=0)
            cov = variances + (current_sigma ** 2)
            
        elif self.covariance_type == "block":
            # Block-diagonal: separate covariance per timestep
            cov = []
            for t in range(horizon):
                start_idx = t * self.action_dim
                end_idx = start_idx + self.action_dim
                block_centered = centered[:, start_idx:end_idx]
                
                block_cov = (block_centered.T @ block_centered) / self.num_elites
                block_cov = block_cov + torch.eye(self.action_dim, device=self.device) * (current_sigma ** 2)
                cov.append(block_cov)
        else:
            raise ValueError(f"Unknown covariance_type: {self.covariance_type}")
        
        return cov
    
    def _evaluate_population(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        population: torch.Tensor,
        horizon: int
    ) -> torch.Tensor:
        """
        Evaluate fitness of population.
        
        Args:
            z_init: Initial state (C, H, W)
            z_goal: Goal state (C, H, W)
            population: Population of action sequences (pop_size, horizon * action_dim)
            horizon: Planning horizon
            
        Returns:
            costs: Fitness for each individual (pop_size,)
        """
        pop_size = len(population)
        
        # Reshape to action sequences
        actions = population.view(pop_size, horizon, self.action_dim)
        actions = self.clip_actions(actions)
        
        # Evaluate all in parallel
        with torch.no_grad():
            trajectory = self.rollout.rollout(z_init, actions)
            costs = self.rollout.total_cost(trajectory, z_goal, actions, self.cost_config)
        
        return costs
    
    def optimize(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        horizon: int,
        initial_actions: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Optimize using CMA-ES.
        
        Args:
            z_init: Initial latent state (C, H, W)
            z_goal: Goal latent state (C, H, W)
            horizon: Planning horizon
            initial_actions: Optional warm-start actions
            verbose: Show progress bar
            
        Returns:
            Dictionary with optimized actions, trajectory, and cost history
        """
        # Handle extra batch dimensions
        if z_init.dim() == 4 and z_init.shape[0] == 1:
            z_init = z_init.squeeze(0)
        if z_goal.dim() == 4 and z_goal.shape[0] == 1:
            z_goal = z_goal.squeeze(0)
        
        # Move inputs to device
        z_init = z_init.to(self.device)
        z_goal = z_goal.to(self.device)
        
        # Initialize distribution
        mean, cov = self._initialize_distribution(horizon, initial_actions)
        current_sigma = self.sigma
        
        # Tracking
        cost_history = []
        sigma_history = []
        best_cost = float('inf')
        best_actions = None
        
        # Evolution loop
        pbar = tqdm(range(self.num_generations), disable=not verbose, desc="CMA-ES")
        
        for generation in pbar:
            # Sample population
            population = self._sample_population(mean, cov, self.population_size, horizon)
            
            # Optionally inject best solution
            if self.retain_best and best_actions is not None:
                population[0] = best_actions.flatten()
            
            # Evaluate fitness
            costs = self._evaluate_population(z_init, z_goal, population, horizon)
            
            # Select elites
            elite_indices = torch.argsort(costs)[:self.num_elites]
            elites = population[elite_indices]
            elite_costs = costs[elite_indices]
            
            # Update best
            if elite_costs[0] < best_cost:
                best_cost = elite_costs[0].item()
                best_actions = elites[0].view(horizon, self.action_dim)
            
            # Update distribution
            mean = elites.mean(dim=0)
            
            # Adaptive sigma
            current_sigma = max(self.min_sigma, current_sigma * self.sigma_decay)
            
            # Update covariance
            cov = self._update_covariance(elites, mean, current_sigma, horizon)
            
            # Log progress
            cost_history.append(best_cost)
            sigma_history.append(current_sigma)
            
            if verbose:
                mean_cost = elite_costs.mean().item()
                pbar.set_postfix({
                    'best': f'{best_cost:.4f}',
                    'mean': f'{mean_cost:.4f}',
                    'sigma': f'{current_sigma:.4f}'
                })
        
        # Final evaluation
        best_actions = self.clip_actions(best_actions)
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
            'sigma_history': sigma_history,
            'num_generations': self.num_generations,
            'final_sigma': current_sigma,
            'covariance_type': self.covariance_type,
            # Cost breakdown
            'goal_cost': goal_cost.item(),
            'trajectory_cost': traj_cost.item(),
            'smoothness_cost': smooth_cost.item(),
        }