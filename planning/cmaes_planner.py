"""
Covariance Matrix Adaptation Evolution Strategy (CMA-ES) planner.
Derivative-free optimization using evolutionary algorithm with adaptive covariance.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple
from .base_planner import BasePlanner


class CMAESPlanner(BasePlanner):
    """
    CMA-ES planner for trajectory optimization.
    Uses evolutionary strategy to optimize action sequences without gradients.
    """
    
    def __init__(
        self,
        jepa_model,
        action_dim: int,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        channel_mask: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        discount: float = 0.99,
        population_size: int = 50,
        num_generations: int = 100,
        sigma: float = 0.3,
        elite_fraction: float = 0.5,
    ):
        """
        Args:
            jepa_model: Trained JEPA model
            action_dim: Dimensionality of action space
            action_bounds: (min, max) bounds for actions
            channel_mask: Optional mask for cost computation
            device: Device to run on
            discount: Discount factor for trajectory cost
            population_size: Number of candidate solutions per generation
            num_generations: Number of generations to evolve
            sigma: Initial standard deviation for sampling
            elite_fraction: Fraction of population to use as elites
        """
        super().__init__(
            jepa_model=jepa_model,
            action_dim=action_dim,
            action_bounds=action_bounds,
            channel_mask=channel_mask,
            device=device,
            discount=discount,
        )
        
        self.population_size = population_size
        self.num_generations = num_generations
        self.sigma = sigma
        self.num_elites = max(1, int(population_size * elite_fraction))
        
    def _initialize_distribution(
        self,
        horizon: int,
        initial_actions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize mean and covariance for CMA-ES.
        
        Returns:
            mean: Initial mean (horizon * action_dim,)
            cov: Initial covariance matrix (horizon * action_dim, horizon * action_dim)
        """
        dim = horizon * self.action_dim
        
        if initial_actions is not None:
            # Warm-start from provided actions
            mean = initial_actions.flatten()
            if len(mean) < dim:
                # Pad with zeros
                padding = torch.zeros(dim - len(mean), device=self.device)
                mean = torch.cat([mean, padding])
            elif len(mean) > dim:
                # Truncate
                mean = mean[:dim]
        else:
            # Initialize with zeros
            mean = torch.zeros(dim, device=self.device)
        
        # Initialize covariance as diagonal with sigma^2
        cov = torch.eye(dim, device=self.device) * (self.sigma ** 2)
        
        return mean, cov
    
    def _sample_population(
        self,
        mean: torch.Tensor,
        cov: torch.Tensor,
        population_size: int
    ) -> torch.Tensor:
        """
        Sample population from multivariate Gaussian.
        
        Args:
            mean: Distribution mean (D,)
            cov: Covariance matrix (D, D)
            population_size: Number of samples
            
        Returns:
            samples: Population samples (population_size, D)
        """
        # Use Cholesky decomposition for sampling
        try:
            L = torch.linalg.cholesky(cov)
        except RuntimeError:
            # If Cholesky fails, add regularization
            cov_reg = cov + torch.eye(len(cov), device=self.device) * 1e-6
            L = torch.linalg.cholesky(cov_reg)
        
        # Sample from standard normal
        z = torch.randn(population_size, len(mean), device=self.device)
        
        # Transform to desired distribution
        samples = mean + (z @ L.T)
        
        return samples
    
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
        
        # Clip to bounds
        actions = self.clip_actions(actions)
        
        # Evaluate all in parallel
        with torch.no_grad():
            # Batch rollout
            trajectory = self.rollout.rollout(z_init, actions)  # (pop_size, horizon+1, C, H, W)
            costs = self.rollout.trajectory_cost(trajectory, z_goal, self.discount)  # (pop_size,)
        
        return costs
    
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
        Optimize using CMA-ES.
        
        Args:
            z_init: Initial latent state (C, H, W)
            z_goal: Goal latent state (C, H, W)
            horizon: Planning horizon
            initial_actions: Optional warm-start actions
            verbose: Print optimization progress
            log_frequency: Print every k generations
            
        Returns:
            Dictionary with optimized actions, trajectory, and cost history
        """
        # Initialize distribution
        mean, cov = self._initialize_distribution(horizon, initial_actions)
        
        # Evolution loop
        cost_history = []
        best_cost = float('inf')
        best_actions = None
        
        for generation in range(self.num_generations):
            # Sample population
            population = self._sample_population(mean, cov, self.population_size)
            
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
            
            # Update distribution (mean and covariance)
            mean = elites.mean(dim=0)
            
            # Update covariance using elite samples
            centered_elites = elites - mean
            cov = (centered_elites.T @ centered_elites) / self.num_elites
            
            # Add regularization to maintain exploration
            cov = cov + torch.eye(len(cov), device=self.device) * (self.sigma ** 2) * 0.1
            
            # Log progress
            cost_history.append(best_cost)
            
            if verbose and (generation % log_frequency == 0 or generation == self.num_generations - 1):
                mean_cost = elite_costs.mean().item()
                std_cost = elite_costs.std().item()
                print(f"Gen {generation:4d}/{self.num_generations}: "
                      f"Best = {best_cost:.6f}, Mean = {mean_cost:.6f}, "
                      f"Std = {std_cost:.6f}")
        
        # Final evaluation with best actions
        best_actions = self.clip_actions(best_actions)
        with torch.no_grad():
            final_trajectory = self.rollout.rollout(z_init, best_actions)
            final_cost = self.rollout.trajectory_cost(final_trajectory, z_goal, self.discount)
        
        return {
            'actions': best_actions,
            'trajectory': final_trajectory,
            'cost': final_cost.item(),
            'cost_history': cost_history,
            'num_generations': self.num_generations,
        }


class CMAESPlannerImproved(CMAESPlanner):
    """
    Improved CMA-ES with adaptive sigma and elite retention.
    """
    
    def __init__(
        self,
        jepa_model,
        action_dim: int,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        channel_mask: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        discount: float = 0.99,
        population_size: int = 50,
        num_generations: int = 100,
        sigma: float = 0.3,
        elite_fraction: float = 0.5,
        sigma_decay: float = 0.995,
        min_sigma: float = 0.01,
        retain_best: bool = True,
    ):
        """
        Args:
            sigma_decay: Decay rate for sigma per generation
            min_sigma: Minimum sigma value
            retain_best: Keep best solution across generations
        """
        super().__init__(
            jepa_model=jepa_model,
            action_dim=action_dim,
            action_bounds=action_bounds,
            channel_mask=channel_mask,
            device=device,
            discount=discount,
            population_size=population_size,
            num_generations=num_generations,
            sigma=sigma,
            elite_fraction=elite_fraction,
        )
        
        self.sigma_decay = sigma_decay
        self.min_sigma = min_sigma
        self.retain_best = retain_best
        
    def optimize(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        horizon: int,
        initial_actions: Optional[torch.Tensor] = None,
        verbose: bool = False,
        log_frequency: int = 10,
    ) -> Dict[str, Any]:
        """Optimize using improved CMA-ES with adaptive sigma."""
        # Initialize distribution
        mean, cov = self._initialize_distribution(horizon, initial_actions)
        current_sigma = self.sigma
        
        # Evolution loop
        cost_history = []
        sigma_history = []
        best_cost = float('inf')
        best_actions = None
        
        for generation in range(self.num_generations):
            # Sample population
            population = self._sample_population(mean, cov, self.population_size)
            
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
            
            # Update covariance
            centered_elites = elites - mean
            cov = (centered_elites.T @ centered_elites) / self.num_elites
            
            # Adaptive sigma with regularization
            current_sigma = max(self.min_sigma, current_sigma * self.sigma_decay)
            cov = cov + torch.eye(len(cov), device=self.device) * (current_sigma ** 2)
            
            # Log progress
            cost_history.append(best_cost)
            sigma_history.append(current_sigma)
            
            if verbose and (generation % log_frequency == 0 or generation == self.num_generations - 1):
                mean_cost = elite_costs.mean().item()
                print(f"Gen {generation:4d}/{self.num_generations}: "
                      f"Best = {best_cost:.6f}, Mean = {mean_cost:.6f}, "
                      f"Sigma = {current_sigma:.6f}")
        
        # Final evaluation
        best_actions = self.clip_actions(best_actions)
        with torch.no_grad():
            final_trajectory = self.rollout.rollout(z_init, best_actions)
            final_cost = self.rollout.trajectory_cost(final_trajectory, z_goal, self.discount)
        
        return {
            'actions': best_actions,
            'trajectory': final_trajectory,
            'cost': final_cost.item(),
            'cost_history': cost_history,
            'sigma_history': sigma_history,
            'num_generations': self.num_generations,
            'final_sigma': current_sigma,
        }
