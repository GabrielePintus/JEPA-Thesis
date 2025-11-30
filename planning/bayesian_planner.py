"""
Bayesian Optimization planner for trajectory optimization.
Uses Gaussian Process surrogate models with acquisition functions for efficient exploration.

Features:
- Gaussian Process regression for modeling cost landscape
- Multiple acquisition functions (UCB, EI, PI)
- Adaptive exploration/exploitation trade-off
- Warm-start support
- Optional dimensionality reduction for high-dimensional action spaces
"""

import torch
from typing import Optional, Dict, Any, Tuple, Literal
from tqdm import tqdm
import math

from .base_planner import BasePlanner, CostConfig


AcquisitionFunction = Literal["ucb", "ei", "pi"]


class BayesianPlanner(BasePlanner):
    """
    Bayesian Optimization planner for trajectory optimization.
    Uses Gaussian Process to model the cost landscape and acquisition functions
    to intelligently select next candidates.
    
    This is useful when:
    - Function evaluations are expensive (rolling out through JEPA)
    - You want sample-efficient optimization
    - You need balance between exploration and exploitation
    - The cost landscape is smooth but with multiple local minima
    """
    
    def __init__(
        self,
        jepa_model,
        action_dim: int,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        channel_mask: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        cost_config: Optional[CostConfig] = None,
        # Bayesian Optimization parameters
        num_initial_samples: int = 20,
        num_iterations: int = 50,
        acquisition_function: AcquisitionFunction = "ucb",
        # UCB parameters
        beta: float = 2.0,
        beta_decay: float = 0.99,
        # EI/PI parameters
        xi: float = 0.01,
        # GP parameters
        noise_variance: float = 1e-4,
        length_scale: float = 0.5,
        signal_variance: float = 1.0,
        # Optimization of acquisition function
        num_restarts: int = 5,
        num_candidates_per_restart: int = 100,
    ):
        """
        Args:
            jepa_model: Trained JEPA model
            action_dim: Dimensionality of action space
            action_bounds: (min, max) bounds for actions
            channel_mask: Optional mask for cost computation
            device: Device to run on
            cost_config: Cost function configuration
            num_initial_samples: Number of random samples for GP initialization
            num_iterations: Number of BO iterations after initialization
            acquisition_function: Which acquisition function to use ('ucb', 'ei', 'pi')
            beta: Exploration parameter for UCB (higher = more exploration)
            beta_decay: Decay rate for beta per iteration
            xi: Exploration parameter for EI/PI
            noise_variance: GP observation noise
            length_scale: GP kernel length scale (controls smoothness assumption)
            signal_variance: GP kernel signal variance
            num_restarts: Number of random restarts for acquisition optimization
            num_candidates_per_restart: Number of candidates per restart
        """
        super().__init__(
            jepa_model=jepa_model,
            action_dim=action_dim,
            action_bounds=action_bounds,
            channel_mask=channel_mask,
            device=device,
            cost_config=cost_config,
        )
        
        self.num_initial_samples = num_initial_samples
        self.num_iterations = num_iterations
        self.acquisition_function = acquisition_function
        self.beta = beta
        self.beta_decay = beta_decay
        self.xi = xi
        self.noise_variance = noise_variance
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.num_restarts = num_restarts
        self.num_candidates_per_restart = num_candidates_per_restart
        
        # Storage for GP data
        self.X_observed = []  # List of observed action sequences
        self.y_observed = []  # List of observed costs
        
    def _initialize_samples(
        self,
        horizon: int,
        initial_actions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate initial random samples for GP.
        
        Returns:
            X: Initial action sequences (num_initial_samples, horizon, action_dim)
            y: Corresponding costs (num_initial_samples,)
        """
        dim = horizon * self.action_dim
        
        # Generate random samples in flattened space
        X_flat = torch.rand(
            self.num_initial_samples, dim, device=self.device
        ) * (self.action_bounds[1] - self.action_bounds[0]) + self.action_bounds[0]
        
        # If we have initial actions, replace first sample
        if initial_actions is not None:
            initial_flat = initial_actions.flatten().to(self.device)
            if len(initial_flat) == dim:
                X_flat[0] = initial_flat
        
        # Reshape to action sequences
        X = X_flat.view(self.num_initial_samples, horizon, self.action_dim)
        X = self.clip_actions(X)
        
        return X
    
    def _rbf_kernel(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        length_scale: float,
        signal_variance: float
    ) -> torch.Tensor:
        """
        Compute RBF (Squared Exponential) kernel matrix.
        
        Args:
            X1: First set of points (N1, D)
            X2: Second set of points (N2, D)
            length_scale: Kernel length scale
            signal_variance: Kernel signal variance
            
        Returns:
            K: Kernel matrix (N1, N2)
        """
        # Compute pairwise squared distances
        X1_norm = (X1 ** 2).sum(dim=1, keepdim=True)
        X2_norm = (X2 ** 2).sum(dim=1, keepdim=True)
        dists_sq = X1_norm + X2_norm.T - 2 * X1 @ X2.T
        
        # RBF kernel
        K = signal_variance * torch.exp(-dists_sq / (2 * length_scale ** 2))
        
        return K
    
    def _gp_predict(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gaussian Process prediction.
        
        Args:
            X_train: Training inputs (N, D)
            y_train: Training targets (N,)
            X_test: Test inputs (M, D)
            
        Returns:
            mean: Predictive mean (M,)
            std: Predictive standard deviation (M,)
        """
        N = len(X_train)
        
        # Compute kernel matrices
        K = self._rbf_kernel(X_train, X_train, self.length_scale, self.signal_variance)
        K_star = self._rbf_kernel(X_train, X_test, self.length_scale, self.signal_variance)
        K_star_star = self._rbf_kernel(X_test, X_test, self.length_scale, self.signal_variance)
        
        # Add noise to diagonal for numerical stability
        K_noisy = K + torch.eye(N, device=self.device) * self.noise_variance
        
        # Solve linear system for mean
        try:
            L = torch.linalg.cholesky(K_noisy)
            alpha = torch.cholesky_solve(y_train.unsqueeze(1), L)
            mean = (K_star.T @ alpha).squeeze(1)
            
            # Compute variance
            v = torch.cholesky_solve(K_star, L)
            variance = torch.diag(K_star_star) - (K_star.T * v.T).sum(dim=1)
            variance = torch.clamp(variance, min=1e-10)  # Numerical stability
            std = torch.sqrt(variance)
        except RuntimeError:
            # Fallback if Cholesky fails
            K_inv = torch.linalg.inv(K_noisy + torch.eye(N, device=self.device) * 1e-6)
            mean = K_star.T @ K_inv @ y_train
            variance = torch.diag(K_star_star) - torch.diag(K_star.T @ K_inv @ K_star)
            variance = torch.clamp(variance, min=1e-10)
            std = torch.sqrt(variance)
        
        return mean, std
    
    def _ucb_acquisition(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        beta: float
    ) -> torch.Tensor:
        """Upper Confidence Bound acquisition function."""
        return mean - beta * std  # Negative because we minimize cost
    
    def _ei_acquisition(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        f_best: float
    ) -> torch.Tensor:
        """Expected Improvement acquisition function."""
        improvement = f_best - mean - self.xi
        Z = improvement / (std + 1e-10)
        
        # Normal CDF and PDF
        normal = torch.distributions.Normal(0, 1)
        cdf = normal.cdf(Z)
        pdf = torch.exp(normal.log_prob(Z))
        
        ei = improvement * cdf + std * pdf
        return ei
    
    def _pi_acquisition(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        f_best: float
    ) -> torch.Tensor:
        """Probability of Improvement acquisition function."""
        improvement = f_best - mean - self.xi
        Z = improvement / (std + 1e-10)
        
        normal = torch.distributions.Normal(0, 1)
        pi = normal.cdf(Z)
        return pi
    
    def _optimize_acquisition(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        horizon: int,
        current_beta: float
    ) -> torch.Tensor:
        """
        Optimize acquisition function to find next candidate.
        Uses multiple random restarts with local optimization.
        
        Returns:
            x_next: Next candidate action sequence (horizon, action_dim)
        """
        dim = horizon * self.action_dim
        best_acq_value = float('-inf')
        best_candidate = None
        
        f_best = y_train.min().item() if len(y_train) > 0 else 0.0
        
        # Multiple random restarts
        for _ in range(self.num_restarts):
            # Generate random candidates
            candidates_flat = torch.rand(
                self.num_candidates_per_restart, dim, device=self.device
            ) * (self.action_bounds[1] - self.action_bounds[0]) + self.action_bounds[0]
            
            # Predict with GP
            mean, std = self._gp_predict(X_train, y_train, candidates_flat)
            
            # Compute acquisition values
            if self.acquisition_function == "ucb":
                acq_values = self._ucb_acquisition(mean, std, current_beta)
            elif self.acquisition_function == "ei":
                acq_values = self._ei_acquisition(mean, std, f_best)
            elif self.acquisition_function == "pi":
                acq_values = self._pi_acquisition(mean, std, f_best)
            else:
                raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
            
            # Find best in this restart
            best_idx = torch.argmax(acq_values)
            if acq_values[best_idx] > best_acq_value:
                best_acq_value = acq_values[best_idx].item()
                best_candidate = candidates_flat[best_idx]
        
        # Reshape to action sequence
        x_next = best_candidate.view(horizon, self.action_dim)
        x_next = self.clip_actions(x_next)
        
        return x_next
    
    def _evaluate_candidates(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        candidates: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate cost for a batch of action sequences.
        
        Args:
            z_init: Initial state (C, H, W)
            z_goal: Goal state (C, H, W)
            candidates: Action sequences (N, horizon, action_dim)
            
        Returns:
            costs: Cost for each candidate (N,)
        """
        with torch.no_grad():
            trajectories = self.rollout.rollout(z_init, candidates)
            costs = self.rollout.total_cost(trajectories, z_goal, candidates, self.cost_config)
        
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
        Optimize using Bayesian Optimization.
        
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
        
        # Initialize with random samples
        X_init = self._initialize_samples(horizon, initial_actions)
        y_init = self._evaluate_candidates(z_init, z_goal, X_init)
        
        # Flatten X for GP
        X_train = X_init.view(len(X_init), -1)
        y_train = y_init
        
        # Track best solution
        best_idx = torch.argmin(y_train)
        best_cost = y_train[best_idx].item()
        best_actions = X_init[best_idx].clone()
        
        # Tracking
        cost_history = [best_cost]
        beta_history = [self.beta]
        current_beta = self.beta
        
        # Bayesian optimization loop
        pbar = tqdm(range(self.num_iterations), disable=not verbose, desc="Bayesian Opt")
        
        for iteration in pbar:
            # Find next candidate using acquisition function
            x_next = self._optimize_acquisition(X_train, y_train, horizon, current_beta)
            
            # Evaluate candidate
            x_next_batch = x_next.unsqueeze(0)
            y_next = self._evaluate_candidates(z_init, z_goal, x_next_batch)
            
            # Update GP data
            X_train = torch.cat([X_train, x_next.flatten().unsqueeze(0)], dim=0)
            y_train = torch.cat([y_train, y_next], dim=0)
            
            # Update best
            if y_next[0] < best_cost:
                best_cost = y_next[0].item()
                best_actions = x_next.clone()
            
            # Decay beta for UCB
            current_beta = max(0.1, current_beta * self.beta_decay)
            
            # Log progress
            cost_history.append(best_cost)
            beta_history.append(current_beta)
            
            if verbose:
                pbar.set_postfix({
                    'best': f'{best_cost:.4f}',
                    'current': f'{y_next[0].item():.4f}',
                    'beta': f'{current_beta:.3f}',
                    'n_obs': len(y_train)
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
            'beta_history': beta_history,
            'num_evaluations': len(y_train),
            'acquisition_function': self.acquisition_function,
            # Cost breakdown
            'goal_cost': goal_cost.item(),
            'trajectory_cost': traj_cost.item(),
            'smoothness_cost': smooth_cost.item(),
        }