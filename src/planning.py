"""
Latent Space Planning Framework for JEPA

This module provides a flexible planning framework for optimizing action sequences
entirely in the JEPA latent space. All planning operations work with latent 
representations - decoders are never used during optimization.

Key Components:
    - BasePlanner: Main planning interface
    - BaseOptimizer: Abstract base class for optimization algorithms
    - GradientOptimizer: Gradient descent with various options
    - CMAESOptimizer: CMA-ES sampling-based optimizer
    - PICOptimizer: Parallel-in-Control (PIC) optimizer
    - CostFunction: Latent space cost function (operates on z_cls, z_patches, z_state)

Key Design Principle:
    - Planning operates ENTIRELY in latent space
    - Cost functions compare latent representations directly
    - Decoders are only used for visualization/debugging (return_trajectory=True)
    - This ensures efficient, differentiable planning without bottlenecks

Model Predictive Control (MPC) with Replanning:
    The planner now supports efficient MPC replanning by accepting latent states
    directly, bypassing the encode-decode cycle:
    
    Example MPC Loop:
    ```python
    # Initialize
    planner = LatentSpacePlanner(jepa_model, cost_fn, optimizer)
    
    # Encode initial and goal states once
    current_latents = jepa_model.encode_state_and_frame(state, frame)
    goal_latents = jepa_model.encode_state_and_frame(goal_state, goal_frame)
    
    # MPC loop
    k_steps = 5  # Replan every 5 steps
    previous_plan = None
    
    for step in range(num_steps):
        # Replan from current latents
        mpc_result = planner.execute_mpc_step(
            current_latents=current_latents,
            goal_latents=goal_latents,
            previous_actions=previous_plan,
            k_steps=k_steps,
            num_iterations=50,
        )
        
        # Execute next k actions
        actions = mpc_result["actions_to_execute"]
        
        # Simulate/execute actions and get next state latents
        next_latents = jepa_model.predict_latents(current_latents, actions)
        
        # Update for next iteration (no decode-encode!)
        current_latents = next_latents
        previous_plan = mpc_result["full_plan"]
    ```
    
    The key advantage: by working with latent states, we avoid the computational
    overhead of decoding to observations and re-encoding at each replanning step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable, Tuple, List
import numpy as np
from tqdm import tqdm
import math


# ============================================================================
# Cost Function
# ============================================================================

class CostFunction:
    """
    Flexible cost function for planning in latent space.
    
    Operates directly on latent representations without decoding.
    """
    
    def __init__(
        self,
        cost_type: str = "mse",
        custom_cost_fn: Optional[Callable] = None,
        use_cls_token: bool = True,
        use_patch_tokens: bool = False,
        use_state_token: bool = True,
        cls_weight: float = 1.0,
        patch_weight: float = 0.0,
        state_weight: float = 1.0,
    ):
        """
        Args:
            cost_type: "mse" (L2), "mae" (L1), or "custom"
            custom_cost_fn: Custom function(pred_latents, goal_latents) -> cost
            use_cls_token: Whether to include CLS token in cost
            use_patch_tokens: Whether to include patch tokens in cost
            use_state_token: Whether to include state token in cost
            cls_weight: Weight for CLS token cost
            patch_weight: Weight for patch token cost
            state_weight: Weight for state token cost
        """
        self.cost_type = cost_type
        self.custom_cost_fn = custom_cost_fn
        self.use_cls_token = use_cls_token
        self.use_patch_tokens = use_patch_tokens
        self.use_state_token = use_state_token
        self.cls_weight = cls_weight
        self.patch_weight = patch_weight
        self.state_weight = state_weight
        
        if cost_type == "custom" and custom_cost_fn is None:
            raise ValueError("custom_cost_fn must be provided when cost_type='custom'")
    
    def compute_distance(self, pred: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between tensors based on cost_type.
        
        Args:
            pred: (..., D) predicted tensor
            goal: (..., D) goal tensor
            
        Returns:
            distance: (...) scalar distance for each sample
        """
        if self.cost_type == "mse":
            return torch.mean((pred - goal) ** 2, dim=-1)
        elif self.cost_type == "mae":
            return torch.mean(torch.abs(pred - goal), dim=-1)
        elif self.cost_type == "custom":
            return self.custom_cost_fn(pred, goal)
        else:
            raise ValueError(f"Unknown cost_type: {self.cost_type}")
    
    def __call__(
        self, 
        pred_latents: Dict[str, torch.Tensor], 
        goal_latents: Dict[str, torch.Tensor],
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        Compute cost between predicted and goal latent representations.
        
        Args:
            pred_latents: Dict with 'z_cls', 'z_patches', 'z_state'
                         z_cls: (B, D)
                         z_patches: (B, N, D)
                         z_state: (B, D)
            goal_latents: Same structure as pred_latents
            return_dict: if True, return dict with breakdown
            
        Returns:
            cost: (B,) cost values or dict with breakdown
        """
        B = pred_latents['z_cls'].shape[0]
        total_cost = torch.zeros(B, device=pred_latents['z_cls'].device)
        cost_components = {}
        
        # CLS token cost
        if self.use_cls_token:
            cls_cost = self.compute_distance(
                pred_latents['z_cls'], 
                goal_latents['z_cls']
            )
            total_cost = total_cost + self.cls_weight * cls_cost
            cost_components['cls_cost'] = cls_cost
        
        # Patch token cost (averaged over all patches)
        if self.use_patch_tokens:
            # (B, N, D) -> (B, N) -> (B,)
            patch_cost = self.compute_distance(
                pred_latents['z_patches'], 
                goal_latents['z_patches']
            )
            if len(patch_cost.shape) > 1:  # If per-patch costs
                patch_cost = patch_cost.mean(dim=-1)
            total_cost = total_cost + self.patch_weight * patch_cost
            cost_components['patch_cost'] = patch_cost
        
        # State token cost
        if self.use_state_token:
            state_cost = self.compute_distance(
                pred_latents['z_state'], 
                goal_latents['z_state']
            )
            total_cost = total_cost + self.state_weight * state_cost
            cost_components['state_cost'] = state_cost
        
        if return_dict:
            return {
                "total_cost": total_cost,
                **cost_components,
            }
        return total_cost


# ============================================================================
# Base Optimizer
# ============================================================================

class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    """
    
    def __init__(self, action_dim: int, horizon: int, device: str = "cpu"):
        """
        Args:
            action_dim: Dimension of action space
            horizon: Planning horizon (number of steps)
            device: Device for computation
        """
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = device
    
    @abstractmethod
    def optimize(
        self,
        cost_fn: Callable,
        initial_actions: Optional[torch.Tensor] = None,
        num_iterations: int = 100,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Optimize action sequence to minimize cost.
        
        Args:
            cost_fn: Function that takes actions (B, T, action_dim) -> (B,) costs
            initial_actions: Initial action sequence (B, T, action_dim)
            num_iterations: Number of optimization iterations
            **kwargs: Algorithm-specific parameters
            
        Returns:
            best_actions: (1, T, action_dim) best action sequence
            info: Dict with optimization information
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset optimizer state."""
        pass





# ============================================================================
# Gradient-Based Optimizer
# ============================================================================

class GradientOptimizer(BaseOptimizer):
    """
    Gradient descent optimizer for action sequences.
    
    Supports various gradient-based optimization methods:
    - SGD
    - Adam
    - SGD with momentum
    - RMSprop
    """
    
    def __init__(
        self,
        action_dim: int,
        horizon: int,
        learning_rate: float = 0.1,
        optimizer_type: str = "adam",
        momentum: float = 0.9,
        action_bounds: Optional[Tuple[float, float]] = None,
        device: str = "cpu",
    ):
        """
        Args:
            learning_rate: Learning rate for gradient descent
            optimizer_type: "sgd", "adam", "momentum", "rmsprop"
            momentum: Momentum coefficient (for momentum/adam)
            action_bounds: (min, max) bounds to clip actions
        """
        super().__init__(action_dim, horizon, device)
        self.lr = learning_rate
        self.optimizer_type = optimizer_type.lower()
        self.momentum = momentum
        self.action_bounds = action_bounds
        
        # Optimizer state
        self.actions = None
        self.optimizer = None
    
    def reset(self):
        """Reset optimizer state."""
        self.actions = None
        self.optimizer = None
    
    def optimize(
        self,
        cost_fn: Callable,
        initial_actions: Optional[torch.Tensor] = None,
        num_iterations: int = 100,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Optimize actions using gradient descent.
        
        Args:
            cost_fn: Function that takes actions (1, T, action_dim) -> scalar cost
            initial_actions: Initial action sequence (1, T, action_dim)
            num_iterations: Number of gradient steps
            verbose: Print progress
            
        Returns:
            best_actions: (1, T, action_dim) optimized actions
            info: Dict with cost_history
        """
        # Initialize actions
        if initial_actions is None:
            self.actions = torch.randn(
                1, self.horizon, self.action_dim, device=self.device
            ) * 0.1
        else:
            self.actions = initial_actions.clone().to(self.device)
        
        self.actions.requires_grad_(True)
        
        # Initialize optimizer
        if self.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD([self.actions], lr=self.lr)
        elif self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam([self.actions], lr=self.lr)
        elif self.optimizer_type == "momentum":
            self.optimizer = torch.optim.SGD(
                [self.actions], lr=self.lr, momentum=self.momentum
            )
        elif self.optimizer_type == "rmsprop":
            self.optimizer = torch.optim.RMSprop([self.actions], lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")
        
        # Optimization loop
        cost_history = []
        best_cost = float('inf')
        best_actions = None
        
        for iteration in range(num_iterations):
            self.optimizer.zero_grad()
            
            # Compute cost
            cost = cost_fn(self.actions)
            
            # Backward pass
            cost.backward()
            
            # Gradient step
            self.optimizer.step()
            
            # Clip actions to bounds
            if self.action_bounds is not None:
                with torch.no_grad():
                    self.actions.clamp_(self.action_bounds[0], self.action_bounds[1])
            
            # Track progress
            cost_val = cost.item()
            cost_history.append(cost_val)
            
            if cost_val < best_cost:
                best_cost = cost_val
                best_actions = self.actions.detach().clone()
            
            if verbose and (iteration % 10 == 0 or iteration == num_iterations - 1):
                print(f"Iteration {iteration}: cost = {cost_val:.6f}")
        
        info = {
            "cost_history": cost_history,
            "best_cost": best_cost,
            "final_cost": cost_history[-1],
            "num_iterations": num_iterations,
        }
        
        return best_actions, info



class LamarckianCMAES:
    """
    Lamarckian CMA-ES: Interleaves CMA-ES with gradient-based refinement.
    
    Combines global evolutionary search with local gradient optimization
    for faster convergence and better final solutions.
    """
    
    def __init__(
        self,
        action_dim: int,
        horizon: int,
        population_size: Optional[int] = None,
        sigma: float = 0.5,
        action_bounds: Optional[Tuple[float, float]] = None,
        device: str = "cpu",
        # Lamarckian parameters
        gradient_steps: int = 5,
        gradient_lr: float = 0.01,
        gradient_frequency: int = 1,
        num_elite_to_refine: int = 1,
    ):
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = torch.device(device)
        self.action_bounds = action_bounds
        
        # Lamarckian parameters
        self.gradient_steps = gradient_steps
        self.gradient_lr = gradient_lr
        self.gradient_frequency = gradient_frequency
        self.num_elite_to_refine = num_elite_to_refine
        
        # Total dimension
        self.N = action_dim * horizon
        
        # Population size
        if population_size is None:
            self.lambda_ = 4 + int(3 * math.log(self.N))
        else:
            self.lambda_ = population_size
        
        self.mu = self.lambda_ // 2
        
        # Recombination weights
        weights = torch.log(torch.tensor(self.mu + 0.5)) - torch.log(torch.arange(1, self.mu + 1, dtype=torch.float32))
        weights = weights / weights.sum()
        self.weights = weights.to(self.device)
        
        self.mu_eff = 1.0 / (weights ** 2).sum().item()
        
        # CMA-ES parameters (same as FastCMAES)
        self.sigma = sigma
        self.c_sigma = (self.mu_eff + 2.0) / (self.N + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0, math.sqrt((self.mu_eff - 1.0) / (self.N + 1.0)) - 1.0) + self.c_sigma
        self.c_c = (4.0 + self.mu_eff / self.N) / (self.N + 4.0 + 2.0 * self.mu_eff / self.N)
        alpha_mu = 2.0
        self.c_1 = alpha_mu / ((self.N + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1.0 - self.c_1,
            alpha_mu * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((self.N + 2.0) ** 2 + alpha_mu * self.mu_eff / 2.0)
        )
        self.chi_N = math.sqrt(self.N) * (1.0 - 1.0 / (4.0 * self.N) + 1.0 / (21.0 * self.N ** 2))
        self.eigen_update_interval = max(1, int(1.0 / (self.c_1 + self.c_mu) / self.N / 10.0))
        
        self.reset()
    
    def reset(self, initial_mean: Optional[torch.Tensor] = None):
        """Reset optimizer state."""
        if initial_mean is None:
            self.mean = torch.zeros(self.N, device=self.device)
        else:
            self.mean = initial_mean.flatten().to(self.device)
        
        self.C = torch.eye(self.N, device=self.device)
        self.B = torch.eye(self.N, device=self.device)
        self.D = torch.ones(self.N, device=self.device)
        self.p_sigma = torch.zeros(self.N, device=self.device)
        self.p_c = torch.zeros(self.N, device=self.device)
        
        self.generation = 0
        self.best_solution = self.mean.clone()
        self.best_fitness = float('inf')
        self.fitness_history = []
    
    def _sample_population(self) -> torch.Tensor:
        """Sample population."""
        z = torch.randn(self.lambda_, self.N, device=self.device)
        y = z * self.D[None, :] @ self.B.T
        population = self.mean[None, :] + self.sigma * y
        
        if self.action_bounds is not None:
            population = torch.clamp(population, self.action_bounds[0], self.action_bounds[1])
        
        return population
    
    def _gradient_refinement(
        self, 
        solutions: torch.Tensor, 
        fitness: torch.Tensor,
        cost_fn: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine elite solutions using gradient descent."""
        sorted_indices = torch.argsort(fitness)
        elite_indices = sorted_indices[:self.num_elite_to_refine]
        elite_solutions = solutions[elite_indices].clone()
        
        refined_solutions = []
        refined_fitness = []
        
        for i in range(self.num_elite_to_refine):
            solution = elite_solutions[i].clone().detach().requires_grad_(True)
            
            for step in range(self.gradient_steps):
                solution_shaped = solution.view(1, self.horizon, self.action_dim)
                
                # Compute cost WITH gradients enabled
                cost = cost_fn(solution_shaped)
                
                # Handle both scalar and tensor returns
                if cost.dim() > 0:
                    cost = cost.sum()
                
                # Check if gradient is possible
                if not cost.requires_grad:
                    # Cost function doesn't support gradients, skip refinement
                    refined_solutions.append(solution.detach())
                    refined_fitness.append(fitness[elite_indices[i]].item())
                    break
                
                # Compute gradient
                if solution.grad is not None:
                    solution.grad.zero_()
                
                try:
                    cost.backward()
                except RuntimeError:
                    # If backward fails, use current solution
                    refined_solutions.append(solution.detach())
                    refined_fitness.append(fitness[elite_indices[i]].item())
                    break
                
                # Gradient descent step
                with torch.no_grad():
                    if solution.grad is not None:
                        solution -= self.gradient_lr * solution.grad
                    if self.action_bounds is not None:
                        solution.clamp_(self.action_bounds[0], self.action_bounds[1])
                
                # Detach and re-enable gradients for next iteration
                solution = solution.detach().requires_grad_(True)
            else:
                # All gradient steps completed successfully
                with torch.no_grad():
                    solution_shaped = solution.view(1, self.horizon, self.action_dim)
                    final_cost = cost_fn(solution_shaped)
                    if final_cost.dim() > 0:
                        final_cost = final_cost[0]
                    refined_solutions.append(solution.detach())
                    refined_fitness.append(final_cost.item())
        
        refined_solutions = torch.stack(refined_solutions)
        refined_fitness = torch.tensor(refined_fitness, device=self.device)
        
        return refined_solutions, refined_fitness
    
    def _update_distribution(self, population: torch.Tensor, fitness: torch.Tensor):
        """Update distribution (same as FastCMAES)."""
        sorted_indices = torch.argsort(fitness)
        sorted_population = population[sorted_indices]
        
        if fitness[sorted_indices[0]] < self.best_fitness:
            self.best_fitness = fitness[sorted_indices[0]].item()
            self.best_solution = sorted_population[0].clone()
        
        selected = sorted_population[:self.mu]
        old_mean = self.mean.clone()
        self.mean = (self.weights[:, None] * selected).sum(dim=0)
        
        mean_shift = (self.mean - old_mean) / self.sigma
        C_inv_half_shift = mean_shift @ self.B / self.D @ self.B.T
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + \
                       math.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * C_inv_half_shift
        
        norm_p_sigma = torch.norm(self.p_sigma).item()
        self.sigma *= math.exp((self.c_sigma / self.d_sigma) * (norm_p_sigma / self.chi_N - 1.0))
        
        left_side = norm_p_sigma / math.sqrt(1.0 - (1.0 - self.c_sigma) ** (2.0 * (self.generation + 1)))
        right_side = (1.4 + 2.0 / (self.N + 1.0)) * self.chi_N
        h_sigma = 1.0 if left_side < right_side else 0.0
        
        self.p_c = (1.0 - self.c_c) * self.p_c + \
                   h_sigma * math.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * mean_shift
        
        delta_h_sigma = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        rank_one = torch.outer(self.p_c, self.p_c)
        
        steps = (selected - old_mean[None, :]) / self.sigma
        weighted_steps = steps * self.weights[:, None].sqrt()
        rank_mu = weighted_steps.T @ weighted_steps
        
        self.C = (1.0 - self.c_1 - self.c_mu + delta_h_sigma) * self.C + \
                 self.c_1 * rank_one + self.c_mu * rank_mu
        
        if self.generation % self.eigen_update_interval == 0:
            self.C = (self.C + self.C.T) / 2.0
            eigenvalues, eigenvectors = torch.linalg.eigh(self.C)
            eigenvalues = torch.clamp(eigenvalues, min=1e-10)
            self.D = torch.sqrt(eigenvalues)
            self.B = eigenvectors
        
        self.generation += 1
    
    def optimize(
        self,
        cost_fn: Callable,
        initial_actions: Optional[torch.Tensor] = None,
        num_iterations: int = 100,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """Run Lamarckian CMA-ES optimization."""
        if initial_actions is not None:
            self.reset(initial_actions)
        else:
            self.reset()
        
        cost_history = []
        gradient_improvements = []
        
        # Test if cost function supports gradients
        supports_gradients = True
        try:
            test_action = torch.randn(1, self.horizon, self.action_dim, device=self.device, requires_grad=True)
            test_cost = cost_fn(test_action)
            if test_cost.dim() > 0:
                test_cost = test_cost.sum()
            if test_cost.requires_grad:
                test_cost.backward()
            else:
                supports_gradients = False
        except:
            supports_gradients = False
        
        if not supports_gradients and self.gradient_steps > 0:
            if verbose:
                print("Warning: Cost function doesn't support gradients. Falling back to pure CMA-ES.")
        
        for gen in range(num_iterations):
            # Sample and evaluate
            population = self._sample_population()
            population_shaped = population.view(self.lambda_, self.horizon, self.action_dim)
            
            with torch.no_grad():
                fitness = cost_fn(population_shaped)
            
            # Gradient refinement (only if supported)
            if supports_gradients and gen % self.gradient_frequency == 0 and self.gradient_steps > 0:
                refined_solutions, refined_fitness = self._gradient_refinement(
                    population, fitness, cost_fn
                )
                
                original_elite_fitness = fitness[torch.argsort(fitness)[:self.num_elite_to_refine]]
                improvement = (original_elite_fitness - refined_fitness).mean().item()
                gradient_improvements.append(improvement)
                
                # Inject refined solutions
                sorted_indices = torch.argsort(fitness, descending=True)
                worst_indices = sorted_indices[:self.num_elite_to_refine]
                population[worst_indices] = refined_solutions
                fitness[worst_indices] = refined_fitness
            
            # Update distribution
            self._update_distribution(population, fitness)
            
            best_gen_cost = fitness.min().item()
            cost_history.append(best_gen_cost)
            self.fitness_history.append(best_gen_cost)
            
            if verbose and (gen % 10 == 0 or gen == num_iterations - 1):
                grad_info = f", GradImp={gradient_improvements[-1]:.6f}" if gradient_improvements else ""
                print(f"Gen {gen:3d}: Best={best_gen_cost:.6f}, Sigma={self.sigma:.4f}{grad_info}")
        
        best_actions = self.best_solution.view(1, self.horizon, self.action_dim)
        
        info = {
            "cost_history": cost_history,
            "best_cost": self.best_fitness,
            "final_cost": cost_history[-1],
            "num_iterations": num_iterations,
            "num_evaluations": num_iterations * self.lambda_,
            "gradient_improvements": gradient_improvements,
            "used_gradients": supports_gradients and self.gradient_steps > 0,
        }
        
        return best_actions, info






# ============================================================================
# CMA-ES Optimizer
# ============================================================================


class CMAESOptimizer:
    """
    Optimized CMA-ES implementation with full batch inference support.
    
    All operations are vectorized for maximum efficiency, especially when
    evaluating fitness functions that support batch inference (e.g., neural networks).
    """
    
    def __init__(
        self,
        action_dim: int,
        horizon: int,
        population_size: Optional[int] = None,
        sigma: float = 0.5,
        action_bounds: Optional[Tuple[float, float]] = None,
        device: str = "cpu",
        # Advanced parameters (tuned defaults from Hansen 2016)
        rank_one_learning_rate: Optional[float] = None,  # c_c
        rank_mu_learning_rate: Optional[float] = None,   # c_1
        covariance_learning_rate: Optional[float] = None, # c_mu
        damping_factor: Optional[float] = None,           # d_sigma
        cumulation_decay: Optional[float] = None,         # c_sigma
        update_eigendecomposition_every: int = None,      # lazy eigendecomposition
    ):
        """
        Initialize CMA-ES optimizer.
        
        Args:
            action_dim: Dimension of action vector at each timestep
            horizon: Planning horizon (number of timesteps)
            population_size: Number of samples per generation (default: 4 + floor(3*log(N)))
            sigma: Initial step size (standard deviation)
            action_bounds: Optional (min, max) tuple for action clipping
            device: 'cpu' or 'cuda'
            
        Advanced parameters (None = use defaults from Hansen 2016):
            rank_one_learning_rate: Learning rate for rank-one update
            rank_mu_learning_rate: Learning rate for rank-mu update  
            covariance_learning_rate: Learning rate for covariance matrix
            damping_factor: Damping for step size adaptation
            cumulation_decay: Time constant for cumulation
            update_eigendecomposition_every: Update eigendecomp every N generations
        """
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = torch.device(device)
        self.action_bounds = action_bounds
        
        # Total dimension
        self.N = action_dim * horizon
        
        # Population size (default from Hansen 2016)
        if population_size is None:
            self.lambda_ = 4 + int(3 * math.log(self.N))
        else:
            self.lambda_ = population_size
        
        # Selection: number of parents (mu)
        self.mu = self.lambda_ // 2
        
        # Recombination weights (positive weights for top mu individuals)
        # Using log-based weighting scheme from Hansen 2016
        weights = torch.log(torch.tensor(self.mu + 0.5)) - torch.log(torch.arange(1, self.mu + 1, dtype=torch.float32))
        weights = weights / weights.sum()  # Normalize to sum to 1
        self.weights = weights.to(self.device)
        
        # Variance effective selection mass
        self.mu_eff = 1.0 / (weights ** 2).sum().item()
        
        # Step size control parameters
        self.sigma = sigma
        if cumulation_decay is None:
            self.c_sigma = (self.mu_eff + 2.0) / (self.N + self.mu_eff + 5.0)
        else:
            self.c_sigma = cumulation_decay
            
        if damping_factor is None:
            self.d_sigma = 1.0 + 2.0 * max(0, math.sqrt((self.mu_eff - 1.0) / (self.N + 1.0)) - 1.0) + self.c_sigma
        else:
            self.d_sigma = damping_factor
        
        # Covariance matrix adaptation parameters
        if rank_one_learning_rate is None:
            self.c_c = (4.0 + self.mu_eff / self.N) / (self.N + 4.0 + 2.0 * self.mu_eff / self.N)
        else:
            self.c_c = rank_one_learning_rate
            
        if rank_mu_learning_rate is None:
            alpha_mu = 2.0
            self.c_1 = alpha_mu / ((self.N + 1.3) ** 2 + self.mu_eff)
        else:
            self.c_1 = rank_mu_learning_rate
            
        if covariance_learning_rate is None:
            self.c_mu = min(
                1.0 - self.c_1,
                alpha_mu * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((self.N + 2.0) ** 2 + alpha_mu * self.mu_eff / 2.0)
            )
        else:
            self.c_mu = covariance_learning_rate
        
        # Expectation of ||N(0,I)||
        self.chi_N = math.sqrt(self.N) * (1.0 - 1.0 / (4.0 * self.N) + 1.0 / (21.0 * self.N ** 2))
        
        # Lazy eigendecomposition (update every N generations for efficiency)
        if update_eigendecomposition_every is None:
            self.eigen_update_interval = int(1.0 / (self.c_1 + self.c_mu) / self.N / 10.0)
            self.eigen_update_interval = max(1, self.eigen_update_interval)
        else:
            self.eigen_update_interval = update_eigendecomposition_every
        
        # Initialize state
        self.reset()
    
    def reset(self, initial_mean: Optional[torch.Tensor] = None):
        """
        Reset the optimizer state.
        
        Args:
            initial_mean: Optional initial mean (shape: (N,) or (1, horizon, action_dim))
        """
        # Mean of the distribution
        if initial_mean is None:
            self.mean = torch.zeros(self.N, device=self.device)
        else:
            if initial_mean.dim() == 3:
                self.mean = initial_mean.flatten().to(self.device)
            else:
                self.mean = initial_mean.flatten().to(self.device)
        
        # Covariance matrix (initially identity)
        self.C = torch.eye(self.N, device=self.device)
        
        # Square root of C (for efficient sampling)
        # We maintain B and D such that C = B @ D^2 @ B^T
        self.B = torch.eye(self.N, device=self.device)  # Eigenvectors
        self.D = torch.ones(self.N, device=self.device)  # sqrt of eigenvalues
        
        # Evolution paths
        self.p_sigma = torch.zeros(self.N, device=self.device)  # For step size
        self.p_c = torch.zeros(self.N, device=self.device)      # For covariance
        
        # Generation counter
        self.generation = 0
        
        # Best solution tracking
        self.best_solution = self.mean.clone()
        self.best_fitness = float('inf')
        
        # History
        self.fitness_history = []
    
    def _sample_population(self) -> torch.Tensor:
        """
        Sample a population from the current distribution.
        
        Uses efficient sampling: x = mean + sigma * B * D * N(0, I)
        
        Returns:
            population: (lambda, N) tensor of candidate solutions
        """
        # Sample from standard normal
        z = torch.randn(self.lambda_, self.N, device=self.device)
        
        # Transform: y = B * D * z
        y = z * self.D[None, :]  # Scale by sqrt(eigenvalues)
        y = y @ self.B.T         # Rotate by eigenvectors
        
        # Scale by step size and add mean
        population = self.mean[None, :] + self.sigma * y
        
        # Apply action bounds if specified
        if self.action_bounds is not None:
            population = torch.clamp(population, self.action_bounds[0], self.action_bounds[1])
        
        return population
    
    def _update_distribution(self, population: torch.Tensor, fitness: torch.Tensor):
        """
        Update mean, covariance, and step size based on fitness values.
        
        Args:
            population: (lambda, N) candidate solutions
            fitness: (lambda,) fitness values (lower is better)
        """
        # Sort by fitness (ascending = best first)
        sorted_indices = torch.argsort(fitness)
        sorted_population = population[sorted_indices]
        
        # Update best solution
        if fitness[sorted_indices[0]] < self.best_fitness:
            self.best_fitness = fitness[sorted_indices[0]].item()
            self.best_solution = sorted_population[0].clone()
        
        # Select top mu individuals
        selected = sorted_population[:self.mu]
        
        # --- Update mean (weighted recombination) ---
        old_mean = self.mean.clone()
        self.mean = (self.weights[:, None] * selected).sum(dim=0)
        
        # --- Step size control (CSA - Cumulative Step-size Adaptation) ---
        # Compute mean shift in coordinate system of C^(-1/2)
        mean_shift = (self.mean - old_mean) / self.sigma
        
        # C^(-1/2) = B * D^(-1) * B^T
        # So C^(-1/2) * mean_shift = B * D^(-1) * B^T * mean_shift
        C_inv_half_shift = mean_shift @ self.B  # Rotate
        C_inv_half_shift = C_inv_half_shift / self.D  # Scale by D^(-1)
        C_inv_half_shift = C_inv_half_shift @ self.B.T  # Rotate back
        
        # Update evolution path for sigma
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + \
                       math.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * C_inv_half_shift
        
        # Update step size
        norm_p_sigma = torch.norm(self.p_sigma).item()
        self.sigma *= math.exp((self.c_sigma / self.d_sigma) * (norm_p_sigma / self.chi_N - 1.0))
        
        # --- Covariance matrix adaptation ---
        # Compute h_sigma (stalling detection)
        left_side = norm_p_sigma / math.sqrt(1.0 - (1.0 - self.c_sigma) ** (2.0 * (self.generation + 1)))
        right_side = (1.4 + 2.0 / (self.N + 1.0)) * self.chi_N
        h_sigma = 1.0 if left_side < right_side else 0.0
        
        # Update evolution path for C
        self.p_c = (1.0 - self.c_c) * self.p_c + \
                   h_sigma * math.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * mean_shift
        
        # Rank-one update
        delta_h_sigma = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        rank_one = torch.outer(self.p_c, self.p_c)
        
        # Rank-mu update (weighted sum of selected steps)
        steps = (selected - old_mean[None, :]) / self.sigma  # (mu, N)
        weighted_steps = steps * self.weights[:, None].sqrt()  # Weight and sqrt for outer product
        rank_mu = weighted_steps.T @ weighted_steps
        
        # Update covariance matrix
        self.C = (1.0 - self.c_1 - self.c_mu + delta_h_sigma) * self.C + \
                 self.c_1 * rank_one + \
                 self.c_mu * rank_mu
        
        # --- Update eigendecomposition (lazy update for efficiency) ---
        if self.generation % self.eigen_update_interval == 0:
            self._update_eigendecomposition()
        
        self.generation += 1
    
    def _update_eigendecomposition(self):
        """
        Update the eigendecomposition of the covariance matrix.
        
        Maintains C = B @ D^2 @ B^T where:
        - B: eigenvectors (orthonormal)
        - D: square root of eigenvalues
        
        This is expensive but only done every few generations.
        """
        # Ensure C is symmetric (numerical stability)
        self.C = (self.C + self.C.T) / 2.0
        
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(self.C)
        
        # Ensure positive eigenvalues (numerical stability)
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)
        
        # Store square root of eigenvalues and eigenvectors
        self.D = torch.sqrt(eigenvalues)
        self.B = eigenvectors
    
    def optimize(
        self,
        cost_fn: Callable[[torch.Tensor], torch.Tensor],
        initial_actions: Optional[torch.Tensor] = None,
        num_iterations: int = 100,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Run CMA-ES optimization with full batch inference.
        
        Args:
            cost_fn: Function that takes (B, horizon, action_dim) and returns (B,) costs
                    This function should support batch inference for efficiency
            initial_actions: Optional initial mean (1, horizon, action_dim)
            num_iterations: Number of generations to run
            verbose: Print progress every 10 iterations
            
        Returns:
            best_actions: (1, horizon, action_dim) best solution found
            info: Dictionary with optimization statistics
        """
        # Reset with initial mean if provided
        if initial_actions is not None:
            self.reset(initial_actions)
        else:
            self.reset()
        
        # Track history
        cost_history = []
        sigma_history = []
        
        for gen in range(num_iterations):
            # Sample population
            population = self._sample_population()  # (lambda, N)
            
            # Reshape for cost function: (lambda, horizon, action_dim)
            population_shaped = population.view(self.lambda_, self.horizon, self.action_dim)
            
            # Evaluate fitness (BATCH INFERENCE - entire population at once!)
            with torch.no_grad():
                fitness = cost_fn(population_shaped)  # (lambda,)
            
            # Update distribution
            self._update_distribution(population, fitness)
            
            # Track statistics
            best_gen_cost = fitness.min().item()
            cost_history.append(best_gen_cost)
            sigma_history.append(self.sigma)
            self.fitness_history.append(best_gen_cost)
            
            # Verbose output
            if verbose and (gen % 10 == 0 or gen == num_iterations - 1):
                mean_fitness = fitness.mean().item()
                std_fitness = fitness.std().item()
                print(f"Gen {gen:3d}: Best={best_gen_cost:.6f}, Mean={mean_fitness:.6f}, "
                      f"Std={std_fitness:.6f}, Sigma={self.sigma:.4f}")
        
        # Return best solution found
        best_actions = self.best_solution.view(1, self.horizon, self.action_dim)
        
        info = {
            "cost_history": cost_history,
            "best_cost": self.best_fitness,
            "final_cost": cost_history[-1],
            "num_iterations": num_iterations,
            "num_evaluations": num_iterations * self.lambda_,
            "sigma_history": sigma_history,
            "final_sigma": self.sigma,
            "population_size": self.lambda_,
        }
        
        return best_actions, info
    
    def ask(self) -> torch.Tensor:
        """
        Sample a new population (ask for candidate solutions).
        
        Useful for custom optimization loops.
        
        Returns:
            population: (lambda, horizon, action_dim) candidate solutions
        """
        population = self._sample_population()
        return population.view(self.lambda_, self.horizon, self.action_dim)
    
    def tell(self, population: torch.Tensor, fitness: torch.Tensor):
        """
        Update distribution based on fitness values (tell the optimizer results).
        
        Useful for custom optimization loops.
        
        Args:
            population: (lambda, horizon, action_dim) candidate solutions
            fitness: (lambda,) fitness values (lower is better)
        """
        population_flat = population.view(self.lambda_, -1)
        self._update_distribution(population_flat, fitness)
    
    def get_state_dict(self) -> Dict:
        """Get optimizer state for checkpointing."""
        return {
            'mean': self.mean,
            'C': self.C,
            'B': self.B,
            'D': self.D,
            'p_sigma': self.p_sigma,
            'p_c': self.p_c,
            'sigma': self.sigma,
            'generation': self.generation,
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load optimizer state from checkpoint."""
        self.mean = state_dict['mean'].to(self.device)
        self.C = state_dict['C'].to(self.device)
        self.B = state_dict['B'].to(self.device)
        self.D = state_dict['D'].to(self.device)
        self.p_sigma = state_dict['p_sigma'].to(self.device)
        self.p_c = state_dict['p_c'].to(self.device)
        self.sigma = state_dict['sigma']
        self.generation = state_dict['generation']
        self.best_solution = state_dict['best_solution'].to(self.device)
        self.best_fitness = state_dict['best_fitness']
        self.fitness_history = state_dict['fitness_history']





# class CMAESOptimizer(BaseOptimizer):
#     """
#     Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    
#     A sampling-based black-box optimization algorithm.
#     Requires: pip install cma
#     """
    
#     def __init__(
#         self,
#         action_dim: int,
#         horizon: int,
#         population_size: Optional[int] = None,
#         sigma: float = 0.5,
#         action_bounds: Optional[Tuple[float, float]] = None,
#         device: str = "cpu",
#     ):
#         """
#         Args:
#             population_size: Number of samples per iteration (default: 4 + 3*log(dim))
#             sigma: Initial standard deviation
#             action_bounds: (min, max) bounds for actions
#         """
#         super().__init__(action_dim, horizon, device)
        
#         try:
#             import cma
#             self.cma = cma
#         except ImportError:
#             raise ImportError("CMA-ES requires: pip install cma")
        
#         self.population_size = population_size
#         self.sigma = sigma
#         self.action_bounds = action_bounds
        
#         self.es = None
    
#     def reset(self):
#         """Reset optimizer state."""
#         self.es = None
    
#     def optimize(
#         self,
#         cost_fn: Callable,
#         initial_actions: Optional[torch.Tensor] = None,
#         num_iterations: int = 100,
#         verbose: bool = False,
#     ) -> Tuple[torch.Tensor, Dict]:
#         """
#         Optimize using CMA-ES.
        
#         Args:
#             cost_fn: Function that takes actions (B, T, action_dim) -> (B,) costs
#             initial_actions: Initial mean (1, T, action_dim)
#             num_iterations: Number of CMA-ES generations
#             verbose: Print progress
            
#         Returns:
#             best_actions: (1, T, action_dim) optimized actions
#             info: Dict with optimization info
#         """
#         # Flatten action dimensions
#         total_dim = self.horizon * self.action_dim
        
#         # Initialize
#         if initial_actions is None:
#             x0 = np.zeros(total_dim)
#         else:
#             x0 = initial_actions.cpu().numpy().flatten()
        
#         # Setup CMA-ES options
#         opts = {
#             'popsize': self.population_size,
#             'verbose': -1 if not verbose else 1,
#         }
        
#         if self.action_bounds is not None:
#             opts['bounds'] = [self.action_bounds[0], self.action_bounds[1]]
        
#         # Initialize CMA-ES
#         self.es = self.cma.CMAEvolutionStrategy(x0, self.sigma, opts)
        
#         # Optimization loop
#         cost_history = []
        
#         for iteration in range(num_iterations):
#             # Sample population
#             solutions = self.es.ask()
            
#             # Evaluate solutions
#             costs = []
#             for sol in solutions:
#                 # Reshape to (1, T, action_dim)
#                 actions = torch.tensor(
#                     sol.reshape(1, self.horizon, self.action_dim),
#                     dtype=torch.float32,
#                     device=self.device
#                 )
                
#                 # Compute cost
#                 with torch.no_grad():
#                     cost = cost_fn(actions).item()
#                 costs.append(cost)
            
#             # Update CMA-ES
#             self.es.tell(solutions, costs)
            
#             # Track best
#             best_cost = min(costs)
#             cost_history.append(best_cost)
            
#             if verbose and (iteration % 10 == 0 or iteration == num_iterations - 1):
#                 print(f"Generation {iteration}: best cost = {best_cost:.6f}, sigma = {self.es.sigma:.4f}")
        
#         # Get best solution
#         best_solution = self.es.result.xbest
#         best_actions = torch.tensor(
#             best_solution.reshape(1, self.horizon, self.action_dim),
#             dtype=torch.float32,
#             device=self.device
#         )
        
#         info = {
#             "cost_history": cost_history,
#             "best_cost": self.es.result.fbest,
#             "num_iterations": num_iterations,
#             "num_evaluations": self.es.result.evaluations,
#         }
        
#         return best_actions, info













class PICOptimizerOptimized:
    """
    Optimized Path Integral Control (MPPI - Model Predictive Path Integral).
    
    Enhanced with:
    - Adaptive temperature and noise scheduling
    - Early stopping for convergence
    - Elite sample retention
    - Momentum updates
    - Covariance adaptation (optional)
    - Top-k averaging for stability
    """
    
    def __init__(
        self,
        action_dim: int,
        horizon: int,
        num_samples: int = 100,
        temperature: float = 1.0,
        noise_sigma: float = 0.5,
        action_bounds: Optional[Tuple[float, float]] = None,
        device: str = "cpu",
        # New optimization parameters
        temperature_schedule: str = "exponential",  # "constant", "exponential", "linear"
        temperature_decay: float = 0.9,  # Decay factor per iteration
        min_temperature: float = 0.1,
        noise_schedule: str = "exponential",  # "constant", "exponential", "linear"
        noise_decay: float = 0.95,
        min_noise_sigma: float = 0.1,
        elite_ratio: float = 0.1,  # Keep top 10% of samples
        momentum: float = 0.0,  # 0.0 = no momentum, 0.9 = high momentum
        use_covariance: bool = False,  # Adapt per-dimension noise
        top_k_ratio: float = 0.5,  # Use top 50% for weighted average
        early_stop_threshold: float = 1e-4,  # Stop if improvement < this
        early_stop_patience: int = 3,  # Number of iterations with no improvement
    ):
        """
        Args:
            num_samples: Number of trajectory samples per iteration
            temperature: Initial temperature for importance sampling (lower = more focused)
            noise_sigma: Initial standard deviation for sampling noise
            action_bounds: (min, max) bounds for actions
            temperature_schedule: How to adjust temperature over iterations
            temperature_decay: Decay rate for temperature
            min_temperature: Minimum temperature value
            noise_schedule: How to adjust noise over iterations
            noise_decay: Decay rate for noise
            min_noise_sigma: Minimum noise standard deviation
            elite_ratio: Fraction of best samples to retain between iterations
            momentum: Momentum coefficient for mean updates (0-1)
            use_covariance: Whether to adapt per-dimension noise (covariance)
            top_k_ratio: Fraction of samples to use for weighted average
            early_stop_threshold: Minimum improvement to continue optimization
            early_stop_patience: Iterations to wait before early stopping
        """
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.initial_temperature = temperature
        self.temperature = temperature
        self.initial_noise_sigma = noise_sigma
        self.noise_sigma = noise_sigma
        self.action_bounds = action_bounds
        self.device = device
        
        # Optimization parameters
        self.temperature_schedule = temperature_schedule
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        self.noise_schedule = noise_schedule
        self.noise_decay = noise_decay
        self.min_noise_sigma = min_noise_sigma
        self.elite_ratio = elite_ratio
        self.momentum = momentum
        self.use_covariance = use_covariance
        self.top_k_ratio = top_k_ratio
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        
        # State
        self.mean_actions = None
        self.previous_mean = None  # For momentum
        self.elite_actions = None
        self.elite_costs = None
        
        # Covariance adaptation
        if self.use_covariance:
            self.noise_covariance = torch.ones(
                horizon, action_dim, device=device
            ) * noise_sigma
        else:
            self.noise_covariance = None
    
    def reset(self, initial_actions: Optional[torch.Tensor] = None):
        """Reset optimizer state."""
        if initial_actions is not None:
            self.mean_actions = initial_actions.clone().to(self.device)
        else:
            self.mean_actions = None
        
        self.previous_mean = None
        self.elite_actions = None
        self.elite_costs = None
        self.temperature = self.initial_temperature
        self.noise_sigma = self.initial_noise_sigma
        
        if self.use_covariance:
            self.noise_covariance = torch.ones(
                self.horizon, self.action_dim, device=self.device
            ) * self.initial_noise_sigma
    
    def _update_temperature(self, iteration: int, num_iterations: int):
        """Update temperature based on schedule."""
        if self.temperature_schedule == "constant":
            return
        elif self.temperature_schedule == "exponential":
            self.temperature = max(
                self.min_temperature,
                self.initial_temperature * (self.temperature_decay ** iteration)
            )
        elif self.temperature_schedule == "linear":
            progress = iteration / max(num_iterations - 1, 1)
            self.temperature = max(
                self.min_temperature,
                self.initial_temperature * (1 - progress) + self.min_temperature * progress
            )
    
    def _update_noise(self, iteration: int, num_iterations: int):
        """Update noise based on schedule."""
        if self.noise_schedule == "constant":
            return
        elif self.noise_schedule == "exponential":
            self.noise_sigma = max(
                self.min_noise_sigma,
                self.initial_noise_sigma * (self.noise_decay ** iteration)
            )
        elif self.noise_schedule == "linear":
            progress = iteration / max(num_iterations - 1, 1)
            self.noise_sigma = max(
                self.min_noise_sigma,
                self.initial_noise_sigma * (1 - progress) + self.min_noise_sigma * progress
            )
    
    def _update_covariance(self, sampled_actions: torch.Tensor, weights: torch.Tensor):
        """
        Update per-dimension noise covariance based on weighted variance.
        This allows different action dimensions to have different exploration scales.
        """
        if not self.use_covariance:
            return
        
        # Compute weighted variance for each dimension
        # sampled_actions: (num_samples, horizon, action_dim)
        # weights: (num_samples, 1, 1)
        weighted_mean = (weights * sampled_actions).sum(dim=0)  # (horizon, action_dim)
        squared_diff = (sampled_actions - weighted_mean.unsqueeze(0)) ** 2
        weighted_var = (weights * squared_diff).sum(dim=0)  # (horizon, action_dim)
        
        # Update covariance with exponential moving average
        alpha = 0.3  # Adaptation rate
        self.noise_covariance = (1 - alpha) * self.noise_covariance + alpha * torch.sqrt(weighted_var + 1e-8)
        
        # Clip to reasonable bounds
        self.noise_covariance = torch.clamp(
            self.noise_covariance,
            self.min_noise_sigma,
            self.initial_noise_sigma * 2
        )
    
    def optimize(
        self,
        cost_fn: Callable,
        initial_actions: Optional[torch.Tensor] = None,
        num_iterations: int = 10,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Optimize using enhanced Path Integral Control.
        
        Args:
            cost_fn: Function that takes actions (B, T, action_dim) -> (B,) costs
            initial_actions: Initial mean sequence (1, T, action_dim)
            num_iterations: Number of MPPI iterations
            verbose: Print progress
            
        Returns:
            best_actions: (1, T, action_dim) optimized actions
            info: Dict with optimization info including convergence metrics
        """
        # Initialize mean
        if initial_actions is None:
            self.mean_actions = torch.zeros(
                1, self.horizon, self.action_dim, device=self.device
            )
        else:
            self.mean_actions = initial_actions.clone().to(self.device)
        
        self.previous_mean = self.mean_actions.clone()
        
        cost_history = []
        best_cost = float('inf')
        no_improvement_count = 0
        num_elite = max(1, int(self.num_samples * self.elite_ratio))
        
        for iteration in range(num_iterations):
            # Update temperature and noise schedules
            self._update_temperature(iteration, num_iterations)
            self._update_noise(iteration, num_iterations)
            
            # Determine number of new samples vs elite samples
            if self.elite_actions is not None and iteration > 0:
                num_new_samples = self.num_samples - num_elite
            else:
                num_new_samples = self.num_samples
            
            # Sample new trajectories around mean
            if self.use_covariance:
                # Use per-dimension noise
                noise = torch.randn(
                    num_new_samples, self.horizon, self.action_dim,
                    device=self.device
                ) * self.noise_covariance.unsqueeze(0)
            else:
                # Use isotropic noise
                noise = torch.randn(
                    num_new_samples, self.horizon, self.action_dim,
                    device=self.device
                ) * self.noise_sigma
            
            # Generate new samples
            new_sampled_actions = self.mean_actions + noise
            
            # Combine with elite samples if available
            if self.elite_actions is not None and iteration > 0:
                sampled_actions = torch.cat([new_sampled_actions, self.elite_actions], dim=0)
            else:
                sampled_actions = new_sampled_actions
            
            # Clip to bounds
            if self.action_bounds is not None:
                sampled_actions = sampled_actions.clamp(
                    self.action_bounds[0], self.action_bounds[1]
                )
            
            # Evaluate costs (batch inference)
            with torch.no_grad():
                costs = cost_fn(sampled_actions)  # (num_samples,)
            
            # Track and store elite samples for next iteration
            sorted_indices = torch.argsort(costs)
            elite_indices = sorted_indices[:num_elite]
            self.elite_actions = sampled_actions[elite_indices].clone()
            self.elite_costs = costs[elite_indices].clone()
            
            # Use top-k samples for weighted average (more stable than all samples)
            top_k = max(1, int(self.num_samples * self.top_k_ratio))
            top_k_indices = sorted_indices[:top_k]
            top_k_costs = costs[top_k_indices]
            top_k_actions = sampled_actions[top_k_indices]
            
            # Compute importance weights on top-k samples
            weights = torch.softmax(-top_k_costs / self.temperature, dim=0)
            weights = weights.view(-1, 1, 1)  # (top_k, 1, 1)
            
            # Update covariance if enabled
            if self.use_covariance:
                # Use all samples for covariance estimation
                all_weights = torch.softmax(-costs / self.temperature, dim=0).view(-1, 1, 1)
                self._update_covariance(sampled_actions, all_weights)
            
            # Compute new mean using weighted average of top-k
            new_mean = (weights * top_k_actions).sum(dim=0, keepdim=True)
            
            # Apply momentum
            if self.momentum > 0 and self.previous_mean is not None:
                new_mean = self.momentum * self.previous_mean + (1 - self.momentum) * new_mean
            
            # Update mean and previous mean
            self.previous_mean = self.mean_actions.clone()
            self.mean_actions = new_mean
            
            # Track best cost
            current_best = costs.min().item()
            cost_history.append(current_best)
            
            # Early stopping check
            if current_best < best_cost - self.early_stop_threshold:
                best_cost = current_best
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if verbose:
                print(f"Iter {iteration}: best={current_best:.6f}, mean={costs.mean().item():.6f}, "
                      f"T={self.temperature:.4f}, σ={self.noise_sigma:.4f}")
            
            # Early stopping
            if no_improvement_count >= self.early_stop_patience and iteration >= 3:
                if verbose:
                    print(f"Early stopping at iteration {iteration} (no improvement for {no_improvement_count} iterations)")
                break
        
        # Evaluate the final mean to get its actual cost
        with torch.no_grad():
            final_mean_cost = cost_fn(self.mean_actions).item()
        
        # Return the better of: final mean or best elite sample
        if self.elite_costs[0].item() < final_mean_cost:
            best_actions = self.elite_actions[0:1]
            final_cost = self.elite_costs[0].item()
        else:
            best_actions = self.mean_actions
            final_cost = final_mean_cost
        
        info = {
            "cost_history": cost_history,
            "best_cost": min(cost_history),
            "final_cost": final_cost,
            "num_iterations": iteration + 1,  # Actual iterations performed
            "converged_early": iteration < num_iterations - 1,
            "final_temperature": self.temperature,
            "final_noise_sigma": self.noise_sigma,
        }
        
        return best_actions, info












# ============================================================================
# Path Integral Control (PIC) Optimizer
# ============================================================================

class PICOptimizer(BaseOptimizer):
    """
    Path Integral Control (MPPI - Model Predictive Path Integral).
    
    A sampling-based method that uses importance sampling to update
    the action distribution.
    """
    
    def __init__(
        self,
        action_dim: int,
        horizon: int,
        num_samples: int = 100,
        temperature: float = 1.0,
        noise_sigma: float = 0.5,
        action_bounds: Optional[Tuple[float, float]] = None,
        device: str = "cpu",
    ):
        """
        Args:
            num_samples: Number of trajectory samples per iteration
            temperature: Temperature for importance sampling (lower = more focused)
            noise_sigma: Standard deviation for sampling noise
            action_bounds: (min, max) bounds for actions
        """
        super().__init__(action_dim, horizon, device)
        self.num_samples = num_samples
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.action_bounds = action_bounds
        
        # Mean action sequence
        self.mean_actions = None
    
    def reset(self):
        """Reset optimizer state."""
        self.mean_actions = None
    
    def optimize(
        self,
        cost_fn: Callable,
        initial_actions: Optional[torch.Tensor] = None,
        num_iterations: int = 10,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Optimize using Path Integral Control.
        
        Args:
            cost_fn: Function that takes actions (B, T, action_dim) -> (B,) costs
            initial_actions: Initial mean sequence (1, T, action_dim)
            num_iterations: Number of MPPI iterations
            verbose: Print progress
            
        Returns:
            best_actions: (1, T, action_dim) optimized actions
            info: Dict with optimization info
        """
        # Initialize mean
        if initial_actions is None:
            self.mean_actions = torch.zeros(
                1, self.horizon, self.action_dim, device=self.device
            )
        else:
            self.mean_actions = initial_actions.clone().to(self.device)
        
        cost_history = []
        
        for iteration in range(num_iterations):
            # Sample trajectories around mean
            noise = torch.randn(
                self.num_samples, self.horizon, self.action_dim,
                device=self.device
            ) * self.noise_sigma
            
            # Broadcast mean and add noise
            sampled_actions = self.mean_actions + noise  # (num_samples, T, action_dim)
            
            # Clip to bounds
            if self.action_bounds is not None:
                sampled_actions = sampled_actions.clamp(
                    self.action_bounds[0], self.action_bounds[1]
                )
            
            # Evaluate costs
            with torch.no_grad():
                costs = cost_fn(sampled_actions)  # (num_samples,)
            
            # Compute importance weights
            weights = torch.softmax(-costs / self.temperature, dim=0)  # (num_samples,)
            
            # Update mean using weighted average
            # Reshape weights for broadcasting
            weights = weights.view(-1, 1, 1)  # (num_samples, 1, 1)
            self.mean_actions = (weights * sampled_actions).sum(dim=0, keepdim=True)
            
            # Track best cost
            best_cost = costs.min().item()
            cost_history.append(best_cost)
            
            if verbose and (iteration % 1 == 0 or iteration == num_iterations - 1):
                print(f"Iteration {iteration}: best cost = {best_cost:.6f}, "
                      f"mean cost = {costs.mean().item():.6f}")
        
        info = {
            "cost_history": cost_history,
            "best_cost": min(cost_history),
            "final_cost": cost_history[-1],
            "num_iterations": num_iterations,
        }
        
        return self.mean_actions, info


# ============================================================================
# Main Planner
# ============================================================================

class LatentSpacePlanner:
    """
    Main planner class for optimizing action sequences in JEPA latent space.
    
    IMPORTANT: All planning happens in latent space - no decoding during optimization!
    - Goal states are encoded once at the start
    - Predictions stay in latent space
    - Cost functions compare latent representations directly
    - Decoders are only used if return_trajectory=True (for visualization)
    
    Supports:
    - Gradient-based optimization
    - Sampling-based optimization (CMA-ES, PIC)
    - Flexible latent-space cost functions
    - Multiple optimization strategies
    """
    
    def __init__(
        self,
        jepa_model,
        cost_function: CostFunction,
        optimizer: BaseOptimizer,
        device: str = "cpu",
        action_repeat: int = 1,
    ):
        """
        Args:
            jepa_model: Trained JEPA model with prediction methods
            cost_function: CostFunction instance
            optimizer: Optimizer instance (GradientOptimizer, CMAESOptimizer, etc.)
            device: Device for computation
            action_repeat: Number of times to repeat each action (default: 1, no repeat)
                          If action_repeat=3, planning horizon is effectively 3x longer
                          with fewer optimization variables
        """
        self.model = jepa_model
        self.model.eval()
        self.cost_fn = cost_function
        self.optimizer = optimizer
        self.device = device
        self.action_repeat = action_repeat
    
    def expand_actions_with_repeat(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Expand actions by repeating each action multiple times.
        
        Args:
            actions: (B, T_opt, action_dim) compressed action sequence
            
        Returns:
            expanded_actions: (B, T_opt * action_repeat, action_dim) expanded sequence
            
        Example:
            If actions = [[a1], [a2]] and action_repeat = 3
            Returns: [[a1], [a1], [a1], [a2], [a2], [a2]]
        """
        if self.action_repeat == 1:
            return actions
        
        B, T_opt, action_dim = actions.shape
        # Repeat each action along the time dimension
        # (B, T_opt, action_dim) -> (B, T_opt, action_repeat, action_dim) -> (B, T_opt * action_repeat, action_dim)
        expanded = actions.unsqueeze(2).expand(B, T_opt, self.action_repeat, action_dim)
        expanded = expanded.reshape(B, T_opt * self.action_repeat, action_dim)
        return expanded
    
    def plan(
        self,
        initial_state: torch.Tensor,
        initial_frame: torch.Tensor,
        goal_state: torch.Tensor,
        goal_frame: torch.Tensor,
        num_iterations: int = 100,
        initial_actions: Optional[torch.Tensor] = None,
        verbose: bool = False,
        return_trajectory: bool = False,
        initial_is_latent: bool = False,
        goal_is_latent: bool = False,
    ) -> Dict:
        """
        Plan action sequence from initial state to goal state.
        
        All computations happen in latent space - no decoding required.
        
        Args:
            initial_state: (1, state_dim) starting state OR dict with latents if initial_is_latent=True
            initial_frame: (1, 3, 64, 64) starting frame OR None if initial_is_latent=True
            goal_state: (1, state_dim) goal state OR dict with latents if goal_is_latent=True
            goal_frame: (1, 3, 64, 64) goal frame OR None if goal_is_latent=True
            num_iterations: Number of optimization iterations
            initial_actions: (1, T, action_dim) initial guess
            verbose: Print optimization progress
            return_trajectory: If True, return predicted trajectory
            initial_is_latent: If True, initial_state is dict with {'z_cls', 'z_patches', 'z_state'}
            goal_is_latent: If True, goal_state is dict with {'z_cls', 'z_patches', 'z_state'}
            
        Returns:
            Dict with:
                - actions: (1, T, action_dim) optimized action sequence
                - final_cost: Final cost value
                - optimization_info: Dict with optimizer-specific info
                - predicted_trajectory: (optional) predicted states/frames
        """
        # Handle initial state - encode or use provided latents
        if initial_is_latent:
            # Initial state is already in latent space
            initial_latents = {
                'z_cls': initial_state['z_cls'].detach().to(self.device),
                'z_patches': initial_state['z_patches'].detach().to(self.device),
                'z_state': initial_state['z_state'].detach().to(self.device),
            }
            # For latent inputs, we don't need the raw state/frame
            init_state_raw = None
            init_frame_raw = None
        else:
            # Encode initial state
            initial_state = initial_state.to(self.device)
            initial_frame = initial_frame.to(self.device)
            with torch.no_grad():
                initial_latents = self.model.encode_state_and_frame(
                    state=initial_state,
                    frame=initial_frame
                )
                initial_latents = {
                    k: v.detach() for k, v in initial_latents.items()
                }
            init_state_raw = initial_state
            init_frame_raw = initial_frame
        
        # Handle goal state - encode or use provided latents
        if goal_is_latent:
            # Goal state is already in latent space
            goal_latents = {
                'z_cls': goal_state['z_cls'].detach().to(self.device),
                'z_patches': goal_state['z_patches'].detach().to(self.device),
                'z_state': goal_state['z_state'].detach().to(self.device),
            }
        else:
            # Encode goal state into latent space (done once)
            goal_state = goal_state.to(self.device)
            goal_frame = goal_frame.to(self.device)
            with torch.no_grad():
                goal_latents = self.model.encode_state_and_frame(
                    state=goal_state,
                    frame=goal_frame
                )
                # Ensure goal latents are detached and don't require gradients
                goal_latents = {
                    k: v.detach() for k, v in goal_latents.items()
                }
        
        # Define cost function wrapper for optimizer
        def rollout_cost(actions: torch.Tensor) -> torch.Tensor:
            """
            Evaluate cost for a batch of action sequences.
            
            Works entirely in latent space - no decoding!
            
            Args:
                actions: (B, T_opt, action_dim) compressed action sequence
            Returns:
                costs: (B,)
            """
            B = actions.shape[0]
            
            # Expand actions if action_repeat > 1
            actions_expanded = self.expand_actions_with_repeat(actions)
            
            # Expand initial latents for batch
            init_latents_batch = {
                'z_cls': initial_latents['z_cls'].expand(B, -1),
                'z_patches': initial_latents['z_patches'].expand(B, -1, -1),
                'z_state': initial_latents['z_state'].expand(B, -1),
            }
            
            # Expand goal latents for batch
            goal_latents_batch = {
                'z_cls': goal_latents['z_cls'].expand(B, -1),
                'z_patches': goal_latents['z_patches'].expand(B, -1, -1),
                'z_state': goal_latents['z_state'].expand(B, -1),
            }
            
            # Rollout predictions in latent space
            needs_grad = isinstance(self.optimizer, (GradientOptimizer))
            with torch.set_grad_enabled(needs_grad):
                # Use latent-based rollout with expanded actions
                rollout_result = self.model.rollout_predictions_from_latents(
                    initial_latents=init_latents_batch,
                    actions=actions_expanded,
                    decode_every=0,  # No decoding - stay in latent space
                )
            
            # Get final predicted latents
            final_latents = rollout_result["latent_trajectory"][-1]
            
            # Compute cost in latent space
            costs = self.cost_fn(final_latents, goal_latents_batch)
            
            return costs
        
        # Optimize
        best_actions, opt_info = self.optimizer.optimize(
            cost_fn=rollout_cost,
            initial_actions=initial_actions,
            num_iterations=num_iterations,
            verbose=verbose,
        )
        
        # Prepare result
        result = {
            "actions": best_actions,
            "final_cost": opt_info.get("best_cost", opt_info.get("final_cost")),
            "optimization_info": opt_info,
        }
        
        # Optionally return predicted trajectory
        if return_trajectory:
            with torch.no_grad():
                # Expand actions for trajectory visualization
                best_actions_expanded = self.expand_actions_with_repeat(best_actions)
                
                # If we have latent initial state, use it; otherwise use raw state/frame
                if initial_is_latent:
                    trajectory = self.model.rollout_predictions_from_latents(
                        initial_latents=initial_latents,
                        actions=best_actions_expanded,
                        decode_every=1,  # Decode for visualization only
                    )
                else:
                    trajectory = self.model.rollout_predictions(
                        initial_state=init_state_raw,
                        initial_frame=init_frame_raw,
                        actions=best_actions_expanded,
                        decode_every=1,  # Decode for visualization only
                    )
            result["predicted_trajectory"] = trajectory
            result["actions_expanded"] = best_actions_expanded  # Include expanded actions
        
        return result
    
    def reset(self):
        """Reset optimizer state."""
        self.optimizer.reset()
    
    def execute_mpc_step(
        self,
        current_latents: Dict[str, torch.Tensor],
        goal_latents: Dict[str, torch.Tensor],
        previous_actions: Optional[torch.Tensor] = None,
        k_steps: int = 1,
        replan_horizon: int = None,
        num_iterations: int = 100,
        verbose: bool = False,
    ) -> Dict:
        """
        Execute one MPC step: plan from current latents and return first k actions.
        
        This is designed for Model Predictive Control with replanning every k steps.
        By accepting latent states directly, it avoids the decode-encode cycle.
        
        Note on action_repeat:
            When action_repeat > 1, k_steps refers to the number of ACTUAL timesteps
            to execute before replanning. The optimizer works with compressed actions,
            so if action_repeat=3 and k_steps=6, the optimizer plans 2 compressed actions
            which expand to 6 actual actions.
        
        Args:
            current_latents: Dict with current state latents {'z_cls', 'z_patches', 'z_state'}
            goal_latents: Dict with goal state latents {'z_cls', 'z_patches', 'z_state'}
            previous_actions: (1, T_opt, action_dim) previous COMPRESSED action plan to warm-start
            k_steps: Number of ACTUAL timesteps to execute before replanning
            replan_horizon: Planning horizon in optimizer steps (default: optimizer.horizon)
            num_iterations: Number of optimization iterations
            verbose: Print optimization progress
            
        Returns:
            Dict with:
                - actions_to_execute: (1, k_steps, action_dim) next k ACTUAL actions to execute
                - actions_compressed: (1, T_opt, action_dim) compressed action sequence
                - full_plan: (1, T_opt * action_repeat, action_dim) full expanded action sequence
                - final_cost: Final cost value
                - optimization_info: Dict with optimizer-specific info
        """
        if replan_horizon is None:
            replan_horizon = self.optimizer.horizon
        
        # Calculate how many compressed actions we need for k_steps actual timesteps
        k_steps_compressed = (k_steps + self.action_repeat - 1) // self.action_repeat  # Ceiling division
        
        # Prepare warm-start actions if provided
        initial_actions = None
        if previous_actions is not None:
            # Shift previous COMPRESSED plan and pad with zeros
            T = replan_horizon
            shifted_actions = torch.zeros(1, T, self.optimizer.action_dim, device=self.device)
            prev_T = min(previous_actions.shape[1] - k_steps_compressed, T)
            if prev_T > 0:
                shifted_actions[:, :prev_T] = previous_actions[:, k_steps_compressed:k_steps_compressed+prev_T]
            initial_actions = shifted_actions
        
        # Plan using latent states directly (optimizer works with compressed actions)
        result = self.plan(
            initial_state=current_latents,
            initial_frame=None,  # Not needed when using latents
            goal_state=goal_latents,
            goal_frame=None,  # Not needed when using latents
            num_iterations=num_iterations,
            initial_actions=initial_actions,
            verbose=verbose,
            return_trajectory=False,
            initial_is_latent=True,
            goal_is_latent=True,
        )
        
        # Extract compressed actions
        actions_compressed = result["actions"]  # (1, T_opt, action_dim)
        
        # Expand actions for execution
        actions_expanded = self.expand_actions_with_repeat(actions_compressed)  # (1, T_opt * action_repeat, action_dim)
        
        # Extract first k_steps ACTUAL actions to execute
        actions_to_execute = actions_expanded[:, :k_steps]
        
        return {
            "actions_to_execute": actions_to_execute,  # (1, k_steps, action_dim) - actual actions
            "actions_compressed": actions_compressed,   # (1, T_opt, action_dim) - for warm-start
            "full_plan": actions_expanded,              # (1, T_opt * action_repeat, action_dim) - full expanded
            "final_cost": result["final_cost"],
            "optimization_info": result["optimization_info"],
        }
