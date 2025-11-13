"""
Latent Space Planning Framework for JEPA

This module provides a flexible planning framework for optimizing action sequences
in the JEPA latent space. It supports both gradient-based and sampling-based
optimization algorithms.

Key Components:
    - BasePlanner: Main planning interface
    - BaseOptimizer: Abstract base class for optimization algorithms
    - GradientOptimizer: Gradient descent with various options
    - CMAESOptimizer: CMA-ES sampling-based optimizer
    - PICOptimizer: Parallel-in-Control (PIC) optimizer
    - CostFunction: Flexible cost function definition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable, Tuple, List
import numpy as np


# ============================================================================
# Cost Function
# ============================================================================

class CostFunction:
    """
    Flexible cost function for planning.
    
    Allows specifying which state features to consider and custom cost computation.
    """
    
    def __init__(
        self,
        feature_indices: Optional[List[int]] = None,
        cost_type: str = "mse",
        custom_cost_fn: Optional[Callable] = None,
    ):
        """
        Args:
            feature_indices: List of indices to extract from state (e.g., [0, 1] for x, y)
                           If None, uses all features
            cost_type: "mse" (L2), "mae" (L1), or "custom"
            custom_cost_fn: Custom function(pred_state, goal_state) -> cost
        """
        self.feature_indices = feature_indices
        self.cost_type = cost_type
        self.custom_cost_fn = custom_cost_fn
        
        if cost_type == "custom" and custom_cost_fn is None:
            raise ValueError("custom_cost_fn must be provided when cost_type='custom'")
    
    def extract_features(self, state: torch.Tensor) -> torch.Tensor:
        """Extract relevant features from state."""
        if self.feature_indices is None:
            return state
        return state[..., self.feature_indices]
    
    def __call__(
        self, 
        pred_state: torch.Tensor, 
        goal_state: torch.Tensor,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        Compute cost between predicted and goal states.
        
        Args:
            pred_state: (B, state_dim) predicted state
            goal_state: (B, state_dim) goal state
            return_dict: if True, return dict with breakdown
            
        Returns:
            cost: (B,) cost values or dict with breakdown
        """
        pred_features = self.extract_features(pred_state)
        goal_features = self.extract_features(goal_state)
        
        if self.cost_type == "mse":
            cost = torch.mean((pred_features - goal_features) ** 2, dim=-1)
        elif self.cost_type == "mae":
            cost = torch.mean(torch.abs(pred_features - goal_features), dim=-1)
        elif self.cost_type == "custom":
            cost = self.custom_cost_fn(pred_features, goal_features)
        else:
            raise ValueError(f"Unknown cost_type: {self.cost_type}")
        
        if return_dict:
            return {
                "total_cost": cost,
                "pred_features": pred_features,
                "goal_features": goal_features,
            }
        return cost


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


# ============================================================================
# Newton-Based Optimizer (Second-Order Gradient Method)
# ============================================================================
# Add this class to planning.py after the GradientOptimizer class

class NewtonOptimizer(BaseOptimizer):
    """
    Newton-based optimizer using second-order derivatives (Hessian).
    
    Uses the Newton method update rule:
        x_{k+1} = x_k - α * H^{-1} * g
    
    where:
        - g is the gradient
        - H is the Hessian matrix
        - α is the step size (learning rate)
    
    For large-scale problems, we use:
        1. Gauss-Newton approximation (for least-squares problems)
        2. Damping (Levenberg-Marquardt style) for stability
        3. L-BFGS for approximate Hessian inverse
    
    This is particularly effective for problems where the cost function
    has strong second-order structure.
    """
    
    def __init__(
        self,
        action_dim: int,
        horizon: int,
        learning_rate: float = 1.0,
        method: str = "gauss_newton",  # "gauss_newton", "lbfgs", "damped_newton"
        damping: float = 1e-3,
        line_search: bool = True,
        action_bounds: Optional[Tuple[float, float]] = None,
        device: str = "cpu",
    ):
        """
        Args:
            learning_rate: Step size multiplier (typically 1.0 for Newton, but can adjust)
            method: 
                - "gauss_newton": Gauss-Newton approximation (good for least-squares)
                - "lbfgs": L-BFGS quasi-Newton method
                - "damped_newton": Levenberg-Marquardt style damped Newton
            damping: Damping parameter for regularization (adds λ*I to Hessian)
            line_search: Use backtracking line search for step size
            action_bounds: (min, max) bounds to clip actions
        """
        super().__init__(action_dim, horizon, device)
        self.lr = learning_rate
        self.method = method.lower()
        self.damping = damping
        self.line_search = line_search
        self.action_bounds = action_bounds
        
        # Optimizer state
        self.actions = None
        self.optimizer = None
        
        # For L-BFGS
        if self.method == "lbfgs":
            self.history_size = 10
    
    def reset(self):
        """Reset optimizer state."""
        self.actions = None
        self.optimizer = None
    
    def _gauss_newton_step(self, actions, cost_fn):
        """
        Gauss-Newton update for least-squares problems.
        
        For cost = ||f(x)||^2, the Gauss-Newton approximation is:
            H ≈ J^T J
            g = J^T f(x)
        
        where J is the Jacobian of f with respect to x.
        """
        actions.requires_grad_(True)
        
        # Compute cost and gradient
        cost = cost_fn(actions).sum()  # Sum over batch (typically batch=1)
        cost.backward(create_graph=True)
        
        gradient = actions.grad.clone()
        
        # Flatten for easier manipulation
        flat_grad = gradient.view(-1)
        
        # Compute Hessian using second derivatives
        # For efficiency, we use the Gauss-Newton approximation
        # H ≈ J^T J where J is Jacobian of residuals
        
        # Approximate Hessian by computing gradient of gradient
        # This gives us the actual Hessian for quadratic costs
        hessian_diag = torch.zeros_like(flat_grad)
        
        # Compute diagonal of Hessian (for efficiency)
        # Full Hessian computation is O(n^2) which is expensive
        for i in range(flat_grad.shape[0]):
            if flat_grad[i].requires_grad:
                # Compute second derivative
                grad_grad = torch.autograd.grad(
                    flat_grad[i], 
                    actions, 
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                if grad_grad is not None:
                    hessian_diag[i] = grad_grad.view(-1)[i]
        
        # Add damping for numerical stability (Levenberg-Marquardt)
        hessian_diag = hessian_diag + self.damping
        
        # Newton step: Δx = -H^{-1} g
        # With diagonal approximation: Δx = -g / diag(H)
        step = -flat_grad / (hessian_diag + 1e-8)
        step = step.view(actions.shape)
        
        return step, cost.item(), gradient
    
    def _damped_newton_step(self, actions, cost_fn):
        """
        Damped Newton method (Levenberg-Marquardt style).
        
        Uses: Δx = -(H + λI)^{-1} g
        
        This is more stable than pure Newton method.
        """
        # Similar to Gauss-Newton but with explicit damping
        step, cost, gradient = self._gauss_newton_step(actions, cost_fn)
        
        # Additional damping control based on cost improvement
        # (could be made adaptive in future)
        return step, cost, gradient
    
    def _compute_hessian_vector_product(self, gradient, actions, v):
        """
        Compute Hessian-vector product: H*v
        
        This is more efficient than computing full Hessian.
        Uses: H*v = ∇(g^T v) where g is the gradient
        """
        # Flatten
        flat_grad = gradient.view(-1)
        flat_v = v.view(-1)
        
        # Compute g^T v
        grad_v = (flat_grad * flat_v).sum()
        
        # Compute ∇(g^T v) = H*v
        hv = torch.autograd.grad(
            grad_v, 
            actions, 
            retain_graph=True,
            allow_unused=True
        )[0]
        
        if hv is None:
            return torch.zeros_like(v)
        
        return hv
    
    def _conjugate_gradient_solve(self, actions, gradient, cost_fn, max_iter=10):
        """
        Solve H*x = -g using Conjugate Gradient method.
        
        This avoids explicit Hessian construction and inversion.
        """
        b = -gradient  # Right-hand side
        x = torch.zeros_like(gradient)
        r = b.clone()
        p = r.clone()
        
        rsold = (r * r).sum()
        
        for i in range(max_iter):
            # Compute H*p using Hessian-vector product
            Hp = self._compute_hessian_vector_product(gradient, actions, p)
            
            # Add damping
            Hp = Hp + self.damping * p
            
            pHp = (p * Hp).sum()
            
            if pHp.abs() < 1e-10:
                break
            
            alpha = rsold / pHp
            x = x + alpha * p
            r = r - alpha * Hp
            
            rsnew = (r * r).sum()
            
            if rsnew < 1e-10:
                break
            
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
        
        return x
    
    def optimize(
        self,
        cost_fn: Callable,
        initial_actions: Optional[torch.Tensor] = None,
        num_iterations: int = 50,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Optimize actions using Newton method.
        
        Args:
            cost_fn: Function that takes actions (1, T, action_dim) -> scalar cost
            initial_actions: Initial action sequence (1, T, action_dim)
            num_iterations: Number of Newton iterations
            verbose: Print progress
            
        Returns:
            best_actions: (1, T, action_dim) optimized actions
            info: Dict with cost_history
        """
        # For L-BFGS, use PyTorch's built-in optimizer
        if self.method == "lbfgs":
            return self._optimize_lbfgs(cost_fn, initial_actions, num_iterations, verbose)
        
        # Initialize actions
        if initial_actions is None:
            self.actions = torch.randn(
                1, self.horizon, self.action_dim, device=self.device
            ) * 0.1
        else:
            self.actions = initial_actions.clone().to(self.device)
        
        cost_history = []
        best_cost = float('inf')
        best_actions = self.actions.clone()
        
        for iteration in range(num_iterations):
            self.actions = self.actions.detach()
            self.actions.requires_grad_(True)
            
            # Compute Newton step
            if self.method == "gauss_newton":
                step, cost, gradient = self._gauss_newton_step(self.actions, cost_fn)
            elif self.method == "damped_newton":
                step, cost, gradient = self._damped_newton_step(self.actions, cost_fn)
            elif self.method == "cg":
                # Conjugate gradient method
                cost = cost_fn(self.actions).sum()
                cost.backward(create_graph=True)
                gradient = self.actions.grad.clone()
                step = self._conjugate_gradient_solve(self.actions, gradient, cost_fn)
                cost = cost.item()
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Line search (optional)
            if self.line_search:
                step_size = self._backtracking_line_search(
                    self.actions, step, cost, cost_fn
                )
            else:
                step_size = self.lr
            
            # Update actions
            with torch.no_grad():
                self.actions = self.actions + step_size * step
                
                # Clip to bounds
                if self.action_bounds is not None:
                    self.actions = self.actions.clamp(
                        self.action_bounds[0], self.action_bounds[1]
                    )
            
            # Evaluate new cost
            with torch.no_grad():
                new_cost = cost_fn(self.actions).sum().item()
            
            cost_history.append(new_cost)
            
            # Track best
            if new_cost < best_cost:
                best_cost = new_cost
                best_actions = self.actions.clone()
            
            if verbose and (iteration % 10 == 0 or iteration == num_iterations - 1):
                print(f"Iteration {iteration}: cost = {new_cost:.6f}, "
                      f"step_size = {step_size:.4f}, ||grad|| = {gradient.norm().item():.6f}")
        
        info = {
            "cost_history": cost_history,
            "best_cost": best_cost,
            "final_cost": cost_history[-1],
            "num_iterations": num_iterations,
        }
        
        return best_actions, info
    
    def _backtracking_line_search(
        self, 
        actions, 
        step, 
        current_cost, 
        cost_fn,
        alpha=0.3,
        beta=0.8,
        max_iter=10
    ):
        """
        Backtracking line search to find appropriate step size.
        
        Args:
            alpha: Sufficient decrease parameter (typically 0.1-0.3)
            beta: Backtracking parameter (typically 0.5-0.9)
        """
        step_size = self.lr
        
        with torch.no_grad():
            for _ in range(max_iter):
                # Try step
                new_actions = actions + step_size * step
                
                # Clip to bounds
                if self.action_bounds is not None:
                    new_actions = new_actions.clamp(
                        self.action_bounds[0], self.action_bounds[1]
                    )
                
                # Evaluate cost
                new_cost = cost_fn(new_actions).sum().item()
                
                # Check sufficient decrease (Armijo condition)
                if new_cost < current_cost:
                    return step_size
                
                # Backtrack
                step_size *= beta
            
        return step_size
    
    def _optimize_lbfgs(
        self,
        cost_fn: Callable,
        initial_actions: Optional[torch.Tensor],
        num_iterations: int,
        verbose: bool,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Optimize using L-BFGS (Limited-memory BFGS).
        
        L-BFGS is a quasi-Newton method that approximates the inverse Hessian
        using only gradient information from recent iterations.
        """
        # Initialize actions
        if initial_actions is None:
            self.actions = torch.randn(
                1, self.horizon, self.action_dim, device=self.device
            ) * 0.1
        else:
            self.actions = initial_actions.clone().to(self.device)
        
        self.actions.requires_grad_(True)
        
        # Create L-BFGS optimizer
        optimizer = torch.optim.LBFGS(
            [self.actions],
            lr=self.lr,
            max_iter=20,
            history_size=self.history_size,
            line_search_fn="strong_wolfe",
        )
        
        cost_history = []
        iteration_count = [0]
        
        def closure():
            optimizer.zero_grad()
            cost = cost_fn(self.actions).sum()
            cost.backward()
            
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_([self.actions], max_norm=10.0)
            
            cost_val = cost.item()
            cost_history.append(cost_val)
            
            if verbose and (iteration_count[0] % 10 == 0):
                print(f"Iteration {iteration_count[0]}: cost = {cost_val:.6f}")
            
            iteration_count[0] += 1
            return cost
        
        # Run optimization
        for _ in range(num_iterations):
            optimizer.step(closure)
            
            # Clip to bounds after each step
            with torch.no_grad():
                if self.action_bounds is not None:
                    self.actions.data = self.actions.data.clamp(
                        self.action_bounds[0], self.action_bounds[1]
                    )
            
            if iteration_count[0] >= num_iterations:
                break
        
        info = {
            "cost_history": cost_history,
            "best_cost": min(cost_history) if cost_history else float('inf'),
            "final_cost": cost_history[-1] if cost_history else float('inf'),
            "num_iterations": len(cost_history),
        }
        
        return self.actions.detach(), info





# ============================================================================
# CMA-ES Optimizer
# ============================================================================

class CMAESOptimizer(BaseOptimizer):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    
    A sampling-based black-box optimization algorithm.
    Requires: pip install cma
    """
    
    def __init__(
        self,
        action_dim: int,
        horizon: int,
        population_size: Optional[int] = None,
        sigma: float = 0.5,
        action_bounds: Optional[Tuple[float, float]] = None,
        device: str = "cpu",
    ):
        """
        Args:
            population_size: Number of samples per iteration (default: 4 + 3*log(dim))
            sigma: Initial standard deviation
            action_bounds: (min, max) bounds for actions
        """
        super().__init__(action_dim, horizon, device)
        
        try:
            import cma
            self.cma = cma
        except ImportError:
            raise ImportError("CMA-ES requires: pip install cma")
        
        self.population_size = population_size
        self.sigma = sigma
        self.action_bounds = action_bounds
        
        self.es = None
    
    def reset(self):
        """Reset optimizer state."""
        self.es = None
    
    def optimize(
        self,
        cost_fn: Callable,
        initial_actions: Optional[torch.Tensor] = None,
        num_iterations: int = 100,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Optimize using CMA-ES.
        
        Args:
            cost_fn: Function that takes actions (B, T, action_dim) -> (B,) costs
            initial_actions: Initial mean (1, T, action_dim)
            num_iterations: Number of CMA-ES generations
            verbose: Print progress
            
        Returns:
            best_actions: (1, T, action_dim) optimized actions
            info: Dict with optimization info
        """
        # Flatten action dimensions
        total_dim = self.horizon * self.action_dim
        
        # Initialize
        if initial_actions is None:
            x0 = np.zeros(total_dim)
        else:
            x0 = initial_actions.cpu().numpy().flatten()
        
        # Setup CMA-ES options
        opts = {
            'popsize': self.population_size,
            'verbose': -1 if not verbose else 1,
        }
        
        if self.action_bounds is not None:
            opts['bounds'] = [self.action_bounds[0], self.action_bounds[1]]
        
        # Initialize CMA-ES
        self.es = self.cma.CMAEvolutionStrategy(x0, self.sigma, opts)
        
        # Optimization loop
        cost_history = []
        
        for iteration in range(num_iterations):
            # Sample population
            solutions = self.es.ask()
            
            # Evaluate solutions
            costs = []
            for sol in solutions:
                # Reshape to (1, T, action_dim)
                actions = torch.tensor(
                    sol.reshape(1, self.horizon, self.action_dim),
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Compute cost
                with torch.no_grad():
                    cost = cost_fn(actions).item()
                costs.append(cost)
            
            # Update CMA-ES
            self.es.tell(solutions, costs)
            
            # Track best
            best_cost = min(costs)
            cost_history.append(best_cost)
            
            if verbose and (iteration % 10 == 0 or iteration == num_iterations - 1):
                print(f"Generation {iteration}: best cost = {best_cost:.6f}")
        
        # Get best solution
        best_solution = self.es.result.xbest
        best_actions = torch.tensor(
            best_solution.reshape(1, self.horizon, self.action_dim),
            dtype=torch.float32,
            device=self.device
        )
        
        info = {
            "cost_history": cost_history,
            "best_cost": self.es.result.fbest,
            "num_iterations": num_iterations,
            "num_evaluations": self.es.result.evaluations,
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
    
    Supports:
    - Gradient-based optimization
    - Sampling-based optimization (CMA-ES, PIC)
    - Flexible cost functions
    - Multiple optimization strategies
    """
    
    def __init__(
        self,
        jepa_model,
        cost_function: CostFunction,
        optimizer: BaseOptimizer,
        device: str = "cpu",
    ):
        """
        Args:
            jepa_model: Trained JEPA model with prediction methods
            cost_function: CostFunction instance
            optimizer: Optimizer instance (GradientOptimizer, CMAESOptimizer, etc.)
            device: Device for computation
        """
        self.model = jepa_model
        self.model.eval()
        self.cost_fn = cost_function
        self.optimizer = optimizer
        self.device = device
    
    def plan(
        self,
        initial_state: torch.Tensor,
        initial_frame: torch.Tensor,
        goal_state: torch.Tensor,
        num_iterations: int = 100,
        initial_actions: Optional[torch.Tensor] = None,
        verbose: bool = False,
        return_trajectory: bool = False,
    ) -> Dict:
        """
        Plan action sequence from initial state to goal state.
        
        Args:
            initial_state: (1, state_dim) starting state
            initial_frame: (1, 3, 64, 64) starting frame
            goal_state: (1, state_dim) goal state
            num_iterations: Number of optimization iterations
            initial_actions: (1, T, action_dim) initial guess
            verbose: Print optimization progress
            return_trajectory: If True, return predicted trajectory
            
        Returns:
            Dict with:
                - actions: (1, T, action_dim) optimized action sequence
                - final_cost: Final cost value
                - optimization_info: Dict with optimizer-specific info
                - predicted_trajectory: (optional) predicted states/frames
        """
        initial_state = initial_state.to(self.device)
        initial_frame = initial_frame.to(self.device)
        goal_state = goal_state.to(self.device)
        
        # Define cost function wrapper for optimizer
        def rollout_cost(actions: torch.Tensor) -> torch.Tensor:
            """
            Evaluate cost for a batch of action sequences.
            
            Args:
                actions: (B, T, action_dim)
            Returns:
                costs: (B,)
            """
            B = actions.shape[0]
            
            # Expand initial conditions for batch
            init_state = initial_state.expand(B, -1)
            init_frame = initial_frame.expand(B, -1, -1, -1)
            goal = goal_state.expand(B, -1)
            
            # Rollout predictions
            # Enable gradients for both GradientOptimizer and NewtonOptimizer
            needs_grad = isinstance(self.optimizer, (GradientOptimizer, NewtonOptimizer))
            with torch.set_grad_enabled(needs_grad):
                rollout_result = self.model.rollout_predictions(
                    initial_state=init_state,
                    initial_frame=init_frame,
                    actions=actions,
                    decode_every=0,  # Only need latent trajectory
                )
            
            # Get final predicted state
            final_latent = rollout_result["latent_trajectory"][-1]
            final_state_latent = final_latent["z_state"]
            
            # Decode final state
            with torch.set_grad_enabled(needs_grad):
                final_state_pred = self.model.proprio_decoder(final_state_latent)
            
            # Compute cost
            costs = self.cost_fn(final_state_pred, goal)
            
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
                trajectory = self.model.rollout_predictions(
                    initial_state=initial_state,
                    initial_frame=initial_frame,
                    actions=best_actions,
                    decode_every=1,
                )
            result["predicted_trajectory"] = trajectory
        
        return result
    
    def reset(self):
        """Reset optimizer state."""
        self.optimizer.reset()