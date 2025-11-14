"""
Uncertainty-Aware Planning for JEPA

Extends the base planning framework to incorporate epistemic uncertainty estimation
using Monte Carlo Dropout as a Bayesian approximation. This helps guide the planner
toward on-manifold states where the model is more confident.

Key Concepts:
    - Enable dropout during inference to get stochastic predictions
    - Run multiple forward passes to estimate prediction variance
    - Add variance to cost function to penalize high-uncertainty states
    - Encourages planner to stay in well-modeled regions of latent space

Usage Example:
    ```python
    # Create uncertainty-aware cost function
    cost_fn = UncertaintyCostFunction(
        base_cost_type="mse",
        uncertainty_weight=0.1,  # Balance between goal reaching and confidence
        num_samples=10,          # MC dropout samples
        use_cls_token=True,
        use_state_token=True
    )
    
    # Use with any optimizer
    planner = LatentSpacePlanner(jepa_model, cost_fn, optimizer)
    result = planner.plan(...)
    ```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty estimation results"""
    mean_latents: Dict[str, torch.Tensor]
    variance: Dict[str, torch.Tensor]
    std_dev: Dict[str, torch.Tensor]
    confidence: torch.Tensor  # 1 / (1 + total_variance)


class DropoutEnabler:
    """
    Context manager to temporarily enable dropout in a model during inference.
    
    Usage:
        with DropoutEnabler(model):
            # Model now has dropout active even in eval mode
            predictions = model(x)
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_training_states = {}
    
    def __enter__(self):
        """Enable dropout by switching relevant modules to train mode"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout):
                self.original_training_states[name] = module.training
                module.train()  # Enable dropout
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original training states"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout):
                if name in self.original_training_states:
                    module.train(self.original_training_states[name])


class UncertaintyEstimator:
    """
    Estimates epistemic uncertainty using Monte Carlo Dropout.
    
    This treats dropout as a Bayesian approximation - by running multiple
    forward passes with dropout enabled, we get a distribution over predictions
    which approximates the model's uncertainty.
    """
    
    def __init__(self, num_samples: int = 10):
        """
        Args:
            num_samples: Number of MC dropout samples to collect
        """
        self.num_samples = num_samples
    
    def estimate_uncertainty(
        self,
        predictor: nn.Module,
        z_cls: torch.Tensor,
        z_patches: torch.Tensor,
        z_state: torch.Tensor,
        z_action: torch.Tensor,
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty by running multiple forward passes with dropout.
        
        Args:
            predictor: The JEPA predictor model
            z_cls: (B, D) or (B, 1, D) CLS token
            z_patches: (B, Np, D) patch tokens
            z_state: (B, D) or (B, 1, D) state token
            z_action: (B, D) action embedding
            
        Returns:
            UncertaintyEstimate containing mean predictions and variances
        """
        # Ensure proper shapes
        if z_cls.dim() == 2:
            z_cls = z_cls.unsqueeze(1)
        if z_state.dim() == 2:
            z_state = z_state.unsqueeze(1)
        
        B, D = z_cls.shape[0], z_cls.shape[-1]
        Np = z_patches.shape[1]
        
        # Collect samples
        samples_cls = []
        samples_patches = []
        samples_state = []
        
        with DropoutEnabler(predictor):
            with torch.no_grad():  # No gradients needed for uncertainty estimation
                for _ in range(self.num_samples):
                    pred_cls, pred_patches, pred_state = predictor(
                        z_cls, z_patches, z_state, z_action
                    )
                    samples_cls.append(pred_cls)
                    samples_patches.append(pred_patches)
                    samples_state.append(pred_state)
        
        # Stack samples: (num_samples, B, ...)
        samples_cls = torch.stack(samples_cls, dim=0)         # (S, B, D)
        samples_patches = torch.stack(samples_patches, dim=0) # (S, B, Np, D)
        samples_state = torch.stack(samples_state, dim=0)     # (S, B, D)
        
        # Compute statistics
        mean_cls = samples_cls.mean(dim=0)                    # (B, D)
        mean_patches = samples_patches.mean(dim=0)            # (B, Np, D)
        mean_state = samples_state.mean(dim=0)                # (B, D)
        
        var_cls = samples_cls.var(dim=0)                      # (B, D)
        var_patches = samples_patches.var(dim=0)              # (B, Np, D)
        var_state = samples_state.var(dim=0)                  # (B, D)
        
        # Aggregate variance as a single confidence score per sample
        total_var = (
            var_cls.mean(dim=-1) +                            # (B,)
            var_patches.mean(dim=(-2, -1)) +                  # (B,)
            var_state.mean(dim=-1)                            # (B,)
        )
        confidence = 1.0 / (1.0 + total_var)                  # (B,)
        
        return UncertaintyEstimate(
            mean_latents={
                'z_cls': mean_cls,
                'z_patches': mean_patches,
                'z_state': mean_state,
            },
            variance={
                'z_cls': var_cls,
                'z_patches': var_patches,
                'z_state': var_state,
            },
            std_dev={
                'z_cls': var_cls.sqrt(),
                'z_patches': var_patches.sqrt(),
                'z_state': var_state.sqrt(),
            },
            confidence=confidence,
        )


class UncertaintyCostFunction:
    """
    Cost function that combines task-based cost with epistemic uncertainty penalty.
    
    Total cost = task_cost + uncertainty_weight * uncertainty_cost
    
    The uncertainty penalty encourages the planner to stay in regions of the
    latent space where the model is confident (low variance in predictions).
    """
    
    def __init__(
        self,
        base_cost_type: str = "mse",
        custom_cost_fn: Optional[Callable] = None,
        uncertainty_weight: float = 0.1,
        num_samples: int = 10,
        use_cls_token: bool = True,
        use_patch_tokens: bool = False,
        use_state_token: bool = True,
        cls_weight: float = 1.0,
        patch_weight: float = 0.0,
        state_weight: float = 1.0,
        uncertainty_aggregation: str = "mean",  # "mean", "max", or "sum"
    ):
        """
        Args:
            base_cost_type: Type of base distance metric ("mse", "mae", "custom")
            custom_cost_fn: Custom cost function if base_cost_type="custom"
            uncertainty_weight: Weight for uncertainty penalty term
            num_samples: Number of MC dropout samples for uncertainty estimation
            use_cls_token: Include CLS token in cost computation
            use_patch_tokens: Include patch tokens in cost computation
            use_state_token: Include state token in cost computation
            cls_weight: Weight for CLS token in base cost
            patch_weight: Weight for patch tokens in base cost
            state_weight: Weight for state token in base cost
            uncertainty_aggregation: How to aggregate variance ("mean", "max", "sum")
        """
        self.base_cost_type = base_cost_type
        self.custom_cost_fn = custom_cost_fn
        self.uncertainty_weight = uncertainty_weight
        self.num_samples = num_samples
        
        self.use_cls_token = use_cls_token
        self.use_patch_tokens = use_patch_tokens
        self.use_state_token = use_state_token
        
        self.cls_weight = cls_weight
        self.patch_weight = patch_weight
        self.state_weight = state_weight
        
        self.uncertainty_aggregation = uncertainty_aggregation
        
        self.uncertainty_estimator = UncertaintyEstimator(num_samples)
    
    def compute_distance(self, pred: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Compute distance between tensors based on base_cost_type."""
        if self.base_cost_type == "mse":
            return torch.mean((pred - goal) ** 2, dim=-1)
        elif self.base_cost_type == "mae":
            return torch.mean(torch.abs(pred - goal), dim=-1)
        elif self.base_cost_type == "custom":
            return self.custom_cost_fn(pred, goal)
        else:
            raise ValueError(f"Unknown cost_type: {self.base_cost_type}")
    
    def aggregate_variance(self, variance: torch.Tensor) -> torch.Tensor:
        """
        Aggregate variance tensor to scalar per sample.
        
        Args:
            variance: (..., D) tensor of variances
            
        Returns:
            Scalar variance per sample
        """
        if self.uncertainty_aggregation == "mean":
            return variance.mean(dim=-1)
        elif self.uncertainty_aggregation == "max":
            return variance.max(dim=-1)[0]
        elif self.uncertainty_aggregation == "sum":
            return variance.sum(dim=-1)
        else:
            raise ValueError(f"Unknown aggregation: {self.uncertainty_aggregation}")
    
    def compute_uncertainty_cost(
        self,
        variance: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute uncertainty penalty from variance estimates.
        
        Args:
            variance: Dict with 'z_cls', 'z_patches', 'z_state' variances
            
        Returns:
            uncertainty_cost: (B,) uncertainty penalty per sample
        """
        costs = []
        
        if self.use_cls_token:
            var_cls = self.aggregate_variance(variance['z_cls'])
            costs.append(self.cls_weight * var_cls)
        
        if self.use_patch_tokens:
            var_patches = self.aggregate_variance(
                variance['z_patches'].flatten(-2, -1)
            )
            costs.append(self.patch_weight * var_patches)
        
        if self.use_state_token:
            var_state = self.aggregate_variance(variance['z_state'])
            costs.append(self.state_weight * var_state)
        
        if not costs:
            raise ValueError("At least one token type must be used")
        
        return torch.stack(costs, dim=0).sum(dim=0)
    
    def __call__(
        self,
        pred_latents: Dict[str, torch.Tensor],
        goal_latents: Dict[str, torch.Tensor],
        predictor: Optional[nn.Module] = None,
        z_action: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        """
        Compute total cost = base_cost + uncertainty_weight * uncertainty_cost.
        
        Args:
            pred_latents: Dict with 'z_cls', 'z_patches', 'z_state' predictions
            goal_latents: Dict with 'z_cls', 'z_patches', 'z_state' goals
            predictor: JEPA predictor model (required if uncertainty_weight > 0)
            z_action: (B, D) action embeddings (required if uncertainty_weight > 0)
            return_dict: If True, return detailed breakdown
            
        Returns:
            total_cost: (B,) scalar cost per sample
            OR dict with cost breakdown if return_dict=True
        """
        # Base task cost (goal reaching)
        base_costs = []
        
        if self.use_cls_token:
            cls_cost = self.compute_distance(
                pred_latents['z_cls'],
                goal_latents['z_cls']
            )
            base_costs.append(self.cls_weight * cls_cost)
        
        if self.use_patch_tokens:
            patch_cost = self.compute_distance(
                pred_latents['z_patches'].flatten(-2, -1),
                goal_latents['z_patches'].flatten(-2, -1)
            )
            base_costs.append(self.patch_weight * patch_cost)
        
        if self.use_state_token:
            state_cost = self.compute_distance(
                pred_latents['z_state'],
                goal_latents['z_state']
            )
            base_costs.append(self.state_weight * state_cost)
        
        if not base_costs:
            raise ValueError("At least one token type must be used")
        
        base_cost = torch.stack(base_costs, dim=0).sum(dim=0)
        
        # Uncertainty cost (epistemic uncertainty penalty)
        uncertainty_cost = torch.zeros_like(base_cost)
        
        if self.uncertainty_weight > 0.0:
            if predictor is None or z_action is None:
                raise ValueError(
                    "predictor and z_action required when uncertainty_weight > 0"
                )
            
            # Estimate uncertainty at predicted state
            uncertainty_est = self.uncertainty_estimator.estimate_uncertainty(
                predictor,
                pred_latents['z_cls'],
                pred_latents['z_patches'],
                pred_latents['z_state'],
                z_action,
            )
            
            uncertainty_cost = self.compute_uncertainty_cost(uncertainty_est.variance)
        
        # Total cost
        total_cost = base_cost + self.uncertainty_weight * uncertainty_cost
        
        if return_dict:
            return {
                'total_cost': total_cost,
                'base_cost': base_cost,
                'uncertainty_cost': uncertainty_cost,
                'confidence': 1.0 / (1.0 + uncertainty_cost) if self.uncertainty_weight > 0 else torch.ones_like(total_cost),
            }
        
        return total_cost


class UncertaintyAwarePlanner:
    """
    Extension of BasePlanner that incorporates uncertainty estimation.
    
    This is a wrapper around your existing planner that adds uncertainty-aware
    cost functions and provides utilities for analyzing plan confidence.
    """
    
    def __init__(
        self,
        base_planner,  # Your existing LatentSpacePlanner
        uncertainty_weight: float = 0.1,
        num_uncertainty_samples: int = 10,
    ):
        """
        Args:
            base_planner: Your existing LatentSpacePlanner instance
            uncertainty_weight: Weight for uncertainty penalty
            num_uncertainty_samples: Number of MC dropout samples
        """
        self.base_planner = base_planner
        self.uncertainty_weight = uncertainty_weight
        self.num_uncertainty_samples = num_uncertainty_samples
        
        # Wrap the cost function to add uncertainty
        original_cost_fn = base_planner.cost_fn
        self.cost_fn = self._create_uncertainty_cost_fn(original_cost_fn)
        base_planner.cost_fn = self.cost_fn
    
    def _create_uncertainty_cost_fn(self, original_cost_fn):
        """Create uncertainty-aware version of original cost function."""
        # This is a placeholder - you'd need to adapt based on your CostFunction class
        return UncertaintyCostFunction(
            base_cost_type="mse",
            uncertainty_weight=self.uncertainty_weight,
            num_samples=self.num_uncertainty_samples,
            use_cls_token=True,
            use_state_token=True,
        )
    
    def plan(self, *args, analyze_uncertainty: bool = True, **kwargs):
        """
        Plan with optional uncertainty analysis.
        
        Args:
            analyze_uncertainty: If True, compute uncertainty along the planned trajectory
            *args, **kwargs: Arguments passed to base planner
            
        Returns:
            Planning result with optional uncertainty analysis
        """
        result = self.base_planner.plan(*args, **kwargs)
        
        if analyze_uncertainty:
            result['uncertainty_analysis'] = self._analyze_plan_uncertainty(result)
        
        return result
    
    def _analyze_plan_uncertainty(self, plan_result: Dict) -> Dict:
        """
        Analyze uncertainty along a planned trajectory.
        
        Returns:
            Dict containing uncertainty metrics for each timestep
        """
        # This would need to be implemented based on your specific planner interface
        # Placeholder for now
        return {
            'mean_confidence': None,
            'min_confidence': None,
            'confidence_trajectory': None,
        }


# ============================================================================
# Integration Examples
# ============================================================================

def example_uncertainty_planning():
    """
    Example showing how to use uncertainty-aware planning.
    
    This demonstrates the full workflow from creating the cost function
    to running planning with uncertainty penalties.
    """
    # Assume you have your JEPA model and optimizer already set up
    # jepa_model = ...
    # optimizer = GradientOptimizer(...)
    
    # 1. Create uncertainty-aware cost function
    cost_fn = UncertaintyCostFunction(
        base_cost_type="mse",
        uncertainty_weight=0.1,     # Tune this: higher = more conservative
        num_samples=10,             # More samples = better estimate but slower
        use_cls_token=True,
        use_state_token=True,
        uncertainty_aggregation="mean",
    )
    
    # 2. Use with your existing planner
    # planner = LatentSpacePlanner(jepa_model, cost_fn, optimizer)
    
    # 3. During planning, the cost function will:
    #    - Compute task cost (distance to goal)
    #    - Estimate uncertainty at predicted states
    #    - Add uncertainty penalty to favor confident regions
    
    # 4. Plan as usual
    # result = planner.plan(
    #     current_state=...,
    #     goal_state=...,
    #     horizon=20,
    #     num_iterations=100,
    # )
    
    print("Uncertainty-aware planning example")


def example_adaptive_uncertainty_weight():
    """
    Example showing how to adaptively adjust uncertainty weight during planning.
    
    You might want to:
    - Start with high uncertainty weight (conservative exploration)
    - Gradually reduce it as the plan gets closer to goal
    - Increase it again in challenging regions
    """
    
    class AdaptiveUncertaintyScheduler:
        def __init__(self, initial_weight=0.5, final_weight=0.05):
            self.initial_weight = initial_weight
            self.final_weight = final_weight
        
        def get_weight(self, progress: float) -> float:
            """
            Args:
                progress: Planning progress from 0 to 1
            """
            # Linear decay from initial to final weight
            return self.initial_weight + (self.final_weight - self.initial_weight) * progress
    
    # Usage would involve updating cost_fn.uncertainty_weight during optimization
    print("Adaptive uncertainty scheduling example")


if __name__ == "__main__":
    print("Uncertainty-Aware Planning Module")
    print("=" * 50)
    print()
    print("This module adds epistemic uncertainty estimation to JEPA planning.")
    print("Key features:")
    print("  - Monte Carlo Dropout for uncertainty estimation")
    print("  - Uncertainty penalty in cost function")
    print("  - Guides planner toward confident (on-manifold) states")
    print()
    print("See docstrings and examples for usage details.")