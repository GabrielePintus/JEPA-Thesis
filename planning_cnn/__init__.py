"""
Planning module for latent space trajectory optimization in JEPA models.

This module provides multiple optimization algorithms for planning in latent space:
- GradientPlanner: Adam-based gradient optimization
- GradientPlannerWithMomentum: SGD with momentum and adaptive learning rate
- CMAESPlanner: Covariance Matrix Adaptation Evolution Strategy
- CMAESPlannerImproved: Enhanced CMA-ES with adaptive sigma and elite retention
- MPPIPlanner: Model Predictive Path Integral Control
- MPPIPlannerWithElites: Enhanced MPPI with elite retention
- MPCController: Model Predictive Control with replanning

Example usage:
    ```python
    from planning import GradientPlanner, MPCController
    import torch
    
    # Create a planner
    planner = GradientPlanner(
        jepa_model=model,
        action_dim=2,
        action_bounds=(-1.0, 1.0),
        channel_mask=visual_mask,  # Optional: only use visual channels
        lr=0.01,
        num_iterations=100
    )
    
    # Optimize a trajectory
    result = planner.optimize(
        z_init=initial_latent,
        z_goal=goal_latent,
        horizon=20
    )
    
    actions = result['actions']  # (horizon, action_dim)
    trajectory = result['trajectory']  # (horizon+1, C, H, W)
    cost = result['cost']  # scalar
    
    # Or use MPC for closed-loop control
    mpc = MPCController(
        planner=planner,
        jepa_model=model,
        replan_frequency=5,
        horizon=20
    )
    
    result = mpc.execute(
        env=environment,
        initial_state=state,
        initial_frame=frame,
        goal_state=goal_state,
        goal_frame=goal_frame,
        max_steps=100
    )
    
    # Using channel mask to only consider visual channels in cost
    visual_channels = torch.arange(16)  # First 16 channels are visual
    channel_mask = torch.zeros(18, dtype=torch.bool)
    channel_mask[visual_channels] = True
    
    planner_visual_only = CMAESPlanner(
        jepa_model=model,
        action_dim=2,
        action_bounds=(-1.0, 1.0),
        channel_mask=channel_mask,
        population_size=100,
        num_generations=50
    )
    ```

Channel Masking:
    The channel_mask parameter allows you to specify which channels to use
    in the cost computation. This is useful when you want to:
    - Only consider visual information (first 16 channels)
    - Ignore proprioceptive information (last 2 channels)
    - Focus on specific features
    
    Example:
        # Only visual channels (0-15)
        visual_mask = torch.zeros(18, dtype=torch.bool)
        visual_mask[:16] = True
        
        # Only proprioceptive channels (16-17)
        proprio_mask = torch.zeros(18, dtype=torch.bool)
        proprio_mask[16:] = True
"""

from .base_planner import BasePlanner, LatentRollout
from .gradient_planner import GradientPlanner
from .cmaes_planner import CMAESPlanner
from .mppi_planner import MPPIPlanner
from .mpc_controller import MPCController

__all__ = [
    # Base classes
    'BasePlanner',
    'LatentRollout',
    
    # Gradient-based planners
    'GradientPlanner',
    'GradientPlannerWithMomentum',
    
    # CMA-ES planners
    'CMAESPlanner',
    'CMAESPlannerImproved',
    
    # MPPI planners
    'MPPIPlanner',
    'MPPIPlannerWithElites',
    
    # MPC controller
    'MPCController',
]

__version__ = '1.0.0'
