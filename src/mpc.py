"""
Model Predictive Control (MPC) for JEPA

Implements receding-horizon MPC with replanning every K steps.
Reuses all existing optimizers and planning infrastructure.

Key Features:
    - Modular: Works with any optimizer (Gradient, CMA-ES, PIC, etc.)
    - Efficient: Reuses previous solutions as warm starts
    - Flexible: Configurable replanning frequency
    - Pure latent space: No decoding during control loop
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Callable
from src.planning import LatentSpacePlanner, BaseOptimizer, CostFunction


class ModelPredictiveController:
    """
    Model Predictive Control in JEPA latent space.
    
    MPC repeatedly:
    1. Plans an action sequence over horizon H
    2. Executes the first K actions
    3. Observes new state
    4. Replans from new state (warm-starting with previous solution)
    
    This implements the standard receding-horizon control paradigm.
    """
    
    def __init__(
        self,
        planner: LatentSpacePlanner,
        horizon: int,
        replan_every: int = 1,
        num_iterations_per_plan: int = 50,
        warm_start: bool = True,
        warm_start_shift: bool = True,
    ):
        """
        Args:
            planner: LatentSpacePlanner instance (with optimizer already configured)
            horizon: Planning horizon (number of steps to plan ahead)
            replan_every: Replan every K steps (K=1 means replan every step)
            num_iterations_per_plan: Optimization iterations per planning call
            warm_start: Use previous solution as initial guess for next plan
            warm_start_shift: If True, shift previous solution and append zeros/noise
                            If False, just use previous solution as-is
        """
        self.planner = planner
        self.horizon = horizon
        self.replan_every = replan_every
        self.num_iterations = num_iterations_per_plan
        self.warm_start = warm_start
        self.warm_start_shift = warm_start_shift
        
        # Cache for warm starting
        self._last_action_plan = None
        
        # Statistics
        self.total_replans = 0
        self.total_steps_executed = 0
    
    def reset(self):
        """Reset controller state (clears cached action plan)."""
        self._last_action_plan = None
        self.total_replans = 0
        self.total_steps_executed = 0
        self.planner.optimizer.reset()
    
    def _get_initial_actions(self) -> Optional[torch.Tensor]:
        """
        Get initial action guess for next planning call.
        
        Returns:
            (1, H, action_dim) initial actions or None
        """
        if not self.warm_start or self._last_action_plan is None:
            return None
        
        prev_actions = self._last_action_plan  # (1, H, A)
        
        if not self.warm_start_shift:
            # Just reuse previous plan as-is
            return prev_actions.clone()
        
        # Shift previous plan: drop executed actions, append new ones
        # If we executed K actions, shift by K
        num_executed = min(self.replan_every, prev_actions.shape[1])
        
        if num_executed >= prev_actions.shape[1]:
            # Executed entire plan, start fresh
            return None
        
        # Shift: [a_K, a_{K+1}, ..., a_{H-1}] + [new_actions]
        shifted = prev_actions[:, num_executed:, :]  # (1, H-K, A)
        
        # Append new actions (zeros or small noise)
        num_new = num_executed
        new_actions = torch.randn(
            1, num_new, prev_actions.shape[2],
            device=prev_actions.device
        ) * 0.01  # Small noise
        
        initial_guess = torch.cat([shifted, new_actions], dim=1)  # (1, H, A)
        
        return initial_guess
    
    def control_step(
        self,
        current_state: torch.Tensor,
        current_frame: torch.Tensor,
        goal_state: torch.Tensor,
        goal_frame: torch.Tensor,
        verbose: bool = False,
    ) -> Dict:
        """
        Single MPC control step: plan and return next action(s) to execute.
        
        Args:
            current_state: (1, state_dim) current state
            current_frame: (1, 3, H, W) current observation
            goal_state: (1, state_dim) goal state
            goal_frame: (1, 3, H, W) goal frame
            verbose: Print planning progress
            
        Returns:
            Dict with:
                - actions_to_execute: (1, K, action_dim) actions to execute
                - full_plan: (1, H, action_dim) full planned sequence
                - planning_info: Dict with optimization info
                - should_replan_next: bool indicating if next step requires replanning
        """
        # Determine if we need to replan
        should_replan = (self.total_steps_executed % self.replan_every == 0)
        
        if should_replan or self._last_action_plan is None:
            # Plan new action sequence
            initial_actions = self._get_initial_actions()
            
            result = self.planner.plan(
                initial_state=current_state,
                initial_frame=current_frame,
                goal_state=goal_state,
                goal_frame=goal_frame,
                num_iterations=self.num_iterations,
                initial_actions=initial_actions,
                verbose=verbose,
                return_trajectory=False,
            )
            
            self._last_action_plan = result['actions']  # (1, H, A)
            self.total_replans += 1
            
            if verbose:
                print(f"[MPC] Replanned (replan #{self.total_replans}), "
                      f"cost: {result['final_cost']:.6f}")
        
        # Extract actions to execute (first K actions from plan)
        K = min(self.replan_every, self._last_action_plan.shape[1])
        actions_to_execute = self._last_action_plan[:, :K, :]  # (1, K, A)
        
        # Update step counter
        self.total_steps_executed += K
        
        # Check if we need to replan next time
        next_step = self.total_steps_executed
        should_replan_next = (next_step % self.replan_every == 0)
        
        return {
            "actions_to_execute": actions_to_execute,
            "full_plan": self._last_action_plan.clone(),
            "planning_info": result if should_replan else None,
            "should_replan_next": should_replan_next,
            "total_replans": self.total_replans,
            "total_steps": self.total_steps_executed,
        }
    
    def rollout_control(
        self,
        initial_state: torch.Tensor,
        initial_frame: torch.Tensor,
        goal_state: torch.Tensor,
        goal_frame: torch.Tensor,
        max_steps: int,
        early_stop_threshold: Optional[float] = None,
        verbose: bool = False,
        save_trajectory: bool = True,
    ) -> Dict:
        """
        Execute full MPC rollout: replan and execute until goal reached or max_steps.
        
        Args:
            initial_state: (1, state_dim) starting state
            initial_frame: (1, 3, H, W) starting frame
            goal_state: (1, state_dim) goal state
            goal_frame: (1, 3, H, W) goal frame
            max_steps: Maximum number of steps to execute
            early_stop_threshold: Stop if cost below this threshold
            verbose: Print progress
            save_trajectory: Save full state/frame/action trajectory
            
        Returns:
            Dict with:
                - success: bool indicating if goal was reached
                - final_cost: Final cost to goal
                - total_replans: Number of times we replanned
                - steps_executed: Number of steps taken
                - trajectory: (optional) List of states, frames, actions
        """
        self.reset()
        
        current_state = initial_state.clone()
        current_frame = initial_frame.clone()
        
        # Storage for trajectory
        states = [current_state.cpu()] if save_trajectory else None
        frames = [current_frame.cpu()] if save_trajectory else None
        actions_executed = [] if save_trajectory else None
        costs = []
        
        # Encode goal once
        with torch.no_grad():
            goal_latents = self.planner.model.encode_state_and_frame(
                state=goal_state,
                frame=goal_frame
            )
        
        for step in range(max_steps):
            # Get next action(s) via MPC
            control_result = self.control_step(
                current_state=current_state,
                current_frame=current_frame,
                goal_state=goal_state,
                goal_frame=goal_frame,
                verbose=verbose,
            )
            
            actions = control_result['actions_to_execute']  # (1, K, A)
            K = actions.shape[1]
            
            # Execute actions in environment (via model prediction)
            with torch.no_grad():
                # Rollout K steps
                rollout_result = self.planner.model.rollout_predictions(
                    initial_state=current_state,
                    initial_frame=current_frame,
                    actions=actions,
                    decode_every=0,  # Stay in latent space
                )
                
                # Get final state after K steps
                final_latent = rollout_result['latent_trajectory'][-1]
                
                # Decode only to get next state/frame for next iteration
                # (In real environment, you'd get this from actual execution)
                next_state = self.planner.model.proprio_decoder(final_latent['z_state'])
                
                # For frame, we need to reconstruct from cls + patches
                patches_with_cls = torch.cat([
                    final_latent['z_cls'].unsqueeze(1),  # (B, 1, D)
                    final_latent['z_patches']             # (B, N, D)
                ], dim=1)
                next_frame = self.planner.model.visual_decoder(patches_with_cls)
            
            # Compute cost to goal
            with torch.no_grad():
                current_latents = self.planner.model.encode_state_and_frame(
                    state=next_state,
                    frame=next_frame
                )
                cost = self.planner.cost_fn(current_latents, goal_latents).item()
            
            costs.append(cost)
            
            # Save trajectory
            if save_trajectory:
                # Save all intermediate states if K > 1
                for k in range(K):
                    if k < len(rollout_result['latent_trajectory']):
                        lat = rollout_result['latent_trajectory'][k]
                        s = self.planner.model.proprio_decoder(lat['z_state'])
                        states.append(s.cpu())
                        
                        # Reconstruct frame
                        p_cls = torch.cat([
                            lat['z_cls'].unsqueeze(1),
                            lat['z_patches']
                        ], dim=1)
                        f = self.planner.model.visual_decoder(p_cls)
                        frames.append(f.cpu())
                    
                    actions_executed.append(actions[:, k:k+1, :].cpu())
            
            if verbose:
                print(f"[MPC] Step {step * self.replan_every + K}/{max_steps}, "
                      f"cost: {cost:.6f}, replans: {control_result['total_replans']}")
            
            # Check early stopping
            if early_stop_threshold is not None and cost < early_stop_threshold:
                if verbose:
                    print(f"[MPC] Goal reached! Final cost: {cost:.6f}")
                break
            
            # Update current state for next iteration
            current_state = next_state
            current_frame = next_frame
            
            # Check if we've executed max_steps
            if control_result['total_steps'] >= max_steps:
                break
        
        # Prepare result
        result = {
            "success": cost < early_stop_threshold if early_stop_threshold else False,
            "final_cost": cost,
            "total_replans": self.total_replans,
            "steps_executed": self.total_steps_executed,
            "cost_history": costs,
        }
        
        if save_trajectory:
            result["trajectory"] = {
                "states": states,
                "frames": frames,
                "actions": actions_executed,
            }
        
        return result


class AdaptiveMPC(ModelPredictiveController):
    """
    Adaptive MPC that adjusts replanning frequency based on prediction error.
    
    If predictions are accurate, replan less frequently.
    If predictions diverge, replan more frequently.
    """
    
    def __init__(
        self,
        planner: LatentSpacePlanner,
        horizon: int,
        min_replan_every: int = 1,
        max_replan_every: int = 5,
        error_threshold_increase: float = 0.1,  # Increase frequency if error > this
        error_threshold_decrease: float = 0.05,  # Decrease frequency if error < this
        **kwargs
    ):
        """
        Args:
            min_replan_every: Minimum K (most frequent replanning)
            max_replan_every: Maximum K (least frequent replanning)
            error_threshold_increase: Replan more if prediction error exceeds this
            error_threshold_decrease: Replan less if prediction error below this
        """
        super().__init__(planner, horizon, replan_every=min_replan_every, **kwargs)
        
        self.min_replan_every = min_replan_every
        self.max_replan_every = max_replan_every
        self.error_threshold_increase = error_threshold_increase
        self.error_threshold_decrease = error_threshold_decrease
        
        self._recent_errors = []
        self._max_error_history = 10
    
    def _update_replan_frequency(self, prediction_error: float):
        """Adjust replanning frequency based on recent prediction error."""
        self._recent_errors.append(prediction_error)
        if len(self._recent_errors) > self._max_error_history:
            self._recent_errors.pop(0)
        
        avg_error = sum(self._recent_errors) / len(self._recent_errors)
        
        if avg_error > self.error_threshold_increase:
            # Increase frequency (decrease K)
            self.replan_every = max(
                self.min_replan_every,
                self.replan_every - 1
            )
        elif avg_error < self.error_threshold_decrease:
            # Decrease frequency (increase K)
            self.replan_every = min(
                self.max_replan_every,
                self.replan_every + 1
            )
    
    def control_step(self, current_state, current_frame, goal_state, 
                     goal_frame, prediction_error=None, verbose=False):
        """
        Control step with adaptive replanning.
        
        Args:
            prediction_error: Optional prediction error from last step
        """
        if prediction_error is not None:
            self._update_replan_frequency(prediction_error)
            if verbose:
                print(f"[Adaptive MPC] Adjusted replan_every to {self.replan_every}")
        
        return super().control_step(
            current_state, current_frame, goal_state, goal_frame, verbose
        )


# ============================================================================
# Utility Functions
# ============================================================================

def create_mpc(
    jepa_model,
    cost_function: CostFunction,
    optimizer: BaseOptimizer,
    horizon: int = 10,
    replan_every: int = 1,
    num_iterations_per_plan: int = 50,
    device: str = "cuda",
    adaptive: bool = False,
    **mpc_kwargs
) -> ModelPredictiveController:
    """
    Convenience function to create MPC controller.
    
    Args:
        jepa_model: JEPA model
        cost_function: CostFunction instance
        optimizer: BaseOptimizer instance
        horizon: Planning horizon
        replan_every: Replan every K steps
        num_iterations_per_plan: Optimization iterations per plan
        device: Device for computation
        adaptive: Use adaptive MPC
        **mpc_kwargs: Additional args for MPC constructor
        
    Returns:
        ModelPredictiveController or AdaptiveMPC instance
    """
    # Create planner
    planner = LatentSpacePlanner(
        jepa_model=jepa_model,
        cost_function=cost_function,
        optimizer=optimizer,
        device=device,
    )
    
    # Create MPC
    mpc_class = AdaptiveMPC if adaptive else ModelPredictiveController
    
    mpc = mpc_class(
        planner=planner,
        horizon=horizon,
        replan_every=replan_every,
        num_iterations_per_plan=num_iterations_per_plan,
        **mpc_kwargs
    )
    
    return mpc