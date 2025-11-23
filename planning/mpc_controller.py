"""
Model Predictive Control (MPC) with replanning for closed-loop control.
Combines open-loop optimization with periodic replanning in the real environment.

Features:
- Executes optimized actions and replans periodically
- Proper handling of PointMaze and similar gym environments
- Open-loop execution mode for evaluation
- Rollout comparison between predicted and actual trajectories

Key fixes in this version:
- Horizon/replan_frequency consistency enforced
- Proper state/action history alignment (T actions -> T+1 states)
- Cost history now consistently measures AFTER each action
- Clear separation between "planned" and "executed" action sequences
- Warm-start properly handles remaining plan
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
import time
from tqdm import tqdm

from .base_planner import BasePlanner, CostConfig


class MPCController:
    """
    Model Predictive Control controller with replanning.
    Executes optimized actions in the environment and replans periodically.
    """
    
    def __init__(
        self,
        planner: BasePlanner,
        jepa_model,
        replan_frequency: int = 5,
        horizon: int = 20,
        device: str = 'cuda'
    ):
        """
        Args:
            planner: Base planner to use for trajectory optimization
            jepa_model: JEPA model for encoding states
            replan_frequency: Execute k actions from each plan before replanning
            horizon: Planning horizon for each optimization (must be >= replan_frequency)
            device: Device to run on
        """
        # FIX: Enforce horizon >= replan_frequency
        if horizon < replan_frequency:
            raise ValueError(
                f"horizon ({horizon}) must be >= replan_frequency ({replan_frequency}). "
                f"Otherwise, you'll run out of planned actions before replanning."
            )
        
        self.planner = planner
        self.jepa = jepa_model
        self.replan_frequency = replan_frequency
        self.horizon = horizon
        self.device = device
        
        # Move JEPA to device and eval mode
        self.jepa = self.jepa.to(device)
        self.jepa.eval()
        
        # History tracking
        self.reset_history()
        
    def reset_history(self):
        """Reset execution history."""
        self.state_history = []      # T+1 states (initial + after each action)
        self.action_history = []     # T executed actions
        self.latent_history = []     # T+1 latent states
        self.cost_history = []       # T+1 costs (after each transition, including initial)
        self.plan_history = []       # Planning events
        
    def encode_state(
        self,
        state: Union[torch.Tensor, np.ndarray],
        frame: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Encode environment state to latent representation.
        
        Args:
            state: Proprioceptive state (state_dim,) - typically [vx, vy] after dropping position
            frame: Visual observation (C, H, W) or (H, W, C)
            
        Returns:
            z: Latent state (C_latent, H_latent, W_latent) - (18, 26, 26) for JEPA
        """
        # Convert numpy to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame).float()
            
        # Handle HWC to CHW conversion for frame
        if frame.dim() == 3 and frame.shape[-1] in [1, 3, 4]:
            frame = frame.permute(2, 0, 1)
            
        # Normalize frame to [0, 1] if needed
        if frame.max() > 1.0:
            frame = frame / 255.0
        
        with torch.no_grad():
            state = state.to(self.device)
            frame = frame.to(self.device)
            
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if frame.dim() == 3:
                frame = frame.unsqueeze(0)
                
            z = self.jepa.encode_state(state, frame)
            
            # Remove batch and time dims if present
            if z.dim() == 5:  # (B, T, C, H, W)
                z = z.squeeze(0).squeeze(0)
            elif z.dim() == 4:  # (B, C, H, W)
                z = z.squeeze(0)
                
            return z
    
    def _extract_state_and_frame(
        self,
        env,
        obs: Union[Dict, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract proprioceptive state and visual frame from environment.
        """
        # Handle dict observations (PointMaze style)
        if isinstance(obs, dict):
            raw_obs = obs.get('observation', obs.get('obs', None))
            if raw_obs is None:
                raise ValueError(
                    f"Cannot find observation in dict. Keys: {list(obs.keys())}"
                )
        else:
            raw_obs = obs
            
        # Convert to numpy if needed
        if isinstance(raw_obs, torch.Tensor):
            raw_obs = raw_obs.cpu().numpy()
        
        # Extract velocity (drop x, y position as per JEPA training)
        if len(raw_obs) >= 4:
            state = torch.tensor(raw_obs[2:4], dtype=torch.float32)
        else:
            state = torch.tensor(raw_obs, dtype=torch.float32)
        
        # Get frame from render
        frame = env.render()
        
        if frame is None:
            raise RuntimeError(
                "env.render() returned None. Make sure render_mode='rgb_array' is set."
            )
        
        frame = torch.from_numpy(frame.copy()).float()
        
        # Handle HWC -> CHW
        if frame.dim() == 3 and frame.shape[-1] in [1, 3, 4]:
            frame = frame.permute(2, 0, 1)
            
        # Normalize to [0, 1]
        if frame.max() > 1.0:
            frame = frame / 255.0
            
        # Resize if needed (JEPA expects 64x64)
        if frame.shape[-2:] != (64, 64):
            frame = F.interpolate(
                frame.unsqueeze(0),
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return state, frame
    
    def _get_initial_observation(self, env) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial state and frame from environment."""
        initial_frame = env.render()
        if initial_frame is None:
            raise RuntimeError(
                "env.render() returned None. Make sure render_mode='rgb_array' is set."
            )
        initial_frame = torch.from_numpy(initial_frame.copy()).float()
        
        # Handle HWC -> CHW
        if initial_frame.dim() == 3 and initial_frame.shape[-1] in [1, 3, 4]:
            initial_frame = initial_frame.permute(2, 0, 1)
        
        # Normalize
        if initial_frame.max() > 1.0:
            initial_frame = initial_frame / 255.0
        
        # Resize if needed
        if initial_frame.shape[-2:] != (64, 64):
            initial_frame = F.interpolate(
                initial_frame.unsqueeze(0),
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Get velocity from environment data
        if hasattr(env, 'data') and hasattr(env.data, 'qvel'):
            initial_state = torch.tensor(env.data.qvel[:2].copy(), dtype=torch.float32)
        elif hasattr(env, 'point_env') and hasattr(env.point_env, 'data'):
            initial_state = torch.tensor(env.point_env.data.qvel[:2].copy(), dtype=torch.float32)
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'data'):
            initial_state = torch.tensor(env.unwrapped.data.qvel[:2].copy(), dtype=torch.float32)
        else:
            initial_state = torch.zeros(2, dtype=torch.float32)
        
        return initial_state, initial_frame
    
    def _prepare_goal(
        self,
        goal_state: Union[torch.Tensor, np.ndarray],
        goal_frame: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare goal state and frame for encoding."""
        if isinstance(goal_state, np.ndarray):
            goal_state = torch.from_numpy(goal_state).float()
        if isinstance(goal_frame, np.ndarray):
            goal_frame = torch.from_numpy(goal_frame).float()
            
        if goal_frame.dim() == 3 and goal_frame.shape[-1] in [1, 3, 4]:
            goal_frame = goal_frame.permute(2, 0, 1)
        if goal_frame.max() > 1.0:
            goal_frame = goal_frame / 255.0
        if goal_frame.shape[-2:] != (64, 64):
            goal_frame = F.interpolate(
                goal_frame.unsqueeze(0),
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return goal_state, goal_frame
            
    def step_environment(
        self,
        env,
        action: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, Dict]:
        """Execute action in environment and return next state."""
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = np.asarray(action)
            
        action_np = action_np.flatten().copy()
        
        result = env.step(action_np)
        
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        elif len(result) == 4:
            obs, reward, done, info = result
        else:
            raise ValueError(f"Unexpected step return length: {len(result)}")
        
        state, frame = self._extract_state_and_frame(env, obs)
        
        return state, frame, done, info
        
    def execute(
        self,
        env,
        goal_state: Union[torch.Tensor, np.ndarray],
        goal_frame: Union[torch.Tensor, np.ndarray],
        max_steps: int = 100,
        stop_threshold: float = 0.1,
        verbose: bool = False,
        **planning_kwargs
    ) -> Dict[str, Any]:
        """
        Execute MPC control loop with replanning.
        
        The environment should already be reset to the initial state.
        
        Args:
            env: Environment instance (already reset)
            goal_state: Goal proprioceptive state (state_dim,) - velocity only
            goal_frame: Goal visual observation (C, H, W)
            max_steps: Maximum number of steps to execute
            stop_threshold: Stop if cost below this threshold
            verbose: Show progress bar
            **planning_kwargs: Additional arguments for planner.optimize()
            
        Returns:
            Dictionary containing execution results and statistics
        """
        self.reset_history()
        
        # Get current observation from environment
        try:
            current_state, current_frame = self._get_initial_observation(env)
        except Exception as e:
            raise RuntimeError(
                f"Could not get initial observation from environment: {e}. "
                "Make sure env.reset() was called and render_mode='rgb_array'."
            )
        
        # Prepare goal
        goal_state, goal_frame = self._prepare_goal(goal_state, goal_frame)
        z_goal = self.encode_state(goal_state, goal_frame)
        
        # Encode initial state
        z_current = self.encode_state(current_state, current_frame)
        
        # FIX: Store initial state and cost BEFORE any actions
        self.state_history.append(current_state.clone())
        self.latent_history.append(z_current.clone())
        initial_cost = self.planner.rollout.state_distance(z_current, z_goal).item()
        self.cost_history.append(initial_cost)
        
        # Planning state
        planned_actions = None
        plan_index = 0  # Index into current plan
        
        # Statistics
        num_replans = 0
        total_planning_time = 0
        
        # Execution loop
        pbar = tqdm(range(max_steps), disable=not verbose, desc="MPC Execution")
        
        for step in pbar:
            # Check if we need to replan
            needs_replan = (
                planned_actions is None or
                plan_index >= self.replan_frequency
            )
            
            if needs_replan:
                start_time = time.time()
                
                # FIX: Proper warm-start handling
                # If we have remaining actions from the previous plan, use them as warm-start
                warm_start = None
                if planned_actions is not None and plan_index < len(planned_actions):
                    remaining_actions = planned_actions[plan_index:]
                    # Pad remaining actions to horizon length for warm-start
                    if len(remaining_actions) < self.horizon:
                        padding = torch.zeros(
                            self.horizon - len(remaining_actions),
                            remaining_actions.shape[-1],
                            device=remaining_actions.device
                        )
                        warm_start = torch.cat([remaining_actions, padding], dim=0)
                    else:
                        warm_start = remaining_actions[:self.horizon]
                
                # Optimize from current state
                try:
                    result = self.planner.optimize(
                        z_init=z_current,
                        z_goal=z_goal,
                        horizon=self.horizon,
                        initial_actions=warm_start,
                        verbose=False,
                        **planning_kwargs
                    )
                except Exception as e:
                    if verbose:
                        print(f"\nPlanning failed at step {step}: {e}")
                    break
                
                planning_time = time.time() - start_time
                total_planning_time += planning_time
                num_replans += 1
                
                planned_actions = result['actions']
                plan_index = 0
                
                if planned_actions is None or len(planned_actions) == 0:
                    if verbose:
                        print(f"\nPlanner returned no actions at step {step}")
                    break
                
                # Store plan info
                self.plan_history.append({
                    'step': step,
                    'actions': planned_actions.clone(),
                    'predicted_cost': result['cost'],
                    'planning_time': planning_time,
                })
            
            # Execute next action from the plan
            action = planned_actions[plan_index]
            plan_index += 1
            
            # Step in environment
            try:
                next_state, next_frame, done, info = self.step_environment(env, action)
            except Exception as e:
                if verbose:
                    print(f"\nError stepping environment: {e}")
                break
            
            # Encode new state
            z_next = self.encode_state(next_state, next_frame)
            
            # FIX: Store action that was executed
            self.action_history.append(action.clone())
            
            # FIX: Store state AFTER action (consistent: T actions -> T+1 states)
            self.state_history.append(next_state.clone())
            self.latent_history.append(z_next.clone())
            
            # FIX: Cost is always measured AFTER the action
            current_cost = self.planner.rollout.state_distance(z_next, z_goal).item()
            self.cost_history.append(current_cost)
            
            if verbose:
                pbar.set_postfix({
                    'cost': f'{current_cost:.4f}',
                    'replans': num_replans
                })
            
            # Check stopping conditions
            if current_cost < stop_threshold:
                if verbose:
                    pbar.set_description("Goal reached!")
                break
                
            if done:
                if verbose:
                    pbar.set_description("Episode terminated")
                break
            
            # Update for next iteration
            current_state = next_state
            current_frame = next_frame
            z_current = z_next
        
        # Compute final cost (last entry in cost_history)
        final_cost = self.cost_history[-1] if self.cost_history else float('inf')
        
        return {
            'success': final_cost < stop_threshold,
            'final_cost': final_cost,
            'num_steps': len(self.action_history),
            'num_replans': num_replans,
            'total_planning_time': total_planning_time,
            'avg_planning_time': total_planning_time / max(1, num_replans),
            'state_history': torch.stack(self.state_history) if self.state_history else None,
            'action_history': torch.stack(self.action_history) if self.action_history else None,
            'latent_history': torch.stack(self.latent_history) if self.latent_history else None,
            'cost_history': self.cost_history,
            'plan_history': self.plan_history,
        }
        
    def execute_open_loop(
        self,
        env,
        actions: torch.Tensor,
        z_goal: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a pre-planned action sequence without replanning.
        Useful for evaluating open-loop plans in the real environment.
        
        Args:
            env: Environment instance (already reset to initial state)
            actions: Pre-planned action sequence (T, action_dim)
            z_goal: Optional goal latent for cost computation
            verbose: Show progress bar
            
        Returns:
            Dictionary containing execution results
        """
        self.reset_history()
        
        # Get initial observation
        try:
            current_state, current_frame = self._get_initial_observation(env)
        except Exception as e:
            raise RuntimeError(f"Could not get initial observation: {e}")
        
        z_current = self.encode_state(current_state, current_frame)
        
        # Store initial state
        self.state_history.append(current_state.clone())
        self.latent_history.append(z_current.clone())
        if z_goal is not None:
            initial_cost = self.planner.rollout.state_distance(z_current, z_goal).item()
            self.cost_history.append(initial_cost)
        
        pbar = tqdm(enumerate(actions), total=len(actions), disable=not verbose, desc="Open-loop")
        
        for t, action in pbar:
            # Execute action
            try:
                next_state, next_frame, done, info = self.step_environment(env, action)
            except Exception as e:
                if verbose:
                    print(f"\nError at step {t}: {e}")
                break
            
            z_next = self.encode_state(next_state, next_frame)
            
            # Store action and resulting state
            self.action_history.append(action.clone())
            self.state_history.append(next_state.clone())
            self.latent_history.append(z_next.clone())
            
            if z_goal is not None:
                cost = self.planner.rollout.state_distance(z_next, z_goal).item()
                self.cost_history.append(cost)
            
            if done:
                if verbose:
                    pbar.set_description(f"Terminated at step {t}")
                break
            
            # Update for next iteration
            current_state = next_state
            current_frame = next_frame
            z_current = z_next
        
        result = {
            'num_steps': len(self.action_history),
            'state_history': torch.stack(self.state_history) if self.state_history else None,
            'action_history': torch.stack(self.action_history) if self.action_history else None,
            'latent_history': torch.stack(self.latent_history) if self.latent_history else None,
        }
        
        if z_goal is not None:
            result['cost_history'] = self.cost_history
            result['final_cost'] = self.cost_history[-1] if self.cost_history else None
        
        return result
    
    def compare_rollouts(
        self,
        env,
        actions: torch.Tensor,
        z_goal: torch.Tensor,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Compare predicted vs actual rollout.
        
        Args:
            env: Environment instance (already reset)
            actions: Action sequence (T, action_dim)
            z_goal: Goal latent state for cost computation
            verbose: Print comparison
            
        Returns:
            Dictionary with predicted and actual trajectories and costs
        """
        try:
            initial_state, initial_frame = self._get_initial_observation(env)
        except Exception as e:
            raise RuntimeError(f"Could not get initial observation: {e}")
        
        z_init = self.encode_state(initial_state, initial_frame)
        
        # Predicted rollout
        with torch.no_grad():
            predicted_traj = self.planner.rollout.rollout(z_init, actions)
            predicted_costs = [
                self.planner.rollout.state_distance(predicted_traj[t], z_goal).item()
                for t in range(len(predicted_traj))
            ]
        
        # Actual rollout
        actual_result = self.execute_open_loop(env, actions, z_goal=z_goal, verbose=False)
        
        actual_costs = actual_result.get('cost_history', [])
        
        if verbose:
            print("\nRollout Comparison:")
            print(f"  Steps: {len(predicted_costs)} predicted vs {len(actual_costs)} actual")
            if predicted_costs and actual_costs:
                print(f"  Final cost: {predicted_costs[-1]:.6f} predicted vs {actual_costs[-1]:.6f} actual")
                print(f"  Cost error: {abs(predicted_costs[-1] - actual_costs[-1]):.6f}")
            
        return {
            'predicted_trajectory': predicted_traj,
            'predicted_costs': predicted_costs,
            'actual_trajectory': actual_result['latent_history'],
            'actual_costs': actual_costs,
            'actual_states': actual_result['state_history'],
            'actions': actions,
            'model_error': abs(predicted_costs[-1] - actual_costs[-1]) if predicted_costs and actual_costs else None,
        }