"""
Model Predictive Control (MPC) with replanning for closed-loop control.
Combines open-loop optimization with periodic replanning in the real environment.
"""

import torch
from typing import Optional, Dict, Any, List, Tuple
import time


class MPCController:
    """
    Model Predictive Control controller with replanning.
    Executes optimized actions in the environment and replans periodically.
    """
    
    def __init__(
        self,
        planner,
        jepa_model,
        replan_frequency: int = 5,
        horizon: int = 20,
        device: str = 'cuda'
    ):
        """
        Args:
            planner: Base planner to use for trajectory optimization
            jepa_model: JEPA model for encoding states
            replan_frequency: Replan every k steps
            horizon: Planning horizon for each optimization
            device: Device to run on
        """
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
        self.state_history = []
        self.action_history = []
        self.latent_history = []
        self.cost_history = []
        self.plan_history = []
        
    def encode_state(self, state: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        """
        Encode environment state to latent representation.
        
        Args:
            state: Environment state (state_dim,) - proprioceptive info
            frame: Visual observation (C, H, W)
            
        Returns:
            z: Latent state (C_latent, H_latent, W_latent)
        """
        with torch.no_grad():
            # Ensure tensors are on correct device
            state = state.to(self.device)
            frame = frame.to(self.device)
            
            if state.dim() == 1:
                state = state.unsqueeze(0)  # Add batch dim
            if frame.dim() == 3:
                frame = frame.unsqueeze(0)  # Add batch dim
                
            z = self.jepa.encode_state(state, frame)
            return z.squeeze(0).squeeze(0)  # Remove batch and time dims
            
    def step_environment(
        self,
        env,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, Dict]:
        """
        Execute action in environment and return next state.
        
        Args:
            env: Environment instance with step() method
            action: Action to execute (action_dim,)
            
        Returns:
            state: Next state
            frame: Next frame
            done: Episode termination flag
            info: Additional info from environment
        """
        # Convert to numpy if needed
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = action
            
        # Step environment
        obs, reward, done, truncated, info = env.step(action_np)
        
        # Extract state and frame from observation
        if isinstance(obs, dict):
            frame = torch.from_numpy(obs['image']).float()
            state = torch.from_numpy(obs['state']).float()
        elif hasattr(env, 'get_state_and_frame'):
            # Custom method to extract state and frame
            state, frame = env.get_state_and_frame(obs)
            state = torch.from_numpy(state).float()
            frame = torch.from_numpy(frame).float()
        else:
            # Try to parse observation
            if len(obs) > 4:
                state = torch.from_numpy(obs[:4]).float()
                remaining = obs[4:]
                if len(remaining) == 3 * 64 * 64:
                    frame = torch.from_numpy(remaining).float().view(3, 64, 64)
                else:
                    raise ValueError(f"Cannot parse observation of length {len(obs)}")
            else:
                raise ValueError("Cannot extract state and frame from observation")
        
        # Ensure correct shapes
        if frame.dim() == 1:
            # Flattened image, try to reshape
            if len(frame) == 3 * 64 * 64:
                frame = frame.view(3, 64, 64)
            else:
                raise ValueError(f"Cannot reshape frame of length {len(frame)}")
                
        done = done or truncated
        
        return state, frame, done, info
        
    def execute(
        self,
        env,
        initial_state: torch.Tensor,
        initial_frame: torch.Tensor,
        goal_state: torch.Tensor,
        goal_frame: torch.Tensor,
        max_steps: int = 100,
        stop_threshold: float = 0.1,
        verbose: bool = False,
        **planning_kwargs
    ) -> Dict[str, Any]:
        """
        Execute MPC control loop with replanning.
        
        Args:
            env: Environment instance
            initial_state: Initial proprioceptive state (state_dim,)
            initial_frame: Initial visual observation (C, H, W)
            goal_state: Goal proprioceptive state (state_dim,)
            goal_frame: Goal visual observation (C, H, W)
            max_steps: Maximum number of steps to execute
            stop_threshold: Stop if cost below this threshold
            verbose: Print execution progress
            **planning_kwargs: Additional arguments for planner.optimize()
            
        Returns:
            Dictionary containing execution results and statistics
        """
        self.reset_history()
        
        # Encode initial and goal states
        z_current = self.encode_state(initial_state, initial_frame)
        z_goal = self.encode_state(goal_state, goal_frame)
        
        # Initialize
        current_state = initial_state.clone()
        current_frame = initial_frame.clone()
        planned_actions = None
        step_count = 0
        done = False
        
        # Track replanning info
        num_replans = 0
        total_planning_time = 0
        
        while step_count < max_steps and not done:
            # Check if we need to replan
            should_replan = (
                planned_actions is None or 
                step_count % self.replan_frequency == 0 or
                len(planned_actions) == 0
            )
            
            if should_replan:
                # Optimize trajectory from current state
                if verbose:
                    print(f"\nReplanning at step {step_count}...")
                    
                start_time = time.time()
                
                result = self.planner.optimize(
                    z_init=z_current,
                    z_goal=z_goal,
                    horizon=self.horizon,
                    initial_actions=planned_actions,  # Warm-start if available
                    **planning_kwargs
                )
                
                planning_time = time.time() - start_time
                total_planning_time += planning_time
                num_replans += 1
                
                planned_actions = result['actions']
                
                if verbose:
                    print(f"Planning completed in {planning_time:.3f}s, "
                          f"predicted cost: {result['cost']:.6f}")
                    
                # Store plan
                self.plan_history.append({
                    'step': step_count,
                    'actions': planned_actions.clone(),
                    'predicted_cost': result['cost'],
                    'planning_time': planning_time,
                })
                
            # Execute next action
            action = planned_actions[0]
            
            # Step in environment
            try:
                next_state, next_frame, done, info = self.step_environment(env, action)
            except Exception as e:
                if verbose:
                    print(f"Error stepping environment: {e}")
                break
            
            # Encode new state
            z_next = self.encode_state(next_state, next_frame)
            
            # Compute actual cost
            current_cost = self.planner.rollout.compute_cost(z_current, z_goal).item()
            
            # Store history
            self.state_history.append(current_state.clone())
            self.action_history.append(action.clone())
            self.latent_history.append(z_current.clone())
            self.cost_history.append(current_cost)
            
            if verbose and step_count % 5 == 0:
                print(f"Step {step_count:3d}: Cost = {current_cost:.6f}")
                
            # Check stopping condition
            if current_cost < stop_threshold:
                if verbose:
                    print(f"\nGoal reached! Final cost: {current_cost:.6f}")
                break
                
            # Update for next iteration
            current_state = next_state
            current_frame = next_frame
            z_current = z_next
            
            # Shift planned actions (remove executed action)
            if planned_actions is not None and len(planned_actions) > 1:
                planned_actions = planned_actions[1:]
            else:
                planned_actions = None
                
            step_count += 1
            
        # Store final state
        self.state_history.append(current_state.clone())
        self.latent_history.append(z_current.clone())
        final_cost = self.planner.rollout.compute_cost(z_current, z_goal).item()
        self.cost_history.append(final_cost)
        
        if verbose:
            print(f"\nExecution completed:")
            print(f"  Total steps: {step_count}")
            print(f"  Final cost: {final_cost:.6f}")
            print(f"  Number of replans: {num_replans}")
            if num_replans > 0:
                print(f"  Total planning time: {total_planning_time:.3f}s")
                print(f"  Avg planning time: {total_planning_time/num_replans:.3f}s")
            
        return {
            'success': final_cost < stop_threshold,
            'final_cost': final_cost,
            'num_steps': step_count,
            'num_replans': num_replans,
            'total_planning_time': total_planning_time,
            'avg_planning_time': total_planning_time / num_replans if num_replans > 0 else 0,
            'state_history': torch.stack(self.state_history) if self.state_history else None,
            'action_history': torch.stack(self.action_history) if self.action_history else None,
            'latent_history': torch.stack(self.latent_history) if self.latent_history else None,
            'cost_history': self.cost_history,
            'plan_history': self.plan_history,
        }
        
    def execute_open_loop(
        self,
        env,
        initial_state: torch.Tensor,
        initial_frame: torch.Tensor,
        actions: torch.Tensor,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a pre-planned action sequence without replanning.
        Useful for evaluating open-loop plans in the real environment.
        
        Args:
            env: Environment instance
            initial_state: Initial proprioceptive state
            initial_frame: Initial visual observation
            actions: Pre-planned action sequence (T, action_dim)
            verbose: Print execution progress
            
        Returns:
            Dictionary containing execution results
        """
        self.reset_history()
        
        current_state = initial_state.clone()
        current_frame = initial_frame.clone()
        z_current = self.encode_state(current_state, current_frame)
        
        for t, action in enumerate(actions):
            # Store current state
            self.state_history.append(current_state.clone())
            self.action_history.append(action.clone())
            self.latent_history.append(z_current.clone())
            
            # Execute action
            try:
                next_state, next_frame, done, info = self.step_environment(env, action)
            except Exception as e:
                if verbose:
                    print(f"Error at step {t}: {e}")
                break
            
            if verbose and t % 5 == 0:
                print(f"Step {t:3d}/{len(actions)}")
                
            if done:
                if verbose:
                    print(f"Episode terminated at step {t}")
                break
                
            # Update
            current_state = next_state
            current_frame = next_frame
            z_current = self.encode_state(current_state, current_frame)
            
        # Store final state
        self.state_history.append(current_state.clone())
        self.latent_history.append(z_current.clone())
        
        return {
            'num_steps': len(self.action_history),
            'state_history': torch.stack(self.state_history) if self.state_history else None,
            'action_history': torch.stack(self.action_history) if self.action_history else None,
            'latent_history': torch.stack(self.latent_history) if self.latent_history else None,
        }
    
    def rollout_comparison(
        self,
        env,
        initial_state: torch.Tensor,
        initial_frame: torch.Tensor,
        actions: torch.Tensor,
        z_goal: torch.Tensor,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Compare predicted vs actual rollout.
        
        Args:
            env: Environment instance
            initial_state: Initial state
            initial_frame: Initial frame
            actions: Action sequence (T, action_dim)
            z_goal: Goal latent state for cost computation
            verbose: Print comparison
            
        Returns:
            Dictionary with predicted and actual trajectories and costs
        """
        # Predicted rollout
        z_init = self.encode_state(initial_state, initial_frame)
        with torch.no_grad():
            predicted_traj = self.planner.rollout.rollout(z_init, actions)
            predicted_costs = [
                self.planner.rollout.compute_cost(predicted_traj[t], z_goal).item()
                for t in range(len(predicted_traj))
            ]
        
        # Actual rollout
        actual_result = self.execute_open_loop(
            env, initial_state, initial_frame, actions, verbose=False
        )
        
        actual_costs = [
            self.planner.rollout.compute_cost(z, z_goal).item()
            for z in actual_result['latent_history']
        ]
        
        if verbose:
            print("\nRollout Comparison:")
            print(f"  Steps: {len(predicted_costs)} predicted vs {len(actual_costs)} actual")
            print(f"  Final cost: {predicted_costs[-1]:.6f} predicted vs {actual_costs[-1]:.6f} actual")
            
        return {
            'predicted_trajectory': predicted_traj,
            'predicted_costs': predicted_costs,
            'actual_trajectory': actual_result['latent_history'],
            'actual_costs': actual_costs,
            'actual_states': actual_result['state_history'],
            'actions': actions,
        }
