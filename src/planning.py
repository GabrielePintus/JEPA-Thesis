"""
Planning algorithms for goal-conditioned navigation in PointMaze using learned value function.

This module implements three planning approaches:
1. Gradient-based optimization: Direct optimization of action sequences using gradients
2. CMA-ES: Covariance Matrix Adaptation Evolution Strategy (gradient-free)
3. Model Predictive Control (MPC): Receding horizon control with replanning
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import cma


@dataclass
class PlanningConfig:
    """Configuration for planning algorithms."""
    horizon: int = 10  # Planning horizon (number of steps)
    action_dim: int = 2  # Action dimensionality (x, y velocities)
    action_low: float = -1.0  # Minimum action value
    action_high: float = 1.0  # Maximum action value
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class GradientPlannerConfig(PlanningConfig):
    """Configuration for gradient-based planner (sampling-based)."""
    num_iterations: int = 100
    learning_rate: float = 0.5  # Larger for sampling-based updates
    num_samples: int = 20  # Number of samples per iteration
    noise_std: float = 0.1  # Perturbation noise std
    temperature: float = 0.1  # Softmax temperature for weighting


@dataclass
class CMAESConfig(PlanningConfig):
    """Configuration for CMA-ES planner."""
    population_size: int = 50
    max_iterations: int = 100
    sigma0: float = 0.5  # Initial standard deviation
    tolerance: float = 1e-6


@dataclass
class MPCConfig(PlanningConfig):
    """Configuration for MPC."""
    horizon: int = 15
    num_iterations: int = 50  # Optimization iterations per replan
    learning_rate: float = 0.1
    replan_frequency: int = 3  # Replan every N steps
    max_steps: int = 100  # Maximum steps before giving up


class ValueFunctionWrapper:
    """Wrapper for the learned value function (isometry model)."""
    
    def __init__(self, model, device: str = "cuda"):
        """
        Args:
            model: Trained Isometry model with visual_encoder and head
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)
        
        # Cache goal embedding for efficiency
        self.goal_embedding = None
    
    def set_goal(self, goal_frame: torch.Tensor):
        """
        Set the goal frame and cache its embedding.
        
        Args:
            goal_frame: Goal image tensor of shape (C, H, W) or (1, C, H, W)
        """
        if goal_frame.dim() == 3:
            goal_frame = goal_frame.unsqueeze(0)
        
        with torch.no_grad():
            z_goal = self.model.visual_encoder(goal_frame.to(self.device))
            self.goal_embedding = self.model.head(z_goal)
    
    def compute_value(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Compute value (negative distance to goal) for a batch of frames.
        
        Args:
            frames: Batch of images (B, C, H, W)
            
        Returns:
            Values (B,) - higher is better (negative distance)
        """
        if self.goal_embedding is None:
            raise ValueError("Goal not set! Call set_goal() first.")
        
        z_frames = self.model.visual_encoder(frames.to(self.device))
        z_frames = self.model.head(z_frames)
        
        # Cosine similarity distance: 1 - cos_sim
        # We want to maximize similarity (minimize distance)
        distances = 1 - F.cosine_similarity(z_frames, self.goal_embedding)

        # If there is std prediction, we could incorporate it here (optional)
        if hasattr(self.model, 'logstd_head'):
            # Prepare input for std prediction
            std_in = torch.cat([z_frames, self.goal_embedding.repeat(z_frames.size(0), 1)], dim=1)
            logstd_pred = self.model.logstd_head(std_in).squeeze(-1)
            std_pred = logstd_pred.exp()
            # Optionally adjust distance by uncertainty (e.g., distance / std)
            distances = distances * (1 + std_pred * 5e-1)  # Weight uncertainty
        
        # Return negative distance (value = -distance)
        return -distances
    
    def compute_distance(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Compute distance to goal for a batch of frames.
        
        Args:
            frames: Batch of images (B, C, H, W)
            
        Returns:
            Distances (B,) - lower is better
        """
        return -self.compute_value(frames)


class EnvironmentRollout:
    """Helper class to rollout action sequences in the environment."""
    
    def __init__(self, env, device: str = "cuda"):
        """
        Args:
            env: PointMazeEnv instance
            device: Device for tensor operations
        """
        self.env = env
        self.device = device
    
    def rollout(
        self, 
        start_position: np.ndarray, 
        actions: torch.Tensor,
        return_frames: bool = True
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Execute a sequence of actions from a start position.
        
        Args:
            start_position: Starting (x, y) position
            actions: Action sequence (T, 2) as torch tensor
            return_frames: Whether to render and return frames
            
        Returns:
            positions: List of (x, y) positions
            frames: List of rendered frames (if return_frames=True)
        """
        # Convert actions to numpy
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        
        # Reset to start position
        self.env.set_position(start_position)
        
        positions = [start_position.copy()]
        frames = []
        
        if return_frames:
            frames.append(self.env.render())
        
        # Execute actions
        for action in actions:
            obs_dict, _, _, _, _ = self.env.step(action)
            # pos = self.env.get_position()
            pos = obs_dict['observation'][:2]  # Assuming first two entries are (x, y)
            positions.append(pos)
            
            if return_frames:
                frames.append(self.env.render())
        
        return positions, frames


class GradientPlanner:
    """Gradient-based trajectory optimization."""
    
    def __init__(
        self, 
        value_fn: ValueFunctionWrapper,
        env_rollout: EnvironmentRollout,
        config: GradientPlannerConfig
    ):
        self.value_fn = value_fn
        self.env_rollout = env_rollout
        self.config = config
    
    def plan(
        self, 
        start_position: np.ndarray,
        initial_actions: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Plan optimal action sequence using gradient-based optimization with sampling.
        
        Uses a sampling-based gradient estimator (like REINFORCE/MPPI gradients)
        since the environment dynamics are not differentiable.
        
        Args:
            start_position: Starting (x, y) position
            initial_actions: Initial action sequence (H, 2). If None, initialized randomly.
            verbose: Whether to print progress
            
        Returns:
            actions: Optimized action sequence (H, 2)
            info: Dictionary with optimization statistics
        """
        # Initialize actions - random is better than zeros!
        if initial_actions is None:
            actions = torch.randn(
                self.config.horizon, 
                self.config.action_dim,
                device=self.config.device
            ) * 0.3  # Small random initialization
        else:
            actions = initial_actions.clone().detach().to(self.config.device)
        
        best_actions = actions.clone()
        best_value = float('-inf')
        history = {'values': [], 'losses': [], 'action_mags': []}
        
        for iteration in range(self.config.num_iterations):
            # Evaluate current actions
            current_value = self._evaluate_actions(start_position, actions)
            
            # Generate perturbations
            noise = torch.randn(
                self.config.num_samples,
                self.config.horizon,
                self.config.action_dim,
                device=self.config.device
            ) * self.config.noise_std
            
            # Sample perturbed action sequences
            perturbed_actions = actions.unsqueeze(0) + noise  # (num_samples, H, 2)
            
            # Clamp to bounds
            perturbed_actions = torch.clamp(
                perturbed_actions,
                self.config.action_low,
                self.config.action_high
            )
            
            # Evaluate all samples
            values = []
            for i in range(self.config.num_samples):
                value = self._evaluate_actions(start_position, perturbed_actions[i])
                values.append(value)
            
            values = torch.tensor(values, device=self.config.device)
            
            # Compute gradient estimate using values as weights
            # This is similar to REINFORCE or reward-weighted regression
            # Normalize values to get weights
            weights = torch.softmax(values / self.config.temperature, dim=0)
            
            # Weighted average of perturbations gives gradient direction
            gradient_estimate = torch.sum(
                weights.view(-1, 1, 1) * noise,
                dim=0
            )
            
            # Update actions using gradient ascent (we want to maximize value)
            actions = actions + self.config.learning_rate * gradient_estimate
            
            # Clip to bounds
            actions = torch.clamp(actions, self.config.action_low, self.config.action_high)
            
            # Track best solution
            if current_value > best_value:
                best_value = current_value
                best_actions = actions.clone()
            
            action_mag = actions.abs().mean().item()
            history['values'].append(current_value)
            history['action_mags'].append(action_mag)
            
            if verbose and (iteration % 10 == 0 or iteration == self.config.num_iterations - 1):
                print(f"Iter {iteration:3d}: Value = {current_value:.4f}, "
                      f"|actions| = {action_mag:.3f}, "
                      f"best = {best_value:.4f}")
        
        info = {
            'best_value': best_value,
            'final_value': history['values'][-1],
            'history': history
        }
        
        return best_actions, info
    
    def _evaluate_actions(self, start_position: np.ndarray, actions: torch.Tensor) -> float:
        """Evaluate action sequence by computing terminal value."""
        actions_np = actions.detach().cpu().numpy()
        
        # Rollout
        _, frames = self.env_rollout.rollout(start_position, actions_np, return_frames=True)
        
        # Convert frames and compute value
        frames_tensor = self._frames_to_tensor(frames)
        
        with torch.no_grad():
            # Compute terminal value (primary objective)
            terminal_value = self.value_fn.compute_value(frames_tensor[-1:]).item()
            
            # Optional: penalize distance traveled for efficiency
            # path_length_penalty = -0.01 * len(frames)
            # return terminal_value + path_length_penalty
        
        return terminal_value
    
    def _frames_to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Convert list of frames to batched tensor."""
        frames = np.array(frames)  # (T, H, W, C)
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        return frames_tensor.to(self.config.device)


class CMAESPlanner:
    """CMA-ES trajectory optimization (gradient-free)."""
    
    def __init__(
        self,
        value_fn: ValueFunctionWrapper,
        env_rollout: EnvironmentRollout,
        config: CMAESConfig
    ):
        self.value_fn = value_fn
        self.env_rollout = env_rollout
        self.config = config
    
    def plan(
        self,
        start_position: np.ndarray,
        initial_actions: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Plan optimal action sequence using CMA-ES.
        
        Args:
            start_position: Starting (x, y) position
            initial_actions: Initial action sequence (H, 2). If None, initialized to zeros.
            verbose: Whether to print progress
            
        Returns:
            actions: Optimized action sequence (H, 2)
            info: Dictionary with optimization statistics
        """
        # Flatten action space for CMA-ES
        action_size = self.config.horizon * self.config.action_dim
        
        if initial_actions is None:
            x0 = np.zeros(action_size)
        else:
            x0 = initial_actions.flatten()
        
        # Setup CMA-ES optimizer
        es = cma.CMAEvolutionStrategy(
            x0,
            self.config.sigma0,
            {
                'popsize': self.config.population_size,
                'maxiter': self.config.max_iterations,
                'tolx': self.config.tolerance,
                'verbose': -1  # Suppress CMA-ES output
            }
        )
        
        history = {'values': [], 'best_values': []}
        best_value = float('-inf')
        
        iteration = 0
        while not es.stop():
            # Sample population
            solutions = es.ask()
            
            # Evaluate fitness (negative value since CMA-ES minimizes)
            fitness_list = []
            for solution in solutions:
                # Reshape and clip actions
                actions = np.clip(
                    solution.reshape(self.config.horizon, self.config.action_dim),
                    self.config.action_low,
                    self.config.action_high
                )
                
                # Evaluate
                value = self._evaluate_actions(start_position, actions)
                fitness_list.append(-value)  # Negative because CMA-ES minimizes
            
            # Update CMA-ES
            es.tell(solutions, fitness_list)
            
            # Track best
            current_best_value = -min(fitness_list)
            if current_best_value > best_value:
                best_value = current_best_value
            
            history['values'].append(-np.mean(fitness_list))
            history['best_values'].append(best_value)
            
            if verbose and (iteration % 10 == 0 or es.stop()):
                print(f"Iter {iteration:3d}: Mean Value = {history['values'][-1]:.4f}, "
                      f"Best Value = {best_value:.4f}")
            
            iteration += 1
        
        # Get best solution
        best_solution = es.result.xbest
        best_actions = np.clip(
            best_solution.reshape(self.config.horizon, self.config.action_dim),
            self.config.action_low,
            self.config.action_high
        )
        
        info = {
            'best_value': best_value,
            'history': history,
            'iterations': iteration
        }
        
        return torch.from_numpy(best_actions).float(), info
    
    def _evaluate_actions(self, start_position: np.ndarray, actions: np.ndarray) -> float:
        """Evaluate action sequence by rolling out and computing terminal value."""
        _, frames = self.env_rollout.rollout(start_position, actions, return_frames=True)
        
        # Convert final frame to tensor
        final_frame = frames[-1]
        frame_tensor = torch.from_numpy(final_frame.copy()).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(self.config.device)
        
        # Compute value
        with torch.no_grad():
            value = self.value_fn.compute_value(frame_tensor).item()
        
        return value


class MPCPlanner:
    """Model Predictive Control with replanning."""
    
    def __init__(
        self,
        value_fn: ValueFunctionWrapper,
        env_rollout: EnvironmentRollout,
        config: MPCConfig,
        planner_type: str = "gradient"  # "gradient" or "cmaes"
    ):
        self.value_fn = value_fn
        self.env_rollout = env_rollout
        self.config = config
        self.planner_type = planner_type
        
        # Create base planner for optimization
        if planner_type == "gradient":
            planner_config = GradientPlannerConfig(
                horizon=config.horizon,
                action_dim=config.action_dim,
                action_low=config.action_low,
                action_high=config.action_high,
                num_iterations=config.num_iterations,
                learning_rate=config.learning_rate,
                num_samples=15,  # Fewer samples for faster replanning
                noise_std=0.1,
                temperature=0.1,
                device=config.device
            )
            self.base_planner = GradientPlanner(value_fn, env_rollout, planner_config)
        elif planner_type == "cmaes":
            planner_config = CMAESConfig(
                horizon=config.horizon,
                action_dim=config.action_dim,
                action_low=config.action_low,
                action_high=config.action_high,
                population_size=30,  # Smaller for faster replanning
                max_iterations=config.num_iterations,
                device=config.device
            )
            self.base_planner = CMAESPlanner(value_fn, env_rollout, planner_config)
        else:
            raise ValueError(f"Unknown planner type: {planner_type}")
    
    def execute(
        self,
        start_position: np.ndarray,
        goal_threshold: float = 0.1,
        verbose: bool = True
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
        """
        Execute MPC with replanning to reach the goal.
        
        Args:
            start_position: Starting (x, y) position
            goal_threshold: Distance threshold to consider goal reached
            verbose: Whether to print progress
            
        Returns:
            positions: List of visited positions
            frames: List of rendered frames
            info: Execution statistics
        """
        current_position = start_position.copy()
        all_positions = [current_position.copy()]
        all_frames = []
        all_actions = []
        
        # Get initial frame
        self.env_rollout.env.set_position(current_position)
        all_frames.append(self.env_rollout.env.render())
        
        # Warm start with zero actions
        planned_actions = None
        step = 0
        replans = 0
        
        while step < self.config.max_steps:
            # Check if goal reached
            current_frame = all_frames[-1]
            frame_tensor = torch.from_numpy(current_frame.copy()).permute(2, 0, 1).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(self.config.device)
            
            with torch.no_grad():
                distance = self.value_fn.compute_distance(frame_tensor).item()
            
            if distance < goal_threshold:
                if verbose:
                    print(f"\n✓ Goal reached at step {step}! Distance: {distance:.4f}")
                break
            
            # Replan if needed
            if step % self.config.replan_frequency == 0:
                if verbose:
                    print(f"\n--- Replanning at step {step} (distance: {distance:.4f}) ---")
                
                # Use previous plan as warm start (shift and pad)
                if planned_actions is not None and planned_actions.shape[0] > self.config.replan_frequency:
                    initial_actions = planned_actions[self.config.replan_frequency:]
                    # Pad with last action
                    padding = planned_actions[-1:].repeat(self.config.replan_frequency, 1)
                    initial_actions = torch.cat([initial_actions, padding], dim=0)
                else:
                    initial_actions = None
                
                # Plan new action sequence
                planned_actions, plan_info = self.base_planner.plan(
                    current_position,
                    initial_actions=initial_actions,
                    verbose=verbose
                )
                replans += 1
            
            # Execute next action
            next_action = planned_actions[0].detach().cpu().numpy()
            all_actions.append(next_action)
            
            # Step in environment
            _, _, _, _ = self.env_rollout.env.step(next_action)
            current_position = self.env_rollout.env.get_position()
            all_positions.append(current_position.copy())
            all_frames.append(self.env_rollout.env.render())
            
            # Shift planned actions
            if planned_actions.shape[0] > 1:
                planned_actions = planned_actions[1:]
            
            step += 1
        
        info = {
            'steps': step,
            'replans': replans,
            'success': distance < goal_threshold,
            'final_distance': distance
        }
        
        if verbose:
            if info['success']:
                print(f"\n✓ Success! Reached goal in {step} steps with {replans} replans")
            else:
                print(f"\n✗ Failed to reach goal in {self.config.max_steps} steps")
        
        return all_positions, all_frames, info, planned_actions


# Utility functions
def visualize_plan(positions: List[np.ndarray], env, save_path: Optional[str] = None):
    """Visualize planned trajectory on the maze."""
    import matplotlib.pyplot as plt
    
    positions = np.array(positions)
    
    # Render background
    env.reset()
    bg_frame = env.render()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(bg_frame, extent=(-3.25, 3.25, -3.25, 3.25))
    
    # Plot trajectory
    # Rotate positions for visualization
    positions_rot = np.column_stack([-positions[:, 1], -positions[:, 0]])
    plt.plot(positions_rot[:, 0], positions_rot[:, 1], 'b-', linewidth=2, alpha=0.7)
    plt.scatter(positions_rot[0, 0], positions_rot[0, 1], c='green', s=200, 
                edgecolor='black', label='Start', zorder=5)
    plt.scatter(positions_rot[-1, 0], positions_rot[-1, 1], c='red', s=200,
                edgecolor='black', label='End', marker='*', zorder=5)
    
    plt.title("Planned Trajectory")
    plt.legend()
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def save_trajectory_video(frames: List[np.ndarray], save_path: str, fps: int = 10):
    """Save trajectory as video."""
    import imageio
    
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Saved trajectory video to {save_path}")