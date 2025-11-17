"""
Optimization algorithms for PointMaze environment planning.

Implements:
- CMA-ES: Covariance Matrix Adaptation Evolution Strategy
- MPPIC: Model Predictive Path Integral Control
- MPCController: Model Predictive Control with Replanning

These planners work directly with the PointMaze environment's step function,
rolling out trajectories in the true environment dynamics.
"""

from __future__ import annotations
import abc
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any, List
import copy

import numpy as np
from tqdm import tqdm

# -------------------------------------------------------------------
# Base Cost Function
# -------------------------------------------------------------------

class BaseCostFunction(metaclass=abc.ABCMeta):
    """Abstract base class for trajectory cost in PointMaze."""

    @abc.abstractmethod
    def __call__(
        self,
        trajectories: np.ndarray,  # (B, T+1, 2) - x, y positions
        actions: np.ndarray,        # (B, T, action_dim)
        goal: np.ndarray,           # (2,) or (B, 2)
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Returns:
            total_cost: (B,) cost per trajectory
            components: dict of component costs (B,) each
        """
        raise NotImplementedError()


@dataclass
class GoalCostConfig:
    """Configuration for goal-reaching cost function."""
    distance_type: str = "l2"           # "l2" or "squared_l2"
    goal_weight: float = 1.0            # terminal cost weight
    
    # Trajectory-wide costs
    soft_goal_weight: float = 0.0       # distance to goal at intermediate steps
    action_smooth_weight: float = 0.0   # ||a_{t+1} - a_t||^2
    state_smooth_weight: float = 0.0    # ||s_{t+1} - s_t||^2
    action_magnitude_weight: float = 0.0  # ||a_t||^2 (control effort)
    soft_goal_discount: float = 1.0     # discount for soft goal (<=1)


class GoalCost(BaseCostFunction):
    """
    Cost function for reaching a goal position in PointMaze.
    
    Main term: distance between final position and goal
    Optional: trajectory smoothness, control effort, intermediate goal distance
    """

    def __init__(self, cfg: GoalCostConfig):
        self.cfg = cfg

    def _dist(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute distance between x and y positions.
        
        Args:
            x: (B, 2) positions
            y: (2,) or (B, 2) goal(s)
        
        Returns:
            (B,) distances
        """
        diff = x - y  # Broadcasting handles (B,2) - (2,) -> (B,2)
        
        if self.cfg.distance_type == "l2":
            return np.sqrt(np.sum(diff ** 2, axis=-1))
        elif self.cfg.distance_type == "squared_l2":
            return np.sum(diff ** 2, axis=-1)
        else:
            raise ValueError(f"Unknown distance_type: {self.cfg.distance_type}")

    def __call__(
        self,
        trajectories: np.ndarray,  # (B, T+1, 2) - positions only
        actions: np.ndarray,        # (B, T, action_dim)
        goal: np.ndarray,           # (2,) or (B, 2)
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        
        B, T1, pos_dim = trajectories.shape
        T = T1 - 1
        
        assert pos_dim == 2, "Trajectories should be (B, T+1, 2) with x, y positions"
        
        components: Dict[str, np.ndarray] = {}
        
        # Terminal goal cost (MAIN TERM)
        final_pos = trajectories[:, -1]  # (B, 2)
        goal_dist = self._dist(final_pos, goal)  # (B,)
        components["goal"] = self.cfg.goal_weight * goal_dist
        
        # Soft trajectory-wide goal distance
        if self.cfg.soft_goal_weight > 0.0:
            # Distance at each timestep
            if goal.ndim == 1:
                goal_exp = goal[np.newaxis, np.newaxis, :]  # (1, 1, 2)
            else:
                goal_exp = goal[:, np.newaxis, :]  # (B, 1, 2)
            
            diff = trajectories - goal_exp  # (B, T+1, 2)
            soft_dist = np.sqrt(np.sum(diff ** 2, axis=-1))  # (B, T+1)
            
            if self.cfg.soft_goal_discount < 1.0:
                # Geometric discount over time
                gamma = self.cfg.soft_goal_discount
                t_idx = np.arange(T1)
                disc = gamma ** t_idx  # (T+1,)
                soft_dist = np.sum(soft_dist * disc[np.newaxis, :], axis=1) / disc.sum()
            else:
                soft_dist = soft_dist.mean(axis=1)  # (B,)
            
            components["soft_goal"] = self.cfg.soft_goal_weight * soft_dist
        
        # Action smoothness
        if self.cfg.action_smooth_weight > 0.0:
            diff = actions[:, 1:] - actions[:, :-1]  # (B, T-1, A)
            action_smooth = np.mean(diff ** 2, axis=(1, 2))  # (B,)
            components["action_smooth"] = self.cfg.action_smooth_weight * action_smooth
        
        # State (position) smoothness
        if self.cfg.state_smooth_weight > 0.0:
            diff = trajectories[:, 1:] - trajectories[:, :-1]  # (B, T, 2)
            state_smooth = np.mean(diff ** 2, axis=(1, 2))  # (B,)
            components["state_smooth"] = self.cfg.state_smooth_weight * state_smooth
        
        # Action magnitude (control effort)
        if self.cfg.action_magnitude_weight > 0.0:
            action_mag = np.mean(actions ** 2, axis=(1, 2))  # (B,)
            components["action_magnitude"] = self.cfg.action_magnitude_weight * action_mag
        
        # Sum all components
        total_cost = sum(components.values())
        
        return total_cost, components


# -------------------------------------------------------------------
# Environment Rollout Utilities
# -------------------------------------------------------------------

def rollout_trajectory(
    env: Any,
    initial_pos: np.ndarray,  # (2,) - x, y position
    actions: np.ndarray,       # (T, action_dim)
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Roll out a single action sequence in the environment.
    
    Args:
        env: PointMaze environment
        initial_pos: (2,) initial x, y position
        actions: (T, action_dim) action sequence
    
    Returns:
        trajectory: (T+1, 2) position trajectory (x, y at each timestep)
        infos: List of info dicts from each step
    """
    T = len(actions)
    
    trajectory = np.zeros((T + 1, 2))
    trajectory[0] = initial_pos
    
    infos = []
    
    # Save initial environment state
    saved_qpos = env.point_env.data.qpos.copy()
    saved_qvel = env.point_env.data.qvel.copy()
    saved_goal = env.goal.copy()
    
    # Set to initial position (zero velocity)
    env.point_env.data.qpos[:2] = initial_pos
    env.point_env.data.qvel[:] = 0.0
    
    for t in range(T):
        obs, reward, terminated, truncated, info = env.step(actions[t])
        trajectory[t + 1] = obs['achieved_goal']  # Extract x, y position
        infos.append(info)
    
    # Restore environment state
    env.point_env.data.qpos[:] = saved_qpos
    env.point_env.data.qvel[:] = saved_qvel
    env.goal[:] = saved_goal
    
    return trajectory, infos


def rollout_trajectories_parallel(
    env: Any,
    initial_pos: np.ndarray,   # (2,) - x, y position
    actions: np.ndarray,        # (B, T, action_dim)
) -> np.ndarray:
    """
    Roll out multiple action sequences in parallel (sequentially executed).
    
    Args:
        env: PointMaze environment
        initial_pos: (2,) initial x, y position
        actions: (B, T, action_dim) action sequences
    
    Returns:
        trajectories: (B, T+1, 2) position trajectories
    """
    B, T, action_dim = actions.shape
    
    trajectories = np.zeros((B, T + 1, 2))
    
    for b in range(B):
        traj, _ = rollout_trajectory(env, initial_pos, actions[b])
        trajectories[b] = traj
    
    return trajectories


# -------------------------------------------------------------------
# Base Planner
# -------------------------------------------------------------------

class BasePlanner(metaclass=abc.ABCMeta):
    """Abstract base class for trajectory planners."""

    @abc.abstractmethod
    def plan(
        self,
        env: Any,
        initial_pos: np.ndarray,  # (2,) - x, y position
        goal: np.ndarray,          # (2,) - x, y goal position
        num_iterations: int,
        init_actions: Optional[np.ndarray] = None,
        verbose: bool = False,
        return_trajectory: bool = False,
    ) -> Dict[str, Any]:
        """
        Plan an action sequence to reach the goal.
        
        Args:
            env: PointMaze environment
            initial_pos: (2,) initial x, y position
            goal: (2,) goal x, y position
            num_iterations: Number of optimization iterations
            init_actions: Optional initial action sequence (T, action_dim)
            verbose: Whether to print progress
            return_trajectory: Whether to return the resulting trajectory
        
        Returns:
            Dictionary with:
                - 'actions': (T, action_dim) planned action sequence
                - 'cost': scalar cost value
                - 'trajectory': (T+1, 2) if return_trajectory=True
                - 'history': list of iteration info if verbose=True
        """
        raise NotImplementedError()


# -------------------------------------------------------------------
# CMA-ES Planner
# -------------------------------------------------------------------

# class CMAESPlanner(BasePlanner):
#     """
#     CMA-ES (Covariance Matrix Adaptation Evolution Strategy) planner.
    
#     A gradient-free evolutionary algorithm that adapts its search distribution
#     based on the success of previous samples. Particularly effective for
#     non-convex optimization problems.
    
#     Key features:
#     - Population-based sampling
#     - Adaptive covariance matrix for search direction
#     - Automatic step-size control
#     - Rank-based selection
#     """

#     def __init__(
#         self,
#         cost_fn: BaseCostFunction,
#         horizon: int,
#         action_dim: int,
#         action_bounds: Optional[Tuple[float, float]] = None,
#         population_size: Optional[int] = None,
#         sigma: float = 0.5,
#         use_best_sample: bool = True,
#     ):
#         """
#         Args:
#             cost_fn: Cost function to minimize
#             horizon: Planning horizon (number of timesteps)
#             action_dim: Dimension of action space
#             action_bounds: (min, max) bounds for actions
#             population_size: Number of samples per iteration (default: 4 + 3*log(n))
#             sigma: Initial standard deviation
#             use_best_sample: Return best sample (True) or mean (False)
#         """
#         self.cost_fn = cost_fn
#         self.horizon = horizon
#         self.action_dim = action_dim
#         self.use_best_sample = use_best_sample
        
#         if action_bounds is not None:
#             self.action_min, self.action_max = action_bounds
#         else:
#             self.action_min, self.action_max = -1.0, 1.0
        
#         # Problem dimension
#         self.n = horizon * action_dim
        
#         # Population size
#         if population_size is None:
#             self.population_size = int(4 + 3 * np.log(self.n))
#         else:
#             self.population_size = population_size
        
#         # Selection parameters
#         self.mu = self.population_size // 2  # Number of parents
        
#         # Weights for recombination (positive weights, normalized)
#         self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
#         self.weights /= self.weights.sum()
#         self.mu_eff = 1.0 / (self.weights ** 2).sum()
        
#         # Adaptation parameters
#         self.cc = (4 + self.mu_eff / self.n) / (self.n + 4 + 2 * self.mu_eff / self.n)
#         self.cs = (self.mu_eff + 2) / (self.n + self.mu_eff + 5)
#         self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mu_eff)
#         self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.n + 2) ** 2 + self.mu_eff))
#         self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.n + 1)) - 1) + self.cs
        
#         # Expected value of ||N(0,I)||
#         self.chiN = np.sqrt(self.n) * (1 - 1/(4*self.n) + 1/(21*self.n**2))
        
#         # Initial step size
#         self.sigma = sigma

#     def _clip_actions(self, actions: np.ndarray) -> np.ndarray:
#         """Clip actions to valid bounds."""
#         return np.clip(actions, self.action_min, self.action_max)

#     def plan(
#         self,
#         env: Any,
#         initial_pos: np.ndarray,  # (2,) - x, y position
#         goal: np.ndarray,          # (2,) - x, y goal position
#         num_iterations: int,
#         init_actions: Optional[np.ndarray] = None,
#         verbose: bool = False,
#         return_trajectory: bool = False,
#     ) -> Dict[str, Any]:
        
#         T = self.horizon
#         A = self.action_dim
        
#         # Initialize mean
#         if init_actions is None:
#             mean = np.zeros(self.n)
#         else:
#             if init_actions.shape != (T, A):
#                 raise ValueError(f"init_actions must have shape ({T}, {A}), got {init_actions.shape}")
#             mean = init_actions.flatten()
        
#         # Initialize covariance matrix and evolution paths
#         C = np.eye(self.n)
#         pc = np.zeros(self.n)  # Evolution path for C
#         ps = np.zeros(self.n)  # Evolution path for sigma
#         sigma = self.sigma
        
#         history = []
#         best_x = None
#         best_cost = np.inf
#         best_trajectory = None
        
#         progressbar = tqdm(range(num_iterations), desc="CMA-ES", disable=not verbose)
        
#         for iteration in progressbar:
#             # Generate population
#             # Sample from N(0, C)
#             z_samples = np.random.randn(self.population_size, self.n)
            
#             # Compute eigendecomposition for sampling
#             D, B = np.linalg.eigh(C)
#             D = np.sqrt(np.maximum(D, 0))  # Ensure non-negative
            
#             # Transform samples: x = mean + sigma * B * D * z
#             x_samples = mean + sigma * (z_samples @ np.diag(D) @ B.T)
            
#             # Clip to action bounds
#             actions_samples = x_samples.reshape(self.population_size, T, A)
#             actions_samples = self._clip_actions(actions_samples)
            
#             # Evaluate costs
#             trajectories = rollout_trajectories_parallel(env, initial_pos, actions_samples)
#             costs, _ = self.cost_fn(trajectories, actions_samples, goal)
            
#             # Sort by cost
#             sorted_indices = np.argsort(costs)
            
#             # Track best sample
#             if costs[sorted_indices[0]] < best_cost:
#                 best_cost = costs[sorted_indices[0]]
#                 best_x = actions_samples[sorted_indices[0]].flatten()
#                 if return_trajectory:
#                     best_trajectory = trajectories[sorted_indices[0]]
            
#             # Select top mu parents
#             elite_indices = sorted_indices[:self.mu]
#             elite_x = x_samples[elite_indices]
#             elite_z = z_samples[elite_indices]
            
#             # Recombination: compute new mean
#             mean_old = mean.copy()
#             mean = (self.weights[:, np.newaxis] * elite_x).sum(axis=0)
#             mean_z = (self.weights[:, np.newaxis] * elite_z).sum(axis=0)
            
#             # Update evolution paths
#             ps = (1 - self.cs) * ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (B @ mean_z)
            
#             hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (iteration + 1))) / self.chiN 
#                     < 1.4 + 2 / (self.n + 1))
            
#             pc = (1 - self.cc) * pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * (mean - mean_old) / sigma
            
#             # Update covariance matrix
#             artmp = (elite_x - mean_old) / sigma
#             C = ((1 - self.c1 - self.cmu) * C + 
#                  self.c1 * (np.outer(pc, pc) + (1 - hsig) * self.cc * (2 - self.cc) * C) +
#                  self.cmu * (artmp.T @ np.diag(self.weights) @ artmp))
            
#             # Update step size
#             sigma = sigma * np.exp((self.cs / self.damps) * (np.linalg.norm(ps) / self.chiN - 1))
            
#             if verbose:
#                 info = {
#                     "iter": iteration,
#                     "mean_cost": float(costs.mean()),
#                     "min_cost": float(costs[sorted_indices[0]]),
#                     "best_cost_so_far": float(best_cost),
#                     "sigma": float(sigma),
#                 }
#                 history.append(info)
#                 progressbar.set_postfix(info)
        
#         # Final output
#         if self.use_best_sample and best_x is not None:
#             final_actions = best_x.reshape(T, A)
#             final_cost = best_cost
#             final_trajectory = best_trajectory
#         else:
#             final_actions = mean.reshape(T, A)
#             final_actions = self._clip_actions(final_actions)
#             # Recompute cost for mean
#             traj, _ = rollout_trajectory(env, initial_pos, final_actions)
#             final_cost, _ = self.cost_fn(traj[np.newaxis, :, :], 
#                                          final_actions[np.newaxis, :, :], goal)
#             final_trajectory = traj
        
#         out = {
#             "actions": self._clip_actions(final_actions),
#             "cost": float(final_cost),
#         }
#         if return_trajectory and final_trajectory is not None:
#             out["trajectory"] = final_trajectory
#         if verbose:
#             out["history"] = history
        
#         return out


class CMAESPlanner(BasePlanner):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) planner.

    covariance_mode:
      - "full": standard full CMA-ES
      - "diag": diagonal covariance (no correlations between parameters)
      - "block": block-banded covariance; correlations only within a temporal window
                 of size `block_size` timesteps (window size in flat space =
                 block_size * action_dim).
    """

    def __init__(
        self,
        cost_fn: BaseCostFunction,
        horizon: int,
        action_dim: int,
        action_bounds: Optional[Tuple[float, float]] = None,
        population_size: Optional[int] = None,
        sigma: float = 0.5,
        use_best_sample: bool = True,
        covariance_mode: str = "full",   # <--- NEW
        block_size: Optional[int] = None # <--- NEW (timesteps)
    ):
        """
        Args:
            cost_fn: Cost function to minimize
            horizon: Planning horizon (number of timesteps)
            action_dim: Dimension of action space
            action_bounds: (min, max) bounds for actions
            population_size: Number of samples per iteration (default: 4 + 3*log(n))
            sigma: Initial standard deviation
            use_best_sample: Return best sample (True) or mean (False)
            covariance_mode: "full", "diag", or "block"
            block_size: if covariance_mode == "block", number of timesteps
                        that are mutually correlated in the covariance.
        """
        self.cost_fn = cost_fn
        self.horizon = horizon
        self.action_dim = action_dim
        self.use_best_sample = use_best_sample

        if action_bounds is not None:
            self.action_min, self.action_max = action_bounds
        else:
            self.action_min, self.action_max = -1.0, 1.0

        # Problem dimension
        self.n = horizon * action_dim

        # Covariance structure
        assert covariance_mode in {"full", "diag", "block"}, \
            f"Unknown covariance_mode: {covariance_mode}"
        self.covariance_mode = covariance_mode

        if covariance_mode == "block":
            if block_size is None or block_size < 1:
                raise ValueError("block_size must be a positive integer when covariance_mode='block'")
            self.block_size = block_size
            self.block_dim = block_size * action_dim
        else:
            self.block_size = None
            self.block_dim = None

        # Population size
        if population_size is None:
            self.population_size = int(4 + 3 * np.log(self.n))
        else:
            self.population_size = population_size

        # Selection parameters
        self.mu = self.population_size // 2  # Number of parents

        # Weights for recombination (positive weights, normalized)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mu_eff = 1.0 / (self.weights ** 2).sum()

        # Adaptation parameters (standard CMA-ES)
        self.cc = (4 + self.mu_eff / self.n) / (self.n + 4 + 2 * self.mu_eff / self.n)
        self.cs = (self.mu_eff + 2) / (self.n + self.mu_eff + 5)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.n + 2) ** 2 + self.mu_eff)
        )
        self.damps = 1 + 2 * max(
            0, np.sqrt((self.mu_eff - 1) / (self.n + 1)) - 1
        ) + self.cs

        # Expected value of ||N(0,I)||
        self.chiN = np.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n**2))

        # Initial step size
        self.sigma = sigma

    def _clip_actions(self, actions: np.ndarray) -> np.ndarray:
        """Clip actions to valid bounds."""
        return np.clip(actions, self.action_min, self.action_max)

    def _apply_covariance_structure(self, C: np.ndarray) -> np.ndarray:
        """
        Enforce the chosen covariance structure.

        - "full": no change
        - "diag": zero-out off-diagonal entries
        - "block": zero-out entries whose |i-j| >= block_dim
                   (i.e. correlations only within windows of size block_dim
                    in the flattened parameter vector, corresponding to
                    block_size timesteps * action_dim).
        """
        if self.covariance_mode == "full":
            return C

        if self.covariance_mode == "diag":
            return np.diag(np.diag(C))

        if self.covariance_mode == "block":
            if self.block_dim is None:
                return C
            n = C.shape[0]
            # Create band mask: |i - j| < block_dim
            idx = np.arange(n)
            diff = np.abs(idx[:, None] - idx[None, :])
            mask = (diff < self.block_dim).astype(C.dtype)
            return C * mask

        return C

    def plan(
        self,
        env: Any,
        initial_pos: np.ndarray,  # (2,) - x, y position
        goal: np.ndarray,          # (2,) - x, y goal position
        num_iterations: int,
        init_actions: Optional[np.ndarray] = None,
        verbose: bool = False,
        return_trajectory: bool = False,
    ) -> Dict[str, Any]:

        T = self.horizon
        A = self.action_dim

        # Initialize mean
        if init_actions is None:
            mean = np.zeros(self.n)
        else:
            if init_actions.shape != (T, A):
                raise ValueError(f"init_actions must have shape ({T}, {A}), got {init_actions.shape}")
            mean = init_actions.flatten()

        # Initialize covariance matrix and evolution paths
        C = np.eye(self.n)
        pc = np.zeros(self.n)  # Evolution path for C
        ps = np.zeros(self.n)  # Evolution path for sigma
        sigma = self.sigma

        history = []
        best_x = None
        best_cost = np.inf
        best_trajectory = None

        progressbar = tqdm(range(num_iterations), desc="CMA-ES", disable=not verbose)

        for iteration in progressbar:
            # Sample from N(0, C)
            z_samples = np.random.randn(self.population_size, self.n)  # (λ, n)

            # Compute sampling transform depending on covariance_mode
            if self.covariance_mode == "diag":
                # C is diagonal => no need for full eigendecomposition
                diag_C = np.diag(C)
                stds = np.sqrt(np.maximum(diag_C, 1e-12))  # (n,)
                x_samples = mean + sigma * (z_samples * stds[None, :])
                # For path updates, we can treat C^(1/2) as diag(stds)
                B = np.eye(self.n)
                D = stds
            else:
                # Full or block: use eigendecomposition
                D_eig, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D_eig, 0.0))  # (n,)
                # Transform samples: x = mean + sigma * B * diag(D) * z
                x_samples = mean + sigma * (z_samples @ (B * D[None, :]).T)

            # Clip to action bounds
            actions_samples = x_samples.reshape(self.population_size, T, A)
            actions_samples = self._clip_actions(actions_samples)

            # Evaluate costs
            trajectories = rollout_trajectories_parallel(env, initial_pos, actions_samples)
            costs, _ = self.cost_fn(trajectories, actions_samples, goal)

            # Sort by cost
            sorted_indices = np.argsort(costs)

            # Track best sample
            if costs[sorted_indices[0]] < best_cost:
                best_cost = float(costs[sorted_indices[0]])
                best_x = actions_samples[sorted_indices[0]].flatten()
                if return_trajectory:
                    best_trajectory = trajectories[sorted_indices[0]]

            # Select top mu parents
            elite_indices = sorted_indices[:self.mu]
            elite_x = x_samples[elite_indices]   # (μ, n)
            elite_z = z_samples[elite_indices]   # (μ, n)

            # Recombination: compute new mean
            mean_old = mean.copy()
            mean = (self.weights[:, None] * elite_x).sum(axis=0)
            mean_z = (self.weights[:, None] * elite_z).sum(axis=0)

            # Update evolution paths
            ps = (1 - self.cs) * ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (B @ mean_z)

            hsig = (
                np.linalg.norm(ps)
                / np.sqrt(1 - (1 - self.cs) ** (2 * (iteration + 1)))
                / self.chiN
                < 1.4 + 2 / (self.n + 1)
            )

            pc = (1 - self.cc) * pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * (mean - mean_old) / sigma

            # Update covariance matrix
            artmp = (elite_x - mean_old) / sigma  # (μ, n)
            C = (
                (1 - self.c1 - self.cmu) * C
                + self.c1 * (np.outer(pc, pc) + (1 - hsig) * self.cc * (2 - self.cc) * C)
                + self.cmu * (artmp.T @ (np.diag(self.weights) @ artmp))
            )

            # Enforce desired covariance structure
            C = self._apply_covariance_structure(C)

            # Update step size
            sigma = sigma * np.exp((self.cs / self.damps) * (np.linalg.norm(ps) / self.chiN - 1))

            if verbose:
                info = {
                    "iter": iteration,
                    "mean_cost": float(costs.mean()),
                    "min_cost": float(costs[sorted_indices[0]]),
                    "best_cost_so_far": float(best_cost),
                    "sigma": float(sigma),
                }
                history.append(info)
                progressbar.set_postfix(info)

        # Final output
        if self.use_best_sample and best_x is not None:
            final_actions = best_x.reshape(T, A)
            final_cost = best_cost
            final_trajectory = best_trajectory
        else:
            final_actions = mean.reshape(T, A)
            final_actions = self._clip_actions(final_actions)
            traj, _ = rollout_trajectory(env, initial_pos, final_actions)
            final_cost, _ = self.cost_fn(
                traj[np.newaxis, :, :],
                final_actions[np.newaxis, :, :],
                goal
            )
            final_trajectory = traj

        out = {
            "actions": self._clip_actions(final_actions),
            "cost": float(final_cost),
        }
        if return_trajectory and final_trajectory is not None:
            out["trajectory"] = final_trajectory
        if verbose:
            out["history"] = history

        return out





# -------------------------------------------------------------------
# MPPIC (Model Predictive Path Integral Control)
# -------------------------------------------------------------------

class MPPICPlanner(BasePlanner):
    """
    Model Predictive Path Integral Control (MPPIC).
    
    A sampling-based method that:
    - Maintains a mean action sequence μ
    - Samples K noisy trajectories around μ
    - Computes importance weights based on trajectory costs
    - Updates μ using weighted average of samples
    
    Key features:
    - No gradients required
    - Naturally handles stochastic dynamics
    - Temperature parameter β controls exploration
    """

    def __init__(
        self,
        cost_fn: BaseCostFunction,
        horizon: int,
        action_dim: int,
        action_bounds: Optional[Tuple[float, float]] = None,
        num_samples: int = 256,
        noise_std: float = 0.3,
        beta: float = 1.0,
        init_std: float = 0.5,
        normalize_cost: bool = True,
        use_best_sample: bool = False,
    ):
        """
        Args:
            cost_fn: Cost function to minimize
            horizon: Planning horizon
            action_dim: Dimension of action space
            action_bounds: (min, max) bounds for actions
            num_samples: Number of samples per iteration
            noise_std: Standard deviation for exploration noise
            beta: Temperature parameter (higher = more exploitation)
            init_std: Initial standard deviation for random initialization
            normalize_cost: Whether to normalize costs for stable weights
            use_best_sample: Return best sample (True) or mean (False)
        """
        self.cost_fn = cost_fn
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.beta = beta
        self.init_std = init_std
        self.normalize_cost = normalize_cost
        self.use_best_sample = use_best_sample
        
        if action_bounds is not None:
            self.action_min, self.action_max = action_bounds
        else:
            self.action_min, self.action_max = -1.0, 1.0

    def _clip_actions(self, actions: np.ndarray) -> np.ndarray:
        """Clip actions to valid bounds."""
        return np.clip(actions, self.action_min, self.action_max)

    def plan(
        self,
        env: Any,
        initial_pos: np.ndarray,  # (2,) - x, y position
        goal: np.ndarray,          # (2,) - x, y goal position
        num_iterations: int,
        init_actions: Optional[np.ndarray] = None,
        verbose: bool = False,
        return_trajectory: bool = False,
    ) -> Dict[str, Any]:
        
        T = self.horizon
        A = self.action_dim
        
        # Initialize mean action sequence μ
        if init_actions is None:
            mu = self.init_std * np.random.randn(T, A)
        else:
            if init_actions.shape != (T, A):
                raise ValueError(f"init_actions must have shape ({T}, {A}), got {init_actions.shape}")
            mu = init_actions.copy()
        
        mu = self._clip_actions(mu)
        
        history = []
        best_actions = None
        best_cost = np.inf
        best_trajectory = None
        
        progressbar = tqdm(range(num_iterations), desc="MPPIC", disable=not verbose)
        
        for iteration in progressbar:
            # Sample K trajectories around mean μ
            noise = self.noise_std * np.random.randn(self.num_samples, T, A)
            actions = mu[np.newaxis, :, :] + noise  # (K, T, A)
            actions = self._clip_actions(actions)
            
            # Rollout and evaluate
            trajectories = rollout_trajectories_parallel(env, initial_pos, actions)
            costs, _ = self.cost_fn(trajectories, actions, goal)  # (K,)
            
            # Track best sample
            min_idx = np.argmin(costs)
            min_cost = costs[min_idx]
            if min_cost < best_cost:
                best_cost = min_cost
                best_actions = actions[min_idx].copy()
                if return_trajectory:
                    best_trajectory = trajectories[min_idx].copy()
            
            # Compute Path Integral weights
            J_min = costs.min()
            deltaJ = costs - J_min  # (K,)
            
            if self.normalize_cost:
                # Adaptive effective β
                scale = deltaJ.std() + 1e-6
                beta_eff = self.beta / scale
            else:
                beta_eff = self.beta
            
            # Softmax weights for numerical stability
            exp_vals = np.exp(-beta_eff * deltaJ)
            weights = exp_vals / exp_vals.sum()  # (K,)
            
            # Update mean μ as weighted average
            mu = (weights[:, np.newaxis, np.newaxis] * actions).sum(axis=0)  # (T, A)
            mu = self._clip_actions(mu)
            
            if verbose:
                info = {
                    "iter": iteration,
                    "mean_cost": float(costs.mean()),
                    "min_cost": float(min_cost),
                    "best_cost_so_far": float(best_cost),
                    "beta_eff": float(beta_eff),
                }
                history.append(info)
                progressbar.set_postfix(info)
        
        # Final output
        if self.use_best_sample and best_actions is not None:
            final_actions = best_actions
            final_cost = best_cost
            final_trajectory = best_trajectory
        else:
            final_actions = mu
            # Recompute trajectory and cost for mean
            traj, _ = rollout_trajectory(env, initial_pos, final_actions)
            final_cost, _ = self.cost_fn(traj[np.newaxis, :, :], 
                                         final_actions[np.newaxis, :, :], goal)
            final_trajectory = traj if return_trajectory else None
        
        out = {
            "actions": self._clip_actions(final_actions),
            "cost": float(final_cost),
        }
        if return_trajectory and final_trajectory is not None:
            out["trajectory"] = final_trajectory
        if verbose:
            out["history"] = history
        
        return out


# -------------------------------------------------------------------
# MPC Controller with Replanning
# -------------------------------------------------------------------

class MPCController:
    """
    Model Predictive Control (MPC) with replanning.
    
    Executes a receding horizon control strategy:
    1. Plan a horizon-length action sequence
    2. Execute first k actions
    3. Replan from new state
    4. Repeat until goal is reached or max steps exceeded
    
    Supports:
    - Configurable replanning interval
    - Warm starting with shifted previous plan
    - Early termination when goal is reached
    - Multiple planner backends (MPPIC, CMA-ES, etc.)
    """

    def __init__(
        self,
        planner: BasePlanner,
        horizon: int,
        replan_interval: int,
        action_dim: int,
        goal_tolerance: float = 0.45,
        max_steps: int = 100,
        planning_iterations: int = 10,
        use_warm_start: bool = True,
    ):
        """
        Args:
            planner: BasePlanner instance (MPPIC, CMA-ES, etc.)
            horizon: Planning horizon
            replan_interval: Number of steps between replanning
            action_dim: Dimension of action space
            goal_tolerance: Distance to goal for success
            max_steps: Maximum steps before termination
            planning_iterations: Number of optimization iterations per replan
            use_warm_start: Whether to warm start with shifted previous plan
        """
        self.planner = planner
        self.horizon = horizon
        self.replan_interval = replan_interval
        self.action_dim = action_dim
        self.goal_tolerance = goal_tolerance
        self.max_steps = max_steps
        self.planning_iterations = planning_iterations
        self.use_warm_start = use_warm_start

    def _shift_actions(self, actions: np.ndarray, shift: int) -> np.ndarray:
        """
        Shift action sequence for warm starting.
        
        Args:
            actions: (T, A) action sequence
            shift: Number of actions already executed
        
        Returns:
            (T, A) shifted actions (padded with zeros)
        """
        T, A = actions.shape
        shifted = np.zeros((T, A))
        if shift < T:
            shifted[:T-shift] = actions[shift:]
        return shifted

    def execute(
        self,
        env: Any,
        initial_pos: np.ndarray,  # (2,) - x, y position
        goal: np.ndarray,          # (2,) - x, y goal position
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute MPC control to reach the goal.
        
        Args:
            env: PointMaze environment
            initial_pos: (2,) initial x, y position
            goal: (2,) goal x, y position
            verbose: Whether to print progress
        
        Returns:
            Dictionary with:
                - 'actions': (N, action_dim) executed actions
                - 'trajectory': (N+1, 2) position trajectory
                - 'success': bool indicating if goal was reached
                - 'num_steps': number of steps taken
                - 'replans': list of replan info dicts
        """
        current_pos = initial_pos.copy()
        
        all_actions = []
        all_positions = [initial_pos.copy()]
        replans = []
        
        planned_actions = None
        steps_since_replan = 0
        
        # Save initial environment state
        saved_qpos = env.point_env.data.qpos.copy()
        saved_qvel = env.point_env.data.qvel.copy()
        saved_goal = env.goal.copy()
        
        # Set initial position
        env.point_env.data.qpos[:2] = initial_pos
        env.point_env.data.qvel[:] = 0.0
        
        progressbar = tqdm(range(self.max_steps), desc="MPC Execution", disable=not verbose)
        
        for step in progressbar:
            # Check if goal reached
            dist_to_goal = np.linalg.norm(current_pos - goal)
            if dist_to_goal <= self.goal_tolerance:
                if verbose:
                    print(f"\nGoal reached at step {step}! Distance: {dist_to_goal:.4f}")
                break
            
            # Replan if needed
            if planned_actions is None or steps_since_replan >= self.replan_interval:
                # Warm start initialization
                init_actions = None
                if self.use_warm_start and planned_actions is not None:
                    init_actions = self._shift_actions(planned_actions, steps_since_replan)
                
                # Plan from current position
                plan_result = self.planner.plan(
                    env=env,
                    initial_pos=current_pos,
                    goal=goal,
                    num_iterations=self.planning_iterations,
                    init_actions=init_actions,
                    verbose=False,
                    return_trajectory=False,
                )
                
                planned_actions = plan_result['actions']
                steps_since_replan = 0
                
                replans.append({
                    'step': step,
                    'cost': plan_result['cost'],
                    'position': current_pos.copy(),
                })
                
                if verbose:
                    progressbar.set_postfix({
                        'dist': f'{dist_to_goal:.3f}',
                        'cost': f'{plan_result["cost"]:.3f}',
                    })
            
            # Execute next action
            action = planned_actions[steps_since_replan]
            obs, reward, terminated, truncated, info = env.step(action)
            
            current_pos = obs['achieved_goal']
            all_actions.append(action)
            all_positions.append(current_pos.copy())
            steps_since_replan += 1
        
        # Restore environment state
        env.point_env.data.qpos[:] = saved_qpos
        env.point_env.data.qvel[:] = saved_qvel
        env.goal[:] = saved_goal
        
        success = np.linalg.norm(current_pos - goal) <= self.goal_tolerance
        
        return {
            'actions': np.array(all_actions),
            'trajectory': np.array(all_positions),
            'success': success,
            'num_steps': len(all_actions),
            'replans': replans,
        }


# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------

def create_mppic_planner(
    horizon: int = 20,
    action_dim: int = 2,
    action_bounds: Tuple[float, float] = (-1.0, 1.0),
    num_samples: int = 256,
    beta: float = 5.0,
) -> MPPICPlanner:
    """Create a standard MPPIC planner configuration."""
    cost_cfg = GoalCostConfig(
        distance_type="l2",
        goal_weight=1.0,
        soft_goal_weight=0.1,
        action_smooth_weight=0.01,
    )
    cost_fn = GoalCost(cost_cfg)
    
    return MPPICPlanner(
        cost_fn=cost_fn,
        horizon=horizon,
        action_dim=action_dim,
        action_bounds=action_bounds,
        num_samples=num_samples,
        beta=beta,
        normalize_cost=True,
        use_best_sample=False,
    )


def create_cmaes_planner(
    horizon: int = 20,
    action_dim: int = 2,
    action_bounds: Tuple[float, float] = (-1.0, 1.0),
    population_size: Optional[int] = None,
    sigma: float = 0.5,
) -> CMAESPlanner:
    """Create a standard CMA-ES planner configuration."""
    cost_cfg = GoalCostConfig(
        distance_type="l2",
        goal_weight=1.0,
        soft_goal_weight=0.1,
        action_smooth_weight=0.01,
    )
    cost_fn = GoalCost(cost_cfg)
    
    return CMAESPlanner(
        cost_fn=cost_fn,
        horizon=horizon,
        action_dim=action_dim,
        action_bounds=action_bounds,
        population_size=population_size,
        sigma=sigma,
        use_best_sample=True,
    )


def create_mpc_controller(
    planner: BasePlanner,
    horizon: int = 20,
    replan_interval: int = 5,
    action_dim: int = 2,
    planning_iterations: int = 10,
) -> MPCController:
    """Create a standard MPC controller configuration."""
    return MPCController(
        planner=planner,
        horizon=horizon,
        replan_interval=replan_interval,
        action_dim=action_dim,
        goal_tolerance=0.45,
        max_steps=100,
        planning_iterations=planning_iterations,
        use_warm_start=True,
    )