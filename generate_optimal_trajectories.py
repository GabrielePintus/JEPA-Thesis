import numpy as np
import json
import matplotlib.pyplot as plt
import os
import imageio  # No longer strictly needed, but might be a dependency of other code
from tqdm import tqdm
import atexit
import sys
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
from typing import Dict, Tuple, Optional

from src.planning_utils.utils_env import (
    GoalCost,
    GoalCostConfig,
    CMAESPlanner,
    MPPICPlanner,
    MPCController,
    create_cmaes_planner,
    create_mppic_planner,
    create_mpc_controller,
)
from envs.pointmaze import PointMazeEnv


# Set environment variables BEFORE any MuJoCo imports
os.environ["MUJOCO_GL"] = "egl"  # Use EGL for headless rendering (better than osmesa)
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["MPLBACKEND"] = "Agg"
os.environ["RENDER_MODE"] = "cli"


# Global list to track environments for proper cleanup
_active_envs = []

def cleanup_envs():
    """Clean up all active environments on exit to avoid context errors"""
    global _active_envs
    for env in _active_envs:
        try:
            if hasattr(env, 'mujoco_renderer') and env.mujoco_renderer is not None:
                env.mujoco_renderer.close()
            env.close()
        except Exception:
            pass
    _active_envs = []

atexit.register(cleanup_envs)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate optimal trajectories for PointMaze environments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument('--num-trajectories', '-n', type=int, default=10,
                        help='Number of trajectories to generate')
    parser.add_argument('--maze-file', type=str, default='data/single_layout.json',
                        help='Path to maze layout JSON file')
    parser.add_argument('--output-dir', '-o', type=str, default='.',
                        help='Output directory for trajectories')
    parser.add_argument('--output-name', type=str, default='trajectories',
                        help='Base name for output files (without extension)')
    
    # Sampling parameters
    parser.add_argument('--min-separation', type=float, default=1.0,
                        help='Minimum distance between start and goal positions')
    parser.add_argument('--success-threshold', type=float, default=0.5,
                        help='Maximum distance from goal to consider trajectory successful')
    
    # Planner parameters
    parser.add_argument('--horizon', type=int, default=100,
                        help='Planning horizon (number of steps)')
    parser.add_argument('--population-size', type=int, default=128,
                        help='CMA-ES population size')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='CMA-ES initial standard deviation')
    parser.add_argument('--num-iterations', type=int, default=10,
                        help='Number of optimization iterations')
    
    # Rendering parameters
    parser.add_argument('--save-frames', action='store_true',
                        help='Render and save trajectory frames to the .npz file')
    parser.add_argument('--render-size', type=int, nargs=2, default=[64, 64],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Render image size (width height)')
    
    # Parallelization
    parser.add_argument('--num-workers', '-j', type=int, default=1,
                        help='Number of parallel workers (1=sequential, -1=all CPUs)')
    
    # Verbosity
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress bars and detailed output')
    
    return parser.parse_args()


def create_env(maze_map, render_size=(64, 64)):
    """Create a PointMaze environment"""
    side = max(len(maze_map), len(maze_map[0])) + 2
    env = PointMazeEnv(
        maze_map=maze_map,
        render_goal=False,
        render_img_size=tuple(render_size),
        camera_distance=side
    )
    env.reset()
    return env


def sample_valid_position(env, min_distance_from_walls=0.3):
    """
    Sample a valid position in the maze that is not in a wall.
    Uses the maze's empty cell locations and adds small random noise.
    
    Args:
        env: PointMazeEnv instance
        min_distance_from_walls: Minimum distance from walls (for safety margin)
    
    Returns:
        np.ndarray: Valid (x, y) position
    """
    # Get all valid locations (reset + goal locations cover all empty cells)
    valid_locations = env.maze.unique_reset_locations + env.maze.unique_goal_locations
    
    if not valid_locations:
        # Fallback: sample uniformly if no explicit valid locations
        return np.random.uniform(low=-1.5, high=1.5, size=(2,))
    
    # Choose a random valid cell
    cell_idx = np.random.randint(0, len(valid_locations))
    base_pos = valid_locations[cell_idx].copy()
    
    # Add small noise within the cell (keeping away from walls)
    noise_range = min(0.3, min_distance_from_walls)
    noise = np.random.uniform(low=-noise_range, high=noise_range, size=(2,))
    
    return base_pos + noise


def sample_goal_and_start(env, min_separation=1.0):
    """
    Sample both goal and initial positions ensuring they are sufficiently separated.
    
    Args:
        env: PointMazeEnv instance
        min_separation: Minimum distance between start and goal
    
    Returns:
        tuple: (initial_pos, goal_pos)
    """
    max_attempts = 100
    
    for _ in range(max_attempts):
        initial_pos = sample_valid_position(env)
        goal_pos = sample_valid_position(env)
        
        # Check if they are sufficiently separated
        distance = np.linalg.norm(initial_pos - goal_pos)
        if distance >= min_separation:
            return initial_pos, goal_pos
    
    # Fallback: just return the last sampled positions
    warnings.warn(f"Could not find positions with separation >= {min_separation}, using best found")
    return initial_pos, goal_pos


def optimize_trajectory(env, initial_pos, goal_pos, planner_config, verbose=False):
    """
    Optimize a trajectory using CMA-ES planner.
    
    Args:
        env: PointMazeEnv instance
        initial_pos: Initial position
        goal_pos: Goal position
        planner_config: Dictionary with planner parameters
        verbose: Whether to print optimization progress
    
    Returns:
        np.ndarray: Optimized actions
    """
    cost_cfg = GoalCostConfig(
        distance_type="l2",
        goal_weight=1.0,
        soft_goal_weight=1e-4,
        action_smooth_weight=1e-4,
    )
    
    planner = CMAESPlanner(
        cost_fn=GoalCost(cost_cfg),
        horizon=planner_config['horizon'],
        action_dim=2,
        action_bounds=(-1.0, 1.0),
        population_size=planner_config['population_size'],
        sigma=planner_config['sigma'],
        use_best_sample=True,
    )
    
    result = planner.plan(
        env=env,
        initial_pos=initial_pos,
        goal=goal_pos,
        num_iterations=planner_config['num_iterations'],
        verbose=verbose,
        return_trajectory=True,
    )
    return result['actions']


def rollout_and_check(env, initial_pos, goal_pos, actions, success_threshold, collect_frames=True):
    """
    Rollout trajectory, collect frames, and check success.
    
    Args:
        env: PointMazeEnv instance
        initial_pos: Initial position
        goal_pos: Goal position
        actions: Action sequence
        success_threshold: Maximum distance from goal to consider successful
        collect_frames: Whether to collect rendered frames
    
    Returns:
        dict: Contains states, frames (if collected), final_distance, and success flag
    """
    trajectory_states = []
    trajectory_frames = [] if collect_frames else None
    
    env.set_position(initial_pos)
    
    # Initial state
    trajectory_states.append(np.concatenate([initial_pos, np.zeros(2)], axis=0))
    
    # Initial frame
    if collect_frames:
        try:
            frame = env.render()
            trajectory_frames.append(frame)
        except Exception as e:
            warnings.warn(f"Failed to render initial frame: {e}")
            trajectory_frames = None
            collect_frames = False
    
    # Rollout trajectory
    for action in actions:
        obs_dict, _, _, _, _ = env.step(action)
        trajectory_states.append(obs_dict['observation'])
        
        if collect_frames and trajectory_frames is not None:
            try:
                frame = env.render()
                trajectory_frames.append(frame)
            except Exception as e:
                warnings.warn(f"Failed to render frame: {e}")
                trajectory_frames = None
                collect_frames = False
    
    # Check success: final position should be close to goal
    final_pos = trajectory_states[-1][:2]  # Extract x, y from [x, y, vx, vy]
    final_distance = np.linalg.norm(final_pos - goal_pos)
    success = final_distance <= success_threshold
    
    return {
        'states': trajectory_states,
        'frames': trajectory_frames,
        'final_distance': final_distance,
        'success': success,
    }


def generate_single_trajectory(args_tuple):
    """
    Generate a single trajectory (for parallel execution).
    
    Args:
        args_tuple: Tuple of (trajectory_id, maze_map, config_dict)
    
    Returns:
        dict: Trajectory data or None if failed
    """
    traj_id, maze_map, config = args_tuple
    
    try:
        # Create environment (not tracked globally in worker process)
        env = create_env(maze_map, config['render_size'])
        
        # Sample positions
        initial_pos, goal_pos = sample_goal_and_start(env, config['min_separation'])
        init_distance = np.linalg.norm(goal_pos - initial_pos)
        
        # Optimize trajectory
        actions = optimize_trajectory(
            env, initial_pos, goal_pos, 
            config['planner_config'], 
            verbose=config['verbose']
        )
        
        # Rollout and check success
        rollout_result = rollout_and_check(
            env, initial_pos, goal_pos, actions,
            config['success_threshold'],
            collect_frames=config['save_frames']  # CHANGED: Use save_frames
        )
        
        # Clean up environment
        try:
            if hasattr(env, 'mujoco_renderer') and env.mujoco_renderer is not None:
                env.mujoco_renderer.close()
            env.close()
        except Exception:
            pass
        
        # Prepare result
        result = {
            'id': traj_id,
            'states': rollout_result['states'],
            'frames': rollout_result['frames'],
            'actions': actions,
            'initial_pos': initial_pos,
            'goal_pos': goal_pos,
            'init_distance': init_distance,
            'final_distance': rollout_result['final_distance'],
            'success': rollout_result['success'],
            'maze_map': maze_map,
        }
        
        if not config['quiet']:
            status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
            print(f"\nTrajectory {traj_id}: {status} | "
                  f"Start-Goal: {init_distance:.2f} | "
                  f"Final dist: {rollout_result['final_distance']:.3f}")
        
        return result
        
    except Exception as e:
        if not config['quiet']:
            print(f"\nTrajectory {traj_id}: ERROR - {str(e)}")
        return None


def save_results(trajectories, args):
    """Save trajectory results to disk"""
    
    # Filter successful trajectories
    successful = [t for t in trajectories if t is not None and t['success']]
    failed = [t for t in trajectories if t is not None and not t['success']]
    
    print(f"\n{'='*60}")
    print(f"Results Summary:")
    print(f"  Total trajectories: {len(trajectories)}")
    print(f"  Successful: {len(successful)} ({len(successful)/len(trajectories)*100:.1f}%)")
    print(f"  Failed: {len(failed)} ({len(failed)/len(trajectories)*100:.1f}%)")
    if failed:
        avg_final_dist = np.mean([t['final_distance'] for t in failed])
        print(f"  Average final distance (failed): {avg_final_dist:.3f}")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # REMOVED: Video saving block
    
    # Save trajectory data
    output_path = os.path.join(args.output_dir, f"{args.output_name}.npz")
    
    save_dict = {}
    for traj in trajectories:
        if traj is not None:
            traj_key = f"traj_{traj['id']:03d}"
            traj_data = {
                'states': np.array(traj['states']),
                'actions': np.array(traj['actions']),
                'initial_pos': np.array(traj['initial_pos']),
                'goal_pos': np.array(traj['goal_pos']),
                'init_distance': traj['init_distance'],
                'final_distance': traj['final_distance'],
                'success': traj['success'],
            }
            
            # ADDED: Save frames to the npz file if they were collected
            if args.save_frames and traj['frames'] is not None:
                traj_data['frames'] = np.array(traj['frames'])
            
            save_dict[traj_key] = traj_data
    
    np.savez_compressed(output_path, **save_dict)
    print(f"Saved trajectory data to {output_path}")
    
    # Save summary statistics
    summary_path = os.path.join(args.output_dir, f"{args.output_name}_summary.json")
    summary = {
        'total_trajectories': len(trajectories),
        'successful': len(successful),
        'failed': len(failed),
        'success_rate': len(successful) / len(trajectories) if trajectories else 0,
        'config': {
            'num_trajectories': args.num_trajectories,
            'min_separation': args.min_separation,
            'success_threshold': args.success_threshold,
            'horizon': args.horizon,
            'population_size': args.population_size,
            'num_iterations': args.num_iterations,
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


def main():
    """Main execution function"""
    args = parse_args()
    
    # Determine number of workers
    if args.num_workers == -1:
        num_workers = cpu_count()
    else:
        num_workers = max(1, args.num_workers)
    
    print(f"{'='*60}")
    print(f"Optimal Trajectory Generation")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Trajectories: {args.num_trajectories}")
    print(f"  Workers: {num_workers}")
    print(f"  Horizon: {args.horizon}")
    print(f"  Success threshold: {args.success_threshold}")
    print(f"  Min separation: {args.min_separation}")
    print(f"  Save frames: {args.save_frames}")  # CHANGED
    print(f"  Output: {os.path.join(args.output_dir, args.output_name)}")
    print(f"{'='*60}\n")
    
    # Load maze layouts
    with open(args.maze_file, "r") as f:
        layouts = json.load(f)
    maze_maps = layouts["maps"]
    
    # Prepare configuration for workers
    config = {
        'min_separation': args.min_separation,
        'success_threshold': args.success_threshold,
        'save_frames': args.save_frames,  # CHANGED
        'render_size': args.render_size,
        'verbose': args.verbose,
        'quiet': args.quiet,
        'planner_config': {
            'horizon': args.horizon,
            'population_size': args.population_size,
            'sigma': args.sigma,
            'num_iterations': args.num_iterations,
        }
    }
    
    # Prepare arguments for parallel execution
    task_args = [
        (i, maze_maps[i % len(maze_maps)], config)
        for i in range(args.num_trajectories)
    ]
    
    # Generate trajectories
    try:
        if num_workers == 1:
            # Sequential execution
            trajectories = []
            for task_arg in tqdm(task_args, desc="Generating trajectories", disable=args.quiet):
                result = generate_single_trajectory(task_arg)
                trajectories.append(result)
        else:
            # Parallel execution
            with Pool(num_workers) as pool:
                trajectories = list(tqdm(
                    pool.imap(generate_single_trajectory, task_args),
                    total=len(task_args),
                    desc="Generating trajectories",
                    disable=args.quiet
                ))
        
        # Save results
        save_results(trajectories, args)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cleanup_envs()
        print("Cleanup complete")


if __name__ == "__main__":
    main()