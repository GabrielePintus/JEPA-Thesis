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
import networkx as nx

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
    
    # Graph-based sampling parameters
    parser.add_argument('--use-graph-sampling', action='store_true',
                        help='Use graph-based sampling to prioritize paths requiring detours')
    parser.add_argument('--detour-ratio', type=float, default=1.5,
                        help='Minimum threshold ratio for detour detection (graph_path_len / euclidean_dist)')
    parser.add_argument('--graph-max-dist', type=float, default=1.0,
                        help='Maximum distance to consider cells as connected in the graph')
    parser.add_argument('--sample-tries', type=int, default=500,
                        help='Number of sampling attempts to find good pairs')
    parser.add_argument('--sampling-mode', type=str, default='best', choices=['best', 'threshold'],
                        help='Sampling strategy: "best" picks highest scoring pair, "threshold" picks first valid')
    parser.add_argument('--visualize-graph', action='store_true',
                        help='Save a visualization of the maze graph')
    
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


# ============================================================================
# Graph-based sampling functions
# ============================================================================

def build_maze_graph(locations, max_dist=1.0):
    """
    Build a graph from maze locations where nodes are connected if they are
    within max_dist of each other (adjacent cells).
    
    Args:
        locations: List of (x, y) positions from the maze
        max_dist: Maximum distance to consider cells as connected
    
    Returns:
        networkx.Graph: Graph with nodes representing maze locations
    """
    G = nx.Graph()
    locations_array = np.array(locations)
    
    # Add all locations as nodes with their positions
    for i, pos in enumerate(locations_array):
        G.add_node(i, pos=tuple(pos))
    
    # Connect adjacent locations
    for i in range(len(locations_array)):
        for j in range(i + 1, len(locations_array)):
            dist = np.linalg.norm(locations_array[i] - locations_array[j])
            if dist <= max_dist + 1e-9:  # Small epsilon for numerical stability
                G.add_edge(i, j)
    
    return G, locations_array


def compute_detour_score(G, locations, i, j):
    """
    Compute a score that rewards high graph distance and low Euclidean distance.
    Higher scores indicate better pairs for learning topology.
    
    Args:
        G: NetworkX graph of maze locations
        locations: Array of (x, y) positions
        i, j: Indices of start and goal locations
    
    Returns:
        float: Detour score (higher is better), or -inf if unreachable
    """
    euclidean_dist = np.linalg.norm(locations[i] - locations[j])
    
    # Avoid division by zero or very small distances
    if euclidean_dist < 0.5:
        return -np.inf
    
    try:
        # Find shortest path through the graph
        path = nx.shortest_path(G, i, j)
        graph_path_length = len(path) - 1  # Number of edges
    except nx.NetworkXNoPath:
        return -np.inf  # Unreachable
    
    # We want to maximize: graph_distance / euclidean_distance
    # This ratio captures how much the maze forces a detour
    detour_ratio = graph_path_length / (euclidean_dist / 0.5)
    
    return detour_ratio


def path_requires_detour(G, locations, i, j, threshold_ratio=1.0):
    """
    Check if the path from location i to j requires a significant detour
    compared to the straight-line distance.
    
    Args:
        G: NetworkX graph of maze locations
        locations: Array of (x, y) positions
        i, j: Indices of start and goal locations
        threshold_ratio: Path is considered a detour if graph_length > ratio * euclidean_dist
    
    Returns:
        bool: True if path requires a detour
    """
    score = compute_detour_score(G, locations, i, j)
    return score >= threshold_ratio


def sample_detour_pair(G, locations, tries=500, threshold_ratio=1.0, mode='best'):
    """
    Sample a pair of start/goal locations that maximizes topology learning.
    
    Two modes available:
    - 'best': Sample many pairs and return the one with highest detour score
    - 'threshold': Return first pair meeting threshold (faster but less optimal)
    
    Args:
        G: NetworkX graph of maze locations
        locations: Array of (x, y) positions
        tries: Number of sampling attempts
        threshold_ratio: Minimum detour ratio required
        mode: Sampling strategy ('best' or 'threshold')
    
    Returns:
        tuple: (start_idx, goal_idx) or (None, None) if no suitable pair found
    """
    n = len(locations)
    
    if mode == 'best':
        # Sample many pairs and keep the best one
        best_score = -np.inf
        best_pair = (None, None)
        
        for _ in range(tries):
            # Randomly sample two different locations
            i, j = np.random.choice(n, size=2, replace=False)
            
            score = compute_detour_score(G, locations, i, j)
            
            # Update best if this pair has higher score and meets threshold
            if score >= threshold_ratio and score > best_score:
                best_score = score
                best_pair = (i, j)
        
        return best_pair
    
    else:  # mode == 'threshold'
        # Return first pair that meets threshold (original behavior)
        for _ in range(tries):
            i, j = np.random.choice(n, size=2, replace=False)
            
            if path_requires_detour(G, locations, i, j, threshold_ratio):
                return i, j
        
        return None, None


def visualize_maze_graph(G, locations, output_path, start_idx=None, goal_idx=None):
    """
    Visualize the maze graph with locations and connections.
    
    Args:
        G: NetworkX graph
        locations: Array of (x, y) positions
        output_path: Path to save the visualization
        start_idx: Optional starting location to highlight
        goal_idx: Optional goal location to highlight
    """
    plt.figure(figsize=(8, 8))
    
    # Create position dictionary
    pos = {i: tuple(locations[i]) for i in G.nodes()}
    
    # Draw edges first (so they appear behind nodes)
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=2, alpha=0.6)
    
    # Draw all nodes
    node_colors = ['skyblue'] * len(G.nodes())
    if start_idx is not None:
        node_colors[start_idx] = 'green'
    if goal_idx is not None:
        node_colors[goal_idx] = 'red'
    
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add edge labels with distances
    edge_labels = {
        (u, v): f"{np.linalg.norm(locations[u] - locations[v]):.2f}"
        for u, v in G.edges()
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.title('Maze Graph Topology', fontsize=14, fontweight='bold')
    
    if start_idx is not None or goal_idx is not None:
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='skyblue', label='Available locations'),
        ]
        if start_idx is not None:
            legend_elements.append(Patch(facecolor='green', label='Start'))
        if goal_idx is not None:
            legend_elements.append(Patch(facecolor='red', label='Goal'))
        plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Original sampling functions (with modifications)
# ============================================================================

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


def sample_goal_and_start_graph(env, graph_config):
    """
    Sample start and goal positions using graph-based approach to maximize
    topology learning by finding pairs with high graph distance but low Euclidean distance.
    
    Args:
        env: PointMazeEnv instance
        graph_config: Dictionary with graph sampling parameters
    
    Returns:
        tuple: (initial_pos, goal_pos, metadata)
    """
    # Get all valid locations
    locations = env.maze.unique_reset_locations
    if not locations:
        locations = env.maze.unique_reset_locations + env.maze.unique_goal_locations
    
    locations = np.array(locations)
    
    # Build graph
    G, locations_array = build_maze_graph(
        locations, 
        max_dist=graph_config['max_dist']
    )
    
    # Sample pair using specified strategy
    start_idx, goal_idx = sample_detour_pair(
        G, 
        locations_array,
        tries=graph_config['sample_tries'],
        threshold_ratio=graph_config['detour_ratio'],
        mode=graph_config['sampling_mode']
    )
    
    if start_idx is None or goal_idx is None:
        # Fallback to random sampling if no suitable pair found
        warnings.warn("Could not find detour-requiring pair, falling back to random sampling")
        start_idx = np.random.randint(0, len(locations_array))
        goal_idx = np.random.randint(0, len(locations_array))
        while goal_idx == start_idx:
            goal_idx = np.random.randint(0, len(locations_array))
    
    # Get base positions
    initial_pos = locations_array[start_idx].copy()
    goal_pos = locations_array[goal_idx].copy()
    
    # Add small random noise to avoid exact grid positions
    initial_pos += np.random.uniform(-0.15, 0.15, size=(2,))
    goal_pos += np.random.uniform(-0.15, 0.15, size=(2,))
    
    # Calculate metadata
    euclidean_dist = np.linalg.norm(initial_pos - goal_pos)
    detour_score = compute_detour_score(G, locations_array, start_idx, goal_idx)
    
    try:
        path = nx.shortest_path(G, start_idx, goal_idx)
        graph_path_length = len(path) - 1
    except nx.NetworkXNoPath:
        graph_path_length = -1
    
    metadata = {
        'start_idx': start_idx,
        'goal_idx': goal_idx,
        'euclidean_distance': euclidean_dist,
        'graph_path_length': graph_path_length,
        'detour_score': detour_score,
        'graph': G,
        'locations': locations_array,
    }
    
    return initial_pos, goal_pos, metadata


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

        # Add noise to initial and goal positions
        initial_pos += np.random.uniform(-0.15, 0.15, size=(2,))
        goal_pos += np.random.uniform(-0.15, 0.15, size=(2,))
        
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
        dict: Result containing states, frames, final_distance, and success flag
    """
    # Reset environment to initial position
    obs, _ = env.reset()
    
    # Set the agent to the initial position
    qpos = env.data.qpos.copy()
    qpos[:2] = initial_pos
    qvel = env.data.qvel.copy()
    qvel[:] = 0.0
    env.set_state(qpos, qvel)
    
    # Collect trajectory
    trajectory_states = []
    trajectory_frames = [] if collect_frames else None
    
    for action in actions:
        # Get current state
        state = np.concatenate([env.data.qpos[:2].copy(), env.data.qvel[:2].copy()])
        trajectory_states.append(state)
        
        # Render frame if requested
        if collect_frames:
            frame = env.render()
            trajectory_frames.append(frame)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
    
    # Get final state
    final_state = np.concatenate([env.data.qpos[:2].copy(), env.data.qvel[:2].copy()])
    trajectory_states.append(final_state)
    
    if collect_frames:
        final_frame = env.render()
        trajectory_frames.append(final_frame)
    
    # Check success
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
        
        # Sample positions (using graph-based or standard approach)
        if config['use_graph_sampling']:
            initial_pos, goal_pos, metadata = sample_goal_and_start_graph(
                env, 
                config['graph_config']
            )
            graph_path_length = metadata['graph_path_length']
            detour_score = metadata.get('detour_score', -1)
        else:
            initial_pos, goal_pos = sample_goal_and_start(env, config['min_separation'])
            graph_path_length = -1
            detour_score = -1
        
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
            collect_frames=config['save_frames']
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
            'graph_path_length': graph_path_length,
            'detour_score': detour_score,
            'maze_map': maze_map,
        }
        
        if not config['quiet']:
            status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
            if detour_score > 0:
                graph_info = f"| Graph: {graph_path_length} edges | Score: {detour_score:.2f}"
            elif graph_path_length > 0:
                graph_info = f"| Graph: {graph_path_length} edges"
            else:
                graph_info = ""
            print(f"\nTrajectory {traj_id}: {status} | "
                  f"Euclidean: {init_distance:.2f} {graph_info} | "
                  f"Final dist: {rollout_result['final_distance']:.3f}")
        
        return result
        
    except Exception as e:
        if not config['quiet']:
            print(f"\nTrajectory {traj_id}: ERROR - {str(e)}")
        import traceback
        traceback.print_exc()
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
    
    if args.use_graph_sampling:
        valid_graph_trajs = [t for t in trajectories if t is not None and t['graph_path_length'] > 0]
        if valid_graph_trajs:
            avg_graph_length = np.mean([t['graph_path_length'] for t in valid_graph_trajs])
            avg_euclidean = np.mean([t['init_distance'] for t in valid_graph_trajs])
            
            # Calculate detour statistics
            detour_scores = [t['detour_score'] for t in valid_graph_trajs if t['detour_score'] > 0]
            
            print(f"\n  Graph Sampling Statistics:")
            print(f"    Average graph path length: {avg_graph_length:.2f} edges")
            print(f"    Average Euclidean distance: {avg_euclidean:.2f}")
            print(f"    Average detour ratio: {avg_graph_length * 0.5 / avg_euclidean:.2f}x")
            
            if detour_scores:
                print(f"    Average detour score: {np.mean(detour_scores):.2f}")
                print(f"    Max detour score: {np.max(detour_scores):.2f}")
                print(f"    Min detour score: {np.min(detour_scores):.2f}")
    
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
                'graph_path_length': traj['graph_path_length'],
                'detour_score': traj.get('detour_score', -1),
            }
            
            # Save frames to the npz file if they were collected
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
            'use_graph_sampling': args.use_graph_sampling,
        }
    }
    
    if args.use_graph_sampling:
        summary['config']['detour_ratio'] = args.detour_ratio
        summary['config']['graph_max_dist'] = args.graph_max_dist
        summary['config']['sampling_mode'] = args.sampling_mode
    
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
    print(f"  Graph sampling: {args.use_graph_sampling}")
    if args.use_graph_sampling:
        print(f"    Detour ratio threshold: {args.detour_ratio}")
        print(f"    Max edge distance: {args.graph_max_dist}")
        print(f"    Sampling mode: {args.sampling_mode}")
        print(f"    Sample tries: {args.sample_tries}")
    print(f"  Save frames: {args.save_frames}")
    print(f"  Output: {os.path.join(args.output_dir, args.output_name)}")
    print(f"{'='*60}\n")
    
    # Load maze layouts
    with open(args.maze_file, "r") as f:
        layouts = json.load(f)
    maze_maps = layouts["maps"]
    
    # Optionally visualize the maze graph
    if args.visualize_graph and args.use_graph_sampling:
        print("Generating maze graph visualization...")
        env = create_env(maze_maps[0], args.render_size)
        locations = env.maze.unique_reset_locations
        if not locations:
            locations = env.maze.unique_reset_locations + env.maze.unique_goal_locations
        locations = np.array(locations)
        
        G, locations_array = build_maze_graph(locations, max_dist=args.graph_max_dist)
        
        viz_path = os.path.join(args.output_dir, f"{args.output_name}_graph.png")
        visualize_maze_graph(G, locations_array, viz_path)
        print(f"Saved graph visualization to {viz_path}")
        
        # Clean up
        try:
            if hasattr(env, 'mujoco_renderer') and env.mujoco_renderer is not None:
                env.mujoco_renderer.close()
            env.close()
        except Exception:
            pass
    
    # Prepare configuration for workers
    config = {
        'min_separation': args.min_separation,
        'success_threshold': args.success_threshold,
        'save_frames': args.save_frames,
        'render_size': args.render_size,
        'verbose': args.verbose,
        'quiet': args.quiet,
        'use_graph_sampling': args.use_graph_sampling,
        'graph_config': {
            'detour_ratio': args.detour_ratio,
            'max_dist': args.graph_max_dist,
            'sample_tries': args.sample_tries,
            'sampling_mode': args.sampling_mode,
        },
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