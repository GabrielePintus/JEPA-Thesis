"""
Utility functions for planning analysis and visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import seaborn as sns


def create_channel_mask(
    total_channels: int,
    mask_type: str = 'visual_only',
    visual_channels: int = 16,
    proprio_channels: int = 2
) -> torch.Tensor:
    """
    Create a channel mask for cost computation.
    
    Args:
        total_channels: Total number of channels in latent state
        mask_type: Type of mask - 'visual_only', 'proprio_only', or 'all'
        visual_channels: Number of visual channels
        proprio_channels: Number of proprioceptive channels
        
    Returns:
        Boolean mask tensor of shape (total_channels,)
    """
    mask = torch.zeros(total_channels, dtype=torch.bool)
    
    if mask_type == 'visual_only':
        mask[:visual_channels] = True
    elif mask_type == 'proprio_only':
        mask[visual_channels:visual_channels + proprio_channels] = True
    elif mask_type == 'all':
        mask[:] = True
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")
        
    return mask


def compare_planners(
    results: Dict[str, Dict],
    save_path: Optional[str] = None
) -> None:
    """
    Compare multiple planner results and visualize.
    
    Args:
        results: Dictionary mapping planner names to result dictionaries
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Cost convergence
    ax = axes[0, 0]
    for name, result in results.items():
        if 'info' in result and 'cost_history' in result['info']:
            ax.plot(result['info']['cost_history'], label=name, linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cost', fontsize=12)
    ax.set_title('Cost Convergence', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Final costs comparison
    ax = axes[0, 1]
    planner_names = list(results.keys())
    final_costs = [results[name]['cost'] for name in planner_names]
    colors = sns.color_palette("husl", len(planner_names))
    bars = ax.bar(planner_names, final_costs, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Final Cost', fontsize=12)
    ax.set_title('Final Cost Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Action sequences
    ax = axes[1, 0]
    for i, (name, result) in enumerate(results.items()):
        actions = result['actions'].cpu().numpy()
        timesteps = np.arange(len(actions))
        # Plot first action dimension
        ax.plot(timesteps, actions[:, 0], label=f'{name} (dim 0)', 
                linewidth=2, alpha=0.7)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Action Value', fontsize=12)
    ax.set_title('Action Sequences (Dimension 0)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Computation statistics
    ax = axes[1, 1]
    stats_data = []
    labels = []
    for name, result in results.items():
        if 'info' in result:
            info = result['info']
            if 'num_iterations' in info:
                stats_data.append(info['num_iterations'])
                labels.append(f'{name}\nIterations')
            if 'population_size' in info:
                stats_data.append(info['population_size'])
                labels.append(f'{name}\nPopulation')
            if 'num_samples' in info:
                stats_data.append(info['num_samples'])
                labels.append(f'{name}\nSamples')
    
    if stats_data:
        ax.bar(range(len(stats_data)), stats_data, color=colors[:len(stats_data)], 
               alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(stats_data)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Algorithm Parameters', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No statistics available', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def visualize_mpc_execution(
    mpc_result: Dict,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize MPC execution results.
    
    Args:
        mpc_result: Result dictionary from MPCController.execute()
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Cost over time
    ax = axes[0, 0]
    steps = np.arange(len(mpc_result['cost_history']))
    ax.plot(steps, mpc_result['cost_history'], linewidth=2, color='steelblue')
    ax.axhline(y=0.1, color='red', linestyle='--', label='Success threshold', alpha=0.5)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Cost to Goal', fontsize=12)
    ax.set_title('MPC Cost Evolution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Actions over time
    ax = axes[0, 1]
    actions = mpc_result['action_history'].cpu().numpy()
    timesteps = np.arange(len(actions))
    for dim in range(actions.shape[1]):
        ax.plot(timesteps, actions[:, dim], label=f'Action dim {dim}', 
                linewidth=2, alpha=0.7)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Action Value', fontsize=12)
    ax.set_title('Executed Actions', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Replanning timeline
    ax = axes[1, 0]
    if mpc_result['plan_history']:
        replan_steps = [p['step'] for p in mpc_result['plan_history']]
        predicted_costs = [p['predicted_cost'] for p in mpc_result['plan_history']]
        
        ax.scatter(replan_steps, predicted_costs, s=100, alpha=0.6, 
                   c=range(len(replan_steps)), cmap='viridis', edgecolors='black')
        ax.plot(replan_steps, predicted_costs, '--', alpha=0.3, color='gray')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                   norm=plt.Normalize(vmin=0, vmax=len(replan_steps)-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Replan Index', fontsize=10)
        
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Predicted Cost', fontsize=12)
        ax.set_title('Replanning Events', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No replanning data', 
                ha='center', va='center', transform=ax.transAxes)
    
    # Plot 4: Statistics summary
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    MPC Execution Summary
    {'='*40}
    
    Success: {'✓' if mpc_result['success'] else '✗'}
    Final Cost: {mpc_result['final_cost']:.6f}
    
    Total Steps: {mpc_result['num_steps']}
    Number of Replans: {mpc_result['num_replans']}
    
    Total Planning Time: {mpc_result['total_planning_time']:.3f}s
    Avg Planning Time: {mpc_result['avg_planning_time']:.3f}s
    
    Initial Cost: {mpc_result['cost_history'][0]:.6f}
    Cost Reduction: {(1 - mpc_result['final_cost']/mpc_result['cost_history'][0])*100:.1f}%
    """
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='center', 
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MPC execution plot saved to {save_path}")
    
    plt.show()


def analyze_trajectory_smoothness(actions: torch.Tensor) -> Dict[str, float]:
    """
    Analyze smoothness metrics of an action trajectory.
    
    Args:
        actions: Action sequence (T, action_dim)
        
    Returns:
        Dictionary of smoothness metrics
    """
    actions_np = actions.cpu().numpy()
    
    # Compute derivatives
    velocities = np.diff(actions_np, axis=0)
    accelerations = np.diff(velocities, axis=0)
    jerks = np.diff(accelerations, axis=0)
    
    metrics = {
        'mean_velocity': np.abs(velocities).mean(),
        'max_velocity': np.abs(velocities).max(),
        'mean_acceleration': np.abs(accelerations).mean(),
        'max_acceleration': np.abs(accelerations).max(),
        'mean_jerk': np.abs(jerks).mean(),
        'max_jerk': np.abs(jerks).max(),
        'action_range': actions_np.max() - actions_np.min(),
    }
    
    return metrics


def create_planning_report(
    planner_name: str,
    result: Dict,
    save_path: Optional[str] = None
) -> str:
    """
    Create a detailed text report of planning results.
    
    Args:
        planner_name: Name of the planner
        result: Result dictionary from planner.optimize()
        save_path: Optional path to save report
        
    Returns:
        Report string
    """
    report_lines = [
        "=" * 60,
        f"Planning Report: {planner_name}",
        "=" * 60,
        "",
        "Final Results:",
        f"  Final Cost: {result['cost']:.8f}",
        f"  Trajectory Length: {result['trajectory'].shape[0]}",
        f"  Action Dimension: {result['actions'].shape[1]}",
        "",
    ]
    
    # Add algorithm-specific info
    if 'info' in result:
        report_lines.append("Algorithm Information:")
        info = result['info']
        
        for key, value in info.items():
            if key == 'cost_history':
                report_lines.append(f"  Cost Improvement: {info['initial_cost']:.6f} → {info['final_cost']:.6f}")
                improvement = (1 - info['final_cost'] / info['initial_cost']) * 100
                report_lines.append(f"  Improvement: {improvement:.2f}%")
            elif key not in ['initial_cost', 'final_cost']:
                if isinstance(value, float):
                    report_lines.append(f"  {key}: {value:.6f}")
                else:
                    report_lines.append(f"  {key}: {value}")
        report_lines.append("")
    
    # Analyze trajectory smoothness
    smoothness = analyze_trajectory_smoothness(result['actions'])
    report_lines.extend([
        "Trajectory Smoothness Metrics:",
        f"  Mean Velocity: {smoothness['mean_velocity']:.6f}",
        f"  Max Velocity: {smoothness['max_velocity']:.6f}",
        f"  Mean Acceleration: {smoothness['mean_acceleration']:.6f}",
        f"  Max Acceleration: {smoothness['max_acceleration']:.6f}",
        f"  Mean Jerk: {smoothness['mean_jerk']:.6f}",
        f"  Action Range: {smoothness['action_range']:.6f}",
        "",
    ])
    
    report_lines.append("=" * 60)
    
    report = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {save_path}")
    
    return report
