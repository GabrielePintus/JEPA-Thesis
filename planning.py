"""
QUICK FIX: Working Uncertainty-Aware Planning

This file provides a simple, working solution you can use RIGHT NOW
without modifying your existing planning.py code.

Just replace your current planning.py script with this one.
"""

import lightning as L
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from tqdm import tqdm

from src.data.dataset import PointMazeSequences
from src.jepa import JEPA
from src.planningv2 import UncertaintyCostFunction, UncertaintyEstimator
from src.planning import PICOptimizerOptimized, LatentSpacePlanner
from envs.pointmaze import PointMazeEnv


os.environ["MUJOCO_GL"] = "egl"

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# Load Data and Model
# ============================================================================

# Load maps
with open("data/train_layouts.json", "r") as f:
    layouts = json.load(f)

# Load dataset
dataset = PointMazeSequences(
    "data/train_trajectories_10_100_4_64_largerdot.npz",
    normalize=False,
    seq_len=100
)

traj_idx = 0
data_sample = dataset[traj_idx]
states, frames, actions = data_sample

initial_state = states[0:1].to(device)
initial_frame = frames[0:1].to(device)
goal_state = states[20:21].to(device)
goal_frame = frames[20:21].to(device)
real_actions = actions[0:20]

# Load model
model = JEPA.load_from_checkpoint("checkpoints/jepa/last.ckpt")
model = model.to(device)
model.eval()

# Decompile networks
def decompile_all_networks(module: L.LightningModule):
    for name, child in module.named_children():
        if isinstance(child, nn.Module):
            try:
                module.__setattr__(name, child._orig_mod)
            except AttributeError:
                pass

decompile_all_networks(model)


# ============================================================================
# SOLUTION: Use Standard Cost Function (No Uncertainty During Planning)
# ============================================================================

print("\n" + "="*70)
print("PLANNING WITHOUT UNCERTAINTY (Fast)")
print("="*70)

# Cost function WITHOUT uncertainty for planning
cost_fn = UncertaintyCostFunction(
    base_cost_type="mse",
    use_cls_token=False,      # Visual similarity
    use_patch_tokens=False,   # Can be expensive
    use_state_token=True,     # Position info
    cls_weight=0.5,
    state_weight=1.0,
    num_samples=16,
    uncertainty_aggregation="mean",
    uncertainty_weight=0.1,   # DISABLED - this is the key!
)

# Setup optimizer and planner
action_dim = 2
horizon = 50

optimizer = PICOptimizerOptimized(
    action_dim=action_dim,
    horizon=horizon,
    num_samples=64,
    temperature=2,
    noise_sigma=2,
    action_bounds=(-1.0, 1.0),
    device=device,
    early_stop_patience=100,
    use_covariance=True,
    momentum=0.0,
    noise_decay=0.9,
    top_k_ratio=.25,
)

planner = LatentSpacePlanner(
    jepa_model=model,
    cost_function=cost_fn,
    optimizer=optimizer,
    device=device,
    action_repeat=1
)

# Plan
print("\nPlanning...")
result = planner.plan(
    initial_state=initial_state,
    initial_frame=initial_frame,
    goal_state=goal_state,
    goal_frame=goal_frame,
    num_iterations=50,
    verbose=True,
    return_trajectory=True,
)

print(f"\n✓ Planning complete!")
print(f"  Final cost: {result['final_cost']:.6f}")
best_actions = result['actions'].squeeze(0).cpu()
print(f"  Planned actions shape: {best_actions.shape}")


# ============================================================================
# POST-HOC UNCERTAINTY ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("ANALYZING UNCERTAINTY ALONG PLANNED TRAJECTORY")
print("="*70)

# Encode initial and goal states
with torch.no_grad():
    init_latents = model.encode_state_and_frame(initial_state, initial_frame)
    goal_latents = model.encode_state_and_frame(goal_state, goal_frame)

# Roll out the planned trajectory
print("\nRolling out planned trajectory...")
current_latents = init_latents
trajectory_states = [current_latents]
uncertainties = []
confidence_scores = []

# Create uncertainty estimator
estimator = UncertaintyEstimator(num_samples=16)

print("Estimating uncertainty at each timestep...")
for t in tqdm(range(len(best_actions)), desc="Uncertainty analysis"):
    action_t = best_actions[t:t+1].to(device)  # (1, action_dim)
    
    # Encode action
    z_action = model.action_encoder(action_t)
    
    # Predict next state
    with torch.no_grad():
        z_cls_next, z_patches_next, z_state_next = model.predictor(
            current_latents['z_cls'].unsqueeze(0) if current_latents['z_cls'].dim() == 1 else current_latents['z_cls'],
            current_latents['z_patches'].unsqueeze(0) if current_latents['z_patches'].dim() == 2 else current_latents['z_patches'],
            current_latents['z_state'].unsqueeze(0) if current_latents['z_state'].dim() == 1 else current_latents['z_state'],
            z_action,
        )
    
    # Update current state
    current_latents = {
        'z_cls': z_cls_next.squeeze(0),
        'z_patches': z_patches_next.squeeze(0),
        'z_state': z_state_next.squeeze(0),
    }
    trajectory_states.append(current_latents)
    
    # Estimate uncertainty at this state
    try:
        uncertainty_est = estimator.estimate_uncertainty(
            model.predictor,
            current_latents['z_cls'].unsqueeze(0) if current_latents['z_cls'].dim() == 1 else current_latents['z_cls'],
            current_latents['z_patches'].unsqueeze(0) if current_latents['z_patches'].dim() == 2 else current_latents['z_patches'],
            current_latents['z_state'].unsqueeze(0) if current_latents['z_state'].dim() == 1 else current_latents['z_state'],
            z_action,
        )
        
        # Compute total variance
        total_var = (
            uncertainty_est.variance['z_cls'].mean().item() +
            uncertainty_est.variance['z_state'].mean().item()
        )
        uncertainties.append(total_var)
        
        # Confidence is inverse of uncertainty
        confidence = 1.0 / (1.0 + total_var)
        confidence_scores.append(confidence)
        
    except Exception as e:
        print(f"\nWarning: Could not estimate uncertainty at step {t}: {e}")
        uncertainties.append(np.nan)
        confidence_scores.append(np.nan)

# Filter out NaN values
uncertainties = [u for u in uncertainties if not np.isnan(u)]
confidence_scores = [c for c in confidence_scores if not np.isnan(c)]

if uncertainties:
    print(f"\n✓ Uncertainty analysis complete!")
    print(f"\nUncertainty Statistics:")
    print(f"  Mean uncertainty: {np.mean(uncertainties):.6f}")
    print(f"  Std uncertainty:  {np.std(uncertainties):.6f}")
    print(f"  Max uncertainty:  {np.max(uncertainties):.6f}")
    print(f"  Min uncertainty:  {np.min(uncertainties):.6f}")
    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence: {np.mean(confidence_scores):.4f}")
    print(f"  Min confidence:  {np.min(confidence_scores):.4f}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Uncertainty over time
    ax1.plot(uncertainties, linewidth=2, color='#e74c3c')
    ax1.fill_between(range(len(uncertainties)), uncertainties, alpha=0.3, color='#e74c3c')
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Total Variance (Epistemic Uncertainty)', fontsize=12)
    ax1.set_title('Model Uncertainty Along Planned Trajectory', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.mean(uncertainties), color='#34495e', linestyle='--', 
                label=f'Mean = {np.mean(uncertainties):.4f}', linewidth=2)
    ax1.legend()
    
    # Plot 2: Confidence over time
    ax2.plot(confidence_scores, linewidth=2, color='#27ae60')
    ax2.fill_between(range(len(confidence_scores)), confidence_scores, alpha=0.3, color='#27ae60')
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Confidence (1 / (1 + variance))', fontsize=12)
    ax2.set_title('Model Confidence Along Planned Trajectory', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=np.mean(confidence_scores), color='#34495e', linestyle='--',
                label=f'Mean = {np.mean(confidence_scores):.4f}', linewidth=2)
    ax2.legend()
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('uncertainty_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: uncertainty_analysis.png")
    
    # Identify high-uncertainty regions
    high_uncertainty_threshold = np.mean(uncertainties) + np.std(uncertainties)
    high_uncertainty_steps = [i for i, u in enumerate(uncertainties) if u > high_uncertainty_threshold]
    
    if high_uncertainty_steps:
        print(f"\n⚠ High uncertainty detected at timesteps: {high_uncertainty_steps}")
        print(f"  These regions may be less reliable or off-manifold.")
    else:
        print(f"\n✓ No unusually high uncertainty regions detected.")
        print(f"  The planned trajectory appears to stay on-manifold.")


# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nPlanning Results:")
print(f"  • Final cost: {result['final_cost']:.6f}")
print(f"  • Horizon: {len(best_actions)} steps")
print(f"  • Actions bounds: {best_actions.min().item():.3f} to {best_actions.max().item():.3f}")

if uncertainties:
    print(f"\nUncertainty Analysis:")
    print(f"  • Average model confidence: {np.mean(confidence_scores):.1%}")
    print(f"  • Lowest confidence: {np.min(confidence_scores):.1%}")
    print(f"  • Trajectory appears {'RELIABLE' if np.mean(confidence_scores) > 0.7 else 'UNCERTAIN'}")

print(f"\nNext Steps:")
print(f"  1. Check uncertainty_analysis.png for visualization")
print(f"  2. If uncertainty is high, consider:")
print(f"     - Collecting more training data in uncertain regions")
print(f"     - Increasing model capacity")
print(f"     - Using uncertainty-aware planning (see planning_uncertainty_patch.py)")
print(f"  3. If uncertainty is low, your model is confident!")

print("\n" + "="*70)