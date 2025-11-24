"""
Pure Isometric Value Function - Simple & Elegant

V(z, g) = -d(z, g) where d is distance in isometric embedding space.

Key insight: Your JEPA already learns isometric representations!
We just need to:
1. Project z to isometric embeddings
2. Compute distance to goal
3. Value = negative distance

That's it!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class IsometricValueNetwork(nn.Module):
    """
    Fully spatial isometric value network.
    
    Architecture:
        z (18, 26, 26) → CNN → h (emb_dim, H, W)
        V(z, g) = -||h_z - h_g||₂  (distance over ALL spatial features)
    
    Computes distance directly in spatial feature space - no pooling!
    """
    
    def __init__(
        self,
        state_channels: int = 18,
        emb_dim: int = 128
    ):
        super().__init__()
        self.emb_dim = emb_dim
        
        # Spatial encoder - preserves ALL spatial information
        self.encoder = nn.Sequential(
            nn.Conv2d(state_channels, 32, kernel_size=3, stride=1, padding=1),  # 18x26x26 → 32x26x26
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),               # 32x26x26 → 64x24x24
        )
    
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode state to spatial features (NO pooling!).
        
        Args:
            z: (B, 18, 26, 26)
        Returns:
            h: (B, emb_dim, 7, 7) - full spatial features
        """
        # Spatial features
        h_spatial = self.encoder(z)  # (B, emb_dim, 7, 7)
        
        # Normalize per-channel (preserve spatial structure)
        h_spatial = F.normalize(h_spatial, p=2, dim=1)  # Normalize channels
        
        return h_spatial  # (B, emb_dim, 7, 7)
    
    def compute_value(self, z_state: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """
        V(z, g) = -||h_z - h_g||  over FULL spatial feature maps
        
        Args:
            z_state: (B, 18, 26, 26)
            z_goal: (B, 18, 26, 26)
        Returns:
            value: (B,)
        """
        h_state = self.encode(z_state)  # (B, emb_dim, 7, 7)
        h_goal = self.encode(z_goal)    # (B, emb_dim, 7, 7)
        
        # Distance over ALL spatial features
        diff = h_state - h_goal  # (B, emb_dim, 7, 7)
        distance = torch.sqrt((diff ** 2).sum(dim=[1, 2, 3]))  # (B,)
        
        return -distance


class IsometricQLearning(nn.Module):
    """
    Implicit Q-learning with isometric value function.
    
    Key idea: Use JEPA's predictor for Q(z,a,g) = V(predictor(z,a), g)
    """
    
    def __init__(
        self,
        state_channels: int = 18,
        emb_dim: int = 128,
        gamma: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.7,
        hindsight_ratio: float = 0.8,
    ):
        super().__init__()
        
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.hindsight_ratio = hindsight_ratio
        
        # Value network
        self.value_net = IsometricValueNetwork(state_channels, emb_dim)
        
        # Target network
        self.value_target = IsometricValueNetwork(state_channels, emb_dim)
        self.value_target.load_state_dict(self.value_net.state_dict())
        for param in self.value_target.parameters():
            param.requires_grad = False
    
    def compute_reward(
        self, 
        z_state: torch.Tensor, 
        z_goal: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Sparse reward: +1 if close to goal, -0.1 otherwise."""
        if z_state.dim() == 5:  # (B, T, C, H, W)
            B, T, C, H, W = z_state.shape
            z_state_flat = z_state.flatten(0, 1)
            z_goal_expanded = z_goal.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
            
            diff = z_state_flat - z_goal_expanded
            distance = torch.sqrt((diff ** 2).sum(dim=[1, 2, 3]))
            reward = (distance < threshold).float() * 1.0 - (distance >= threshold).float() * 0.1
            
            return reward.view(B, T)
        else:  # (B, C, H, W)
            diff = z_state - z_goal
            distance = torch.sqrt((diff ** 2).sum(dim=[1, 2, 3]))
            reward = (distance < threshold).float() * 1.0 - (distance >= threshold).float() * 0.1
            return reward
    
    def hindsight_relabel(
        self,
        z_states: torch.Tensor,
        z_goals: torch.Tensor,
    ) -> torch.Tensor:
        """Relabel goals with achieved states (HER)."""
        B, T, C, H, W = z_states.shape
        
        # Randomly select which trajectories to relabel
        relabel_mask = torch.rand(B, device=z_states.device) < self.hindsight_ratio
        n_relabel = relabel_mask.sum().item()
        
        if n_relabel == 0:
            return z_goals
        
        # Relabel with random future state
        relabel_indices = torch.where(relabel_mask)[0]
        future_t = torch.randint(1, T, (n_relabel,), device=z_states.device)
        new_goals = z_states[relabel_indices, future_t]
        
        z_goals_relabeled = z_goals.clone()
        z_goals_relabeled[relabel_indices] = new_goals
        
        return z_goals_relabeled
    
    def expectile_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute expectile regression loss used in IQL.
        
        Args:
            prediction: predicted values.
            target: target values (treated as constants).
        
        Returns:
            Scalar expectile loss encouraging overestimation when expectile>0.5.
        """
        diff = target - prediction
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return (weight * diff.pow(2)).mean()
    
    def compute_q_loss(
        self,
        z_states: torch.Tensor,
        z_next_states: torch.Tensor,
        z_goals: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Q-learning loss.
        
        Args:
            z_states: (B, T, 18, 26, 26) - current states
            z_next_states: (B, T, 18, 26, 26) - predicted next states from JEPA
            z_goals: (B, 18, 26, 26) - goals
            rewards: (B, T) - optional precomputed rewards
            
        Returns:
            dict with 'loss' and metrics
        """
        B, T, C, H, W = z_states.shape
        
        # Hindsight relabeling
        z_goals = self.hindsight_relabel(z_states, z_goals)
        
        # Compute rewards if not provided
        if rewards is None:
            rewards = self.compute_reward(z_states, z_goals)
        
        # Flatten batch and time
        z_curr = z_states.flatten(0, 1)  # (B*T, 18, 26, 26)
        z_next = z_next_states.flatten(0, 1)  # (B*T, 18, 26, 26)
        z_goals_expanded = z_goals.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
        rewards_flat = rewards.flatten()  # (B*T,)
        
        # Compute current values (for next states)
        v_next = self.value_net.compute_value(z_next, z_goals_expanded)
        
        # Compute target values
        with torch.no_grad():
            v_target_next = self.value_target.compute_value(z_next, z_goals_expanded)
            v_target = rewards_flat + self.gamma * v_target_next
        
        # Value loss (Implicit Q-Learning with expectile regression)
        value_loss = self.expectile_loss(v_next, v_target)
        
        # Metrics
        with torch.no_grad():
            v_error = (v_next - v_target).abs().mean()
            v_mean = v_next.mean()
            v_std = v_next.std()
        
        return {
            'loss': value_loss,
            'value_loss': value_loss,
            'value_error': v_error,
            'value_mean': v_mean,
            'value_std': v_std,
            'reward_mean': rewards_flat.mean(),
        }
    
    @torch.no_grad()
    def update_target(self):
        """Soft update of target network."""
        for param, target_param in zip(
            self.value_net.parameters(),
            self.value_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    @torch.no_grad()
    def get_value(
        self,
        z_state: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get value and distance.
        
        Returns:
            dict with 'value' and 'distance'
        """
        value = self.value_net.compute_value(z_state, z_goal)
        distance = -value  # Since V = -distance
        
        return {
            'value': value,
            'distance': distance,
        }