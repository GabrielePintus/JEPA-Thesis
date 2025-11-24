"""
Implicit Q-Learning (IQL) Implementation - Following Kostrikov et al. 2021

Key components:
1. V-network: Learns upper expectile of Q-values via expectile regression (Eq. 5)
2. Q-network: Standard Bellman backup using V-network (Eq. 6)  
3. Isometric assumption: Both V and Q use distance-based representations

CRITICAL FIX: Rewards MUST be non-positive to align with V = -distance geometry.
- Goal reward: 0.0 (maximum possible value)
- Step penalty: -1.0 (cost for each step)
This ensures TD targets are always non-positive, matching V's output range.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class IsometricValueNetwork(nn.Module):
    """
    Isometric value network: V(z, g) = -||h(z) - h(g)||₂
    
    Maps states to an embedding space where value = negative distance to goal.
    Since distance >= 0, value is always <= 0, requiring non-positive rewards.
    """
    
    def __init__(
        self,
        state_channels: int = 18,
        emb_dim: int = 64,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        
        # Encoder: preserves spatial structure
        self.encoder = nn.Sequential(
            nn.Conv2d(state_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, emb_dim, kernel_size=3, stride=1, padding=0),  # 26x26 -> 24x24
        )
    
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode state to isometric embedding.
        
        Args:
            z: (B, 18, 26, 26)
        Returns:
            h: (B, emb_dim, 24, 24)
        """
        h = self.encoder(z)
        # L2 normalize per channel for stable distance computation
        h = F.normalize(h, p=2, dim=1)
        return h
    
    def compute_value(self, z_state: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """
        V(z, g) = -||h(z) - h(g)||₂
        
        Args:
            z_state: (B, 18, 26, 26)
            z_goal: (B, 18, 26, 26)
        Returns:
            value: (B,) - always non-positive
        """
        h_state = self.encode(z_state)
        h_goal = self.encode(z_goal)
        
        diff = h_state - h_goal
        distance = torch.sqrt((diff ** 2).sum(dim=[1, 2, 3]) + 1e-8)
        
        return -distance


class IsometricQNetwork(nn.Module):
    """
    Q-function using isometric representation.
    
    Q(z, a, g) can be implicitly computed as V(predictor(z, a), g),
    but for true IQL we need an explicit Q-network that we can query
    and use for the two-loss training procedure.
    """
    
    def __init__(
        self,
        state_channels: int = 18,
        action_dim: int = 2,
        emb_dim: int = 64,
    ):
        super().__init__()
        
        # Option 1: Use the same isometric encoder as V
        # Q will be distance-based but conditioned on actions
        self.state_encoder = nn.Sequential(
            nn.Conv2d(state_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        
        # Action encoder: broadcast action to spatial dimensions
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.GELU(),
        )
        
        # Combined encoder
        self.combined_encoder = nn.Sequential(
            nn.Conv2d(32 + 32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, emb_dim, kernel_size=3, padding=0),  # -> 24x24
        )
    
    def encode(self, z_state: torch.Tensor, action: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """
        Encode (state, action, goal) to compute Q-value.
        
        Args:
            z_state: (B, 18, 26, 26)
            action: (B, 2)
            z_goal: (B, 18, 26, 26)
        Returns:
            q_value: (B,)
        """
        B = z_state.shape[0]
        H, W = z_state.shape[2], z_state.shape[3]
        
        # Encode state
        h_state = self.state_encoder(z_state)  # (B, 32, 26, 26)
        
        # Encode action and tile to spatial dimensions
        h_action = self.action_encoder(action)  # (B, 32)
        h_action = h_action.view(B, -1, 1, 1).expand(-1, -1, H, W)  # (B, 32, 26, 26)
        
        # Concatenate
        h_combined = torch.cat([h_state, h_action], dim=1)  # (B, 64, 26, 26)
        
        # Final encoding
        h_state_action = self.combined_encoder(h_combined)  # (B, emb_dim, 24, 24)
        h_state_action = F.normalize(h_state_action, p=2, dim=1)
        
        # Encode goal
        # We need a separate encoder for goal to match dimensions
        h_goal = self.combined_encoder(torch.cat([
            self.state_encoder(z_goal),
            torch.zeros_like(h_action)  # Zero action for goal
        ], dim=1))
        h_goal = F.normalize(h_goal, p=2, dim=1)
        
        # Compute distance -> Q value
        diff = h_state_action - h_goal
        distance = torch.sqrt((diff ** 2).sum(dim=[1, 2, 3]) + 1e-8)
        
        return -distance


class ImplicitQLearning(nn.Module):
    """
    Implicit Q-Learning following Kostrikov et al. 2021.
    
    Two-network architecture:
    1. V-network: V(s) = E^τ_{a~π_β}[Q(s,a)] (expectile regression, Eq. 5)
    2. Q-network: Q(s,a) = r + γV(s') (Bellman backup, Eq. 6)
    
    Key insight: V learns the expectile of Q-values, approximating max when τ→1.
    """
    
    def __init__(
        self,
        state_channels: int = 18,
        action_dim: int = 2,
        emb_dim: int = 64,
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
        
        # Value network (learns expectile of Q-values)
        self.value_net = IsometricValueNetwork(state_channels, emb_dim)
        
        # Q-network (learns Bellman backup)
        self.q_net = IsometricQNetwork(state_channels, action_dim, emb_dim)
        
        # Target networks
        self.value_target = IsometricValueNetwork(state_channels, emb_dim)
        self.value_target.load_state_dict(self.value_net.state_dict())
        
        self.q_target = IsometricQNetwork(state_channels, action_dim, emb_dim)
        self.q_target.load_state_dict(self.q_net.state_dict())
        
        # Freeze target networks
        for param in self.value_target.parameters():
            param.requires_grad = False
        for param in self.q_target.parameters():
            param.requires_grad = False
    
    def compute_reward(
        self, 
        z_state: torch.Tensor, 
        z_goal: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        CORRECTED REWARD: Must be non-positive!
        
        - Goal (within threshold): 0.0 (maximum possible value)
        - Not at goal: -1.0 (step cost)
        
        This aligns with V = -distance which is always <= 0.
        
        Args:
            z_state: (B, T, C, H, W) or (B, C, H, W)
            z_goal: (B, C, H, W)
            threshold: distance threshold for "reaching" goal
        
        Returns:
            reward: (B, T) or (B,) - always non-positive
        """
        if z_state.dim() == 5:  # (B, T, C, H, W)
            B, T, C, H, W = z_state.shape
            z_state_flat = z_state.flatten(0, 1)
            z_goal_expanded = z_goal.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
            
            diff = z_state_flat - z_goal_expanded
            distance = torch.sqrt((diff ** 2).sum(dim=[1, 2, 3]) + 1e-8)
            
            # Goal: 0.0, Step: -1.0
            is_goal = (distance < threshold).float()
            reward = is_goal * 0.0 + (1 - is_goal) * (-1.0)
            
            return reward.view(B, T)
        else:  # (B, C, H, W)
            diff = z_state - z_goal
            distance = torch.sqrt((diff ** 2).sum(dim=[1, 2, 3]) + 1e-8)
            
            is_goal = (distance < threshold).float()
            reward = is_goal * 0.0 + (1 - is_goal) * (-1.0)
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
    
    def expectile_loss(self, diff: torch.Tensor, expectile: float) -> torch.Tensor:
        """
        Asymmetric squared loss for expectile regression (IQL Eq. 5).
        
        L^τ_2(u) = |τ - 1(u < 0)| * u^2
        
        Args:
            diff: target - prediction
            expectile: τ ∈ (0,1), larger values give more weight to positive diffs
        
        Returns:
            Expectile loss
        """
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return (weight * (diff ** 2)).mean()
    
    def compute_losses(
        self,
        z_states: torch.Tensor,
        actions: torch.Tensor,
        z_next_states: torch.Tensor,
        z_goals: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute IQL losses following the paper's two-step procedure.
        
        Args:
            z_states: (B, T, 18, 26, 26) - current states
            actions: (B, T, 2) - actions taken
            z_next_states: (B, T, 18, 26, 26) - next states (can be predicted or true)
            z_goals: (B, 18, 26, 26) - goals
            rewards: (B, T) - optional precomputed rewards
            
        Returns:
            dict with separate 'value_loss', 'q_loss', and metrics
        """
        B, T, C, H, W = z_states.shape
        
        # Hindsight relabeling
        z_goals = self.hindsight_relabel(z_states, z_goals)
        
        # Compute rewards if not provided
        if rewards is None:
            rewards = self.compute_reward(z_states, z_goals)
        
        # Flatten batch and time for network inputs
        z_curr_flat = z_states.flatten(0, 1)  # (B*T, 18, 26, 26)
        z_next_flat = z_next_states.flatten(0, 1)  # (B*T, 18, 26, 26)
        actions_flat = actions.flatten(0, 1)  # (B*T, 2)
        z_goals_expanded = z_goals.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
        rewards_flat = rewards.flatten()  # (B*T,)
        
        # ========================================
        # IQL Loss 1: Value Expectile Regression (Eq. 5)
        # ========================================
        # V(s) learns to predict the expectile of Q(s, a) over actions in dataset
        # L_V = E[(L^τ_2(Q_target(s,a) - V(s)))]
        
        with torch.no_grad():
            # Get Q-values from target Q-network
            q_target_values = self.q_target.encode(z_curr_flat, actions_flat, z_goals_expanded)
        
        # Get V predictions
        v_pred = self.value_net.compute_value(z_curr_flat, z_goals_expanded)
        
        # Expectile loss: target - prediction
        diff_v = q_target_values - v_pred
        value_loss = self.expectile_loss(diff_v, self.expectile)
        
        # ========================================
        # IQL Loss 2: Q-function Bellman Backup (Eq. 6)
        # ========================================
        # Q(s,a) = r + γ V_target(s')
        
        # Compute TD target using target V-network
        with torch.no_grad():
            v_next = self.value_target.compute_value(z_next_flat, z_goals_expanded)
            q_target = rewards_flat + self.gamma * v_next
        
        # Get Q predictions
        q_pred = self.q_net.encode(z_curr_flat, actions_flat, z_goals_expanded)
        
        # MSE loss for Q-function
        q_loss = F.mse_loss(q_pred, q_target)
        
        # ========================================
        # Metrics
        # ========================================
        with torch.no_grad():
            v_error = (v_pred - q_target_values).abs().mean()
            q_error = (q_pred - q_target).abs().mean()
            v_mean = v_pred.mean()
            v_std = v_pred.std()
            q_mean = q_pred.mean()
            q_std = q_pred.std()
        
        return {
            'value_loss': value_loss,
            'q_loss': q_loss,
            'total_loss': value_loss + q_loss,
            'value_error': v_error,
            'q_error': q_error,
            'value_mean': v_mean,
            'value_std': v_std,
            'q_mean': q_mean,
            'q_std': q_std,
            'reward_mean': rewards_flat.mean(),
        }
    
    @torch.no_grad()
    def update_target(self):
        """Soft update of target networks."""
        for param, target_param in zip(
            self.value_net.parameters(),
            self.value_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        for param, target_param in zip(
            self.q_net.parameters(),
            self.q_target.parameters()
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
        Get value and distance for planning.
        
        Returns:
            dict with 'value' and 'distance'
        """
        value = self.value_net.compute_value(z_state, z_goal)
        distance = -value  # Since V = -distance
        
        return {
            'value': value,
            'distance': distance,
        }


# ============================================================================
# Simplified Version: Single V-network (Model-based IQL Variant)
# ============================================================================
# If you want to keep the simpler structure but fix the critical bugs,
# use this version instead:


class IsometricVLearning(nn.Module):
    """
    Simplified model-based IQL variant using only V-network.
    
    This is what your original code was attempting, but with fixes:
    1. Non-positive rewards
    2. Expectile regression on correct target
    3. Use TRUE next states (from encoder) not just predicted ones
    """
    
    def __init__(
        self,
        state_channels: int = 18,
        emb_dim: int = 64,
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
        """Non-positive rewards aligned with V = -distance."""
        if z_state.dim() == 5:
            B, T, C, H, W = z_state.shape
            z_state_flat = z_state.flatten(0, 1)
            z_goal_expanded = z_goal.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
            
            diff = z_state_flat - z_goal_expanded
            distance = torch.sqrt((diff ** 2).sum(dim=[1, 2, 3]) + 1e-8)
            
            is_goal = (distance < threshold).float()
            reward = is_goal * 0.0 + (1 - is_goal) * (-1.0)
            
            return reward.view(B, T)
        else:
            diff = z_state - z_goal
            distance = torch.sqrt((diff ** 2).sum(dim=[1, 2, 3]) + 1e-8)
            
            is_goal = (distance < threshold).float()
            reward = is_goal * 0.0 + (1 - is_goal) * (-1.0)
            return reward
    
    def hindsight_relabel(self, z_states, z_goals):
        """HER goal relabeling."""
        B, T, C, H, W = z_states.shape
        relabel_mask = torch.rand(B, device=z_states.device) < self.hindsight_ratio
        n_relabel = relabel_mask.sum().item()
        
        if n_relabel == 0:
            return z_goals
        
        relabel_indices = torch.where(relabel_mask)[0]
        future_t = torch.randint(1, T, (n_relabel,), device=z_states.device)
        new_goals = z_states[relabel_indices, future_t]
        
        z_goals_relabeled = z_goals.clone()
        z_goals_relabeled[relabel_indices] = new_goals
        return z_goals_relabeled
    
    def expectile_loss(self, diff, expectile):
        """Asymmetric squared loss."""
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return (weight * (diff ** 2)).mean()
    
    def compute_value_loss(
        self,
        z_states: torch.Tensor,
        z_next_states_true: torch.Tensor,  # IMPORTANT: Use TRUE encoded next states
        z_goals: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Simplified value learning using true next states.
        
        Key fix: Use z_next_states_true from encoder, not predicted ones!
        """
        B, T, C, H, W = z_states.shape
        
        z_goals = self.hindsight_relabel(z_states, z_goals)
        
        if rewards is None:
            rewards = self.compute_reward(z_states, z_goals)
        
        z_curr_flat = z_states.flatten(0, 1)
        z_next_flat = z_next_states_true.flatten(0, 1)  # TRUE next states
        z_goals_expanded = z_goals.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
        rewards_flat = rewards.flatten()
        
        # Compute V(s')
        v_next = self.value_net.compute_value(z_next_flat, z_goals_expanded)
        
        # TD target: r + γ V_target(s')
        with torch.no_grad():
            v_target_next = self.value_target.compute_value(z_next_flat, z_goals_expanded)
            td_target = rewards_flat + self.gamma * v_target_next
        
        # Expectile regression
        diff = td_target - v_next
        value_loss = self.expectile_loss(diff, self.expectile)
        
        # Metrics
        with torch.no_grad():
            v_error = (v_next - td_target).abs().mean()
        
        return {
            'value_loss': value_loss,
            'value_error': v_error,
            'value_mean': v_next.mean(),
            'value_std': v_next.std(),
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
    def get_value(self, z_state, z_goal):
        """Get value for planning."""
        value = self.value_net.compute_value(z_state, z_goal)
        return {
            'value': value,
            'distance': -value,
        }
    
    