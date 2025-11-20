import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import LambdaLR

from src.components.decoder import IDMDecoder
from src.losses import TemporalVCRegLossOptimized as TemporalVCRegLoss
from src.components.encoder import MeNet6, Expander2D, CNNEncoder, MLPHead
from src.losses import VCRegLoss


class Isometry(L.LightningModule):

    def __init__(
        self,
        initial_lr: float = 3e-4,
        final_lr: float = 1e-5,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        emb_dim: int = 128,
        horizon: int = 10,
        compile: bool = False,
        num_samples_std: int = 8,
        regression_weight: float = 1.0,
        uncertainty_weight: float = 1e-2,
        triplet_weight: float = 0.5,  # NEW: triplet loss weight
        triplet_margin: float = 0.5,  # NEW: margin for triplet loss
    ):
        super().__init__()

        self.save_hyperparameters()

        # --- Encoder: small CNN ---
        self.visual_encoder = CNNEncoder(input_channels=3)

        # --- Head: global pool + MLP with LayerNorm ---
        self.head = MLPHead(emb_dim=emb_dim)
        
        # --- Regression head: predicts mean distance directly ---
        self.regression_head = nn.Sequential(
            nn.LayerNorm(emb_dim*2),
            nn.Linear(emb_dim*2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )
        
        # --- Uncertainty head: predicts log std ---
        self.logstd_head = nn.Sequential(
            nn.LayerNorm(emb_dim*2),
            nn.Linear(emb_dim*2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )

        if compile:
            self.visual_encoder = torch.compile(self.visual_encoder)
            self.head = torch.compile(self.head)
            self.regression_head = torch.compile(self.regression_head)
            self.logstd_head = torch.compile(self.logstd_head)

    def compute_triplet_loss(self, h, target_distances_matrix, num_random_samples=500):
        """
        Hybrid triplet loss: local + random sampling
        
        Total complexity: O(T) for local + O(num_samples) for random = O(T + num_samples)
        """
        B, T, D = h.shape
        
        if T < 3:
            return torch.tensor(0.0, device=h.device)
        
        # Compute all pairwise distances once - O(T²) but only done once
        h_expanded_i = h.unsqueeze(2)  # (B, T, 1, D)
        h_expanded_j = h.unsqueeze(1)  # (B, 1, T, D)
        emb_dist_matrix = torch.norm(h_expanded_i - h_expanded_j, dim=-1)  # (B, T, T)
        
        losses = []
        
        # ========== Part 1: Consecutive triplets - O(T) ==========
        # Most important: enforce metric properties along trajectory
        if T >= 3:
            i_consecutive = torch.arange(T - 2, device=h.device)
            j_consecutive = i_consecutive + 1
            k_consecutive = i_consecutive + 2
            
            emb_d_ij = emb_dist_matrix[:, i_consecutive, j_consecutive]  # (B, T-2)
            emb_d_jk = emb_dist_matrix[:, j_consecutive, k_consecutive]  # (B, T-2)
            emb_d_ik = emb_dist_matrix[:, i_consecutive, k_consecutive]  # (B, T-2)
            
            # For consecutive points: d(i,k) = d(i,j) + d(j,k)
            consecutive_loss = torch.abs(emb_d_ik - (emb_d_ij + emb_d_jk)).mean()
            losses.append(consecutive_loss)
        
        # ========== Part 2: Random sampling - O(num_samples) ==========
        # Capture longer-range dependencies and wall-crossing scenarios
        if num_random_samples > 0:
            # Generate valid random triplets
            attempts = 0
            valid_triplets = []
            
            while len(valid_triplets) < num_random_samples and attempts < num_random_samples * 3:
                i = torch.randint(0, T, (1,), device=h.device).item()
                j = torch.randint(0, T, (1,), device=h.device).item()
                k = torch.randint(0, T, (1,), device=h.device).item()
                
                if i != j and i != k and j != k:
                    valid_triplets.append((i, j, k))
                attempts += 1
            
            if valid_triplets:
                i_rand = torch.tensor([t[0] for t in valid_triplets], device=h.device)
                j_rand = torch.tensor([t[1] for t in valid_triplets], device=h.device)
                k_rand = torch.tensor([t[2] for t in valid_triplets], device=h.device)
                
                emb_d_ij = emb_dist_matrix[:, i_rand, j_rand]  # (B, num_valid)
                emb_d_ik = emb_dist_matrix[:, i_rand, k_rand]
                emb_d_kj = emb_dist_matrix[:, k_rand, j_rand]
                
                # Triangle inequality
                violations = F.relu(emb_d_ij - emb_d_ik - emb_d_kj)
                random_loss = violations.mean()
                losses.append(random_loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=h.device)

    def shared_step(self, batch):
        # Assuming batch = (states, frames, actions)
        states, frames, _ = batch
        B, T, C, H, W = frames.shape
        
        # Encode frames once
        frames_flat = frames.flatten(0, 1)  # (B*T, C, H, W)
        z_frames = self.visual_encoder(frames_flat)  # (B*T, D_enc)
        
        # Get embeddings for all frames
        h = self.head(z_frames)  # (B*T, D_emb)
        h = h.view(B, T, -1)  # (B, T, D)
        
        # Compute state pairs
        state_pairs = self.unique_state_pairs(h)  # (B, N_pairs, 2, D)
        h_i = state_pairs[:, :, 0, :]
        h_j = state_pairs[:, :, 1, :]
        
        # Concatenate pair embeddings for heads
        h_pairs = torch.cat([h_i, h_j], dim=-1)  # (B, N_pairs, 2*D)
        B_size, N_pairs, pair_dim = h_pairs.shape
        h_pairs_flat = h_pairs.view(B_size * N_pairs, pair_dim)
        
        # ==================== COMPUTE TARGET DISTANCES ====================
        # Option 1: Use temporal differences (assumes constant velocity)
        time_indices = torch.arange(T, device=self.device, dtype=h.dtype)
        time_pairs = self.unique_state_pairs(time_indices.unsqueeze(0).unsqueeze(-1))
        t_i = time_pairs[:, :, 0, 0]
        t_j = time_pairs[:, :, 1, 0]
        target_distances = (t_i - t_j).abs()
        
        # Option 2: Use actual cumulative path lengths (BETTER!)
        # Compute cumulative distances from state positions
        positions = states[:, :, :2]  # (B, T, 2) - x, y positions
        deltas = positions[:, 1:] - positions[:, :-1]  # (B, T-1, 2)
        segment_lengths = torch.norm(deltas, dim=-1)  # (B, T-1)
        cumulative_lengths = torch.cat([
            torch.zeros(B, 1, device=self.device),
            torch.cumsum(segment_lengths, dim=1)
        ], dim=1)  # (B, T)
        
        # Use cumulative lengths as target
        length_pairs = self.unique_state_pairs(cumulative_lengths.unsqueeze(-1))
        l_i = length_pairs[:, :, 0, 0]
        l_j = length_pairs[:, :, 1, 0]
        target_distances = (l_i - l_j).abs()  # (B, N_pairs)
        
        # Build full distance matrix for triplet loss
        # (B, T, T) matrix of ground truth distances
        target_dist_matrix = torch.zeros(B, T, T, device=self.device)
        idx_i, idx_j = torch.triu_indices(T, T, offset=1)
        target_dist_matrix[:, idx_i, idx_j] = target_distances
        target_dist_matrix[:, idx_j, idx_i] = target_distances  # Symmetric
        
        # ==================== REGRESSION TASK ====================
        # Predict mean distance directly from pair embeddings
        distance_pred = self.regression_head(h_pairs_flat).squeeze(-1)
        distance_pred = distance_pred.view(B_size, N_pairs)  # (B, N_pairs)
        
        # Direct MSE regression loss
        regression_loss = F.mse_loss(distance_pred, target_distances)
        
        # ==================== TRIPLET LOSS ====================
        # Enforce triangle inequality and topological constraints
        triplet_loss = self.compute_triplet_loss(h, target_dist_matrix)
        
        # ==================== UNCERTAINTY TASK ====================
        num_samples = self.hparams.num_samples_std
        distance_samples = []
        
        self.head.train()  # Ensure dropout is active
        
        for _ in range(num_samples):
            h_sample = self.head(z_frames)
            h_sample = h_sample.view(B, T, -1)
            
            state_pairs_sample = self.unique_state_pairs(h_sample)
            h_i_sample = state_pairs_sample[:, :, 0, :]
            h_j_sample = state_pairs_sample[:, :, 1, :]
            distances = torch.norm(h_i_sample - h_j_sample, dim=-1)
            
            distance_samples.append(distances)
        
        if not self.training:
            self.head.eval()
        
        distance_samples = torch.stack(distance_samples, dim=0)
        distances_std = distance_samples.std(dim=0)
        
        logstd_pred = self.logstd_head(h_pairs_flat).squeeze(-1).view(B_size, N_pairs)
        logstd_target = torch.log(distances_std + 1e-8)
        uncertainty_loss = F.mse_loss(logstd_pred, logstd_target.detach())
        
        # ==================== COMBINED LOSS ====================
        total_loss = (
            self.hparams.regression_weight * regression_loss + 
            self.hparams.uncertainty_weight * uncertainty_loss +
            self.hparams.triplet_weight * triplet_loss
        )
        
        # ==================== METRICS ====================
        distances_mean_sampled = distance_samples.mean(dim=0)
        
        return {
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            "loss": total_loss,
            "regression_loss": regression_loss,
            "uncertainty_loss": uncertainty_loss,
            "triplet_loss": triplet_loss,
            "mean_pred_distance": distance_pred.mean(),
            "mean_target_distance": target_distances.mean(),
            "distance_ratio": distance_pred.mean() / (target_distances.mean() + 1e-8),
            "mean_distance_std": distances_std.mean(),
            "max_distance_std": distances_std.max(),
        }

    def training_step(self, batch, batch_idx):
        losses = self.shared_step(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            prog_bar=True,
            sync_dist=True,
        )
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        losses = self.shared_step(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items()},
            prog_bar=True,
            sync_dist=True,
        )
        return losses["loss"]

    @staticmethod
    def unique_state_pairs(states):
        B, T, D = states.shape
        idx_i, idx_j = torch.triu_indices(T, T, offset=1)
        s_i = states[:, idx_i]
        s_j = states[:, idx_j]
        return torch.stack([s_i, s_j], dim=2)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.initial_lr,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return (step + 1) / float(self.hparams.warmup_steps)

            total_steps = getattr(self.trainer, "max_steps", None)
            
            if total_steps is None or total_steps == -1:
                max_epochs = getattr(self.trainer, "max_epochs", None)
                if max_epochs is not None and hasattr(self.trainer, "estimated_stepping_batches"):
                    total_steps = self.trainer.estimated_stepping_batches
                elif max_epochs is not None:
                    if self.trainer.num_training_batches != float('inf'):
                        total_steps = max_epochs * self.trainer.num_training_batches
                    else:
                        total_steps = None
            
            if total_steps is None or total_steps <= self.hparams.warmup_steps:
                print(f"Warning: Could not determine total_steps (got {total_steps}), keeping LR constant after warmup")
                return 1.0

            progress = (step - self.hparams.warmup_steps) / max(
                1, (total_steps - self.hparams.warmup_steps)
            )
            progress = min(1.0, progress)
            min_ratio = self.hparams.final_lr / self.hparams.initial_lr
            return max(min_ratio, 0.5 * (1 + math.cos(math.pi * progress)))

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda=lr_lambda),
            "interval": "step",
        }

        return [optimizer], [scheduler]