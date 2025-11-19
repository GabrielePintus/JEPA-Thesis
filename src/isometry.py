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
        initial_lr: float = 3e-4,      # FIXED: Reduced from 2e-3
        final_lr: float = 1e-5,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        emb_dim: int = 128,
        horizon: int = 10,
        compile: bool = False,
        num_samples_std: int = 8,
    ):
        super().__init__()

        self.save_hyperparameters()

        # --- Encoder: small CNN ---
        self.visual_encoder = CNNEncoder(input_channels=3)

        # --- Head: global pool + MLP with LayerNorm ---
        # FIXED: Added LayerNorm at output for stability
        self.head = MLPHead(emb_dim=emb_dim)#, spatial_features=32*4*4)
        self.logstd_head = nn.Sequential(
            nn.LayerNorm(emb_dim*2),
            nn.Linear(emb_dim*2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )

        if compile:
            self.visual_encoder = torch.compile(self.visual_encoder)
            self.head = torch.compile(self.head)
            self.logstd_head = torch.compile(self.logstd_head)


    def shared_step(self, batch):
        # Assuming batch = (states, frames, actions)
        _, frames, _ = batch
        B, T, C, H, W = frames.shape
        
        # Encode frames once
        frames_flat = frames.flatten(0, 1)  # (B*T, C, H, W)
        z_frames = self.visual_encoder(frames_flat)  # (B*T, D_enc)
        
        # Sample multiple embeddings using dropout
        num_samples = self.hparams.num_samples_std
        distance_samples = []
        
        for _ in range(num_samples):
            h = self.head(z_frames)  # (B*T, D_emb) - dropout active
            h = h.view(B, T, -1)  # (B, T, D)
            
            # Compute pairwise distances for this sample
            state_pairs = self.unique_state_pairs(h)
            h_i = state_pairs[:, :, 0, :]
            h_j = state_pairs[:, :, 1, :]
            distances = torch.sqrt((h_i - h_j).pow(2).sum(dim=-1) + 1e-8)  # (B, N_pairs)
            
            # Normalize for cosine similarity
            distances_norm = F.normalize(distances, dim=-1)  # (B, N_pairs)
            distance_samples.append(distances_norm)
        
        # Stack samples: (num_samples, B, N_pairs)
        distance_samples = torch.stack(distance_samples, dim=0)
        
        # Compute mean and std of DISTANCE predictions
        distances_mean = distance_samples.mean(dim=0)  # (B, N_pairs)
        distances_std = distance_samples.std(dim=0)    # (B, N_pairs)
        
        # Get one more forward pass for logstd prediction
        h_for_logstd = self.head(z_frames).view(B, T, -1)  # (B, T, D)
        
        # Compute state pairs for logstd prediction
        state_pairs_logstd = self.unique_state_pairs(h_for_logstd)  # (B, N_pairs, 2, D)
        # Concatenate the pair embeddings
        h_pairs = state_pairs_logstd.view(B, -1, 2 * h_for_logstd.shape[-1])  # (B, N_pairs, 2*D)
        
        # Predict log std for each pair
        B_size, N_pairs, pair_dim = h_pairs.shape
        h_pairs_flat = h_pairs.view(B_size * N_pairs, pair_dim)
        print(h_pairs_flat.shape)
        logstd_pred = self.logstd_head(h_pairs_flat).squeeze(-1)  # (B*N_pairs,)
        logstd_pred = logstd_pred.view(B_size, N_pairs)  # (B, N_pairs)
        
        # Compute target logstd from sampled distances
        logstd_target = torch.log(distances_std + 1e-8)  # (B, N_pairs)
        
        # LogStd regression loss
        print(logstd_pred.shape, logstd_target.shape)
        logstd_loss = F.mse_loss(logstd_pred, logstd_target)
        
        # Compute target distances (temporal differences)
        time_indices = torch.arange(T, device=self.device, dtype=distances_mean.dtype)
        time_indices = time_indices.unsqueeze(0)
        time_pairs = self.unique_state_pairs(time_indices.unsqueeze(-1))
        t_i = time_pairs[:, :, 0, 0]
        t_j = time_pairs[:, :, 1, 0]
        target_distances = (t_i - t_j).abs().to(dtype=distances_mean.dtype)
        target_distances_norm = F.normalize(target_distances, dim=-1)
        
        # Isometry loss using mean distances
        cosine_sim = (distances_mean * target_distances_norm).sum(dim=-1)
        isometry_loss = (1 - cosine_sim).mean()
        
        # Combined loss
        total_loss = isometry_loss + logstd_loss * 1e-2
        
        return {
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            "loss": total_loss,
            "isometry_loss": isometry_loss,
            "logstd_loss": logstd_loss,
            "mean_distance_std": distances_std.mean(),  # Monitor average prediction uncertainty
            "max_distance_std": distances_std.max(),    # Monitor maximum uncertainty
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
        """
        states: (B, T, D)
        returns:
            pairs: (B, N_pairs, 2, D)
                pairs[b, k] = (states[b, i], states[b, j]) with i < j
        """
        B, T, D = states.shape
        idx_i, idx_j = torch.triu_indices(T, T, offset=1)

        s_i = states[:, idx_i]   # (B, N_pairs, D)
        s_j = states[:, idx_j]   # (B, N_pairs, D)

        return torch.stack([s_i, s_j], dim=2)  # (B, N_pairs, 2, D)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.initial_lr,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(step):
            # Warmup
            if step < self.hparams.warmup_steps:
                return (step + 1) / float(self.hparams.warmup_steps)

            # After warmup: cosine decay
            # Try to get total_steps from trainer
            total_steps = getattr(self.trainer, "max_steps", None)
            
            # If max_steps not set, calculate from max_epochs
            if total_steps is None or total_steps == -1:
                max_epochs = getattr(self.trainer, "max_epochs", None)
                if max_epochs is not None and hasattr(self.trainer, "estimated_stepping_batches"):
                    # Lightning 2.0+ has this property
                    total_steps = self.trainer.estimated_stepping_batches
                elif max_epochs is not None:
                    # Estimate based on current epoch
                    if self.trainer.num_training_batches != float('inf'):
                        total_steps = max_epochs * self.trainer.num_training_batches
                    else:
                        total_steps = None
            
            # If we still don't have total_steps, keep constant LR after warmup
            if total_steps is None or total_steps <= self.hparams.warmup_steps:
                print(f"Warning: Could not determine total_steps (got {total_steps}), keeping LR constant after warmup")
                return 1.0

            progress = (step - self.hparams.warmup_steps) / max(
                1, (total_steps - self.hparams.warmup_steps)
            )
            progress = min(1.0, progress)  # Cap at 1.0
            min_ratio = self.hparams.final_lr / self.hparams.initial_lr
            return max(min_ratio, 0.5 * (1 + math.cos(math.pi * progress)))

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda=lr_lambda),
            "interval": "step",
        }

        return [optimizer], [scheduler]
