import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import LambdaLR

from src.components.decoder import IDMDecoder
from src.losses import TemporalVCRegLossOptimized as TemporalVCRegLoss
from src.components.encoder import MeNet6
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
    ):
        super().__init__()

        self.save_hyperparameters()

        # --- Encoder: small CNN ---
        self.visual_encoder = MeNet6(input_channels=3)

        # --- Head: global pool + MLP with LayerNorm ---
        # FIXED: Added LayerNorm at output for stability
        self.head = nn.Sequential(
            nn.Flatten(),              # (B, 32)
            nn.LayerNorm(32*4*4),
            nn.Linear(32*4*4, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        # self.proj = nn.Linear(emb_dim, emb_dim*2) # For VCReg
        self.vcreg = TemporalVCRegLoss()

        # Assuming uniform time steps of 1 between frames
        self.mu = horizon / 2.0
        self.std = horizon / 3.46

        if compile:
            self.visual_encoder = torch.compile(self.visual_encoder)
            self.head = torch.compile(self.head)


    def shared_step(self, batch):
        # Assuming batch = (states, frames, actions)
        _, frames, _ = batch

        
        # Reshape for encoding
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        z_frames = self.visual_encoder(frames)  # (B*T, D_enc)
        h = self.head(z_frames)  # (B*T, D_emb)

        # Reshape back to (B, T, D)
        D = h.shape[-1]
        h = h.view(B, T, D)

        # Apply Temporal VCReg Loss
        vcreg_loss = self.vcreg(h)

        # Compute pairwise distances
        state_pairs = self.unique_state_pairs(h)  # (B, N_pairs, 2, D)
        h_i = state_pairs[:, :, 0, :]  # (B, N_pairs, D)
        h_j = state_pairs[:, :, 1, :]  # (B, N_pairs, D)
        distances = (h_i - h_j).pow(2).sum(dim=-1)  # (B, N_pairs)

        # Compute target distances
        time_indices = torch.arange(T, device=self.device, dtype=distances.dtype)  # (T,)
        # Expand to (1, T)
        time_indices = time_indices.unsqueeze(0)  # (1, T)
        time_pairs = self.unique_state_pairs(time_indices.unsqueeze(-1))  # (1, N_pairs, 2, 1)
        t_i = time_pairs[:, :, 0, 0]  # (1, N_pairs)
        t_j = time_pairs[:, :, 1, 0]  # (1, N_pairs)
        target_distances = (t_i - t_j).pow(2)  # (1, N_pairs)

        # Normalize target distances
        target_distances = (target_distances - self.mu) / self.std
        distances = (distances - self.mu) / self.std

        # Compute loss
        isometry_loss = (distances - target_distances).pow(2).mean()

        total_loss = isometry_loss + vcreg_loss['loss'] * 1e-1
        

        return {
            "loss": total_loss,
            "isometry_loss": isometry_loss,
            "vcreg_loss": vcreg_loss['loss'],
            "vcreg_var": vcreg_loss['var-loss'],
            "vcreg_cov": vcreg_loss['cov-loss'],
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

            # After warmup: cosine decay, but only if max_steps is set > warmup_steps
            total_steps = getattr(self.trainer, "max_steps", -1)
            if total_steps is None or total_steps <= self.hparams.warmup_steps:
                # Fallback: keep LR constant after warmup
                return 1.0

            progress = (step - self.hparams.warmup_steps) / max(
                1, (total_steps - self.hparams.warmup_steps)
            )
            min_ratio = self.hparams.final_lr / self.hparams.initial_lr
            return max(min_ratio, 0.5 * (1 + math.cos(math.pi * progress)))

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda=lr_lambda),
            "interval": "step",
        }

        return [optimizer], [scheduler]

