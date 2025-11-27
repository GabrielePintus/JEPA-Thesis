import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


# ============================================================
#  POSITION REGRESSION MODEL
# ============================================================
class PositionRegressor(nn.Module):
    def __init__(
        self, 
        hidden_dim=128, 
        patch_size=8, 
        num_heads=4, 
        num_layers=3,
    ):
        super().__init__()

        # ============================================================
        # Vision Transformer Encoder
        # ============================================================
        self.patch_size = patch_size
        num_patches = (64 // patch_size) ** 2  # 64 patches for 64x64 image
        patch_dim = 3 * patch_size * patch_size
        
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # ============================================================
        # Position Prediction Head
        # ============================================================
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Predict (x, y) position
        )

    def patchify(self, images):
        """
        Convert images to patches.
        Args:
            images: (B, 3, H, W)
        Returns:
            patches: (B, num_patches, patch_dim)
        """
        B, C, H, W = images.shape
        p = self.patch_size
        
        patches = images.reshape(B, C, H // p, p, W // p, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(B, (H // p) * (W // p), C * p * p)
        
        return patches

    def encode_visual(self, frame):
        """
        Encode frame with Vision Transformer.
        Args:
            frame: (B, 3, 64, 64)
        Returns:
            cls_token: (B, hidden_dim)
        """
        B = frame.shape[0]
        
        # Patchify and embed
        patches = self.patchify(frame)
        x = self.patch_embed(patches)
        
        # Add CLS token and positional embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        # Transform and extract CLS token
        x = self.transformer(x)
        cls_output = self.norm(x[:, 0])
        
        return cls_output

    def forward(self, frame):
        """
        Predict position from visual observation.
        Args:
            frame: (B, 3, 64, 64) - visual observation
        Returns:
            position: (B, 2) - predicted (x, y) position
        """
        # Encode visual features
        features = self.encode_visual(frame)
        
        # Predict position
        position = self.position_head(features)
        
        return position


# ============================================================
#  LIGHTNING TRAINER
# ============================================================
class PointMazeLearner(L.LightningModule):
    def __init__(
        self,
        lr=1e-3, 
        final_lr=1e-5, 
        warmup_steps=1000,
        hidden_dim=128,
        patch_size=8,
        num_heads=4,
        num_layers=3
    ):  
        super().__init__()
        self.model = PositionRegressor(
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers
        )
        self.lr = lr
        self.save_hyperparameters()

    def shared_step(self, batch, batch_idx):
        """
        Compute position prediction loss.
        """
        (s_t, frame_t), a_t, (s_target, _), _ = batch
        
        # Predict position from current frame
        pred_position = self.model(frame_t)
        
        # Extract target position (first 2 elements of state)
        target_position = s_t[:, :2]
        
        # Position loss
        loss = F.mse_loss(pred_position, target_position)
        
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        losses = self.shared_step(batch, batch_idx)
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()}, 
            prog_bar=True, 
            sync_dist=True
        )
        self.log(
            "train/lr", 
            self.trainer.optimizers[0].param_groups[0]['lr'], 
            prog_bar=True
        )
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        losses = self.shared_step(batch, batch_idx)
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items()}, 
            prog_bar=True, 
            sync_dist=True
        )
        return losses['loss']

    def configure_optimizers(self):
        """
        Configure optimizer with warmup and cosine decay schedulers.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        
        # Get total training steps
        total_steps = self.trainer.estimated_stepping_batches
        
        # Linear warmup
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.hparams.warmup_steps
        )
        
        # Cosine decay after warmup
        decay_steps = max(1, total_steps - self.hparams.warmup_steps)
        decay = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=self.hparams.final_lr
        )
        
        # Sequential scheduler
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[self.hparams.warmup_steps]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }