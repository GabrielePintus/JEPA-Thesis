from src.models.encoder import MeNet6_128
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
from torch.optim.lr_scheduler import LambdaLR


class VisualEncoder(L.LightningModule):
    def __init__(
        self,
        initial_lr=1e-4,
        final_lr=1e-6,
        weight_decay=1e-4,
        warmup_steps=1000,
        proj_dim=2048,
        pred_hidden_dim=512,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ----------------------------
        # 1️⃣ Backbone CNN encoder
        # ----------------------------
        self.encoder = MeNet6_128()  # Output: (B, C, H, W)

        # Dummy forward to infer C dimension (for flexible encoder)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            c_dim = self.encoder(dummy).shape[1]

        # ----------------------------
        # 2️⃣ Projection MLP
        # ----------------------------
        self.projector = nn.Sequential(
            nn.Linear(c_dim, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, self.hparams.proj_dim, bias=False),
            nn.BatchNorm1d(self.hparams.proj_dim, affine=False),
        )
        self.projector = torch.compile(self.projector, mode="default")

        # ----------------------------
        # 3️⃣ Prediction MLP
        # ----------------------------
        self.predictor = nn.Sequential(
            nn.Linear(self.hparams.proj_dim, self.hparams.pred_hidden_dim, bias=False),
            nn.BatchNorm1d(self.hparams.pred_hidden_dim),
            nn.GELU(),
            nn.Linear(self.hparams.pred_hidden_dim, self.hparams.proj_dim),
        )
        self.predictor = torch.compile(self.predictor, mode="default")

    # ----------------------------
    # Forward: encoder + pooling + heads
    # ----------------------------
    def forward_encoder(self, x):
        """Encode image -> pooled feature vector."""
        feat = self.encoder(x)           # (B, C, H, W)
        feat = feat.mean(dim=[2, 3])     # Global average pooling → (B, C)
        return feat

    def forward(self, x):
        """Forward pass through encoder, projection, prediction."""
        h = self.forward_encoder(x)
        z = self.projector(h)
        p = self.predictor(z)
        return z, p

    # ----------------------------
    # Loss (negative cosine similarity)
    # ----------------------------
    @staticmethod
    def neg_cosine_similarity(p, z):
        z = z.detach()  # stop-grad on target branch
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return -(p * z).sum(dim=-1).mean()

    # ----------------------------
    # Training / Validation
    # ----------------------------
    def shared_step(self, batch, batch_idx):
        v1, v2 = batch  # Two augmented views
        z1, p1 = self(v1)
        z2, p2 = self(v2)
        loss = 0.5 * (
            self.neg_cosine_similarity(p1, z2) +
            self.neg_cosine_similarity(p2, z1)
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    # ----------------------------
    # Optimizer + LR Scheduler
    # ----------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.initial_lr,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(step: int):
            if self.trainer is None or not hasattr(self.trainer, "estimated_stepping_batches"):
                max_steps = 100_000
            else:
                max_steps = self.trainer.estimated_stepping_batches

            # Linear warm-up
            if step < self.hparams.warmup_steps:
                return step / float(self.hparams.warmup_steps)

            # Cosine decay
            progress = (step - self.hparams.warmup_steps) / float(
                max(1, max_steps - self.hparams.warmup_steps)
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            decayed = (
                self.hparams.final_lr / self.hparams.initial_lr
                + (1 - self.hparams.final_lr / self.hparams.initial_lr) * cosine_decay
            )
            return decayed

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda=lr_lambda),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
