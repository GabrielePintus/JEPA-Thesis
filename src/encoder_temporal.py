import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import lightning as L
from torch.optim.lr_scheduler import LambdaLR
from vicreg_loss import VICRegLoss
# from src.vicreg import VICRegLoss
from src.focal_loss import FocalReconstructionLoss

from src.components.encoder import VisualEncoder, ProprioEncoder, MLPProjection, LocalizationHead
from src.components.decoder import VisualDecoder, ProprioDecoder, RegressionTransformerDecoder


def focal_mse(pred, target, gamma=2):
    err = (pred - target)
    return ((err.abs() ** gamma) * (err ** 2))

class VICRegJEPAEncoder(L.LightningModule):
    """
    Batch format expectation:
      batch = {
        "image": (B, 3, 64, 64) float in [0,1],
        "pos":   (B, 2) pixel coords [0..63],  # only (x,y) for cross-modal
        # optionally "vel": (B, 2) but UNUSED here
      }
    """
    def __init__(
        self,
        emb_dim=128,
        depth=3,
        heads=4,
        mlp_dim=256,
        proj_dim=1024,
        initial_lr_encoder = 1e-3,
        final_lr_encoder   = 1e-5,
        weight_decay_encoder = 1e-3,
        initial_lr_decoder = 1e-4,
        final_lr_decoder   = 1e-6,
        weight_decay_decoder = 1e-5,
        initial_lr_idm = 1e-3,
        final_lr_idm   = 1e-5,
        weight_decay_idm = 1e-5,
        warmup_steps = 1000,
        vc_global_coeff = 1.0,
        vc_patch_coeff  = 0.5,
        vic_state_coeff  = 0.2,
        decoder_delay_steps = 0,
        compile = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Encoders
        self.visual_encoder     = VisualEncoder(emb_dim=emb_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, patch_size=8)
        self.proprio_encoder    = ProprioEncoder(emb_dim=emb_dim)
        self.action_encoder     = ProprioEncoder(emb_dim=emb_dim, input_dim=2)

        # Decoders
        self.visual_decoder  = VisualDecoder(
            emb_dim=emb_dim,
            patch_size=8,
        )
        self.proprio_decoder = ProprioDecoder(emb_dim=emb_dim, hidden_dim=64)
        self.idm_visual = nn.Sequential(
            nn.LayerNorm(emb_dim * 2),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.idm_proprio = nn.Sequential(
            nn.LayerNorm(emb_dim * 2),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.state_from_patches_decoder = RegressionTransformerDecoder(
            d_model=emb_dim,
            d_out=emb_dim,
            num_layers=1,
            nhead=2,
            dim_feedforward=64,
            num_queries=1,
        )

        # Projections for VICReg
        self.proj_cls       = MLPProjection(in_dim=emb_dim, proj_dim=proj_dim)
        self.proj_patch     = MLPProjection(in_dim=emb_dim, proj_dim=proj_dim)  # applied per token
        self.proj_cross_vis = MLPProjection(in_dim=emb_dim, proj_dim=proj_dim)  # CLS→proj
        self.proj_cross_pos = MLPProjection(in_dim=emb_dim, proj_dim=proj_dim)  # PosToken→proj

        # Compile all the networks
        if compile:
            self.visual_encoder     = torch.compile(self.visual_encoder)
            self.proprio_encoder    = torch.compile(self.proprio_encoder)
            self.action_encoder     = torch.compile(self.action_encoder)
            self.proj_cls           = torch.compile(self.proj_cls)
            self.proj_patch         = torch.compile(self.proj_patch)
            self.proj_cross_vis     = torch.compile(self.proj_cross_vis)
            self.proj_cross_pos     = torch.compile(self.proj_cross_pos)
            self.visual_decoder     = torch.compile(self.visual_decoder)
            self.proprio_decoder    = torch.compile(self.proprio_decoder)
            self.idm_visual        = torch.compile(self.idm_visual)
            self.idm_proprio       = torch.compile(self.idm_proprio)
            self.state_from_patches_decoder = torch.compile(self.state_from_patches_decoder)

        # Losses
        self.vicreg_loss = VICRegLoss()
        self.focal_loss = FocalReconstructionLoss(gamma=2.0, alpha=0.5)

        self._decoder_delay_steps = decoder_delay_steps
        self._global_step = 0

    def shared_step(self, batch, batch_idx):
        (state_curr, frame_curr), action, (state_next, frame_next), _ = batch

        # 1) Encode both views (vision branch)
        cls_curr, patch_curr, _ = self.visual_encoder(frame_curr)      # (B, D), (B, N, D)
        cls_next, patch_next, _ = self.visual_encoder(frame_next)      # (B, D), (B, N, D)
        z_state_curr = self.proprio_encoder(state_curr)
        z_state_next = self.proprio_encoder(state_next)
        z_action = self.action_encoder(action)

        # Temporal VCReg losses on patches
        patch_curr_proj = self.proj_patch(patch_curr).flatten(0, 1)
        patch_next_proj = self.proj_patch(patch_next).flatten(0, 1)
        vcreg_patch_curr = 15 * VICRegLoss.variance_loss(patch_curr_proj, 1.0) + VICRegLoss.covariance_loss(patch_curr_proj)
        vcreg_patch_next = 15 * VICRegLoss.variance_loss(patch_next_proj, 1.0) + VICRegLoss.covariance_loss(patch_next_proj)
        vcreg_patch = (vcreg_patch_curr + vcreg_patch_next) / 2.0

        # Temporal VICReg loss on global CLS token
        cls_curr_proj = self.proj_cls(cls_curr)  # (B, D)
        cls_next_proj = self.proj_cls(cls_next)  # (B, D)
        vcreg_cls_curr = 15 * VICRegLoss.variance_loss(cls_curr_proj, 1.0) + VICRegLoss.covariance_loss(cls_curr_proj)
        vcreg_cls_next = 15 * VICRegLoss.variance_loss(cls_next_proj, 1.0) + VICRegLoss.covariance_loss(cls_next_proj)
        vcreg_cls = (vcreg_cls_curr + vcreg_cls_next) / 2.0

        # Temporal VICReg loss on state tokens
        z_state_curr_proj = self.proj_cross_pos(z_state_curr)
        z_state_next_proj = self.proj_cross_pos(z_state_next)
        vcreg_state_curr = 15 * VICRegLoss.variance_loss(z_state_curr_proj, 1.0) + VICRegLoss.covariance_loss(z_state_curr_proj)
        vcreg_state_next = 15 * VICRegLoss.variance_loss(z_state_next_proj, 1.0) + VICRegLoss.covariance_loss(z_state_next_proj)
        vcreg_state = (vcreg_state_curr + vcreg_state_next) / 2.0

        # Invariance between state and visual CLS tokens
        z_cross_vis_curr = self.proj_cross_vis(cls_curr)
        z_cross_vis_next = self.proj_cross_vis(cls_next)
        vicreg_cross_curr = F.mse_loss(z_cross_vis_curr, z_state_curr_proj)
        vicreg_cross_next = F.mse_loss(z_cross_vis_next, z_state_next_proj)
        vicreg_cross = (vicreg_cross_curr + vicreg_cross_next) / 2.0

        # Decode z_state_curr, z_state_next from visual patches
        state_from_patches_curr = self.state_from_patches_decoder(patch_curr).squeeze(1)  # (B, D)
        state_from_patches_next = self.state_from_patches_decoder(patch_next).squeeze(1)  # (B, D)
        state_from_patches_loss_curr = F.mse_loss(state_from_patches_curr, z_state_curr)
        state_from_patches_loss_next = F.mse_loss(state_from_patches_next, z_state_next)
        state_from_patches_loss = (state_from_patches_loss_curr + state_from_patches_loss_next) / 2.0

        # Decode action from visual cls, proprio state and visual patches
        idm_input_visual = torch.cat([cls_curr, cls_next], dim=-1)
        idm_input_proprio = torch.cat([z_state_curr, z_state_next], dim=-1)
        idm_pred_visual = self.idm_visual(idm_input_visual)
        idm_pred_proprio = self.idm_proprio(idm_input_proprio)
        idm_visual_loss = F.mse_loss(idm_pred_visual, z_action)
        idm_proprio_loss = F.mse_loss(idm_pred_proprio, z_action)

        idm_loss = (idm_visual_loss + idm_proprio_loss) / 2.0


        # Visual reconstruction probe
        frame_curr_recon = self.visual_decoder(patch_curr.detach())  # (B,3,64,64)
        frame_next_recon = self.visual_decoder(patch_next.detach())  # (B,3,64,64)
        alpha = 0.9
        visual_recon_loss_curr = alpha * self.focal_loss(frame_curr_recon, frame_curr) + (1 - alpha) * F.mse_loss(frame_curr_recon, frame_curr)
        visual_recon_loss_next = alpha * self.focal_loss(frame_next_recon, frame_next) + (1 - alpha) * F.mse_loss(frame_next_recon, frame_next)
        visual_recon_loss = (visual_recon_loss_curr + visual_recon_loss_next) / 2.0


        # Proprio reconstruction probe — only supervise (x,y)
        state_curr_recon = self.proprio_decoder(z_state_curr.detach())   # (B,4)
        state_next_recon = self.proprio_decoder(z_state_next.detach())   # (B,4)
        proprio_recon_loss_curr = F.mse_loss(state_curr_recon, state_curr)
        proprio_recon_loss_next = F.mse_loss(state_next_recon, state_next)
        proprio_recon_loss = (proprio_recon_loss_curr + proprio_recon_loss_next) / 2.0
        
        recon_loss_curr = visual_recon_loss_curr + proprio_recon_loss_curr
        recon_loss_next = visual_recon_loss_next + proprio_recon_loss_next
        recon_loss = (recon_loss_curr + recon_loss_next) / 2.0

        # Total loss
        loss = (
            recon_loss +
            idm_loss +
            self.hparams.vc_global_coeff * vcreg_cls +
            self.hparams.vc_patch_coeff  * vcreg_patch +
            self.hparams.vic_state_coeff  * vcreg_state +
            vicreg_cross + state_from_patches_loss
        )

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "recon_loss_curr": recon_loss_curr,
            "recon_loss_next": recon_loss_next,
            "proprio_recon_loss": proprio_recon_loss,
            "proprio_recon_loss_curr": proprio_recon_loss_curr,
            "proprio_recon_loss_next": proprio_recon_loss_next,
            "visual_recon_loss": visual_recon_loss,
            "visual_recon_loss_curr": visual_recon_loss_curr,
            "visual_recon_loss_next": visual_recon_loss_next,
            "idm_loss": idm_loss,       
            "idm_visual_loss": idm_visual_loss,
            "idm_proprio_loss": idm_proprio_loss,
            "vcreg_cls": vcreg_cls,
            "vcreg_cls_curr": vcreg_cls_curr,
            "vcreg_cls_next": vcreg_cls_next,
            "vcreg_patch": vcreg_patch,
            "vcreg_patch_curr": vcreg_patch_curr,
            "vcreg_patch_next": vcreg_patch_next,
            "vcreg_state": vcreg_state,
            "vcreg_state_curr": vcreg_state_curr,
            "vcreg_state_next": vcreg_state_next,
            "vicreg_cross": vicreg_cross,
            "vicreg_cross_curr": vicreg_cross_curr,
            "vicreg_cross_next": vicreg_cross_next,
            "state_from_patches_loss": state_from_patches_loss,
            "state_from_patches_loss_curr": state_from_patches_loss_curr,
            "state_from_patches_loss_next": state_from_patches_loss_next,
        }

    def training_step(self, batch, batch_idx):
        outs = self.shared_step(batch, batch_idx)
        self.log_dict({f"train/{k}": v for k, v in outs.items()}, prog_bar=True, sync_dist=True)
        self._global_step += 1
        return outs["loss"]
    def validation_step(self, batch, batch_idx):
        outs = self.shared_step(batch, batch_idx)
        self.log_dict({f"val/{k}": v for k, v in outs.items()}, prog_bar=True, sync_dist=True)
        return outs["loss"]

    def configure_optimizers(self):
        # ----------------------------
        # Param groups
        # ----------------------------
        encoder_params = {
            "params": list(self.visual_encoder.parameters())
                    + list(self.proprio_encoder.parameters())
                    + list(self.action_encoder.parameters())
                    + list(self.proj_cls.parameters())
                    + list(self.proj_patch.parameters())
                    + list(self.proj_cross_vis.parameters())
                    + list(self.proj_cross_pos.parameters()),
            "lr": self.hparams.initial_lr_encoder,          # encoder LR
            "weight_decay": self.hparams.weight_decay_encoder,
        }

        decoder_params = {
            "params": list(self.visual_decoder.parameters())
                    + list(self.proprio_decoder.parameters())
                    + list(self.state_from_patches_decoder.parameters()),
            "lr": self.hparams.initial_lr_decoder,    # <-- smaller LR for decoder
            "weight_decay": 0.0,                    # <-- common choice: no WD on decoder
        }

        idm_params = {
            "params": list(self.idm_visual.parameters())
                    + list(self.idm_proprio.parameters()),
            "lr": self.hparams.initial_lr_idm,          # same as encoder
            "weight_decay": self.hparams.weight_decay_idm,
        }

        optimizer = torch.optim.AdamW(
            [encoder_params, decoder_params, idm_params],
        )

        # ----------------------------
        # Scheduler (warmup + cosine)
        # ----------------------------
        total_steps = (
            self.trainer.max_epochs * self.trainer.estimated_stepping_batches
        )
        warmup_steps = self.hparams.warmup_steps

        def lr_lambda(step):
            if step < warmup_steps:            # warmup
                return step / max(1, warmup_steps)
            # cosine decay
            t = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
            return 0.5 * (1 + math.cos(math.pi * min(max(t, 0.0), 1.0)))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
