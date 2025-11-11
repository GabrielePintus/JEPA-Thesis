import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import LambdaLR
# from vicreg_loss import VICRegLoss
from losses import VICRegLoss

from src.components.encoder import VisualEncoder, ProprioEncoder, MLPProjection, LocalizationHead
from src.components.decoder import VisualDecoder, ProprioDecoder


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
        initial_lr = 1e-3,
        final_lr   = 1e-5,
        weight_decay = 1e-4,
        warmup_steps = 1000,
        vic_global_coeff = 1.0,
        vic_patch_coeff  = 0.5,
        vic_cross_coeff  = 0.2,
        loc_coeff        = 0.2,
        decoder_delay_steps = 0,
        compile = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Encoders
        self.visual_encoder     = VisualEncoder(emb_dim=emb_dim, depth=depth, heads=heads, mlp_dim=mlp_dim)
        self.proprio_encoder    = ProprioEncoder(emb_dim=emb_dim)

        # Decoders
        self.visual_decoder  = VisualDecoder(
            emb_dim=emb_dim,
        )
        self.proprio_decoder = ProprioDecoder(emb_dim=emb_dim)

        # Projections for VICReg
        self.proj_cls       = MLPProjection(in_dim=emb_dim, proj_dim=proj_dim)
        self.proj_patch     = MLPProjection(in_dim=emb_dim, proj_dim=proj_dim)  # applied per token
        self.proj_cross_vis = MLPProjection(in_dim=emb_dim, proj_dim=proj_dim)  # CLS→proj
        self.proj_cross_pos = MLPProjection(in_dim=emb_dim, proj_dim=proj_dim)  # PosToken→proj

        # Localization head(s) for supervised tasks
        # self.loc_head = LocalizationHead(dim=emb_dim, grid_hw=(8, 8))
        # self.obs_head = nn.Sequential(
        #     nn.LayerNorm(emb_dim),
        #     nn.Linear(emb_dim, 4)   # predict (x,y,vx,vy)
        # )

        # Compile all the networks
        if compile:
            self.visual_encoder     = torch.compile(self.visual_encoder)
            self.proprio_encoder    = torch.compile(self.proprio_encoder)
            self.proj_cls           = torch.compile(self.proj_cls)
            self.proj_patch         = torch.compile(self.proj_patch)
            self.proj_cross_vis     = torch.compile(self.proj_cross_vis)
            self.proj_cross_pos     = torch.compile(self.proj_cross_pos)
            self.visual_decoder     = torch.compile(self.visual_decoder)
            self.proprio_decoder    = torch.compile(self.proprio_decoder)

        # Losses
        self.vicreg_loss = VICRegLoss(inv_coeff=1.0, var_coeff=25.0, cov_coeff=1.0, gamma=1.0)

        self._decoder_delay_steps = decoder_delay_steps
        self._global_step = 0

    def shared_step(self, batch, batch_idx):
        (img, img_aug), obs, protected_patch_idx = batch

        # extract only (x,y) for losses that depend on predictability
        # pos_px = obs[:, :2]        # (B, 2)

        # 1) Encode both views (vision branch)
        cls1, p1, _ = self.visual_encoder(img)
        cls2, p2, _ = self.visual_encoder(img_aug)

        # 2) VICReg on CLS tokens
        zc1 = self.proj_cls(cls1)
        zc2 = self.proj_cls(cls2)
        vic_g = self.vicreg_loss(zc1, zc2)

        # 3) VICReg on patch tokens
        B, N, _ = p1.shape
        zp1 = self.proj_patch(p1).view(B * N, -1)
        zp2 = self.proj_patch(p2).view(B * N, -1)
        vic_p = self.vicreg_loss(zp1, zp2)

        # 4) Cross-modal (VISUAL <-> POSITION ONLY)
        # obs_zero_vel = obs[:, :2]          # (B, 2) only (x,y)
        # obs_zero_vel = torch.cat([obs_zero_vel, torch.zeros_like(obs[:, 2:4])], dim=1)  # (B,4) add zero (vx,vy)
        # pos_token = self.proprio(obs_zero_vel)             # (B, D) encodes (x,y,vx,vy)
        pos_token = self.proprio_encoder(obs)
        z_cross_vis = self.proj_cross_vis(cls1)
        z_cross_pos = self.proj_cross_pos(pos_token)
        vic_x = self.vicreg_loss(z_cross_vis, z_cross_pos)

        # # 5) Localization loss (same: only x,y matter)
        # logits = self.loc_head(p1)
        # # idx = self.xy_to_patch_index(pos_px)
        # # target = self.gaussian_heatmap_index(idx, num=64, sigma=1.0)
        # # loc_loss = F.kl_div(torch.log_softmax(logits, dim=1), target, reduction="batchmean")
        # loc_loss_patch = F.cross_entropy(logits, protected_patch_idx)
        # # Regression on full (x,y,vx,vy) state
        # pred_obs = self.obs_head(cls1)
        # loc_loss_obs = F.mse_loss(pred_obs[:, :2], obs[:, :2])
        # loc_loss = loc_loss_patch + loc_loss_obs

        # ============================================================
        # DECODER PROBES (stopgrad)
        # ============================================================

        if self._global_step >= self._decoder_delay_steps:
            # Visual reconstruction probe
            recon_img_1 = self.visual_decoder(p1.detach())  # (B,3,64,64)
            recon_img_2 = self.visual_decoder(p2.detach())  # (B,3,64,64)    
            recon_loss = F.mse_loss(recon_img_1, img) + 1e-1 * F.l1_loss(recon_img_1, img)
            recon_loss += F.mse_loss(recon_img_2, img) + 1e-1 * F.l1_loss(recon_img_2, img)

            # Proprio reconstruction probe — only supervise (x,y)
            pred_state = self.proprio_decoder(pos_token.detach())   # (B,4)
            # proprio_recon_loss = F.mse_loss(pred_state[:, :2], pos_px)
            proprio_recon_loss = F.mse_loss(pred_state, obs)
        else:
            recon_loss = torch.tensor(0.0, device=self.device)
            proprio_recon_loss = torch.tensor(0.0, device=self.device)

        # Total loss (weighted VICReg + auxiliary probes)
        loss = (
            self.hparams.vic_global_coeff * vic_g['loss'] +
            self.hparams.vic_patch_coeff  * vic_p['loss'] +
            self.hparams.vic_cross_coeff  * vic_x['loss'] +
            #self.hparams.loc_coeff        * loc_loss      +
            recon_loss + proprio_recon_loss
        )

        return {
            "loss": loss,
            "vic_global": vic_g["loss"],
            "vic_patch": vic_p["loss"],
            "vic_cross": vic_x["loss"],
            # "loc_loss": loc_loss,
            # "loc_loss_patch": loc_loss_patch,
            # "loc_loss_obs": loc_loss_obs,
            "recon_loss": recon_loss,
            "proprio_recon_loss": proprio_recon_loss,
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
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.initial_lr, weight_decay=self.hparams.weight_decay)
        total_steps = self.trainer.max_epochs * self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            t = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
            return 0.5 * (1 + math.cos(math.pi * min(max(t, 0.0), 1.0)))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "step"
            }
        }