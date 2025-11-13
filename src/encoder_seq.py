import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import LambdaLR

# from src.losses import TemporalVCRegLoss
from src.losses import TemporalVCRegLossOptimized as TemporalVCRegLoss
from src.components.encoder import VisualEncoder, ProprioEncoder, MLPProjection
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
        self.proprio_decoder = ProprioDecoder(emb_dim=emb_dim, hidden_dim=64, output_dim=4)
        self.action_decoder = ProprioDecoder(emb_dim=emb_dim, hidden_dim=64, output_dim=2)
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
        self.action_from_patches_decoder = RegressionTransformerDecoder(
            d_model=emb_dim,
            d_out=emb_dim,
            num_layers=1,
            nhead=2,
            dim_feedforward=64,
            num_queries=1,
        )

        # Projections for VICReg
        self.proj_cls       = MLPProjection(in_dim=emb_dim, proj_dim=proj_dim)
        self.proj_state     = MLPProjection(in_dim=emb_dim, proj_dim=proj_dim)
        self.proj_patch     = MLPProjection(in_dim=emb_dim, proj_dim=proj_dim)
        self.proj_action    = MLPProjection(in_dim=emb_dim, proj_dim=proj_dim)
        self.proj_coherence_cls = MLPProjection(in_dim=emb_dim, proj_dim=proj_dim)
        self.proj_coherence_patches = RegressionTransformerDecoder(
            d_model=emb_dim,
            d_out=proj_dim,
            num_layers=1,
            nhead=2,
            dim_feedforward=64,
            num_queries=1,
        )

        # Losses
        self.tvcreg_loss = TemporalVCRegLoss()

        # Compile the networks
        if compile:
            self.visual_encoder = torch.compile(self.visual_encoder)
            self.proprio_encoder = torch.compile(self.proprio_encoder)
            self.action_encoder = torch.compile(self.action_encoder)
            self.visual_decoder = torch.compile(self.visual_decoder)
            self.proprio_decoder = torch.compile(self.proprio_decoder)
            self.action_decoder = torch.compile(self.action_decoder)
            self.idm_visual = torch.compile(self.idm_visual)
            self.idm_proprio = torch.compile(self.idm_proprio)
            self.action_from_patches_decoder = torch.compile(self.action_from_patches_decoder)
            self.proj_cls = torch.compile(self.proj_cls)
            self.proj_state = torch.compile(self.proj_state)
            self.proj_patch = torch.compile(self.proj_patch)
            self.proj_coherence_cls = torch.compile(self.proj_coherence_cls)
            self.proj_coherence_patches = torch.compile(self.proj_coherence_patches)        



    def shared_step(self, batch, batch_idx):
        states, frames, actions = batch
        # states:  (B, T+1, state_dim )
        # frames:  (B, T+1, 3, 64, 64 )
        # actions: (B, T  , action_dim)

        B, T, _ = actions.shape

        # Flatten over time
        states_flatten  = states.flatten(0, 1)      # (B * T+1, state_dim)
        frames_flatten  = frames.flatten(0, 1)      # (B * T+1, 3, 64, 64)
        actions_flatten = actions.flatten(0, 1)     # (B * T, action_dim)

        # Encode
        z_states_flatten = self.proprio_encoder(states_flatten)   # (B*T, D)
        z_cls_flatten, z_patches_flatten, _ = self.visual_encoder(frames_flatten)  # (B*T, D), (B*T, N, D)
        z_actions_flatten = self.action_encoder(actions_flatten)  # (B*T, D)

        # Reshape back to (B, T, D)
        z_states  = z_states_flatten.view(B, T+1, -1)        # (B, T+1, D)
        z_cls     = z_cls_flatten.view(B, T+1, -1)           # (B, T+1, D)
        z_patches = z_patches_flatten.view(B, T+1, z_patches_flatten.shape[1], -1) # (B, T+1, N, D)
        z_actions  = z_actions_flatten.view(B, T, -1)         # (B, T, D)


        #
        #   Temporal VCReg
        #
        # the vcreg loss class automatically handles temporal dimension, meaning we can just pass in (B, T, D) or (B, T, N, D)

        # Temporal VCReg on CLS
        z_cls_proj = self.proj_cls(z_cls)
        tvcreg_cls_loss = self.tvcreg_loss(z_cls_proj)

        # Temporal VCReg on states
        z_states_proj = self.proj_state(z_states)
        tvcreg_states_loss = self.tvcreg_loss(z_states_proj)

        # Temporal VCReg on patches
        z_patches_proj = self.proj_patch(z_patches).permute(0, 2, 1, 3).flatten(0, 1)
        tvcreg_patches_loss = self.tvcreg_loss(z_patches_proj)

        # Temporal VCReg on actions
        z_actions_proj = self.proj_action(z_actions)
        tvcreg_actions_loss = self.tvcreg_loss(z_actions_proj)

        # Combine losses
        vcreg_loss = (
            self.hparams.vc_global_coeff * tvcreg_cls_loss["loss"] +
            self.hparams.vic_state_coeff * tvcreg_states_loss["loss"] +
            self.hparams.vc_patch_coeff  * tvcreg_patches_loss["loss"] + 
            1.0  * tvcreg_actions_loss["loss"]
        )


        #
        #   Decode
        #

        # Decode state from z_state
        states_recon_flatten = self.proprio_decoder(z_states_flatten.detach())   # (B*T, state_dim)
        proprio_recon_loss = F.mse_loss(states_recon_flatten, states_flatten)

        # Decode frames from z_patches
        frames_recon_flatten = self.visual_decoder(z_patches_flatten.detach())  # (B*T, 3, 64, 64)
        visual_recon_loss = F.mse_loss(frames_recon_flatten, frames_flatten)

        # Decode actions from z_action
        actions_recon_flatten = self.action_decoder(z_actions_flatten.detach())  # (B*T, action_dim)
        action_recon_loss = F.mse_loss(actions_recon_flatten, actions_flatten)

        # combine recon losses
        recon_loss = proprio_recon_loss + visual_recon_loss + action_recon_loss

        #
        #   Inverse Dynamics Model
        #   

        # From visual CLS
        z_cls_curr, z_cls_next = z_cls[:, :-1], z_cls[:, 1:]
        idm_cls_in = torch.cat([z_cls_curr, z_cls_next], dim=-1)  # (B, T, D*2)
        idm_cls_pred = self.idm_visual(idm_cls_in)  # (B, T, D)
        idm_visual_loss = F.mse_loss(idm_cls_pred, z_actions)

        # From proprio states
        z_state_curr, z_state_next = z_states[:, :-1], z_states[:, 1:]
        idm_state_in = torch.cat([z_state_curr, z_state_next], dim=-1)  # (B, T, D*2)
        idm_state_pred = self.idm_proprio(idm_state_in)  # (B, T, D)    
        idm_proprio_loss = F.mse_loss(idm_state_pred, z_actions)

        # From visual patches
        z_patches_curr, z_patches_next = z_patches[:, :-1], z_patches[:, 1:]  # (B, T, N, D)
        z_patches_curr = z_patches_curr.flatten(0, 1)  # (B*T, N, D)
        z_patches_next = z_patches_next.flatten(0, 1)  # (B*T, N, D)
        idm_patches_in = torch.cat([z_patches_curr, z_patches_next], dim=-2)  # (B, T, N*2, D)
        idm_patches_pred = self.action_from_patches_decoder(idm_patches_in)    # (B, T, D)
        idm_patches_loss = F.mse_loss(idm_patches_pred, z_actions_flatten)

        # Combine IDM losses
        idm_loss = (idm_visual_loss + idm_proprio_loss + idm_patches_loss) / 3.0

        #
        #   Coherence
        #
        state_proj = z_states_proj.flatten(0, 1)  # (B*T, D)
        cls_proj = self.proj_coherence_cls(z_cls_flatten)  # (B*T, D)
        patches_proj = self.proj_coherence_patches(z_patches_flatten)

        # State information in patches
        patch_coherence_loss = F.mse_loss(state_proj, patches_proj)

        # State information in CLS
        cls_coherence_loss = F.mse_loss(state_proj, cls_proj)

        cross_coherence_loss = (patch_coherence_loss + cls_coherence_loss) / 2.0

        #
        #   Total loss
        #
        loss = recon_loss + vcreg_loss + cross_coherence_loss + idm_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "proprio_recon_loss": proprio_recon_loss,
            "visual_recon_loss": visual_recon_loss,
            "action_recon_loss": action_recon_loss,
            "idm_loss": idm_loss,       
            "idm_visual_loss": idm_visual_loss,
            "idm_proprio_loss": idm_proprio_loss,
            "idm_patches_loss": idm_patches_loss,
            "vcreg_cls_loss": tvcreg_cls_loss["loss"],
            "vcreg_state_loss": tvcreg_states_loss["loss"],
            "vcreg_patch_loss": tvcreg_patches_loss["loss"],
            "vcreg_action_loss": tvcreg_actions_loss["loss"],
            "vcreg_loss": vcreg_loss,
        }
    

    def training_step(self, batch, batch_idx):
        outs = self.shared_step(batch, batch_idx)
        self.log_dict({f"train/{k}": v for k, v in outs.items()}, prog_bar=True, sync_dist=True)
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
                    + list(self.proj_state.parameters())
                    + list(self.proj_patch.parameters())
                    + list(self.proj_action.parameters())
                    + list(self.proj_coherence_cls.parameters())
                    + list(self.proj_coherence_patches.parameters()),
            "lr": self.hparams.initial_lr_encoder,          # encoder LR
            "weight_decay": self.hparams.weight_decay_encoder,
        }

        decoder_params = {
            "params": list(self.visual_decoder.parameters())
                    + list(self.proprio_decoder.parameters())
                    + list(self.action_decoder.parameters()),
            "lr": self.hparams.initial_lr_decoder,    # <-- smaller LR for decoder
            "weight_decay": self.hparams.weight_decay_decoder,
        }

        idm_params = {
            "params": list(self.idm_visual.parameters())
                    + list(self.idm_proprio.parameters())
                    + list(self.action_from_patches_decoder.parameters()),
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
