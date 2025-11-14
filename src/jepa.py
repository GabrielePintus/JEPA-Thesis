import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import LambdaLR

# from src.losses import TemporalVCRegLoss
from src.losses import TemporalVCRegLossOptimized as TemporalVCRegLoss
from src.components.encoder import RepeatEncoder, SimpleEncoder, VisualEncoder, ProprioEncoder, MLPProjection
from src.components.decoder import VisualDecoder, ProprioDecoder, RegressionTransformerDecoder
from src.components.predictor import EncoderDecoderPredictor, TransformerDecoderPredictor, TransformerEncoderPredictor



class JEPA(L.LightningModule):
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
        initial_lr_predictor = 1e-3,
        final_lr_predictor   = 1e-5,
        weight_decay_predictor = 1e-5,
        warmup_steps = 1000,
        vc_global_coeff = 1.0,
        vc_patch_coeff  = 0.5,
        vic_state_coeff  = 0.2,
        autoregressive_loss_decay = 0.5,
        compile = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Encoders
        self.visual_encoder     = VisualEncoder(emb_dim=emb_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, patch_size=8)
        self.proprio_encoder    = SimpleEncoder(input_dim=4, emb_dim=emb_dim)
        self.action_encoder = SimpleEncoder(input_dim=2, emb_dim=emb_dim)
        
        # Predictor
        self.predictor = EncoderDecoderPredictor(
            emb_dim=emb_dim,
            num_heads=4,
            num_encoder_layers=3,
            num_decoder_layers=0,
            mlp_dim=128,
            residual=False
        )

        # Decoders
        self.visual_decoder  = VisualDecoder(
            emb_dim=emb_dim,
            patch_size=8,
        )
        self.proprio_decoder = ProprioDecoder(emb_dim=emb_dim, hidden_dim=64, output_dim=4)
        self.action_decoder = ProprioDecoder(emb_dim=emb_dim, hidden_dim=64, output_dim=2)

        # Inverse Dynamics Models
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
            self.predictor = torch.compile(self.predictor)
        


    def encode_state_and_frame(self, state, frame):       
        """
        Encode a single state and frame.
        
        Args:
            state: (B, state_dim)
            frame: (B, 3, 64, 64)
        """
        z_state = self.proprio_encoder(state)
        z_cls, z_patches, _ = self.visual_encoder(frame)
        result = {
            "z_state": z_state,
            "z_cls": z_cls,
            "z_patches": z_patches,
        }
        return result
    
    def predictor_step(self, z_cls, z_patches, z_state, z_action):
        """
        Predict next latent representations given current latents and action.
        
        Args:
            z_cls:     (B, 1, D)
            z_patches: (B, Np, D)
            z_state:   (B, 1, D)
            z_action:  (B, D)
        Returns:
            pred_cls:     (B, D)
            pred_patches: (B, Np, D)
            pred_state:   (B, D)
        """
        pred_cls, pred_patches, pred_state = self.predictor(z_cls, z_patches, z_state, z_action)
        return pred_cls, pred_patches, pred_state

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
        states_recon_flatten = self.proprio_decoder(
            z_states_flatten.detach(),
            *self.proprio_encoder.get_stats()
        )   # (B*T, state_dim)
        proprio_recon_loss = F.mse_loss(states_recon_flatten, states_flatten)

        # Decode frames from z_patches
        frames_recon_flatten = self.visual_decoder(z_patches_flatten.detach())  # (B*T, 3, 64, 64)
        visual_recon_loss = F.mse_loss(frames_recon_flatten, frames_flatten)

        # Decode actions from z_action
        actions_recon_flatten = self.action_decoder(
            z_actions_flatten.detach(),
            *self.action_encoder.get_stats()
        )  # (B*T, action_dim)
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
        #   Prediction
        #

        # 1-step prediction
        z_states_curr, z_states_next = z_states[:, :-1].flatten(0, 1), z_states[:, 1:].flatten(0, 1)
        z_cls_curr, z_cls_next = z_cls[:, :-1].flatten(0, 1), z_cls[:, 1:].flatten(0, 1)
        z_patches_curr, z_patches_next = z_patches[:, :-1].flatten(0, 1), z_patches[:, 1:].flatten(0, 1)
        z_cls_next_pred, z_patches_next_pred, z_states_next_pred = self.predictor(
            z_cls_curr.unsqueeze(1),
            z_patches_curr,
            z_states_curr.unsqueeze(1),
            z_actions_flatten,
        )
        # Prediction losses
        loss_pred_cls       = F.mse_loss(z_cls_next_pred, z_cls_next)
        loss_pred_patches   = F.mse_loss(z_patches_next_pred, z_patches_next)
        loss_pred_states    = F.mse_loss(z_states_next_pred, z_states_next)
        # Combine prediction losses
        prediction_loss_1 = (loss_pred_cls + loss_pred_patches + loss_pred_states) / 3.0

        # Variance loss stacking on time the previous and predicted steps
        tvcreg_states_pred_in = torch.stack([z_states_curr, z_states_next_pred], dim=1)
        tvcreg_cls_pred_in = torch.stack([z_cls_curr, z_cls_next_pred], dim=1)
        tvcreg_patches_pred_in = torch.stack([z_patches_curr, z_patches_next_pred], dim=1).permute(0, 2, 1, 3).flatten(0, 1)
        tvcreg_states_pred_loss = self.tvcreg_loss(self.proj_state(tvcreg_states_pred_in))['loss']
        tvcreg_cls_pred_loss = self.tvcreg_loss(self.proj_cls(tvcreg_cls_pred_in))['loss']
        tvcreg_patches_pred_loss = self.tvcreg_loss(self.proj_patch(tvcreg_patches_pred_in))['loss']
        tvcreg_pred_loss = (tvcreg_states_pred_loss + tvcreg_cls_pred_loss + tvcreg_patches_pred_loss) / 3.0


        # Smoothing 
        smoothing_loss_states = F.mse_loss(z_states_next, z_states_curr)
        smoothing_loss_cls = F.mse_loss(z_cls_next, z_cls_curr)
        smoothing_loss_patches = F.mse_loss(z_patches_next, z_patches_curr)
        smoothing_loss = (smoothing_loss_states + smoothing_loss_cls + smoothing_loss_patches) / 3.0



        # # 2-step prediction
        # # First, we need to get the states, cls, and patches for t+2
        # z_cls_pred_step1 = z_cls_next_pred[:-B]  # (B*(T-1), D)
        # z_patches_pred_step1 = z_patches_next_pred[:-B]  # (B*(T-1), N, D)
        # z_states_pred_step1 = z_states_next_pred[:-B]  # (B*(T-1), D)

        # # Actions at t+1 (skip first action, these are actions that follow our predictions)
        # z_actions_step2 = z_actions_flatten[B:]  # (B*(T-1), D)

        # # Ground truth at t+2
        # z_states_next_2 = z_states[:, 2:].flatten(0, 1)  # (B*(T-1), D)
        # z_cls_next_2 = z_cls[:, 2:].flatten(0, 1)  # (B*(T-1), D)
        # z_patches_next_2 = z_patches[:, 2:].flatten(0, 1)  # (B*(T-1), N, D)

        # # Step 2: Predict t+2 from predicted t+1 using action at t+1 (autoregressive, no teacher forcing)
        # z_cls_next_pred_2, z_patches_next_pred_2, z_states_next_pred_2 = self.predictor(
        #     z_cls_pred_step1.unsqueeze(1),
        #     z_patches_pred_step1,
        #     z_states_pred_step1.unsqueeze(1),
        #     z_actions_step2,
        # )
        # # Prediction losses
        # loss_pred_cls_2       = F.mse_loss(z_cls_next_pred_2, z_cls_next_2)
        # loss_pred_patches_2   = F.mse_loss(z_patches_next_pred_2, z_patches_next_2)
        # loss_pred_states_2    = F.mse_loss(z_states_next_pred_2, z_states_next_2)

        # # Combine prediction losses
        # prediction_loss_2 = (loss_pred_cls_2 + loss_pred_patches_2 + loss_pred_states_2) / 3.0

        # Weighted sum
        # prediction_loss = prediction_loss_1 + self.hparams.autoregressive_loss_decay * prediction_loss_2
        # prediction_loss = prediction_loss / (1.0 + self.hparams.autoregressive_loss_decay)
        prediction_loss = prediction_loss_1
        


        #
        #   Total loss
        #
        loss = recon_loss + vcreg_loss + cross_coherence_loss + idm_loss + prediction_loss + tvcreg_pred_loss + 0.1 * smoothing_loss

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
            "prediction_loss": prediction_loss,
            "prediction_loss_1": prediction_loss_1,
            # "prediction_loss_2": prediction_loss_2,
            "smoothing_loss": smoothing_loss,
            "loss_pred_cls": loss_pred_cls,
            "loss_pred_patches": loss_pred_patches,
            "loss_pred_states": loss_pred_states,
            "cross_coherence_loss": cross_coherence_loss,
            "tvcreg_pred_loss": tvcreg_pred_loss,
            "tvcreg_states_pred_loss": tvcreg_states_pred_loss,
            "tvcreg_cls_pred_loss": tvcreg_cls_pred_loss,
            "tvcreg_patches_pred_loss": tvcreg_patches_pred_loss,
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
        predictor_params = {
            "params": list(self.predictor.parameters()),
            "lr": self.hparams.initial_lr_predictor,
            "weight_decay": self.hparams.weight_decay_predictor,
        }

        optimizer = torch.optim.AdamW(
            [encoder_params, decoder_params, idm_params, predictor_params],
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



    # ========================================
    # Autoregressive Prediction Methods
    # ========================================
    def encode_initial_state(self, state, frame):
        """
        Encode a single initial state and frame.
        
        Args:
            state: (B, state_dim) initial proprioceptive state
            frame: (B, 3, 64, 64) initial frame
            
        Returns:
            dict with:
                - z_state: (B, D) encoded state
                - z_cls: (B, D) encoded cls token
                - z_patches: (B, N, D) encoded patches
        """
        z_state = self.proprio_encoder(state)
        z_cls, z_patches, _ = self.visual_encoder(frame)
        
        return {
            "z_state": z_state,
            "z_cls": z_cls,
            "z_patches": z_patches,
        }
    
    def predict_next_latent(self, z_cls, z_patches, z_state, action):
        """
        Predict next latent representations given current latents and action.

        Args:
            z_cls:     (B, D) current cls token
            z_patches: (B, N, D) current patches
            z_state:   (B, D) current state
            action:    (B, action_dim) action to take

        Returns:
            dict with predicted next latents:
                - z_cls_next:     (B, D)
                - z_patches_next: (B, N, D)
                - z_state_next:   (B, D)
        """
        # Encode action
        z_action = self.action_encoder(action)

        # Predictor outputs in latent space
        z_cls_next, z_patches_next, z_state_next = self.predictor(
            z_cls.unsqueeze(1),   # (B, 1, D)
            z_patches,            # (B, N, D)
            z_state.unsqueeze(1), # (B, 1, D)
            z_action,             # (B, D)
        )

        return {
            "z_cls_next": z_cls_next,
            "z_patches_next": z_patches_next,
            "z_state_next": z_state_next,
        }

    
    def predict_next_latents(self, current_latents, action):
        """
        Predict next latent representations given current latents dict and action.
        """
        if action.ndim == 3:
            action = action.squeeze(1)

        predicted = self.predict_next_latent(
            current_latents["z_cls"],
            current_latents["z_patches"],
            current_latents["z_state"],
            action,
        )

        return {
            "z_cls": predicted["z_cls_next"],
            "z_patches": predicted["z_patches_next"],
            "z_state": predicted["z_state_next"],
        }

    
    @torch.no_grad()
    def decode_latents(self, z_cls, z_patches, z_state):
        """
        Decode latent representations back to observations.
        
        Args:
            z_cls: (B, D) cls token
            z_patches: (B, N, D) patches
            z_state: (B, D) state
            
        Returns:
            dict with:
                - frame_recon: (B, 3, 64, 64) reconstructed frame
                - state_recon: (B, state_dim) reconstructed state
        """
        frame_recon = self.visual_decoder(z_patches)
        state_recon = self.proprio_decoder(z_state)
        
        return {
            "frame_recon": frame_recon,
            "state_recon": state_recon,
        }
    
    def rollout_predictions(self, initial_state, initial_frame, actions, decode_every=1, debug=False, gt_states=None, gt_frames=None):
        """
        Perform autoregressive rollout in latent space with optional decoding.
        
        Args:
            initial_state: (B, state_dim) initial proprioceptive state
            initial_frame: (B, 3, 64, 64) initial frame
            actions: (B, T, action_dim) sequence of actions
            decode_every: decode every N steps (1 = decode all, 0 = decode none)
            debug: if True, compute prediction losses at each step (requires gt_states and gt_frames)
            gt_states: (B, T+1, state_dim) ground truth states for debug mode
            gt_frames: (B, T+1, 3, 64, 64) ground truth frames for debug mode
            
        Returns:
            dict with:
                - latent_trajectory: list of dicts with z_cls, z_patches, z_state for each step
                - decoded_frames: (B, T', 3, 64, 64) if decode_every > 0
                - decoded_states: (B, T', state_dim) if decode_every > 0
                - decode_indices: list of timestep indices that were decoded
                - debug_losses: dict with losses at each decoded step if debug=True
        """
        B, T, _ = actions.shape

        if debug:
            if gt_states is None or gt_frames is None:
                raise ValueError("debug=True requires gt_states and gt_frames to be provided")
            if decode_every <= 0:
                raise ValueError("debug=True requires decode_every > 0 to compute reconstruction losses")

        # Encode initial state
        latents = self.encode_initial_state(initial_state, initial_frame)
        z_cls = latents["z_cls"]
        z_patches = latents["z_patches"]
        z_state = latents["z_state"]

        latent_trajectory = []
        decoded_frames, decoded_states, decode_indices = [], [], []

        if debug:
            debug_losses = {
                "frame_recon_loss": [],
                "state_recon_loss": [],
                "total_recon_loss": [],
                "pred_latent_cls_loss": [],
                "pred_latent_patches_loss": [],
                "pred_latent_state_loss": [],
                "pred_latent_total_loss": [],
                "timesteps": [],
            }

        # Store initial
        latent_trajectory.append({
            "z_cls": z_cls.clone(),
            "z_patches": z_patches.clone(),
            "z_state": z_state.clone(),
        })

        # Decode initial if requested
        if decode_every > 0:
            decoded = self.decode_latents(z_cls, z_patches, z_state)
            decoded_frames.append(decoded["frame_recon"])
            decoded_states.append(decoded["state_recon"])
            decode_indices.append(0)

            if debug:
                frame_loss = F.mse_loss(decoded["frame_recon"], gt_frames[:, 0])
                state_loss = F.mse_loss(decoded["state_recon"], gt_states[:, 0])
                debug_losses["frame_recon_loss"].append(frame_loss.item())
                debug_losses["state_recon_loss"].append(state_loss.item())
                debug_losses["total_recon_loss"].append((frame_loss + state_loss).item())
                debug_losses["timesteps"].append(0)

        # Rollout
        for t in range(T):
            # Predict next latent
            predicted = self.predict_next_latent(z_cls, z_patches, z_state, actions[:, t])
            z_cls = predicted["z_cls_next"]
            z_patches = predicted["z_patches_next"]
            z_state = predicted["z_state_next"]

            latent_trajectory.append({
                "z_cls": z_cls.clone(),
                "z_patches": z_patches.clone(),
                "z_state": z_state.clone(),
            })

            # Decode if requested
            if decode_every > 0 and (t + 1) % decode_every == 0:
                decoded = self.decode_latents(z_cls, z_patches, z_state)
                decoded_frames.append(decoded["frame_recon"])
                decoded_states.append(decoded["state_recon"])
                decode_indices.append(t + 1)

                if debug:
                    gt_idx = t + 1
                    frame_loss = F.mse_loss(decoded["frame_recon"], gt_frames[:, gt_idx])
                    state_loss = F.mse_loss(decoded["state_recon"], gt_states[:, gt_idx])
                    debug_losses["frame_recon_loss"].append(frame_loss.item())
                    debug_losses["state_recon_loss"].append(state_loss.item())
                    debug_losses["total_recon_loss"].append((frame_loss + state_loss).item())

                    # ---- Prediction latent error ----
                    # Encode ground-truth next step
                    with torch.no_grad():
                        gt_latents = self.encode_initial_state(gt_states[:, gt_idx], gt_frames[:, gt_idx])
                    z_cls_gt = gt_latents["z_cls"]
                    z_patches_gt = gt_latents["z_patches"]
                    z_state_gt = gt_latents["z_state"]

                    pred_cls_loss = F.mse_loss(z_cls, z_cls_gt)
                    pred_patches_loss = F.mse_loss(z_patches, z_patches_gt)
                    pred_state_loss = F.mse_loss(z_state, z_state_gt)
                    pred_total = (pred_cls_loss + pred_patches_loss + pred_state_loss).item() / 3.0

                    debug_losses["pred_latent_cls_loss"].append(pred_cls_loss.item())
                    debug_losses["pred_latent_patches_loss"].append(pred_patches_loss.item())
                    debug_losses["pred_latent_state_loss"].append(pred_state_loss.item())
                    debug_losses["pred_latent_total_loss"].append(pred_total)
                    debug_losses["timesteps"].append(gt_idx)

        result = {
            "latent_trajectory": latent_trajectory,
            "decode_indices": decode_indices,
        }

        if decoded_frames:
            result["decoded_frames"] = torch.stack(decoded_frames, dim=1)
            result["decoded_states"] = torch.stack(decoded_states, dim=1)

        if debug:
            result["debug_losses"] = debug_losses

        return result

    def rollout_predictions_from_latents(self, initial_latents, actions, decode_every=1, debug=False, gt_states=None, gt_frames=None):
        """
        Perform autoregressive rollout starting from latent representations (no initial encoding).
        
        This method is designed for efficient MPC replanning where you want to avoid
        the encode-decode cycle. It takes latent states directly and rolls out predictions.
        
        Args:
            initial_latents: dict with:
                - z_cls: (B, D) initial cls token
                - z_patches: (B, N, D) initial patches
                - z_state: (B, D) initial state
            actions: (B, T, action_dim) sequence of actions
            decode_every: decode every N steps (1 = decode all, 0 = decode none)
            debug: if True, compute prediction losses at each step (requires gt_states and gt_frames)
            gt_states: (B, T+1, state_dim) ground truth states for debug mode
            gt_frames: (B, T+1, 3, 64, 64) ground truth frames for debug mode
            
        Returns:
            dict with:
                - latent_trajectory: list of dicts with z_cls, z_patches, z_state for each step
                - decoded_frames: (B, T', 3, 64, 64) if decode_every > 0
                - decoded_states: (B, T', state_dim) if decode_every > 0
                - decode_indices: list of timestep indices that were decoded
                - debug_losses: dict with losses at each decoded step if debug=True
        """
        B, T, _ = actions.shape

        if debug:
            if gt_states is None or gt_frames is None:
                raise ValueError("debug=True requires gt_states and gt_frames to be provided")
            if decode_every <= 0:
                raise ValueError("debug=True requires decode_every > 0 to compute reconstruction losses")

        # Extract initial latents
        z_cls = initial_latents["z_cls"]
        z_patches = initial_latents["z_patches"]
        z_state = initial_latents["z_state"]

        latent_trajectory = []
        decoded_frames, decoded_states, decode_indices = [], [], []

        if debug:
            debug_losses = {
                "frame_recon_loss": [],
                "state_recon_loss": [],
                "total_recon_loss": [],
                "pred_latent_cls_loss": [],
                "pred_latent_patches_loss": [],
                "pred_latent_state_loss": [],
                "pred_latent_total_loss": [],
                "timesteps": [],
            }

        # Store initial latents
        latent_trajectory.append({
            "z_cls": z_cls.clone(),
            "z_patches": z_patches.clone(),
            "z_state": z_state.clone(),
        })

        # Decode initial if requested
        if decode_every > 0:
            decoded = self.decode_latents(z_cls, z_patches, z_state)
            decoded_frames.append(decoded["frame_recon"])
            decoded_states.append(decoded["state_recon"])
            decode_indices.append(0)

            if debug:
                frame_loss = F.mse_loss(decoded["frame_recon"], gt_frames[:, 0])
                state_loss = F.mse_loss(decoded["state_recon"], gt_states[:, 0])
                debug_losses["frame_recon_loss"].append(frame_loss.item())
                debug_losses["state_recon_loss"].append(state_loss.item())
                debug_losses["total_recon_loss"].append((frame_loss + state_loss).item())
                debug_losses["timesteps"].append(0)

        # Rollout
        for t in range(T):
            # Predict next latent
            predicted = self.predict_next_latent(z_cls, z_patches, z_state, actions[:, t])
            z_cls = predicted["z_cls_next"]
            z_patches = predicted["z_patches_next"]
            z_state = predicted["z_state_next"]

            latent_trajectory.append({
                "z_cls": z_cls.clone(),
                "z_patches": z_patches.clone(),
                "z_state": z_state.clone(),
            })

            # Decode if requested
            if decode_every > 0 and (t + 1) % decode_every == 0:
                decoded = self.decode_latents(z_cls, z_patches, z_state)
                decoded_frames.append(decoded["frame_recon"])
                decoded_states.append(decoded["state_recon"])
                decode_indices.append(t + 1)

                if debug:
                    gt_idx = t + 1
                    frame_loss = F.mse_loss(decoded["frame_recon"], gt_frames[:, gt_idx])
                    state_loss = F.mse_loss(decoded["state_recon"], gt_states[:, gt_idx])
                    debug_losses["frame_recon_loss"].append(frame_loss.item())
                    debug_losses["state_recon_loss"].append(state_loss.item())
                    debug_losses["total_recon_loss"].append((frame_loss + state_loss).item())

                    # ---- Prediction latent error ----
                    # Encode ground-truth next step
                    with torch.no_grad():
                        gt_latents = self.encode_initial_state(gt_states[:, gt_idx], gt_frames[:, gt_idx])
                    z_cls_gt = gt_latents["z_cls"]
                    z_patches_gt = gt_latents["z_patches"]
                    z_state_gt = gt_latents["z_state"]

                    pred_cls_loss = F.mse_loss(z_cls, z_cls_gt)
                    pred_patches_loss = F.mse_loss(z_patches, z_patches_gt)
                    pred_state_loss = F.mse_loss(z_state, z_state_gt)
                    pred_total = (pred_cls_loss + pred_patches_loss + pred_state_loss).item() / 3.0

                    debug_losses["pred_latent_cls_loss"].append(pred_cls_loss.item())
                    debug_losses["pred_latent_patches_loss"].append(pred_patches_loss.item())
                    debug_losses["pred_latent_state_loss"].append(pred_state_loss.item())
                    debug_losses["pred_latent_total_loss"].append(pred_total)
                    debug_losses["timesteps"].append(gt_idx)

        result = {
            "latent_trajectory": latent_trajectory,
            "decode_indices": decode_indices,
        }

        if decoded_frames:
            result["decoded_frames"] = torch.stack(decoded_frames, dim=1)
            result["decoded_states"] = torch.stack(decoded_states, dim=1)

        if debug:
            result["debug_losses"] = debug_losses

        return result



