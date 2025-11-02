import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math

from src.models.encoder import MeNet6, Expander2D
from src.models.predictor import ConvPredictor
from src.models.decoder import MeNet6Decoder128, InverseDynamicsModel
from src.losses import TemporalVICRegLoss





class JEPA_TeacherStudent(L.LightningModule):
    def __init__(
        self,
        encoder_weights : str =None,
        encoder_momentum=0.999,
        initial_lr=1e-4,
        final_lr=1e-6,
        weight_decay=1e-4,
        warmup_steps=1000,
        target_shape=(26, 26)
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        # Encoder
        self.encoder_student = MeNet6()
        self.encoder_teacher = MeNet6()
        self.action_encoder = Expander2D(target_shape=target_shape, out_channels=2)
        self.proprio_encoder = Expander2D(target_shape=target_shape, out_channels=2)
        self.decoder = MeNet6Decoder128()

        # Predictor
        self.predictor = ConvPredictor()

        # Inverse Dynamic Modelling
        self.idm = InverseDynamicsModel()

        # If the encoder weights are provided, load them
        if encoder_weights is not None:
            encoder_weights = torch.load(encoder_weights)
            self.encoder_student.load_state_dict(encoder_weights)
            self.encoder_teacher.load_state_dict(encoder_weights)
        # Freeze student encoder parameters
        for param in self.encoder_student.parameters():
            param.requires_grad = False
        self.encoder_student.eval()

        # Losses
        self.prediction_cost        = nn.MSELoss()
        self.idm_cost               = nn.MSELoss()
        self.reconstruction_cost    = nn.MSELoss()


    def encode(
        self,
        state,
        frame,
        action,
        next_state,
        next_frame
    ):
        """Encodes state, action, and next_state into latent representations."""
        z_state         = self.proprio_encoder(state[:, 2:])
        z_frame         = self.encoder_teacher(frame)
        z_action        = self.action_encoder(action)
        with torch.no_grad():
            z_next_state    = self.proprio_encoder(next_state[:, 2:])
            z_next_frame    = self.encoder_student(next_frame)
        z_state = torch.cat([z_state, z_frame], dim=1)
        z_next_state = torch.cat([z_next_state, z_next_frame], dim=1)

        return z_state, z_action, z_next_state

    def forward(self, obs, proprio=None):
        """Encodes an observation (and optional proprio) into a latent representation."""
        z_obs = self.encoder(obs)
        if proprio is not None:
            z_prop = self.proprio_encoder(proprio)
            z = torch.cat([z_obs, z_prop], dim=1)
        else:
            z = z_obs
        return z

    def predict_next(self, z_state, z_action):
        """Predicts the next latent state given context latent and action."""
        z = torch.cat([z_state, z_action], dim=1)
        z_next_state_pred = self.predictor(z)
        return z_next_state_pred
        

    def shared_step(self, batch, batch_idx):
        """
        batch = (obs_t, obs_tp, proprio_t, action_t)
        where:
          obs_t     : context frame
          obs_tp    : target future frame
          proprio_t : optional proprio features
          action_t  : optional actions
        """
        (state, frame), action, (next_state, next_frame) = batch

        # Encode
        z_state, z_action, z_next_state = self.encode(state, frame, action, next_state, next_frame)

        # Predict
        z_next_state_pred = self.predict_next(z_state, z_action)
        action_pred = self.idm(z_state, z_next_state)

        # Compute loss - Stopgrad on z_next_state
        loss_prediction_visual  = self.prediction_cost(
            z_next_state_pred[:, 4:, :, :],
            z_next_state[:, 4:, :, :]
        )
        loss_prediction_proprio = self.prediction_cost(
            z_next_state_pred[:, :4, :, :],
            z_next_state[:, :4, :, :]
        )
        loss_prediction = loss_prediction_visual + loss_prediction_proprio
        loss_idm = self.idm_cost(action_pred, action)
        loss = loss_prediction + loss_idm * 0.1

        # Reconstruct
        frame_recon = self.decoder(z_state.detach())
        # Compute reconstruction loss
        loss_reconstruction = self.reconstruction_cost(frame_recon, frame)

        return {'loss': loss,
                'loss_prediction': loss_prediction,
                'loss_prediction_visual': loss_prediction_visual,
                'loss_prediction_proprio': loss_prediction_proprio,
                'loss_idm': loss_idm,
                'loss_reconstruction': loss_reconstruction
                }
    
    def training_step(self, batch, batch_idx):
        # Get optimizers and schedulers
        opt_jepa, opt_decoder = self.optimizers()
        lr_sched_jepa = self.lr_schedulers()


        loss = self.shared_step(batch, batch_idx)

        self.log_dict({ f'train/{k}': v for k, v in loss.items() }, prog_bar=True, sync_dist=True)

        loss_jepa = loss['loss']
        self.manual_backward(loss_jepa)
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder_teacher.parameters())
            + list(self.predictor.parameters())
            + list(self.idm.parameters())
            + list(self.proprio_encoder.parameters())
            + list(self.action_encoder.parameters()),
            max_norm=1.0
        )
        opt_jepa.step()
        opt_jepa.zero_grad()
        lr_sched_jepa.step()

        with torch.no_grad():
            for student_param, teacher_param in zip(
                self.encoder_student.parameters(),
                self.encoder_teacher.parameters()
            ):
                student_param.data.mul_(self.hparams.encoder_momentum)
                student_param.data.add_(
                    (1 - self.hparams.encoder_momentum) * teacher_param.data
                )

        loss_decoder = loss['loss_reconstruction']
        self.manual_backward(loss_decoder)
        torch.nn.utils.clip_grad_norm_(
            self.decoder.parameters(),
            max_norm=1.0
        )
        opt_decoder.step()
        opt_decoder.zero_grad()

        return loss['loss']
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log_dict({ f'val/{k}': v for k, v in loss.items() }, prog_bar=True, sync_dist=True)
        return loss['loss']

    def configure_optimizers(self):
        # ---------------------------
        # 1️⃣ JEPA Optimizer (main)
        # ---------------------------
        opt_jepa = torch.optim.AdamW(
            list(self.encoder_teacher.parameters())
            + list(self.predictor.parameters())
            + list(self.idm.parameters())
            + list(self.proprio_encoder.parameters())
            + list(self.action_encoder.parameters()),
            lr=self.hparams.initial_lr,
            weight_decay=self.hparams.weight_decay
        )

        def lr_lambda(step: int):
            if self.trainer is None or not hasattr(self.trainer, "estimated_stepping_batches"):
                max_steps = 100_000  # fallback
            else:
                max_steps = self.trainer.estimated_stepping_batches

            # Linear warmup
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

        sched_jepa = {
            "scheduler": LambdaLR(opt_jepa, lr_lambda=lr_lambda),
            "interval": "step",
            "frequency": 1,
        }

        # ---------------------------
        # 2️⃣ Decoder Optimizer (independent)
        # ---------------------------
        opt_decoder = torch.optim.AdamW(
            self.decoder.parameters(),
            lr=5e-5,
            weight_decay=0.0,
        )

        # Return both optimizers
        return [opt_jepa, opt_decoder], [sched_jepa]

    



class JEPA_Regularized(L.LightningModule):
    def __init__(
        self,
        encoder_weights : str =None,
        # Optimization hyperparameters
        initial_lr=1e-4,
        final_lr=1e-6,
        weight_decay=1e-4,
        warmup_steps=1000,
        # Loss hyperparameters
        idm_weight=1.,
        tvcreg_weight=0.1,
        # Model hyperparameters
        target_shape=(26, 26)
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        # Encoder
        self.visual_encoder      = MeNet6()
        self.action_encoder      = Expander2D(target_shape=target_shape, out_channels=2)
        self.proprio_encoder     = Expander2D(target_shape=target_shape, out_channels=2)

        # Predictor
        self.predictor           = ConvPredictor()

        # Decoders
        self.idm                 = InverseDynamicsModel(latent_dim=18*26*26, action_dim=2, hidden_dim=256)
        self.idm                 = torch.compile(self.idm, mode='default')
        self.visual_decoder      = MeNet6Decoder128()

        # Losses
        self.prediction_cost     = nn.MSELoss()
        self.idm_cost            = nn.MSELoss()
        self.reconstruction_cost = nn.MSELoss()
        self.temporal_vcreg      = TemporalVICRegLoss()


    def encode(self, states, frames, actions):
        B, T, _ = actions.shape

        # Flatten batch and time dimensions
        states = states.flatten(0, 1)[:, 2:] # We remove the global x,y position and only keep the velocities
        frames = frames.flatten(0, 1)
        actions = actions.flatten(0, 1)

        # Encode everything
        z_states = self.proprio_encoder(states)
        z_frames = self.visual_encoder(frames)
        z_actions = self.action_encoder(actions)

        # Reshape back to (B, T, C, H, W)
        z_states = z_states.view(B, T+1, 2, 26, 26)
        z_frames = z_frames.view(B, T+1, 16, 26, 26)
        z_actions = z_actions.view(B, T, 2, 26, 26)

        return z_states, z_frames, z_actions

    def forward(self, obs, proprio=None):
        """Encodes an observation (and optional proprio) into a latent representation."""
        z_obs = self.encoder(obs)
        if proprio is not None:
            z_prop = self.proprio_encoder(proprio)
            z = torch.cat([z_obs, z_prop], dim=1)
        else:
            z = z_obs
        return z

    def predict_next(self, z_state, z_action):
        """Predicts the next latent state given context latent and action."""
        z = torch.cat([z_state, z_action], dim=1)
        z_next_state_pred = self.predictor(z)
        return z_next_state_pred
        

    def shared_step(self, batch, batch_idx, optimizer_idx=0):
        """
        batch = (obs_t, obs_tp, proprio_t, action_t)
        where:
          obs_t     : context frame
          obs_tp    : target future frame
          proprio_t : optional proprio features
          action_t  : optional actions
        """
        states, frames, actions = batch
        B, T, _ = actions.shape

        # Encode everything
        z_states, z_frames, z_actions = self.encode(
            states, frames, actions
        )

        # Split between current and next time step
        z_states_curr = z_states[:, :-1]
        z_states_next = z_states[:, 1:]
        z_frames_curr = z_frames[:, :-1]
        z_frames_next = z_frames[:, 1:]

        # Flatten batch and time dimensions
        z_states_curr = z_states_curr.flatten(0,1)
        z_states_next = z_states_next.flatten(0,1)
        z_frames_curr = z_frames_curr.flatten(0,1)
        z_frames_next = z_frames_next.flatten(0,1)
        z_actions     = z_actions.flatten(0,1)

        # A latent state is the concatenation of visual and proprioceptive latents
        z_state_curr = torch.cat([z_states_curr, z_frames_curr], dim=1)
        z_state_next = torch.cat([z_states_next, z_frames_next], dim=1)

        # Predict the next latent state
        z_state_next_pred = self.predict_next(z_state_curr, z_actions)
        loss_prediction   = self.prediction_cost(z_state_next_pred, z_state_next)
        # print("Loss Prediction:", loss_prediction.item())

        # Predict the action taken
        action_pred = self.idm(z_state_curr, z_state_next)
        loss_idm    = self.idm_cost(action_pred, actions.flatten(0,1))
        # print("Loss IDM:", loss_idm.item())

        # Regularization loss
        # Concatenate along channel dimension → (B, T+1, C_total, H, W)
        z_combined = torch.cat([z_states, z_frames], dim=2)
        # Flatten spatial dimensions to match VCReg expected input (B, T+1, D)
        B, T1, C, H, W = z_combined.shape
        z_flat = z_combined.view(B, T1, C * H * W)
        # Compute VCReg loss
        loss_tvcreg = self.temporal_vcreg(z_flat)
        # print("Loss TVCReg:", loss_tvcreg['loss'].item())

        # Total loss
        loss = loss_prediction + self.hparams.idm_weight * loss_idm  + self.hparams.tvcreg_weight * loss_tvcreg['loss']
        # print("Total Loss:", loss.item())
        
        # Reconstruct
        frames_pred = self.visual_decoder(z_combined.flatten(0,1).detach())
        loss_reconstruction = self.reconstruction_cost(frames_pred, frames.flatten(0,1))
        # print("Loss Reconstruction:", loss_reconstruction.item())

        return {'loss': loss,
                'loss_prediction': loss_prediction,
                'loss_idm': loss_idm,
                'loss_tvcreg': loss_tvcreg['loss'],
                'loss_std': loss_tvcreg['std'],
                'loss_cov': loss_tvcreg['cov'],
                'loss_std_t': loss_tvcreg['std_t'],
                'loss_cov_t': loss_tvcreg['cov_t'],
                'loss_reconstruction': loss_reconstruction
                }


    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log_dict({ f'train/{k}': v for k, v in loss.items() }, prog_bar=True, sync_dist=True)
        # print(loss)

        # Get optimizers and schedulers
        opt_jepa, opt_decoder = self.optimizers()
        lr_sched_jepa = self.lr_schedulers()

        loss_jepa = loss['loss']
        self.manual_backward(loss_jepa)
        torch.nn.utils.clip_grad_norm_(
            list(self.visual_encoder.parameters())
            + list(self.predictor.parameters())
            + list(self.idm.parameters())
            + list(self.proprio_encoder.parameters())
            + list(self.action_encoder.parameters()),
            max_norm=1.0
        )
        opt_jepa.step()
        opt_jepa.zero_grad()
        lr_sched_jepa.step()

        loss_decoder = loss['loss_reconstruction']
        self.manual_backward(loss_decoder)
        torch.nn.utils.clip_grad_norm_(
            self.visual_decoder.parameters(),
            max_norm=1.0
        )
        opt_decoder.step()
        opt_decoder.zero_grad()

        return loss['loss']

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log_dict({ f'val/{k}': v for k, v in loss.items() }, prog_bar=True, sync_dist=True)

    
    def configure_optimizers(self):
        # ---------------------------
        # 1️⃣ JEPA Optimizer (main)
        # ---------------------------
        opt_jepa = torch.optim.AdamW(
            list(self.visual_encoder.parameters())
            + list(self.predictor.parameters())
            + list(self.idm.parameters())
            + list(self.proprio_encoder.parameters())
            + list(self.action_encoder.parameters()),
            lr=self.hparams.initial_lr,
            weight_decay=self.hparams.weight_decay
        )

        def lr_lambda(step: int):
            if self.trainer is None or not hasattr(self.trainer, "estimated_stepping_batches"):
                max_steps = 100_000  # fallback
            else:
                max_steps = self.trainer.estimated_stepping_batches

            # Linear warmup
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

        sched_jepa = {
            "scheduler": LambdaLR(opt_jepa, lr_lambda=lr_lambda),
            "interval": "step",
            "frequency": 1,
        }

        # ---------------------------
        # 2️⃣ Decoder Optimizer (independent)
        # ---------------------------
        opt_decoder = torch.optim.AdamW(
            self.visual_decoder.parameters(),
            lr=self.hparams.initial_lr / 5.0,
            weight_decay=0.0,
        )

        # Return both optimizers
        return [opt_jepa, opt_decoder], [sched_jepa]

