import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import LambdaLR

from src.components.decoder import IDMDecoder, MeNet6Decoder, IDMDecoderConv
from src.losses import TemporalVCRegLossOptimized as TemporalVCRegLoss
from src.components.encoder import MeNet6, Expander2D
from src.components.predictor import ConvPredictor


class JEPA(L.LightningModule):

    def __init__(
        self,
        # Architecture hyperparameters
        emb_dim: int = 128,
        # Loss coefficients
        var_coeff: float = 54.5,
        cov_coeff: float = 15.5,
        smooth_coeff: float = 0.1,
        idm_coeff: float = 5.2,
        tvcreg_coeff: float = 1.0,
        prediction_cost_discount: float = 0.99,
        isometry_coeff: float = 1e-1,
        # Optimization hyperparameters
        initial_lr_encoder: float = 2e-3,
        final_lr_encoder: float = 1e-5,
        weight_decay_encoder: float = 1e-4,
        initial_lr_decoder: float = 1e-3,
        final_lr_decoder: float = 1e-5,
        weight_decay_decoder: float = 0,
        initial_lr_predictor: float = 1e-3,
        final_lr_predictor: float = 1e-5,
        weight_decay_predictor: float = 0,
        warmup_steps: int = 1000,
        # Misc
        compile: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Encoders
        self.visual_encoder = MeNet6(input_channels=3)
        self.proprio_encoder = Expander2D(target_shape=(26, 26), out_channels=2, use_batchnorm=True)
        self.action_encoder = Expander2D(target_shape=(26, 26), out_channels=2, use_batchnorm=True)

        # Decoders
        self.visual_decoder = MeNet6Decoder(out_channels=3)
        self.idm_decoder = IDMDecoderConv(input_channels=36, output_dim=2)

        # Predictor
        self.predictor = ConvPredictor(
            in_channels=20,      # 16 visual + 2 proprio + 2 action
            hidden_channels=32,
            out_channels=18,     # 16 visual + 2 proprio
        )

        # Temporal VCReg
        self.tvcreg_proj_1 = nn.Sequential(
            nn.Conv2d(18, 36, kernel_size=3, padding=1),
        )
        self.tvcreg_proj_2 = nn.Sequential(
            nn.Linear(26*26, 26*26*2),
        )
        self.tvcreg_loss = TemporalVCRegLoss(
            var_coeff=self.hparams.var_coeff,
            cov_coeff=self.hparams.cov_coeff,
        )

        # Optional compilation
        if compile:
            self.visual_encoder = torch.compile(self.visual_encoder)
            self.visual_decoder = torch.compile(self.visual_decoder)
            self.predictor = torch.compile(self.predictor)
            self.idm_decoder = torch.compile(self.idm_decoder)
            self.tvcreg_proj_1 = torch.compile(self.tvcreg_proj_1)
            self.tvcreg_proj_2 = torch.compile(self.tvcreg_proj_2)

    # ============================================================================
    # ENCODING METHODS
    # ============================================================================

    def encode_state(self, state, frame):
        state_dim = state.dim()
        if state_dim == 3:
            B, T, _ = state.shape
            # Flatten for encoding
            state = state.flatten(0, 1)
            frame = frame.flatten(0, 1)

        # Encode
        if state.shape[-1] > 2:
            state = state[..., 2:]  # Drop x,y info
        expanded_state = self.proprio_encoder(state)    # (B*T, 2, 26, 26)
        z = self.visual_encoder(frame)                  # (B*T, 16, 26, 26)
        z = torch.cat([z, expanded_state], dim=1)       # (B*T, 18, 26, 26)

        if state_dim == 3:
            # Reshape back
            z = z.view(B, T, 18, 26, 26)  # (B, T, 18, 26, 26)

        return z

    def encode_isometry(self, z):
        B, T, C, H, W = z.shape
        if C == 16:
            z_proprio = torch.zeros(B, T, 2, H, W, device=z.device, dtype=z.dtype)
            z = torch.cat([z, z_proprio], dim=2)  # (B, T, 18, H, W)
        z = z.flatten(0, 1)
        h = self.isometry(z)  # (B*T, emb_dim)
        h = h.view(B, T, -1)  # (B, T, emb_dim)
        return h

    def isometry(self, z):
        return z

    # ============================================================================
    # DECODING METHODS
    # ============================================================================

    def decode_visual(self, z):
        z = z.flatten(0, 1)
        recon_frame = self.visual_decoder(z[:, :16])
        return recon_frame

    # ============================================================================
    # PREDICTION METHODS
    # ============================================================================

    def predict_state(self, z, action):
        """
        z: (B, C, H, W)
        action: (B, action_dim)
        returns: z_pred (B, C, H, W)
        """
        action_expanded = self.action_encoder(action)
        x = torch.cat([z, action_expanded], dim=1)
        z_pred = self.predictor(x)
        return z_pred

    # ============================================================================
    # DISTANCE METHODS
    # ============================================================================

    def distance(self, h1, h2):
        # z: (B, T, D)
        return (h1 - h2).pow(2).sum(-1)

    # ============================================================================
    # TRAINING STEP
    # ============================================================================

    def shared_step(self, batch, batch_idx):
        states, frames, actions = batch
        B, T, _ = actions.shape

        # Drop x,y infomation
        states = states[..., 2:]

        # Encode
        z = self.encode_state(states, frames)  # (B, T, 18, 26, 26)

        # Decode
        recon_frame = self.decode_visual(z.detach())
        recon_loss = F.mse_loss(recon_frame, frames.flatten(0, 1))

        # Predict future states
        prediction_loss = None
        z_current = z[:, 0]  # (B, 18, 26, 26) - initial state
        for t in range(T):
            # Predict next state using current state and action at time t
            z_next_pred = self.predict_state(z_current, actions[:, t])  # (B, 18, 26, 26)
            _prediction_loss = F.mse_loss(z_next_pred, z[:, t+1]) * (self.hparams.prediction_cost_discount ** t)
            prediction_loss = _prediction_loss if prediction_loss is None else prediction_loss + _prediction_loss

        # Smoothness loss
        smooth_loss = F.mse_loss(z[:, 1:], z[:, :-1])

        # Isometry
        h = self.encode_isometry(z)  # (B, T, emb_dim)
        # Compute pairwise distances
        h_pairs = self.unique_state_pairs(h)  # (B, num_pairs, 2, emb_dim)
        h_i = h_pairs[:, :, 0, :]
        h_j = h_pairs[:, :, 1, :]
        pred_distances = self.distance(h_i, h_j)  # (B, num_pairs)
        # Compute target distances
        time_indices = torch.arange(T+1, device=h_pairs.device, dtype=h_pairs.dtype)
        time_pairs = self.unique_state_pairs(time_indices.unsqueeze(0).unsqueeze(-1))
        t_i = time_pairs[:, :, 0, 0]
        t_j = time_pairs[:, :, 1, 0]
        target_distances = (t_i - t_j).abs()
        # Compute isometry loss with cosine similarity
        cosine_similarity = F.cosine_similarity(pred_distances, target_distances, dim=-1)
        isometry_loss = (1 - cosine_similarity).mean()

        # Reshape back for temporal VCReg
        z_proj = self.tvcreg_proj_1(z.flatten(0, 1))  # (B*(T+1), 36, 26, 26)
        z_proj = z_proj.view(B, T+1, 36, 26, 26).permute(0, 2, 1, 3, 4)  # (B, 36, T+1, 26, 26)
        z_proj = z_proj.flatten(0, 1)  # (B*36, T+1, 26, 26)
        z_proj = z_proj.view(B * 36, T+1, 26*26)  # (B*36, T+1, 26*26)
        z_proj = self.tvcreg_proj_2(z_proj)  # (B*36, T+1, 26*26*2)
        loss_tvcreg = self.tvcreg_loss(z_proj)

        # Decode action using inverse dynamics model
        idm_in_1 = z[:, :-1].flatten(0, 1)
        idm_in_2 = z[:, 1:].flatten(0, 1)
        idm_in = torch.cat([idm_in_1, idm_in_2], dim=1)
        idm_pred = self.idm_decoder(idm_in)
        actions_expanded = actions.flatten(0, 1)
        actions_expanded = self.action_encoder(actions_expanded)
        idm_loss = F.mse_loss(idm_pred, actions_expanded)

        # Compute total loss
        jepa_loss = prediction_loss + \
                    smooth_loss * self.hparams.smooth_coeff + \
                    isometry_loss * self.hparams.isometry_coeff + \
                    loss_tvcreg['loss'] * self.hparams.tvcreg_coeff + \
                    idm_loss * self.hparams.idm_coeff
        loss = recon_loss + jepa_loss

        return {
            "loss": loss,
            "jepa_loss": jepa_loss,
            "recon_loss": recon_loss,
            "prediction_loss": prediction_loss,
            "smooth_loss": smooth_loss,
            "isometry_loss": isometry_loss,
            "loss_tvcreg": loss_tvcreg['loss'],
            'loss_tvcreg_var': loss_tvcreg['var-loss'],
            'loss_tvcreg_cov': loss_tvcreg['cov-loss'],
            "lr_encoder": self.trainer.optimizers[0].param_groups[0]['lr'],
            "lr_decoder": self.trainer.optimizers[0].param_groups[1]['lr'],
            "lr_predictor": self.trainer.optimizers[0].param_groups[2]['lr'],
            "idm_loss": idm_loss,
        }

    def training_step(self, batch, batch_idx):
        outs = self.shared_step(batch, batch_idx)
        self.log_dict({f"train/{k}": v for k, v in outs.items()}, prog_bar=True, sync_dist=True)
        return outs["loss"]

    def validation_step(self, batch, batch_idx):
        outs = self.shared_step(batch, batch_idx)
        self.log_dict({f"val/{k}": v for k, v in outs.items()}, prog_bar=True, sync_dist=True)
        return outs["loss"]

    # ============================================================================
    # OPTIMIZER CONFIGURATION
    # ============================================================================

    def configure_optimizers(self):
        # Optimizer with 3 param groups
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(self.visual_encoder.parameters()) +
                              list(self.proprio_encoder.parameters()) +
                              list(self.action_encoder.parameters()) +
                              list(self.tvcreg_proj_1.parameters()) +
                              list(self.tvcreg_proj_2.parameters()) +
                              list(self.idm_decoder.parameters()),
                    "lr": self.hparams.initial_lr_encoder,
                    "weight_decay": self.hparams.weight_decay_encoder,
                },
                {
                    "params": list(self.visual_decoder.parameters()),
                    "lr": self.hparams.initial_lr_decoder,
                    "weight_decay": self.hparams.weight_decay_decoder,
                },
                {
                    "params": list(self.predictor.parameters()),
                    "lr": self.hparams.initial_lr_predictor,
                    "weight_decay": self.hparams.weight_decay_predictor,
                },
            ]
        )

        # LR schedule functions (one per param group)
        def make_lr_lambda(initial_lr, final_lr):
            """Warmup + cosine schedule for one param group."""
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
                min_ratio = final_lr / initial_lr
                return max(min_ratio, 0.5 * (1 + math.cos(math.pi * progress)))
            return lr_lambda

        lr_lambdas = [
            make_lr_lambda(self.hparams.initial_lr_encoder, self.hparams.final_lr_encoder),
            make_lr_lambda(self.hparams.initial_lr_decoder, self.hparams.final_lr_decoder),
            make_lr_lambda(self.hparams.initial_lr_predictor, self.hparams.final_lr_predictor),
        ]

        # Single scheduler with multiple LR-lambdas
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambdas)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    @staticmethod
    def unique_state_pairs(states):
        B, T, D = states.shape
        idx_i, idx_j = torch.triu_indices(T, T, offset=1)
        s_i = states[:, idx_i]
        s_j = states[:, idx_j]
        return torch.stack([s_i, s_j], dim=2)