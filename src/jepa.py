import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import LambdaLR

from src.components.decoder import MeNet6Decoder, IDMDecoderConv, PositionDecoder
from src.losses import TemporalVCRegLossOptimized as TemporalVCRegLoss
from src.components.encoder import MeNet6, Expander2D, SmoothMeNet6
from src.components.predictor import ConvPredictor
from src.components.rl import IsometricQLearning


class JEPA(L.LightningModule):
    """
    JEPA with pure isometric value function.
    
    Key insight: V(z, g) = -distance(z, g) in isometric space.
    Super simple, no overcomplicated MLP heads!
    """

    def __init__(
        self,
        # ============================================================================
        # Architecture hyperparameters
        # ============================================================================
        emb_dim: int = 128,
        
        # ============================================================================
        # JEPA Loss coefficients
        # ============================================================================
        var_coeff: float = 15.0,
        cov_coeff: float = 1.0,
        smooth_coeff: float = 0.0,
        idm_coeff: float = 1.0,
        tvcreg_coeff: float = 1.0,
        prediction_cost_discount: float = 0.95,
        isometry_coeff: float = 0.1,
        entropy_coeff: float = 0.1,
        
        # ============================================================================
        # Value function coefficients
        # ============================================================================
        value_coeff: float = 1.0,  # Weight of value loss
        
        # ============================================================================
        # Value function hyperparameters
        # ============================================================================
        value_gamma: float = 0.99,
        value_tau: float = 0.005,
        value_expectile: float = 0.7,
        value_hindsight_ratio: float = 0.8,
        value_update_freq: int = 2,
        
        # ============================================================================
        # Optimization hyperparameters
        # ============================================================================
        initial_lr_encoder: float = 2e-3,
        final_lr_encoder: float = 1e-6,
        weight_decay_encoder: float = 1e-5,
        initial_lr_decoder: float = 1e-3,
        final_lr_decoder: float = 1e-5,
        weight_decay_decoder: float = 0.0,
        initial_lr_predictor: float = 1e-3,
        final_lr_predictor: float = 1e-6,
        weight_decay_predictor: float = 1e-5,
        initial_lr_value: float = 3e-4,
        final_lr_value: float = 1e-5,
        weight_decay_value: float = 1e-4,
        warmup_steps: int = 1000,
        
        # ============================================================================
        # Misc
        # ============================================================================
        compile: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ========================================================================
        # JEPA Components (unchanged)
        # ========================================================================
        self.visual_encoder = SmoothMeNet6(input_channels=3)
        self.proprio_encoder = Expander2D(target_shape=(26, 26), out_channels=2, use_batchnorm=False)
        self.action_encoder = Expander2D(target_shape=(26, 26), out_channels=2, use_batchnorm=False)
        
        self.visual_decoder = MeNet6Decoder(out_channels=3)
        self.idm_decoder = IDMDecoderConv(input_channels=36, output_dim=2)
        self.position_decoder = PositionDecoder(in_channels=16)
        self.predictor = ConvPredictor(
            in_channels=20,
            hidden_channels=32,
            out_channels=18,
        )
        
        # self.tvcreg_proj = nn.Sequential(
        #     nn.Conv2d(18, 36, kernel_size=3, padding=1),
        # )
        self.tvcreg_loss = TemporalVCRegLoss(
            var_coeff=self.hparams.var_coeff,
            cov_coeff=self.hparams.cov_coeff,
        )
        self.cosine_loss = lambda x , y: 1 - F.cosine_similarity(x, y, dim=-1).mean()

        # Optional compilation
        if compile:
            self.visual_encoder = torch.compile(self.visual_encoder)
            self.visual_decoder = torch.compile(self.visual_decoder)
            self.predictor = torch.compile(self.predictor)
            self.idm_decoder = torch.compile(self.idm_decoder)
            self.position_decoder = torch.compile(self.position_decoder)

    # ============================================================================
    # ENCODING (unchanged)
    # ============================================================================

    def encode_state(self, state, frame):
        state_dim = state.dim()
        if state_dim == 3:
            B, T, _ = state.shape
            state = state.flatten(0, 1)
            frame = frame.flatten(0, 1)

        if state.shape[-1] > 2:
            state = state[..., 2:]
        expanded_state = self.proprio_encoder(state)
        z = self.visual_encoder(frame)
        z = torch.cat([z, expanded_state], dim=1)

        if state_dim == 3:
            z = z.view(B, T, 18, 26, 26)

        return z

    def encode_isometry(self, z):
        B, T, C, H, W = z.shape
        if C == 16:
            z_proprio = torch.zeros(B, T, 2, H, W, device=z.device, dtype=z.dtype)
            z = torch.cat([z, z_proprio], dim=2)
        z = z.flatten(0, 1)
        h = self.isometry(z)
        h = h.view(B, T, -1)
        return h

    def isometry(self, z):
        return z

    def decode_visual(self, z):
        z = z.flatten(0, 1)
        recon_frame = self.visual_decoder(z[:, :16])
        return recon_frame

    def predict_state(self, z, action):
        """Predict next state using JEPA predictor."""
        action_expanded = self.action_encoder(action)
        x = torch.cat([z, action_expanded], dim=1)
        z_pred = self.predictor(x)
        return z_pred

    def distance(self, h1, h2):
        return (h1 - h2).pow(2).sum(-1)

    # ============================================================================
    # GOAL SAMPLING
    # ============================================================================
    
    def sample_goals(self, z_states: torch.Tensor, strategy: str = 'future') -> torch.Tensor:
        """Sample goal states from trajectory."""
        B, T_plus_1, C, H, W = z_states.shape
        
        if strategy == 'final':
            return z_states[:, -1]
        elif strategy == 'future':
            min_horizon = min(3, T_plus_1 - 1)
            t_goals = torch.randint(min_horizon, T_plus_1, (B,), device=z_states.device)
            z_goals = z_states[torch.arange(B, device=z_states.device), t_goals]
            return z_goals
        elif strategy == 'random':
            t_goals = torch.randint(0, T_plus_1, (B,), device=z_states.device)
            z_goals = z_states[torch.arange(B, device=z_states.device), t_goals]
            return z_goals
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # ============================================================================
    # TRAINING STEP
    # ============================================================================

    def shared_step(self, batch, batch_idx, stage='train'):
        states, frames, actions = batch
        B, T, _ = actions.shape

        states = states[..., 2:]  # (B, T+1, 2)
        positions = states[..., :2]  # (B, T+1, 2)

        # ========================================================================
        # JEPA Forward Pass (unchanged)
        # ========================================================================
        z = self.encode_state(states, frames)  # (B, T+1, 18, 26, 26)

        # # -----------------------------------------------------------
        # #   Global Repulsion Loss (for far-away states)
        # # -----------------------------------------------------------
        # z_vis = z[:, :, :16, :, :]  # (B, T+1, 16, 26, 26)

        # # collapse spatial dims → vector embeddings
        # z_vec = z_vis.mean(dim=(3,4)).flatten(0,1)     # (B*(T+1), 16)

        # # sample arbitrary global pairs
        # N = z_vec.shape[0]
        # perm = torch.randperm(N, device=z.device)
        # half = N // 2
        # z1 = z_vec[perm[:half]]
        # z2 = z_vec[perm[half: 2*half]]

        # # repulsion margin
        # margin = 2.0
        # d = (z1 - z2).pow(2).sum(dim=1)
        # repulsion_loss = F.relu(margin - d).mean()

        # Predict position
        pos_pred = self.position_decoder(z.flatten(0,1)[:, :16])[0]
        pos_loss = F.mse_loss(
            pos_pred,
            positions.flatten(0,1),
        )


        # Reconstruction
        recon_frame = self.decode_visual(z.detach())
        recon_loss = F.mse_loss(recon_frame, frames.flatten(0, 1))

        # Prediction
        prediction_loss = None
        z_current = z[:, 0]
        for t in range(T):
            z_next_pred = self.predict_state(z_current, actions[:, t])
            _prediction_loss = F.mse_loss(z_next_pred, z[:, t+1].detach()) * (
                self.hparams.prediction_cost_discount ** t
            )
            prediction_loss = _prediction_loss if prediction_loss is None else prediction_loss + _prediction_loss
            z_current = z_next_pred  # Use predicted state for next step

        # Isometry
        # h = self.encode_isometry(z)
        # h_pairs = self.unique_state_pairs(h)
        # h_i = h_pairs[:, :, 0, :]
        # h_j = h_pairs[:, :, 1, :]
        # pred_distances = self.distance(h_i, h_j)
        
        # time_indices = torch.arange(T+1, device=h_pairs.device, dtype=h_pairs.dtype)
        # time_pairs = self.unique_state_pairs(time_indices.unsqueeze(0).unsqueeze(-1))
        # t_i = time_pairs[:, :, 0, 0]
        # t_j = time_pairs[:, :, 1, 0]
        # target_distances = (t_i - t_j).abs()
        
        # cosine_similarity = F.cosine_similarity(pred_distances, target_distances / 500.0, dim=-1)
        # isometry_loss = (1 - cosine_similarity).mean()
        isometry_loss = torch.tensor(0.0, device=z.device)

        # Temporal VCReg
        z_proj = z.flatten(0, 1)
        z_proj = z_proj.view(B, T+1, 18, 26, 26).permute(0, 2, 1, 3, 4)
        z_proj = z_proj.flatten(0, 1)
        z_proj = z_proj.view(B * 18, T+1, 26*26)
        loss_tvcreg = self.tvcreg_loss(z_proj)

        # Inverse dynamics
        idm_in_1 = z[:, :-1].flatten(0, 1)
        idm_in_2 = z[:, 1:].flatten(0, 1)
        idm_in = torch.cat([idm_in_1, idm_in_2], dim=1)
        idm_pred = self.idm_decoder(idm_in)
        # idm_loss = F.mse_loss(idm_pred, actions.flatten(0, 1))
        idm_loss = self.cosine_loss(idm_pred, actions.flatten(0, 1))

        # Smooth loss
        smooth_loss = self.cosine_loss(z[:, 1:], z[:, :-1])

        # ========================================================================
        # Total Loss
        # ========================================================================
        jepa_loss = (
            prediction_loss +
            smooth_loss * self.hparams.smooth_coeff +
            isometry_loss * self.hparams.isometry_coeff +
            loss_tvcreg['loss'] * self.hparams.tvcreg_coeff +
            idm_loss * self.hparams.idm_coeff + 
            pos_loss * self.hparams.entropy_coeff
        )
        
        total_loss = recon_loss + jepa_loss

        # ========================================================================
        # Metrics
        # ========================================================================
        metrics = {
            "loss": total_loss,
            "jepa_loss": jepa_loss,
            "recon_loss": recon_loss,
            "prediction_loss": prediction_loss,
            "smooth_loss": smooth_loss,
            "isometry_loss": isometry_loss,
            "loss_tvcreg": loss_tvcreg['loss'],
            'loss_tvcreg_var': loss_tvcreg['var-loss'],
            'loss_tvcreg_cov': loss_tvcreg['cov-loss'],
            "idm_loss": idm_loss,
            "pos_loss": pos_loss,
        }
        
        if stage == 'train':
            metrics.update({
                "lr_encoder": self.trainer.optimizers[0].param_groups[0]['lr'],
                "lr_decoder": self.trainer.optimizers[0].param_groups[1]['lr'],
                "lr_predictor": self.trainer.optimizers[0].param_groups[2]['lr'],
            })

        return metrics

    def training_step(self, batch, batch_idx):
        outs = self.shared_step(batch, batch_idx, stage='train')
        self.log_dict({f"train/{k}": v for k, v in outs.items()}, prog_bar=True, sync_dist=True)        
        return outs["loss"]

    def validation_step(self, batch, batch_idx):
        outs = self.shared_step(batch, batch_idx, stage='val')
        self.log_dict({f"val/{k}": v for k, v in outs.items()}, prog_bar=True, sync_dist=True)
        return outs["loss"]

    # ============================================================================
    # OPTIMIZER
    # ============================================================================

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(self.visual_encoder.parameters()) +
                              list(self.proprio_encoder.parameters()) +
                              list(self.action_encoder.parameters()) +
                              list(self.position_decoder.parameters()) +
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

        def make_lr_lambda(initial_lr, final_lr):
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
    # UTILITIES
    # ============================================================================

    @staticmethod
    def unique_state_pairs(states):
        B, T, D = states.shape
        idx_i, idx_j = torch.triu_indices(T, T, offset=1)
        s_i = states[:, idx_i]
        s_j = states[:, idx_j]
        return torch.stack([s_i, s_j], dim=2)
    
    # ============================================================================
    # PLANNING (for inference)
    # ============================================================================
    
    @torch.no_grad()
    def plan_to_goal(
        self,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
        horizon: int = 10,
        n_candidates: int = 100,
    ):
        """
        Simple shooting planner using learned value function.
        
        Returns best action sequence based on final state value.
        """
        B = z_start.shape[0]
        device = z_start.device
        
        # Sample random action sequences
        actions = torch.randn(n_candidates, horizon, 2, device=device) * 0.5
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # Evaluate each sequence
        values = []
        for i in range(n_candidates):
            z_current = z_start.clone()
            for t in range(horizon):
                z_current = self.predict_state(z_current, actions[i:i+1, t])
            
            # Value of final state
            value = self.value_fn.get_value(z_current, z_goal)['value'].item()
            values.append(value)
        
        # Return best sequence
        values = torch.tensor(values, device=device)
        best_idx = values.argmax()
        
        return {
            'actions': actions[best_idx],
            'value': values[best_idx],
            'all_values': values,
        }