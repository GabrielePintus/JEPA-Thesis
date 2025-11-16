import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import LambdaLR


from src.components.decoder import IDMDecoder, MeNet6Decoder
from src.losses import TemporalVCRegLossOptimized as TemporalVCRegLoss
from src.components.encoder import MeNet6, Expander2D
from src.components.predictor import ConvPredictor


class PLDMEncoder(L.LightningModule):

    def __init__(
        self,
        alpha: float = 54.5,
        beta: float = 15.5,
        delta: float = 0.1,
        omega: float = 5.2,
        horizon: int = 16,
        initial_lr_encoder : float = 2e-3,
        final_lr_encoder : float = 1e-5,
        weight_decay_encoder : float = 1e-4,
        initial_lr_decoder : float = 1e-3,
        final_lr_decoder : float = 1e-5,
        weight_decay_decoder : float = 0,
        initial_lr_predictor : float = 1e-3,
        final_lr_predictor : float = 1e-5,
        weight_decay_predictor : float = 0,
        warmup_steps : int = 1000,
        compile : bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()

        # Encoders
        self.visual_encoder = MeNet6(input_channels=3)
        self.proprio_expander = Expander2D(target_shape=(26, 26), out_channels=2)
        self.action_expander = Expander2D(target_shape=(26, 26), out_channels=2)

        # Decoders
        self.visual_decoder = MeNet6Decoder(out_channels=3)
        self.idm = IDMDecoder(input_channels=36, output_dim=2)

        # Predictor
        self.predictor = ConvPredictor(
            in_channels=20,
            hidden_channels=32,
            out_channels=18,
        )
        
        self.proj = nn.Linear(26*26, 26*26*2)
        self.tvcreg_loss = TemporalVCRegLoss(var_coeff=alpha, cov_coeff=beta)

        if compile:
            self.visual_encoder = torch.compile(self.visual_encoder)
            self.visual_decoder = torch.compile(self.visual_decoder)
            self.predictor = torch.compile(self.predictor)
            self.idm = torch.compile(self.idm)
            self.proj = torch.compile(self.proj)


    def encode_state(self, state, frame):
        B, T, _ = state.shape

        # Flatten for encoding
        state = state.flatten(0, 1)
        frame = frame.flatten(0, 1)

        expanded_state = self.proprio_expander(state)
        z = self.visual_encoder(frame)
        z = torch.cat([z, expanded_state], dim=1)

        # Reshape back
        z = z.view(B, T, -1, 26, 26) # (B, T, 16, 26, 26)

        return z
    
    def decode_visual(self, z):
        B, T, C, H, W = z.shape
        z = z.flatten(0,1)
        recon_frame = self.visual_decoder(z[:, :16])
        return recon_frame
    
    def predict_state_parallel(self, z, action):
        print("z: ", z.shape)
        print("action: ", action.shape)
        B, T, C, H, W = z.shape
        print("z input: ", z.shape)
        z = z.flatten(0,1)
        print("z flattened: ", z.shape)
        action_expanded = self.action_expander(action)
        x = torch.cat([z, action_expanded], dim=1)
        print("x input: ", x.shape)
        z_pred = self.predictor(x)
        print("z_pred before view: ", z_pred.shape)
        z_pred = z_pred.view(B, T, C, H, W)
        return z_pred
    def predict_state(self, z, action):
        """
        z: (B, C, H, W)
        action: (B, action_dim)
        returns: z_pred (B, C, H, W)
        """
        action_expanded = self.action_expander(action)
        x = torch.cat([z, action_expanded], dim=1)
        z_pred = self.predictor(x)
        return z_pred
    
    def shared_step(self, batch, batch_idx):
        states, frames, actions = batch
        B, T, _ = actions.shape

        # Drop x,y infomation
        states = states[..., 2:]


        # Encode
        z = self.encode_state(states, frames) # (B, T, 18, 26, 26)

        # Decode
        recon_frame = self.decode_visual(z.detach())
        recon_loss = F.mse_loss(recon_frame, frames.flatten(0,1))

        # Temporal VCReg loss
        z_proj = z.view(B, T+1, 18, 26, 26).permute(0,2,1,3,4)  # (B, C, T, H, W)
        z_proj = z_proj.flatten(0,1) # (B*C, T, H, W)
        z_proj = z_proj.view(B * 18, T+1, 26*26) # (B*C, T, H*W)
        z_proj = self.proj(z_proj)  # (B*C, T, H*W*2)
        loss_tvcreg = self.tvcreg_loss(z_proj)

        # Predictive loss
        z_pred_list = []
        z_current = z[:, 0]  # (B, 18, 26, 26) - initial state
        
        for t in range(T):
            # Predict next state using current state and action at time t
            z_next_pred = self.predict_state(z_current, actions[:, t])  # (B, 18, 26, 26)
            z_pred_list.append(z_next_pred)
            z_current = z_next_pred  # Use predicted state for next step
        
        # Stack predictions: (B, T, 18, 26, 26)
        z_pred = torch.stack(z_pred_list, dim=1)
        
        # Compare predictions with ground truth latents z[:, 1:T+1]
        z_target = z[:, 1:T+1]  # (B, T, 18, 26, 26)
        pred_loss = F.mse_loss(z_pred, z_target)

        # Smoothness loss
        smooth_loss = F.mse_loss(z[:, 1:], z[:, :-1])

        # IDM loss (optional)
        idm_in_1 = z[:, :-1].flatten(0,1)
        idm_in_2 = z[:, 1:].flatten(0,1)
        idm_in = torch.cat([idm_in_1, idm_in_2], dim=1)
        idm_pred = self.idm(idm_in)
        idm_loss = F.mse_loss(idm_pred, actions.flatten(0,1))

        jepa_loss = pred_loss + loss_tvcreg['loss'] + self.hparams.delta * smooth_loss + idm_loss * self.hparams.omega
        loss = recon_loss + jepa_loss

        return {
            "loss": loss,
            "loss_tvcreg": loss_tvcreg['loss'],
            'loss_tvcreg_var': loss_tvcreg['var-loss'],
            'loss_tvcreg_cov': loss_tvcreg['cov-loss'],
            "recon_loss": recon_loss,
            "pred_loss": pred_loss,
            "idm_loss": idm_loss,
            "smooth_loss": smooth_loss,
            "jepa_loss": jepa_loss,
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
        # ----------------------------------------------------
        # 1. OPTIMIZER WITH 3 PARAM GROUPS
        # ----------------------------------------------------
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(self.visual_encoder.parameters()) +
                              list(self.idm.parameters()) +
                              list(self.proj.parameters()),
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

        # ----------------------------------------------------
        # 2. LR SCHEDULE FUNCTIONS (ONE PER PARAM GROUP)
        # ----------------------------------------------------
        total_steps = (
            self.trainer.max_epochs * self.trainer.estimated_stepping_batches
        )
        warmup = self.hparams.warmup_steps

        def make_lr_lambda(initial_lr, final_lr):
            """Warmup + cosine schedule for one param group."""
            def lr_lambda(step):
                # warmup: linear
                if step < warmup:
                    return (step / max(1, warmup))
                # cosine decay
                t = (step - warmup) / max(1, total_steps - warmup)
                return 0.5 * (1 + math.cos(math.pi * min(max(t, 0.0), 1.0)))
            return lr_lambda

        lr_lambdas = [
            make_lr_lambda(self.hparams.initial_lr_encoder,   self.hparams.final_lr_encoder),
            make_lr_lambda(self.hparams.initial_lr_decoder,   self.hparams.final_lr_decoder),
            make_lr_lambda(self.hparams.initial_lr_predictor, self.hparams.final_lr_predictor),
        ]

        # ----------------------------------------------------
        # 3. SINGLE SCHEDULER WITH MULTIPLE LR-LAMBDAS
        # ----------------------------------------------------
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambdas)

        # ----------------------------------------------------
        # 4. LIGHTNING RETURN FORMAT
        # ----------------------------------------------------
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

