import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import LambdaLR


from src.components.decoder import MeNet6Decoder, MeNet6DecoderStrong
from src.losses import TemporalVCRegLossOptimized as TemporalVCRegLoss
from src.components.encoder import MeNet6, Expander2D
from src.components.predictor import ConvPredictor


class PLDMEncoder(L.LightningModule):

    def __init__(
        self,
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
        self.visual_encoder = MeNet6(input_channels=5)
        self.proprio_expander = Expander2D(target_shape=(64, 64), out_channels=2)
        self.action_expander = Expander2D(target_shape=(26, 26), out_channels=2)

        # Decoders
        self.visual_decoder = MeNet6Decoder(out_channels=3)
        # self.visual_decoder = MeNet6DecoderStrong(out_channels=3)

        # Predictor
        self.predictor = ConvPredictor(
            in_channels=18,
            hidden_channels=36,
            out_channels=16,
        )
        
        self.proj = nn.Linear(26*26, 26*26*2)
        self.tvcreg_loss = TemporalVCRegLoss()

        if compile:
            self.visual_encoder = torch.compile(self.visual_encoder)
            self.visual_decoder = torch.compile(self.visual_decoder)
            self.proj = torch.compile(self.proj)


    def encode_state(self, state, frame):
        expanded_state = self.proprio_expander(state)
        x = torch.cat([frame, expanded_state], dim=1)
        z = self.visual_encoder(x)
        return z
    
    def shared_step(self, batch, batch_idx):
        states, frames, actions = batch
        B, T, _ = actions.shape

        # Drop x,y infomation
        states = states[..., 2:]

        # Reshape for encoding
        state = states.flatten(0, 1)
        frame = frames.flatten(0, 1)
        action = actions.flatten(0, 1)

        # Encode
        z = self.encode_state(state, frame)
        action_expanded = self.action_expander(action)

        # Decode
        recon_frame = self.visual_decoder(z.detach())
        recon_loss = F.mse_loss(recon_frame, frame)

        # Temporal VCReg loss
        z_proj = z.view(B, T+1, 16, 26, 26).permute(0,2,1,3,4)  # (B, C, T, H, W)
        z_proj = z_proj.flatten(0, 1) # (B*C, T, H, W)
        z_proj = z_proj.view(B * 16, T+1, 26*26) # (B*C, T, H*W)
        z_proj = self.proj(z_proj) # (B*C, T, H*W*2)
        loss_tvcreg = self.tvcreg_loss(z_proj)

        # Predict future latents
        z_curr = z.view(B, T+1, 16, 26, 26)[:, :-1]  # (B, T, C, H, W)
        z_next = z.view(B, T+1, 16, 26, 26)[:, 1:]   # (B, T, C, H, W)
        z_curr = z_curr.flatten(0,1)  # (B*T, C, H, W)
        z_next = z_next.flatten(0,1)    # (B*T, C, H, W)
        action_expanded = action_expanded.view(B, T, 2, 26, 26).flatten(0,1)  # (B*T, 2, H, W)

        z_curr = torch.cat([z_curr, action_expanded], dim=1)  # (B*T, C+2, H, W)
        z_pred = self.predictor(z_curr)  # (B*T, C, H, W)
        pred_loss = F.mse_loss(z_pred, z_next)


        

        loss = recon_loss + loss_tvcreg['loss'] + pred_loss

        return {
            "loss": loss,
            "loss_tvcreg": loss_tvcreg['loss'],
            'loss_tvcreg_var': loss_tvcreg['var-loss'],
            'loss_tvcreg_cov': loss_tvcreg['cov-loss'],
            "recon_loss": recon_loss,
            "pred_loss": pred_loss,
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
                    "params": list(self.visual_encoder.parameters()) + list(self.proj.parameters()),
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

