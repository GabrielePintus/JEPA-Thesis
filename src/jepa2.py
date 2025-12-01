import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import LambdaLR

from src.components.decoder import IDMDecoderConv
from src.components.encoder import MaskHead, SmoothMeNet6, Expander2D
from src.components.predictor import ConvPredictor


class JEPA(L.LightningModule):
    def __init__(
        self,
        # ============================================================================
        # Architecture hyperparameters
        # ============================================================================
        emb_dim: int = 128,  # unused legacy parameter
        
        # ============================================================================
        # Loss coefficients
        # ============================================================================
        prediction_coeff: float = 1.0,
        prediction_cost_discount: float = 0.95,
        
        # Mask losses
        mask_bg_invariance_coeff: float = 1.0,    # background should be temporally invariant
        mask_alignment_coeff: float = 1.0,         # subject should cover changing regions
        mask_min_area_coeff: float = 0.1,          # subject must cover at least k_min of image
        mask_compactness_coeff: float = 0.1,       # subject should be spatially compact
        mask_sparsity_coeff: float = 0.1,          # encourage small subject area (L1 penalty)
        mask_min_area: float = 0.01,               # minimum fraction for subject
        mask_beta: float = 1.0,                    # temperature for sigmoid (higher = sharper)
        mask_beta_learnable: bool = False,         # whether beta is a learnable parameter
        
        # IDM loss
        idm_coeff: float = 1.0,
        
        # ============================================================================
        # Optimization hyperparameters
        # ============================================================================
        initial_lr_encoder: float = 2e-3,
        final_lr_encoder: float = 1e-6,
        weight_decay_encoder: float = 1e-5,
        
        initial_lr_predictor: float = 1e-3,
        final_lr_predictor: float = 1e-6,
        weight_decay_predictor: float = 1e-5,
        
        warmup_steps: int = 1000,
        
        # ============================================================================
        # Misc
        # ============================================================================
        compile: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ========================================================================
        # Encoders
        # ========================================================================
        self.visual_encoder     = SmoothMeNet6(input_channels=3)
        self.proprio_encoder    = Expander2D(target_shape=(26, 26), out_channels=2, use_batchnorm=False)
        self.action_encoder     = Expander2D(target_shape=(26, 26), out_channels=2, use_batchnorm=False)
        
        # ========================================================================
        # Mask head
        # ========================================================================
        self.mask_head = MaskHead(in_channels=16, hidden_channels=32)
        
        # Temperature parameter for mask sharpness
        # Higher beta → sharper/more binary mask (e.g., beta=5.0)
        # Lower beta → softer/more gradual mask (e.g., beta=0.5)
        if self.hparams.mask_beta_learnable:
            self.mask_beta = nn.Parameter(torch.tensor(self.hparams.mask_beta))
        else:
            self.register_buffer('mask_beta', torch.tensor(self.hparams.mask_beta))
        
        # ========================================================================
        # Subject/Background refinement nets
        # ========================================================================
        self.subject_refine = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
        )
        
        # self.background_refine = nn.Sequential(
        #     nn.Conv2d(16, 16, kernel_size=1, padding=0),
        # )
        self.background_refine = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
        )
        
        # ========================================================================
        # Predictor (predicts subject residual)
        # ========================================================================
        # Input: z_subj (16) + z_bg (16) + action (2) = 34 channels
        # Output: subject residual (16 channels)
        self.predictor = ConvPredictor(
            in_channels=36,
            hidden_channels=32,
            out_channels=18,
        )
        
        # ========================================================================
        # Stabilizer (prevents residual explosion in autoregressive rollouts)
        # ========================================================================
        self.stabilizer = nn.Sequential(
            nn.Conv2d(18, 18, kernel_size=5, padding=2),
        )
        
        # ========================================================================
        # IDM decoder
        # ========================================================================
        # Input: [z_t, z_t+1] = 36 channels (18 + 18)
        self.idm_decoder = IDMDecoderConv(input_channels=36, output_dim=2)

        if compile:
            self.visual_encoder     = torch.compile(self.visual_encoder)
            self.proprio_encoder    = torch.compile(self.proprio_encoder)
            self.action_encoder     = torch.compile(self.action_encoder)
            self.mask_head          = torch.compile(self.mask_head)
            self.subject_refine     = torch.compile(self.subject_refine)
            self.background_refine  = torch.compile(self.background_refine)
            self.predictor          = torch.compile(self.predictor)
            self.stabilizer       = torch.compile(self.stabilizer)
            self.idm_decoder        = torch.compile(self.idm_decoder)

    # ============================================================================
    # ENCODING
    # ============================================================================

    def encode_state(self, state, frame):
        """
        Encode state and frame into latent representation.
        
        Args:
            state: (B, T, 4) or (B, 4) - only last 2 dims (vx, vy) are used
            frame: (B, T, 3, 64, 64) or (B, 3, 64, 64)
        
        Returns:
            z: (B, T, 18, 26, 26) or (B, 18, 26, 26) - latent state
        """
        state_dim = state.dim()
        if state_dim == 3:
            B, T, _ = state.shape
            state = state.flatten(0, 1)
            frame = frame.flatten(0, 1)

        # Extract velocity only
        if state.shape[-1] > 2:
            state = state[..., 2:]  # (B*T, 2) or (B, 2)
        
        # Encode visual and proprio
        expanded_state = self.proprio_encoder(state)  # (B*T, 2, 26, 26)
        z_vis = self.visual_encoder(frame)            # (B*T, 16, 26, 26)
        z = torch.cat([z_vis, expanded_state], dim=1) # (B*T, 18, 26, 26)

        if state_dim == 3:
            z = z.view(B, T, 18, 26, 26)

        return z

    @staticmethod
    def mask_to_centroid(M, threshold=0.9, normalize=True):
        """
        M: (B, 1, H, W) raw mask or heatmap (any value range)
        
        Returns: (B, 2) centroid coordinates in [-1, 1] (cy, cx)
        """
        B, _, H, W = M.shape
        
        # -------------------------------------------------------
        # 1. MIN-MAX normalize mask to [0, 1]
        # -------------------------------------------------------
        if normalize:
            M_min = M.amin(dim=(2,3), keepdim=True)
            M_max = M.amax(dim=(2,3), keepdim=True)
            M = (M - M_min) / (M_max - M_min + 1e-6)
        
        # -------------------------------------------------------
        # 2. Threshold to obtain binary region
        # -------------------------------------------------------
        if threshold is not None:
            binary = (M > threshold).float()   # (B,1,H,W)
        else:
            binary = M                         # soft centroid
        
        # Fallback: if mask is empty, use soft mask
        area = binary.sum(dim=(1,2,3), keepdim=True)
        if (area < 1e-3).any():
            binary = M
            area = binary.sum(dim=(1,2,3), keepdim=True)
        
        # -------------------------------------------------------
        # 3. Create coordinate grid in [-1, 1]
        # -------------------------------------------------------
        ys = torch.linspace(-1, 1, H, device=M.device).view(1,1,H,1)  # shape (1,1,H,1)
        xs = torch.linspace(-1, 1, W, device=M.device).view(1,1,1,W)  # shape (1,1,1,W)
        
        # -------------------------------------------------------
        # 4. Compute centroid (weighted center of mass)
        # -------------------------------------------------------
        cy = (binary * ys).sum(dim=(2,3)) / (area.squeeze(-1).squeeze(-1) + 1e-6)
        cx = (binary * xs).sum(dim=(2,3)) / (area.squeeze(-1).squeeze(-1) + 1e-6)

        coords = torch.stack([cy, cx], dim=1)  # (B, 2)
        return coords

    def estimate_position(self, z, threshold=0.9):
        """
        Estimate 2D position from latent state z.
        
        Args:
            z: (B, 18, H, W) latent state
        Returns:
            coords: (B, 2) estimated (y, x) position in [-1, 1]
        """
        if z.dim() == 3:
            z = z.unsqueeze(0)
        z_vis = z[:, :16]                       # (B, 16, H, W)
        M_logits = self.mask_head(z_vis)        # (B, 1, H, W)
        M = torch.sigmoid(self.mask_beta * M_logits)  # Apply beta scaling
        coords = self.mask_to_centroid(M, threshold=threshold, normalize=True)  # (B, 2)
        return coords

    def compute_mask_and_split(self, z_vis):
        """
        Compute mask and split visual features into subject and background.
        
        Args:
            z_vis: (B, 16, H, W) visual features
        
        Returns:
            M: (B, 1, H, W) mask in [0, 1]
            z_subj: (B, 16, H, W) refined subject features
            z_bg: (B, 16, H, W) refined background features
        """
        # Compute mask with temperature scaling
        M_logits = self.mask_head(z_vis)
        M = torch.sigmoid(self.mask_beta * M_logits)  # Apply beta scaling
        
        # Split and refine
        z_subj_raw = M * z_vis
        z_bg_raw = (1.0 - M) * z_vis
        
        z_subj = self.subject_refine(z_subj_raw)
        z_bg = self.background_refine(z_bg_raw)
        
        return M, z_subj, z_bg

    def predict_state(self, z, action):
        """
        Predict next state using subject/background decomposition.
        
        Args:
            z: (B, 18, 26, 26) - current latent state
            action: (B, 2) - action
        
        Returns:
            z_next: (B, 18, 26, 26) - predicted next latent state
        """
        _, C, _, _ = z.shape
        assert C == 18, f"Expected 18 channels (16 visual + 2 vel), got {C}"

        # Split into visual + velocity
        z_vis = z[:, :16]   # (B, 16, H, W)
        z_vel = z[:, 16:]   # (B, 2, H, W)

        # Compute mask and split (with refinement)
        _, z_subj, z_bg = self.compute_mask_and_split(z_vis)

        # Background is static geometry - detach from prediction gradient
        z_bg_detached = z_bg.detach()

        # Expand action to spatial map
        action_expanded = self.action_encoder(action)  # (B, 2, H, W)

        # Predictor input: subject + static background + action
        pred_input = torch.cat([z_subj, z_bg_detached, z_vel, action_expanded], dim=1)  # (B, 36, H, W)

        # Predict RESIDUAL for subject
        z_subj_residual = self.predictor(pred_input)  # (B, 18, H, W)

        # Combine: add residual to current background
        z_subj_residual[:, :16] = z_bg + z_subj_residual[:, :16]  # (B, 16, H, W)

        z_next = z_subj_residual  # (B, 18, H, W)
        
        # Stabilize to prevent explosion in autoregressive rollouts
        z_next = self.stabilizer(z_next)

        # # Experiment: predict full next state instead of residual
        # z_next = self.predictor(pred_input)  # (B, 18, H, W)

        return z_next

    # ============================================================================
    # MASK LOSSES
    # ============================================================================

    def compute_mask_losses(self, M, z_vis):
        """
        Compute all mask-related losses.
        
        Args:
            M: (B, T+1, 1, H, W) masks
            z_vis: (B, T+1, 16, H, W) visual features
        
        Returns:
            dict of losses
        """
        B, T_plus_1, _, H, W = M.shape
        
        losses = {}
        
        # ========================================================================
        # 1. Background invariance: z_bg should not change over time
        # ========================================================================
        z_bg = (1.0 - M) * z_vis                    # (B, T+1, 16, H, W)
        z_bg_t = z_bg[:, :-1]                       # (B, T, 16, H, W)
        z_bg_tp1 = z_bg[:, 1:].detach()             # stop gradient on target

        # Pixel-wise invariance loss
        mask_bg_invariance_pixel = F.mse_loss(z_bg_t, z_bg_tp1)
        # Item-wise invariance loss
        mask_bg_invariance_item = F.mse_loss(z_bg_t.mean(dim=(2, 3, 4)), z_bg_tp1.mean(dim=(2, 3, 4)))
        
        losses['mask_bg_invariance'] = 0.5 * (mask_bg_invariance_pixel + mask_bg_invariance_item)
        
        # ========================================================================
        # 2. Alignment: subject mask should cover regions with high temporal change
        # ========================================================================
        # Temporal feature change
        dz = (z_vis[:, 1:] - z_vis[:, :-1]).pow(2).mean(dim=2, keepdim=True)  # (B, T, 1, H, W)
        
        # Penalize when mask is LOW where change is HIGH
        losses['mask_alignment'] = ((1.0 - M[:, :-1]) * dz).mean()
        
        # ========================================================================
        # 3. Minimum area: subject must cover at least k_min of the image
        # ========================================================================
        area = M.mean(dim=(2, 3, 4))  # (B, T+1)
        k_min = self.hparams.mask_min_area
        
        # Hinge loss: penalize when area < k_min
        mask_deficit = F.relu(k_min - area)
        losses['mask_min_area'] = (mask_deficit + mask_deficit.pow(2)).mean()
        
        # ========================================================================
        # 4. Compactness: subject should be spatially localized (total variation)
        # ========================================================================
        # Horizontal and vertical gradients of mask
        M_flat = M.view(B * T_plus_1, 1, H, W)
        
        tv_h = (M_flat[:, :, :, 1:] - M_flat[:, :, :, :-1]).abs().mean()
        tv_v = (M_flat[:, :, 1:, :] - M_flat[:, :, :-1, :]).abs().mean()
        
        losses['mask_compactness'] = tv_h + tv_v
        
        # ========================================================================
        # 5. Sparsity: encourage mask to be small (Bernoullian variance reduction)
        # ========================================================================
        # This penalizes the total area covered by the subject
        # Works in tension with mask_min_area to find minimal sufficient coverage
        losses['mask_sparsity'] = (M * (1.0 - M)).mean()
        
        return losses

    # ============================================================================
    # TRAINING STEP
    # ============================================================================

    def shared_step(self, batch, batch_idx):
        states, frames, actions = batch
        B, T, _ = actions.shape

        # Remove x, y from states (keep only vx, vy)
        states = states[..., 2:]  # (B, T+1, 2)

        # ========================================================================
        # Encode all states
        # ========================================================================
        z = self.encode_state(states, frames)  # (B, T+1, 18, 26, 26)
        z_vis = z[:, :, :16, :, :]             # (B, T+1, 16, 26, 26)

        # ========================================================================
        # Compute masks for entire trajectory
        # ========================================================================
        z_vis_flat = z_vis.flatten(0, 1)       # (B*(T+1), 16, 26, 26)
        M_logits = self.mask_head(z_vis_flat)  # (B*(T+1), 1, 26, 26)
        M = torch.sigmoid(self.mask_beta * M_logits).view(B, T+1, 1, 26, 26)  # Apply beta scaling

        # ========================================================================
        # Mask losses
        # ========================================================================
        mask_losses = self.compute_mask_losses(M, z_vis)

        # ========================================================================
        # Autoregressive prediction with discounting
        # ========================================================================
        prediction_loss = 0.0
        z_current = z[:, 0]  # (B, 18, 26, 26)
        total_discount = 0.0
        for t in range(T):
            # Predict next state
            z_next_pred = self.predict_state(z_current, actions[:, t])

            # Discounted MSE loss
            discount = self.hparams.prediction_cost_discount ** t
            prediction_loss = prediction_loss + discount * F.mse_loss(
                z_next_pred, z[:, t+1].detach()
            )
            total_discount = total_discount + discount

            # Autoregressive: use prediction as next input
            z_current = z_next_pred
        prediction_loss = prediction_loss / total_discount

        # ========================================================================
        # Inverse Dynamics Modeling (IDM)
        # ========================================================================
        z_pairs = torch.cat([z[:, :-1], z[:, 1:]], dim=2)  # (B, T, 36, H, W)
        z_pairs_flat = z_pairs.flatten(0, 1)                # (B*T, 36, H, W)
        actions_pred = self.idm_decoder(z_pairs_flat)       # (B*T, 2)
        actions_pred = actions_pred.view(B, T, 2)           # (B, T, 2)
        idm_loss = F.mse_loss(actions_pred, actions)

        # ========================================================================
        # Total loss
        # ========================================================================
        loss = (
            self.hparams.prediction_coeff * prediction_loss +
            self.hparams.mask_bg_invariance_coeff * mask_losses['mask_bg_invariance'] +
            self.hparams.mask_alignment_coeff * mask_losses['mask_alignment'] +
            self.hparams.mask_min_area_coeff * mask_losses['mask_min_area'] +
            self.hparams.mask_compactness_coeff * mask_losses['mask_compactness'] +
            self.hparams.mask_sparsity_coeff * mask_losses['mask_sparsity'] +
            self.hparams.idm_coeff * idm_loss
        )

        return {
            "loss": loss,
            "prediction_loss": prediction_loss,
            "idm_loss": idm_loss,
            **mask_losses,
        }

    def training_step(self, batch, batch_idx):
        outs = self.shared_step(batch, batch_idx)
        self.log_dict(
            {f"train/{k}": v for k, v in outs.items()},
            prog_bar=True,
            sync_dist=True
        )
        return outs["loss"]

    def validation_step(self, batch, batch_idx):
        outs = self.shared_step(batch, batch_idx)
        self.log_dict(
            {f"val/{k}": v for k, v in outs.items()},
            prog_bar=True,
            sync_dist=True
        )
        return outs["loss"]

    # ============================================================================
    # OPTIMIZER
    # ============================================================================

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": (
                        list(self.visual_encoder.parameters()) +
                        list(self.proprio_encoder.parameters()) +
                        list(self.action_encoder.parameters()) +
                        list(self.mask_head.parameters()) +
                        list(self.subject_refine.parameters()) +
                        list(self.background_refine.parameters()) +
                        list(self.idm_decoder.parameters())
                    ),
                    "lr": self.hparams.initial_lr_encoder,
                    "weight_decay": self.hparams.weight_decay_encoder,
                },
                {
                    "params": (
                        list(self.predictor.parameters()) +
                        list(self.stabilizer.parameters())
                    ),
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
        Simple random shooting planner.
        
        NOTE: This requires a value function or distance function to be implemented.
        Placeholder for now.
        """
        B = z_start.shape[0]
        device = z_start.device
        
        # Sample random action sequences
        actions = torch.randn(n_candidates, horizon, 2, device=device) * 0.5
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # Evaluate each sequence (placeholder - needs distance/value function)
        best_actions = actions[0]  # Just return first sequence for now
        
        return {
            'actions': best_actions,
        }