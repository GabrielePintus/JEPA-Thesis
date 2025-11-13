import torch
import torch.nn as nn
from flash_attn import flash_attn_func


class ConvPredictor(nn.Module):
    def __init__(self, in_channels=20, hidden_channels=32, out_channels=18):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.layers(x)



class TransformerDecoderPredictor(nn.Module):
    """
    Action-conditioned predictor for JEPA latent tokens with FiLM conditioning.
    Predicts next-step latent tokens:
        [CLS, PATCHES, STATE]_{t+1}
    conditioned on
        [CLS, PATCHES, STATE]_t and action_t.
    """

    def __init__(self, emb_dim=128, num_heads=4, num_layers=2, mlp_dim=256):
        super().__init__()

        # Token projections
        self.cls_proj = nn.Linear(emb_dim, emb_dim)
        self.patch_proj = nn.Linear(emb_dim, emb_dim)
        self.state_proj = nn.Linear(emb_dim, emb_dim)
        self.action_proj = nn.Linear(emb_dim, emb_dim)

        # --- FiLM conditioning layers ---
        self.film_gamma = nn.Linear(emb_dim, emb_dim)
        self.film_beta = nn.Linear(emb_dim, emb_dim)
        # ---------------------------------

        # Transformer decoder (latent tokens attend to action)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output head (residual delta)
        self.output_head = nn.Linear(emb_dim, emb_dim)

    def forward(self, z_cls, z_patches, z_state, z_action):
        """
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

        # Project each token group
        cls_q = self.cls_proj(z_cls)
        patch_q = self.patch_proj(z_patches)
        state_q = self.state_proj(z_state)

        # Concatenate into one token sequence
        tokens = torch.cat([cls_q, patch_q, state_q], dim=1)   # (B, 1 + Np + 1, D)

        # ----- FiLM conditioning -----
        gamma = self.film_gamma(z_action)  # (B, D)
        beta = self.film_beta(z_action)    # (B, D)
        tokens = tokens * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        # -----------------------------

        # Action → single memory token
        action_mem = self.action_proj(z_action).unsqueeze(1)   # (B, 1, D)

        # Transformer decoder (tokens attend to action token)
        decoded = self.decoder(tgt=tokens, memory=action_mem)  # (B, 1 + Np + 1, D)

        # Residual update
        delta = self.output_head(decoded)
        pred_tokens = tokens + delta

        # Split back into original groups
        pred_cls = pred_tokens[:, :1, :].squeeze(1)
        pred_patches = pred_tokens[:, 1:-1, :]
        pred_state = pred_tokens[:, -1:, :].squeeze(1)

        return pred_cls, pred_patches, pred_state
    

class TransformerEncoderPredictor(nn.Module):
    """
    Encoder-only predictor with FiLM conditioning.
    Tokens = [CLS, PATCHES, STATE] + ACTION as extra token.
    Pure self-attention; FiLM applied before the encoder.
    """

    def __init__(self, emb_dim=128, num_heads=4, num_layers=2, mlp_dim=256, residual=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.residual = residual

        # Token projections
        self.cls_proj    = nn.Linear(emb_dim, emb_dim)
        self.patch_proj  = nn.Linear(emb_dim, emb_dim)
        self.state_proj  = nn.Linear(emb_dim, emb_dim)
        self.action_proj = nn.Linear(emb_dim, emb_dim)

        # ---- FiLM: action → (gamma, beta) ----
        self.film_gamma = nn.Linear(emb_dim, emb_dim)
        self.film_beta  = nn.Linear(emb_dim, emb_dim)
        # --------------------------------------

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Residual output
        self.output_head = nn.Linear(emb_dim, emb_dim)

    def forward(self, z_cls, z_patches, z_state, z_action):
        """
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

        B, _, D = z_cls.shape

        # Project token groups
        cls_tok   = self.cls_proj(z_cls)         # (B, 1, D)
        patch_tok = self.patch_proj(z_patches)   # (B, Np, D)
        state_tok = self.state_proj(z_state)     # (B, 1, D)
        action_tok= self.action_proj(z_action).unsqueeze(1)  # (B, 1, D)

        # Full sequence: latent tokens + action token
        tokens = torch.cat([cls_tok, patch_tok, state_tok, action_tok], dim=1)
        # tokens shape: (B, Np + 3, D)

        # ====== FiLM Conditioning ======
        gamma = self.film_gamma(z_action)          # (B, D)
        beta  = self.film_beta(z_action)           # (B, D)
        tokens = tokens * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        # =================================

        # Self-attention over all tokens (including action token)
        encoded = self.encoder(tokens)  # (B, Np+3, D)

        # Compute delta only for latent tokens (ignore final action token)
        delta = self.output_head(encoded[:, :-1, :])

        # Original latent tokens to apply residual to
        if self.residual:
            orig = torch.cat([z_cls, z_patches, z_state], dim=1)
            pred_tokens = orig + delta
        else:
            pred_tokens = delta

        # Split back into components
        pred_cls     = pred_tokens[:, 0]      # (B, D)
        pred_patches = pred_tokens[:, 1:-1]   # (B, Np, D)
        pred_state   = pred_tokens[:, -1]     # (B, D)

        return pred_cls, pred_patches, pred_state
