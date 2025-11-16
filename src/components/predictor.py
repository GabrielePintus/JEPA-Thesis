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
            nn.Dropout2d(p=0.1),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),

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

        # --- Role embeddings ---
        self.role_cls    = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.role_patch  = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.role_state  = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.role_action = nn.Parameter(torch.randn(1, 1, emb_dim))

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
        # self.output_head = nn.Linear(emb_dim, emb_dim)
        self.output_head = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, emb_dim)
        )

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

        # --- Add role embeddings ---
        cls_tok    = cls_tok    + self.role_cls
        patch_tok  = patch_tok  + self.role_patch
        state_tok  = state_tok  + self.role_state
        action_tok = action_tok + self.role_action

        tokens = torch.cat([cls_tok, patch_tok, state_tok, action_tok], dim=1)

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


#
#           NEW
#

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
#  Small building blocks
# ---------------------------------------------------------------------

class TokenProjection(nn.Module):
    """
    Projects each token group (CLS, PATCH, STATE, ACTION) into a shared embedding dim.
    Assumes inputs are already in some emb_dim, but allows separate linear heads.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.cls   = nn.Linear(emb_dim, emb_dim)
        self.patch = nn.Linear(emb_dim, emb_dim)
        self.state = nn.Linear(emb_dim, emb_dim)
        self.action= nn.Linear(emb_dim, emb_dim)

    def forward(
        self,
        z_cls: torch.Tensor,      # (B, 1, D)
        z_patches: torch.Tensor,  # (B, Np, D)
        z_state: torch.Tensor,    # (B, 1, D)
        z_action: torch.Tensor    # (B, D)
    ):
        cls_tok   = self.cls(z_cls)               # (B, 1, D)
        patch_tok = self.patch(z_patches)         # (B, Np, D)
        state_tok = self.state(z_state)           # (B, 1, D)
        action_tok= self.action(z_action).unsqueeze(1)  # (B, 1, D)
        return cls_tok, patch_tok, state_tok, action_tok


class RoleEmbedding(nn.Module):
    """
    Adds learned role embeddings for each token type: CLS, PATCH, STATE, ACTION.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.cls    = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.patch  = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.state  = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.action = nn.Parameter(torch.randn(1, 1, emb_dim))

    def forward(
        self,
        cls_tok: torch.Tensor,      # (B, 1, D)
        patch_tok: torch.Tensor,    # (B, Np, D)
        state_tok: torch.Tensor,    # (B, 1, D)
        action_tok: torch.Tensor    # (B, 1, D)
    ):
        cls_tok    = cls_tok    + self.cls
        patch_tok  = patch_tok  + self.patch
        state_tok  = state_tok  + self.state
        action_tok = action_tok + self.action
        return cls_tok, patch_tok, state_tok, action_tok


class FiLM(nn.Module):
    """
    Applies FiLM conditioning: tokens -> gamma,beta from action.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.gamma = nn.Linear(emb_dim, emb_dim)
        self.beta  = nn.Linear(emb_dim, emb_dim)

    def forward(
        self,
        tokens: torch.Tensor,   # (B, T, D)
        z_action: torch.Tensor  # (B, D)
    ):
        gamma = self.gamma(z_action).unsqueeze(1)  # (B, 1, D)
        beta  = self.beta(z_action).unsqueeze(1)   # (B, 1, D)
        return tokens * (1 + gamma) + beta


# ---------------------------------------------------------------------
#  Encoder: self-attention over latent tokens
# ---------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """
    Single transformer encoder layer with self-attention over tokens.
    Returns attention weights for inspection.
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            emb_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.ff = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, emb_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        x: (B, T, D)
        """
        # Self-attention block
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            x_norm, x_norm, x_norm,
            need_weights=True,
            average_attn_weights=False,  # (B, num_heads, T, T)
        )
        x = x + self.dropout(attn_out)

        # Feed-forward block
        x = x + self.dropout(self.ff(self.norm2(x)))

        if return_attn:
            return x, attn_weights
        return x, None


class LatentEncoder(nn.Module):
    """
    Stacked encoder layers over latent tokens [CLS, PATCHES, STATE].
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(emb_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        x: (B, T_enc, D)
        """
        all_attn = [] if return_attn else None
        for layer in self.layers:
            x, attn = layer(x, return_attn=return_attn)
            if return_attn:
                all_attn.append(attn)  # each attn: (B, H, T, T)
        return x, all_attn


# ---------------------------------------------------------------------
#  Decoder: self-attn on queries + cross-attn to encoder memory
# ---------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """
    One Transformer-style decoder layer with:
      - self-attention on queries
      - cross-attention to encoder memory
    Returns both self- and cross-attention weights if requested.
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            emb_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            emb_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_self  = nn.LayerNorm(emb_dim)
        self.norm_cross = nn.LayerNorm(emb_dim)
        self.norm_ff    = nn.LayerNorm(emb_dim)

        self.ff = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, emb_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,        # (B, T_dec, D), decoder queries
        memory: torch.Tensor,   # (B, T_enc, D), encoder outputs
        return_attn: bool = False,
    ):
        # Self-attention on decoder tokens
        x_norm = self.norm_self(x)
        self_out, self_attn = self.self_attn(
            x_norm, x_norm, x_norm,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + self.dropout(self_out)

        # Cross-attention: queries = decoder tokens, keys/values = encoder memory
        x_norm2 = self.norm_cross(x)
        cross_out, cross_attn = self.cross_attn(
            x_norm2, memory, memory,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + self.dropout(cross_out)

        # Feed-forward
        x = x + self.dropout(self.ff(self.norm_ff(x)))

        if return_attn:
            return x, self_attn, cross_attn
        return x, None, None


class ActionConditionedDecoder(nn.Module):
    """
    Stacked decoder layers. Operates on:
      - decoder queries (prediction tokens)
      - encoder memory
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(emb_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,        # (B, T_dec, D)
        memory: torch.Tensor,   # (B, T_enc, D)
        return_attn: bool = False,
    ):
        self_attn_all = [] if return_attn else None
        cross_attn_all = [] if return_attn else None

        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, memory, return_attn=return_attn)
            if return_attn:
                self_attn_all.append(self_attn)
                cross_attn_all.append(cross_attn)

        return x, self_attn_all, cross_attn_all


class OutputHead(nn.Module):
    """
    Maps decoder outputs to predicted deltas on latent tokens.
    """

    def __init__(self, emb_dim: int, mlp_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # (B, T, D)


# ---------------------------------------------------------------------
#  Top-level Predictor
# ---------------------------------------------------------------------

class EncoderDecoderPredictor(nn.Module):
    """
    JEPA-style latent dynamics model:
      - Latent encoder: self-attention on [CLS, PATCHES, STATE]_t
      - Action-conditioned decoder:
            predicts [CLS, PATCHES, STATE]_{t+1}
            from encoded memory + action FiLM + decoder self-attn + cross-attn.

    Inputs:
        z_cls:     (B, 1, D)
        z_patches: (B, Np, D)
        z_state:   (B, 1, D)
        z_action:  (B, D)

    Outputs:
        pred_cls:     (B, D)
        pred_patches: (B, Np, D)
        pred_state:   (B, D)
        (optionally) attention maps
    """

    def __init__(
        self,
        emb_dim: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.residual = residual

        # Shared components
        self.tokens = TokenProjection(emb_dim)
        self.roles  = RoleEmbedding(emb_dim)
        self.film   = FiLM(emb_dim)

        # Encoder/decoder stacks
        self.encoder = LatentEncoder(
            emb_dim=emb_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.decoder = ActionConditionedDecoder(
            emb_dim=emb_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        # Output head over decoder outputs
        self.output_head = OutputHead(emb_dim=emb_dim, mlp_dim=mlp_dim)

    def forward(
        self,
        z_cls: torch.Tensor,       # (B, 1, D)
        z_patches: torch.Tensor,   # (B, Np, D)
        z_state: torch.Tensor,     # (B, 1, D)
        z_action: torch.Tensor,    # (B, D)
        return_attn: bool = False,
    ):
        B, Np, D = z_patches.shape

        # -----------------------------------------------------------------
        # 1) Encoder over current latent: build tokens, roles, encode
        # -----------------------------------------------------------------
        enc_cls, enc_patches, enc_state, _ = self.tokens(
            z_cls, z_patches, z_state, z_action
        )
        enc_cls, enc_patches, enc_state, _ = self.roles(
            enc_cls, enc_patches, enc_state, torch.zeros_like(enc_cls)
        )

        enc_tokens = torch.cat([enc_cls, enc_patches, enc_state], dim=1)  # (B, N_enc, D)
        memory, enc_attn = self.encoder(enc_tokens, return_attn=return_attn)

        # -----------------------------------------------------------------
        # 2) Decoder queries: another view of latent_t, FiLM-conditioned by action
        # -----------------------------------------------------------------
        dec_cls, dec_patches, dec_state, _ = self.tokens(
            z_cls, z_patches, z_state, z_action
        )
        dec_cls, dec_patches, dec_state, _ = self.roles(
            dec_cls, dec_patches, dec_state, torch.zeros_like(dec_cls)
        )

        dec_tokens = torch.cat([dec_cls, dec_patches, dec_state], dim=1)  # (B, N_dec, D)
        dec_tokens = self.film(dec_tokens, z_action)  # FiLM from action

        # -----------------------------------------------------------------
        # 3) Decoder: self-attn + cross-attn to encoder memory
        # -----------------------------------------------------------------
        dec_out, dec_self_attn, dec_cross_attn = self.decoder(
            dec_tokens, memory, return_attn=return_attn
        )

        # -----------------------------------------------------------------
        # 4) Predict deltas and apply residual
        # -----------------------------------------------------------------
        delta = self.output_head(dec_out)  # (B, N_dec, D)

        if self.residual:
            orig = torch.cat([z_cls, z_patches, z_state], dim=1)  # (B, N_dec, D)
            pred_tokens = orig + delta
        else:
            pred_tokens = delta

        # Split back into components
        pred_cls     = pred_tokens[:, 0]           # (B, D)
        pred_patches = pred_tokens[:, 1:-1]        # (B, Np, D)
        pred_state   = pred_tokens[:, -1]          # (B, D)

        if return_attn:
            attn_dict = {
                "encoder_self": enc_attn,              # list[num_enc_layers] of (B, H, N_enc, N_enc)
                "decoder_self": dec_self_attn,         # list[num_dec_layers] of (B, H, N_dec, N_dec)
                "decoder_cross": dec_cross_attn,       # list[num_dec_layers] of (B, H, N_dec, N_enc)
            }
            return pred_cls, pred_patches, pred_state, attn_dict

        return pred_cls, pred_patches, pred_state


