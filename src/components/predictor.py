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


class Expander2D(nn.Module):
    """Same definition as in the encoder."""
    def __init__(self, target_shape=(16, 16), out_channels=4):
        super().__init__()
        self.target_shape = target_shape
        self.out_channels = out_channels

    def forward(self, x):
        B, D = x.shape
        out = x.view(B, self.out_channels, 1, 1)
        out = out.expand(-1, -1, *self.target_shape)
        return out



import torch
import torch.nn as nn

# class PredictorTransformer(nn.Module):
#     """
#     JEPA Predictor
#     ----------------
#     Receives latent tokens from encoders (already encoded).

#     Inputs:
#         cls      : (B, D)
#         patches  : (B, N, D)
#         z_prop   : (B, D)   (latent proprio token)
#         z_act    : (B, D)   (latent action token)

#     Outputs:
#         pred_cls    : (B, D)
#         pred_patches: (B, N, D)
#         pred_prop   : (B, 4)  # predicts (x,y,vx,vy)
#     """

#     def __init__(
#         self,
#         emb_dim=128,
#         depth=4,
#         heads=4,
#         mlp_dim=256,
#         num_patches=64,
#     ):
#         super().__init__()

#         self.emb_dim = emb_dim
#         self.num_patches = num_patches

#         # ----------------------------
#         # Transformer blocks
#         # ----------------------------
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(
#                 nn.ModuleDict({
#                     "self_attn": nn.MultiheadAttention(
#                         embed_dim=emb_dim,
#                         num_heads=heads,
#                         batch_first=True
#                     ),
#                     "cross_attn": nn.MultiheadAttention(
#                         embed_dim=emb_dim,
#                         num_heads=heads,
#                         batch_first=True
#                     ),
#                     "ff": nn.Sequential(
#                         nn.LayerNorm(emb_dim),
#                         nn.Linear(emb_dim, mlp_dim),
#                         nn.GELU(),
#                         nn.Linear(mlp_dim, emb_dim),
#                     ),
#                     # FiLM modulation (gamma, beta computed from control tokens)
#                     "film": nn.Linear(emb_dim, emb_dim * 2)
#                 })
#             )

#         self.norm = nn.LayerNorm(emb_dim)

#         # ----------------------------
#         # Output heads
#         # ----------------------------

#         # Visual latent prediction (residual delta)
#         self.pred_vis = nn.Linear(emb_dim, emb_dim)

#         # Proprio prediction (supervised)
#         self.pred_prop_latent = nn.Linear(emb_dim, emb_dim)

#     def forward(self, cls, patches, z_prop, z_act):
#         B = cls.size(0)

#         # ------------------------------------------------------------
#         # token layout: [ CLS | patch₀ | ... | patch₆₃ | proprio | action ]
#         # ------------------------------------------------------------
#         ctrl = torch.stack([z_prop, z_act], dim=1)  # (B, 2, D)
#         tokens = torch.cat([cls.unsqueeze(1), patches, ctrl], dim=1)  # (B, 65+2, D)

#         for blk in self.layers:

#             # ----------------------
#             # 1) SELF-ATTENTION
#             # ----------------------
#             attn_out = blk["self_attn"](tokens, tokens, tokens, need_weights=False)[0]
#             tokens = tokens + attn_out      # <-- SAFE residual, not slicing

#             # ----------------------
#             # 2) CROSS-ATTENTION (visual ← control tokens)
#             # ----------------------
#             visual = tokens[:, : self.num_patches + 1]       # (B, 65, D)
#             ctrl   = tokens[:, self.num_patches + 1 : ]      # (B,  2, D)

#             cross = blk["cross_attn"](visual, ctrl, ctrl, need_weights=False)[0]
#             visual = visual + cross                           # <-- DO NOT assign back inplace

#             # ----------------------
#             # 3) FiLM modulation
#             # ----------------------
#             pooled_ctrl = ctrl.mean(dim=1)                   # (B, D)
#             gamma, beta = blk["film"](pooled_ctrl).chunk(2, dim=-1)

#             visual = visual * gamma.unsqueeze(1) + beta.unsqueeze(1)

#             # ----------------------
#             # 4) Reassemble tokens (NO INPLACE)
#             # ----------------------
#             tokens = torch.cat([visual, ctrl], dim=1)

#             # ----------------------
#             # 5) FEEDFORWARD
#             # ----------------------
#             tokens = tokens + blk["ff"](tokens)

#         tokens = self.norm(tokens)

#         # Split outputs
#         vis_tokens = tokens[:, : self.num_patches + 1]  # (B, 65, D)
#         pred_vis = self.pred_vis(vis_tokens) + vis_tokens

#         pred_cls = pred_vis[:, 0]         # (B, D)
#         pred_patches = pred_vis[:, 1:]    # (B, 64, D)
#         z_pred_prop = self.pred_prop_latent(pred_cls)

#         return pred_cls, pred_patches, z_pred_prop



class PredictorTransformer(nn.Module):
    """
    JEPA Predictor
    ----------------
    Receives latent tokens from encoders (already encoded).

    Inputs:
        cls      : (B, D)
        patches  : (B, N, D)
        z_prop   : (B, D)   (latent proprio token)
        z_act    : (B, D)   (latent action token)

    Outputs:
        pred_cls       : (B, D)         (future visual CLS latent)
        pred_patches   : (B, N, D)      (future visual patch latents)
        pred_prop_lat  : (B, D)         (future proprio latent)
    """

    def __init__(
        self,
        emb_dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        mlp_dim: int = 256,
        num_patches: int = 64,
        dropout_p: float = 0.1,
    ):
        super().__init__()

        assert emb_dim % heads == 0, "emb_dim must be divisible by heads"
        self.emb_dim = emb_dim
        self.num_patches = num_patches
        self.dropout = nn.Dropout(dropout_p)

        # Role embeddings: [CLS | patches(=N) | proprio | action]
        self.role_embed = nn.Parameter(torch.zeros(1, 1 + num_patches + 2, emb_dim))

        # Transformer blocks (Pre-Norm)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict({
                    # Pre-norms
                    "ln_self": nn.LayerNorm(emb_dim),
                    "ln_cross_v": nn.LayerNorm(emb_dim),
                    "ln_cross_c": nn.LayerNorm(emb_dim),
                    "ln_ff": nn.LayerNorm(emb_dim),

                    # Attentions
                    "self_attn": nn.MultiheadAttention(
                        embed_dim=emb_dim,
                        num_heads=heads,
                        batch_first=True,
                        dropout=dropout_p
                    ),
                    "cross_attn": nn.MultiheadAttention(
                        embed_dim=emb_dim,
                        num_heads=heads,
                        batch_first=True,
                        dropout=dropout_p
                    ),

                    # Feedforward
                    "ff": nn.Sequential(
                        nn.Linear(emb_dim, mlp_dim),
                        nn.GELU(),
                        nn.Linear(mlp_dim, emb_dim),
                    ),

                    # FiLM modulation (gamma, beta from control tokens)
                    "film": nn.Linear(emb_dim, 2 * emb_dim),
                    "ln_after_film": nn.LayerNorm(emb_dim),
                })
            )

        self.final_norm = nn.LayerNorm(emb_dim)

        # Output heads
        # Separate heads for CLS and patches (both residual-delta style)
        self.pred_cls_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )
        self.pred_patch_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )

        # Proprio latent prediction head (predicts future proprio *latent*)
        self.pred_prop_latent = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, cls, patches, z_prop, z_act):
        """
        cls     : (B, D)
        patches : (B, N, D)
        z_prop  : (B, D)
        z_act   : (B, D)
        """
        B, N, D = patches.shape
        assert N == self.num_patches, f"Expected {self.num_patches} patches, got {N}"
        assert D == self.emb_dim, "patches last dim must match emb_dim"

        # ------------------------------------------------------------
        # Token layout: [ CLS | patch₀ ... patchₙ₋₁ | proprio | action ]
        # ------------------------------------------------------------
        ctrl = torch.stack([z_prop, z_act], dim=1)             # (B, 2, D)
        tokens = torch.cat([cls.unsqueeze(1), patches, ctrl], dim=1)  # (B, 1+N+2, D)
        tokens = tokens + self.role_embed                      # add role embeddings

        for blk in self.layers:
            # ----------------------
            # 1) SELF-ATTENTION (Pre-Norm)
            # ----------------------
            x = blk["ln_self"](tokens)
            attn_out, _ = blk["self_attn"](x, x, x, need_weights=False)
            tokens = tokens + self.dropout(attn_out)

            # ----------------------
            # 2) CROSS-ATTN: visual ← control (Pre-Norm)
            # ----------------------
            visual = tokens[:, : self.num_patches + 1]         # (B, 1+N, D)
            ctrl   = tokens[:, self.num_patches + 1 : ]        # (B, 2, D)

            v = blk["ln_cross_v"](visual)
            c = blk["ln_cross_c"](ctrl)
            cross_out, _ = blk["cross_attn"](v, c, c, need_weights=False)
            visual = visual + self.dropout(cross_out)

            # ----------------------
            # 3) Safe FiLM modulation on visual tokens
            # ----------------------
            pooled_ctrl = ctrl.mean(dim=1)                     # (B, D)
            gamma, beta = blk["film"](pooled_ctrl).chunk(2, dim=-1)
            # keep modulation near identity for stability
            gamma = 1.0 + 0.1 * torch.tanh(gamma)
            beta  = 0.1 * torch.tanh(beta)
            visual = visual * gamma.unsqueeze(1) + beta.unsqueeze(1)
            visual = blk["ln_after_film"](visual)

            # ----------------------
            # 4) Reassemble tokens (no in-place)
            # ----------------------
            tokens = torch.cat([visual, ctrl], dim=1)

            # ----------------------
            # 5) FEEDFORWARD (Pre-Norm)
            # ----------------------
            x = blk["ln_ff"](tokens)
            tokens = tokens + self.dropout(blk["ff"](x))

        tokens = self.final_norm(tokens)

        # Split & predict with separate heads (residual deltas)
        vis_tokens = tokens[:, : self.num_patches + 1]         # (B, 1+N, D)
        cls_token = vis_tokens[:, :1]                          # (B, 1, D)
        patch_tokens = vis_tokens[:, 1:]                       # (B, N, D)

        pred_cls = self.pred_cls_head(cls_token).squeeze(1) + cls_token.squeeze(1)     # (B, D)
        pred_patches = self.pred_patch_head(patch_tokens) + patch_tokens               # (B, N, D)

        # Proprio latent prediction can benefit from both CLS and pooled patches
        pooled_patches = pred_patches.mean(dim=1)                                     # (B, D)
        prop_query = 0.5 * (pred_cls + pooled_patches)                                # (B, D)
        pred_prop_lat = self.pred_prop_latent(prop_query)                              # (B, D)

        return pred_cls, pred_patches, pred_prop_lat






#
# CUSTOM well done
#



# ============================================================
# Utility: simple MLP
# ============================================================
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ============================================================
# Transformer Predictor Block:
# Self-Attention on latent state tokens
# Cross-Attention to action/control tokens
# ============================================================
class PredictorBlock(nn.Module):
    """
    Implements:
        X <- X + SelfAttention(X)
        X <- X + CrossAttention(X <- C)   # C = control tokens (action-projected)
        X <- X + MLP
    """

    def __init__(self, dim, num_heads=4, mlp_ratio=4, ctrl_slots=6):
        super().__init__()

        # --- normalization ---
        self.ln_self = nn.LayerNorm(dim)
        self.ln_cross = nn.LayerNorm(dim)
        self.ln_mlp = nn.LayerNorm(dim)

        # --- self-attention over state tokens (CLS + patches + proprio) ---
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        # --- cross attention: Q = latent tokens, K/V = control tokens ---
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        # --- small learned scalar gate to regulate cross-attention ---
        self.gate = nn.Parameter(torch.tensor(0.1))  # start small, learn bigger if needed

        # --- feed-forward ---
        self.mlp = MLP(dim, dim * mlp_ratio)

        # --- per-slot projection from action token ---
        self.ctrl_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(ctrl_slots)])
        self.ctrl_slots = ctrl_slots

    def build_control_set(self, a_token, prop_token=None):
        """
        Expand action latent into B slots + optionally include proprio as context token.
          a_token: (B, D)
          prop_token: (B, D) or None
        Returns: (B, M, D) control/context tokens
        """
        B, D = a_token.shape

        controls = []
        for proj in self.ctrl_projs:
            controls.append(proj(a_token))  # each is (B, D)

        C = torch.stack(controls, dim=1)  # (B, ctrl_slots, D)

        if prop_token is not None:
            C = torch.cat([C, prop_token.unsqueeze(1)], dim=1)

        return C  # shape: (B, ctrl_slots (+1), D)

    def forward(self, X, a_token, prop_token=None):
        """
        X : (B, 1+N+1, D)       # CLS + patches + proprio
        a_token : (B, D)
        prop_token : (B, D)
        """
        # ---------------------------------------------------------
        # SELF-ATTENTION (latent dynamics using old latent only)
        # ---------------------------------------------------------
        X = X + self.self_attn(
            query=self.ln_self(X),
            key=self.ln_self(X),
            value=self.ln_self(X)
        )[0]  # output only

        # ---------------------------------------------------------
        # CROSS-ATTENTION (condition on action latent)
        # ---------------------------------------------------------
        C = self.build_control_set(a_token, prop_token)  # (B, M, D)

        X = X + torch.sigmoid(self.gate) * self.cross_attn(
            query=self.ln_cross(X),  # (B, n_tokens, D)
            key=self.ln_cross(C),    # (B, M, D)
            value=self.ln_cross(C)
        )[0]

        # ---------------------------------------------------------
        # MLP
        # ---------------------------------------------------------
        X = X + self.mlp(self.ln_mlp(X))
        return X



# ============================================================
# Full Predictor
# Input: latent_state = [CLS, patches..., PROP]
# Output: predicted next latent_state
# ============================================================
class ActionConditionedPredictor(nn.Module):
    """
    JEPA predictor with:
      - learnable positional embeddings (2D for patches, 1D for CLS & proprio)
      - learnable role embeddings (CLS / PATCH / STATE / ACTION)
      - transformer with latent cross-attention on the action-token

    Inputs:
        visual_tokens: (B, 1+N, D)     # CLS + patch tokens
        proprio_token: (B, D)
        action_token:  (B, D)

    Outputs:
        pred_cls      (B, D)
        pred_patches  (B, N, D)
        pred_proprio  (B, D)
    """

    def __init__(
        self,
        emb_dim=128,
        heads=4,
        depth=3,
        num_patches=64,           # e.g., 8x8
        ctrl_slots=6,
        mlp_dim=512              # mlp hidden size, NOT ratio
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_patches = num_patches

        # --------------------------
        # 1) LEARNABLE POS ENCODING
        # --------------------------
        # CLS + 64 patches + proprio → +2 tokens (CLS + PROP)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 1 + num_patches + 1, emb_dim)
        )

        # --------------------------
        # 2) LEARNABLE ROLE EMBEDDING
        # --------------------------
        # 0=CLS, 1=PATCH, 2=STATE(PROPRIO), 3=ACTION
        self.role_embedding = nn.Embedding(4, emb_dim)

        # --------------------------
        # 3) Transformer blocks
        # --------------------------
        self.blocks = nn.ModuleList([
            PredictorBlock(
                dim=self.emb_dim,
                num_heads=heads,
                mlp_ratio=mlp_dim // emb_dim,   # correct handling
                ctrl_slots=ctrl_slots,
            )
            for _ in range(depth)
        ])

        # --------------------------
        # 4) Output projectors
        # --------------------------
        self.proj_cls = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.proj_patch = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.proj_prop = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

    # ---------------------------------------------------------------------
    def forward(self, visual_tokens, proprio_token, action_token):
        """
        visual_tokens: (B, 1+N, D)  CLS + patches
        proprio_token: (B, D)
        action_token:  (B, D)
        """
        B = visual_tokens.size(0)

        # ---- concat visual + proprio
        proprio_token = proprio_token.unsqueeze(1)                   # (B,1,D)
        X = torch.cat([visual_tokens, proprio_token], dim=1)         # (B,1+N+1,D)

        # ====== POSITIONAL EMBEDDING ======
        # same pos_embedding added to batch
        X = X + self.pos_embedding[:, : X.size(1)]                   # (1,1+N+1,D)

        # ====== ROLE EMBEDDING ======
        # role IDs per token (CLS=0, PATCH=1, PROPRIO=2)
        role_ids = torch.zeros(X.size(1), dtype=torch.long, device=X.device)  # (1+N+1,)
        role_ids[1:1 + self.num_patches] = 1    # PATCH tokens
        role_ids[-1] = 2                        # PROPRIO token

        # Broadcast to batch
        X = X + self.role_embedding(role_ids)[None, :, :]            # (B,1+N+1,D)

        # ====== Transformer ======
        # Action token also gets role embedding
        action_token = action_token + self.role_embedding(
            torch.tensor([3], device=action_token.device)
        )

        for blk in self.blocks:
            X = blk(X, a_token=action_token, prop_token=proprio_token.squeeze(1))

        # ====== SPLIT ======
        pred_cls = X[:, 0]                                  # (B,D)
        pred_patches = X[:, 1:1 + self.num_patches]         # (B,N,D)
        pred_prop = X[:, 1 + self.num_patches]              # (B,D)

        # ====== PROJECTIONS (JEPA BYOL-style) ======
        pred_cls = self.proj_cls(pred_cls)
        pred_patches = self.proj_patch(pred_patches)
        pred_prop = self.proj_prop(pred_prop)

        return pred_cls, pred_patches, pred_prop




# class VanillaBlock(nn.Module):
#     def __init__(self, dim, heads=4, mlp_ratio=4, dropout=0.1):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(
#             embed_dim=dim, num_heads=heads,
#             batch_first=True, dropout=dropout
#         )

#         self.ln2 = nn.LayerNorm(dim)
#         self.ff = MLP(dim, dim * mlp_ratio)
#         self.drop = nn.Dropout(dropout)

#     def forward(self, x):
#         # Self-attention
#         h = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
#         x = x + self.drop(h)

#         # Feedforward
#         h = self.ff(self.ln2(x))
#         x = x + self.drop(h)
#         return x
    



class PredictorMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class VanillaBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.heads = heads
        self.head_dim = dim // heads
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        
        self.ln2 = nn.LayerNorm(dim)
        self.ff = MLP(dim, dim * mlp_ratio)
        self.drop = nn.Dropout(dropout)
        self.attn_drop = dropout

    def forward(self, x):
        # Self-attention with FlashAttention v2
        batch_size, seq_len, dim = x.shape
        
        # Pre-norm and project to QKV
        normed = self.ln1(x)
        qkv = self.qkv(normed)
        
        # Reshape to (batch, seqlen, 3, nheads, headdim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.heads, self.head_dim)
        
        # FlashAttention v2 expects (batch, seqlen, nheads, headdim) for each of q, k, v
        q, k, v = qkv.unbind(dim=2)
        
        # Apply FlashAttention v2
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
            causal=False  # Set to True for causal/autoregressive attention
        )
        
        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, seq_len, dim)
        h = self.out_proj(attn_output)
        x = x + self.drop(h)

        # Feedforward
        h = self.ff(self.ln2(x))
        x = x + self.drop(h)
        return x


class PredictorTransformerVanilla(nn.Module):
    """
    Inputs:
        cls     : (B, D)
        patches : (B, N, D)
        z_prop  : (B, D)
        z_act   : (B, D)

    Output:
        pred_cls     : (B, D)
        pred_patches : (B, N, D)
        pred_prop    : (B, D)
    """

    def __init__(
        self,
        emb_dim=128,
        num_patches=64,
        depth=3,
        heads=4,
        mlp_ratio=4,
        dropout=0.1,
        use_residual=True,
    ):
        super().__init__()
        self.D = emb_dim
        self.N = num_patches
        self.use_residual = use_residual

        assert int(mlp_ratio) == mlp_ratio, "mlp_ratio must be an integer"

        # positional embedding over [CLS | patches | PROP | ACTION]
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + num_patches + 2, emb_dim)
        )

        # role embeddings: CLS=0, PATCH=1, PROP=2, ACTION=3
        self.role_embed = nn.Embedding(4, emb_dim)

        # transformer blocks
        self.blocks = nn.ModuleList([
            VanillaBlock(emb_dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.final_ln = nn.LayerNorm(emb_dim)

        # projection heads
        self.head_cls = nn.Linear(emb_dim, emb_dim)
        self.head_patch = nn.Linear(emb_dim, emb_dim)
        self.head_prop = nn.Linear(emb_dim, emb_dim)

    def forward(self, cls, patches, z_prop, z_act):
        B, N, D = patches.shape
        assert N == self.N

        # build token sequence: [CLS | patches | PROP | ACTION]
        tokens = torch.cat([
            cls.unsqueeze(1),               # (B,1,D)
            patches,                        # (B,N,D)
            z_prop.unsqueeze(1),            # (B,1,D)
            z_act.unsqueeze(1),             # (B,1,D)
        ], dim=1)                           # → (B, 1+N+2, D)

        # add positional embeddings
        tokens = tokens + self.pos_embed[:, : tokens.size(1)]

        # build role ids
        role_ids = torch.zeros(tokens.size(1), dtype=torch.long, device=tokens.device)
        role_ids[1:1+N] = 1      # patches
        role_ids[1+N]   = 2      # proprio
        role_ids[2+N]   = 3      # action
        tokens = tokens + self.role_embed(role_ids)[None, :, :]

        # pass through transformer blocks
        x = tokens
        for blk in self.blocks:
            x = blk(x)

        x = self.final_ln(x)

        # split back to outputs
        cls_out = x[:, 0]                        # (B,D)
        patches_out = x[:, 1:1+N]                # (B,N,D)
        prop_out = x[:, 1+N]                     # (B,D)

        # optional residual prediction
        if self.use_residual:
            cls_out = cls + self.head_cls(cls_out)
            patches_out = patches + self.head_patch(patches_out)
            prop_out = z_prop + self.head_prop(prop_out)
        else:
            cls_out = self.head_cls(cls_out)
            patches_out = self.head_patch(patches_out)
            prop_out = self.head_prop(prop_out)

        return cls_out, patches_out, prop_out

