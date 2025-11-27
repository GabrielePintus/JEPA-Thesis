"""
Simplified test version of ViT-JEPA with lucidrains rotary-embedding-torch.
This uses standard PyTorch attention for testing purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal

# Import the rotary embedding implementation from lucidrains
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from src.components.encoder import TokenExpansionEncoder


class PatchEmbedding(nn.Module):
    """Convert image into patch tokens with optional convolutional stem."""
    def __init__(
        self, 
        img_size=64, 
        patch_size=8, 
        in_channels=3, 
        embed_dim=384,
        use_conv_stem=False,
        stem_channels=[16, 32],
    ):
        """
        Args:
            img_size: Size of input image
            patch_size: Size of patches for tokenization
            in_channels: Number of input channels
            embed_dim: Dimension of token embeddings
            use_conv_stem: Whether to use convolutional stem before patch projection
            stem_channels: List of channel dimensions for convolutional stem layers
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        self.use_conv_stem = use_conv_stem
        
        if use_conv_stem:
            # Build convolutional stem
            stem_layers = []
            prev_channels = in_channels
            
            for i, out_channels in enumerate(stem_channels):
                stem_layers.extend([
                    nn.Conv2d(prev_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                ])
                prev_channels = out_channels
            
            self.stem = nn.Sequential(*stem_layers)
            
            # Patch projection from last stem channel to embed_dim
            self.proj = nn.Conv2d(prev_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.stem = None
            # Direct patch projection from input
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Apply convolutional stem if enabled
        if self.use_conv_stem:
            x = self.stem(x)
        
        # Patch projection
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class SimpleAttention(nn.Module):
    """Standard multi-head attention with optional RoPE and causal masking."""
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        dropout: float = 0.0,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        rope_interpolate_factor: float = 1.0,
        causal: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        self.use_rope = use_rope
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        if use_rope:
            # Initialize RoPE using lucidrains implementation
            self.rope = RotaryEmbedding(
                dim=self.head_dim,
                theta=rope_base,
                interpolate_factor=rope_interpolate_factor,
                freqs_for='lang',  # Standard language model frequencies
                cache_if_possible=True,  # Enable caching for efficiency
            )
        
        # Register buffer for causal mask (will be initialized on first forward)
        self.register_buffer("causal_mask", None, persistent=False)
        
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, dim]
            position_ids: Optional position indices [batch, seq_len] (not used with RoPE)
            attn_mask: Optional attention mask [batch, seq_len, seq_len] or [seq_len, seq_len]
        """
        B, N, D = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each is [B, num_heads, N, head_dim]
        
        # Apply RoPE if enabled
        if self.use_rope:
            # The lucidrains implementation expects seq_dim to be -2 by default
            # Our tensors are [B, num_heads, seq_len, head_dim]
            # We can use rotate_queries_or_keys method which handles this
            q = self.rope.rotate_queries_or_keys(q, seq_dim=-2)
            k = self.rope.rotate_queries_or_keys(k, seq_dim=-2)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        
        # Apply causal mask if enabled
        if self.causal:
            if self.causal_mask is None or self.causal_mask.shape[-1] < N:
                # Create or expand causal mask
                causal_mask = torch.triu(
                    torch.ones(N, N, dtype=torch.bool, device=x.device),
                    diagonal=1
                )
                self.causal_mask = causal_mask
            
            mask = self.causal_mask[:N, :N]
            attn = attn.masked_fill(mask, float('-inf'))
        
        # Apply custom attention mask if provided
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [B, 1, N, N]
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return out



class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0, 
        dropout: float = 0.0,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        rope_interpolate_factor: float = 1.0,
        causal: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimpleAttention(
            dim, 
            num_heads, 
            dropout, 
            use_rope=use_rope,
            rope_base=rope_base,
            rope_interpolate_factor=rope_interpolate_factor,
            causal=causal,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        
    def forward(
        self, 
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), position_ids=position_ids, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Args:
        embed_dim: Output dimension for each position
        pos: A list/tensor of positions to be encoded, shape [M, 2] for 2D positions
    
    Returns:
        emb: [M, embed_dim]
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 4, dtype=torch.float32)
    omega /= embed_dim / 4.0
    omega = 1.0 / (10000 ** omega)  # [embed_dim // 4]
    
    # pos is [M, 2] (y, x coordinates)
    pos = pos.reshape(-1, 2)  # [M, 2]
    
    # Compute embeddings for y-coordinate
    out_y = torch.einsum('m,d->md', pos[:, 0], omega)  # [M, embed_dim // 4]
    emb_sin_y = torch.sin(out_y)
    emb_cos_y = torch.cos(out_y)
    
    # Compute embeddings for x-coordinate  
    out_x = torch.einsum('m,d->md', pos[:, 1], omega)  # [M, embed_dim // 4]
    emb_sin_x = torch.sin(out_x)
    emb_cos_x = torch.cos(out_x)
    
    # Concatenate [sin_y, cos_y, sin_x, cos_x]
    emb = torch.cat([emb_sin_y, emb_cos_y, emb_sin_x, emb_cos_x], dim=1)  # [M, embed_dim]
    
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Create 2D sine-cosine positional embeddings.
    
    Args:
        embed_dim: Embedding dimension (must be divisible by 2)
        grid_size: Grid height/width (assumes square grid)
        cls_token: If True, prepend a zero vector for CLS token
    
    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim]
    """
    grid_h = grid_w = grid_size
    grid_h_coords = torch.arange(grid_h, dtype=torch.float32)
    grid_w_coords = torch.arange(grid_w, dtype=torch.float32)
    
    # Create 2D grid
    grid = torch.stack(torch.meshgrid(grid_h_coords, grid_w_coords, indexing='ij'), dim=0)  # [2, H, W]
    grid = grid.reshape(2, -1).t()  # [H*W, 2]
    
    # Generate sine-cosine embeddings for each dimension
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)  # [H*W, embed_dim]
    
    if cls_token:
        # Prepend zero vector for CLS token
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    
    return pos_embed




class VisionEncoder(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_channels=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.0,
        use_conv_stem=False,
        stem_channels=[16, 32],
    ):
        """
        Args:
            img_size: Size of input image
            patch_size: Size of patches
            in_channels: Number of input channels
            embed_dim: Token embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout probability
            use_conv_stem: Whether to use convolutional stem in patch embedding
            stem_channels: Channel dimensions for convolutional stem layers
        """
        super().__init__()
        
        self.patch_embed = PatchEmbedding(
            img_size, 
            patch_size, 
            in_channels, 
            embed_dim,
            use_conv_stem=use_conv_stem,
            stem_channels=stem_channels,
        )
        num_patches = self.patch_embed.num_patches
        grid_size = self.patch_embed.grid_size

        # -------------------------------------------------
        # Sinusoidal positional embeddings (fixed, not learned)
        # -------------------------------------------------
        pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # [1, N, embed_dim]

        self.pos_drop = nn.Dropout(dropout)

        # -------------------------------------------------
        # Transformer blocks (no RoPE)
        # -------------------------------------------------
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, 
                num_heads, 
                mlp_ratio, 
                dropout,
                use_rope=False,
                causal=False,
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Tokens: [B, N, D]
        """
        B = x.shape[0]

        # Patch embeddings: [B, N, D]
        x = self.patch_embed(x)

        # Add positional embeddings (fixed sinusoidal)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x




class TransformerPredictor(nn.Module):
    """
    Parametric Transformer Predictor that can operate as either:
    - Encoder mode: Bidirectional attention (for batch processing)
    - Decoder mode: Causal attention (for autoregressive prediction)
    
    Uses RoPE for better positional encoding in temporal sequences.
    """
    def __init__(
        self,
        embed_dim   : int   = 384,
        action_dim  : int   = 2,
        depth       : int   = 4,
        num_heads   : int   = 6,
        mlp_ratio   : float = 4.0,
        dropout     : float = 0.0,
        mode        : Literal["encoder", "decoder"] = "decoder",
        rope_base   : float = 10000.0,
        rope_interpolate_factor : float = 1.0,
        use_action_conditioning : bool  = True,
    ):
        """
        Args:
            embed_dim: Dimension of token embeddings
            action_dim: Dimension of action space
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embed_dim
            dropout: Dropout probability
            mode: "encoder" (bidirectional) or "decoder" (causal)
            rope_base: Base (theta) for RoPE frequency computation
            rope_interpolate_factor: Factor for positional interpolation (for longer sequences)
            use_action_conditioning: Whether to use action as conditioning token
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.mode = mode
        self.use_action_conditioning = use_action_conditioning
        
        # Action embedding network
        if use_action_conditioning:
            # self.action_embed = nn.Sequential(
            #     nn.Linear(action_dim, embed_dim),
            #     nn.GELU(),
            #     nn.Linear(embed_dim, embed_dim),
            # )
            self.action_embed = TokenExpansionEncoder(
                input_dim=action_dim,
                embed_dim=embed_dim,
                batch_norm=True,
            )
        
        # Transformer blocks with RoPE and optional causal masking
        causal = (mode == "decoder")
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, 
                num_heads, 
                mlp_ratio, 
                dropout,
                use_rope=True,
                rope_base=rope_base,
                rope_interpolate_factor=rope_interpolate_factor,
                causal=causal,
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self, 
        state_tokens: torch.Tensor, 
        action: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            state_tokens: State token embeddings [batch, num_tokens, embed_dim]
            action: Action vector [batch, action_dim] (optional, used if use_action_conditioning=True)
            position_ids: Custom position indices [batch, seq_len] (optional, not used with RoPE)
            attn_mask: Custom attention mask [batch, seq_len, seq_len] (optional)
        
        Returns:
            predicted_tokens: Predicted next state tokens [batch, num_tokens, embed_dim]
        """
        B = state_tokens.shape[0]
        
        # Optionally concatenate action token
        if self.use_action_conditioning:
            if action is None:
                raise ValueError("action must be provided when use_action_conditioning=True")
            action_token = self.action_embed(action).unsqueeze(1)  # [B, 1, embed_dim]
            x = torch.cat([state_tokens, action_token], dim=1)  # [B, N+1, embed_dim]
        else:
            x = state_tokens
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, position_ids=position_ids, attn_mask=attn_mask)
        
        x = self.norm(x)
        
        # Remove action token if it was added
        if self.use_action_conditioning:
            predicted_tokens = x[:, :-1, :]  # [B, N, embed_dim]
        else:
            predicted_tokens = x
        
        return predicted_tokens
    
    def forward_autoregressive(
        self,
        initial_state_tokens: torch.Tensor,
        actions: torch.Tensor,
        return_all_predictions: bool = False,
    ) -> torch.Tensor:
        """
        Autoregressive forward pass for trajectory prediction.
        
        Args:
            initial_state_tokens: Initial state [batch, num_tokens, embed_dim]
            actions: Sequence of actions [batch, seq_len, action_dim]
            return_all_predictions: If True, return all intermediate predictions
        
        Returns:
            If return_all_predictions:
                All predicted states [batch, seq_len, num_tokens, embed_dim]
            Else:
                Final predicted state [batch, num_tokens, embed_dim]
        """
        B, seq_len, _ = actions.shape
        
        if self.mode != "decoder":
            raise ValueError("Autoregressive prediction requires decoder mode")
        
        current_state = initial_state_tokens
        predictions = []
        
        for t in range(seq_len):
            action_t = actions[:, t, :]  # [B, action_dim]
            next_state = self.forward(current_state, action_t)  # [B, N, embed_dim]
            
            if return_all_predictions:
                predictions.append(next_state)
            
            current_state = next_state
        
        if return_all_predictions:
            return torch.stack(predictions, dim=1)  # [B, seq_len, N, embed_dim]
        else:
            return current_state  # [B, N, embed_dim]


class LightweightDecoder(nn.Module):
    """
    Improved lightweight decoder:
    - Uses MLP + LayerNorm + GELU per patch
    - Reshapes into (B, C, H, W)
    - Applies 2 residual depthwise-separable conv blocks
    - Final light blur + sigmoid for smooth output
    """
    def __init__(
        self,
        embed_dim=384,
        patch_size=8,
        img_size=64,
        out_channels=3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size
        proj_out_dim = patch_size * patch_size * out_channels
        
        # -------------------------
        # Patch-level projection
        # -------------------------
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, proj_out_dim),
        )
        
        # -------------------------
        # Refinement UNet-like block
        # -------------------------
        def dw_sep(in_ch, out_ch):
            """Depthwise-separable conv block."""
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.GELU(),
            )

        self.refine = nn.Sequential(
            dw_sep(out_channels, 32),
            dw_sep(32, 32),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

        # Small smoothing layer (VERY helpful for deblocking)
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, tokens):
        B, N, D = tokens.shape
        
        # Project tokens → patch pixels
        x = self.proj(tokens)                       # (B, N, P*P*C)

        # Reshape into (B, C, H, W)
        x = x.reshape(
            B, self.grid_size, self.grid_size,
            self.patch_size, self.patch_size, -1
        )
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, -1, self.img_size, self.img_size)  # B, C, H, W

        # Refinement (UNet-like residual)
        residual = self.refine(x)
        x = x + residual

        # Final smoothing
        x = self.smooth(x)
        return x



class SimpleIDM(nn.Module):
    def __init__(self, embed_dim=384, action_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, action_dim),
        )
        
    def forward(self, state_tokens, next_state_tokens):
        state_emb = state_tokens.mean(dim=1)
        next_state_emb = next_state_tokens.mean(dim=1)
        combined = torch.cat([state_emb, next_state_emb], dim=-1)
        action = self.mlp(combined)
        return action