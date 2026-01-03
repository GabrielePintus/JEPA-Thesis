"""PointMaze sequential dataset for trajectory learning.

Provides contiguous windows of state-action-frame sequences from
episodic PointMaze data with optional preprocessing and normalization.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PointMazeSequences(Dataset):
    """Sequential dataset for PointMaze trajectories.

    Extracts fixed-length subsequences from episodic PointMaze data,
    returning synchronized states, visual frames, and actions.

    Each sample contains:
        - states: (seq_len+1, state_dim) - state trajectory
        - frames: (seq_len+1, 3, H, W) - RGB visual observations
        - actions: (seq_len, action_dim) - action sequence

    Args:
        npz_path: Path to .npz file containing episodic data
        seq_len: Number of transitions per sequence
        stride: Spacing between sequence starts (defaults to seq_len for non-overlapping)
        normalize: Apply z-score normalization to states and actions
        frame_size: Target (H, W) for frame resizing, or None to keep original size

    Example:
        >>> dataset = PointMazeSequences(
        ...     npz_path="data/pointmaze.npz",
        ...     seq_len=16,
        ...     stride=8,
        ...     frame_size=(64, 64),
        ... )
        >>> states, frames, actions = dataset[0]
        >>> states.shape  # (17, state_dim)
        >>> frames.shape  # (17, 3, 64, 64)
        >>> actions.shape # (16, action_dim)
    """

    def __init__(
        self,
        npz_path: str,
        seq_len: int = 8,
        stride: Optional[int] = None,
        normalize: bool = True,
        frame_size: Optional[tuple[int, int]] = (64, 64),
    ):
        super().__init__()
        
        # Load episodic data
        data = np.load(npz_path)
        self.obs = self._load_array(data, ["obs", "states"]).astype(np.float32)
        self.act = self._load_array(data, ["act", "actions"]).astype(np.float32)
        self.frames = data.get("frames", None)
        
        if self.frames is not None:
            self.frames = self.frames.astype(np.uint8)

        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len

        # Preprocess frames
        if frame_size is not None and self.frames is not None:
            self.frames = self._resize_frames(self.frames, frame_size)

        # Normalize observations and actions
        self.obs_mean = self.obs_std = None
        self.act_mean = self.act_std = None
        if normalize:
            self._normalize()

        # Cache dimensions
        self.E, self.T1, self.D = self.obs.shape  # episodes, timesteps+1, state_dim
        self.T = self.T1 - 1  # number of transitions per episode

        # Precompute all valid sequence indices
        self.indices = self._compute_indices()

        print(
            f"[PointMazeSequences] {len(self.indices)} sequences from {self.E} episodes "
            f"(seq_len={self.seq_len}, stride={self.stride})"
        )

    def _load_array(self, data: np.lib.npyio.NpzFile, keys: list[str]) -> np.ndarray:
        """Load array from npz file, trying multiple possible keys.

        Args:
            data: Loaded npz file
            keys: List of possible key names to try

        Returns:
            Array data from first matching key

        Raises:
            KeyError: If none of the keys exist in the file
        """
        for key in keys:
            if key in data:
                return data[key]
        raise KeyError(f"None of {keys} found in dataset")

    def _resize_frames(
        self, 
        frames: np.ndarray, 
        frame_size: tuple[int, int]
    ) -> np.ndarray:
        """Resize frames to target dimensions.

        Args:
            frames: Input frames of shape (E, T+1, H_in, W_in, C)
            frame_size: Target (H_out, W_out)

        Returns:
            Resized frames of shape (E, T+1, H_out, W_out, C)
        """
        H, W = frame_size
        E, T1, _, _, C = frames.shape
        resized = np.empty((E, T1, H, W, C), dtype=np.uint8)
        
        for e in range(E):
            for t in range(T1):
                resized[e, t] = cv2.resize(
                    frames[e, t], 
                    (W, H), 
                    interpolation=cv2.INTER_AREA
                )
        
        return resized

    def _normalize(self) -> None:
        """Apply z-score normalization to observations and actions.

        Computes global mean and std across all episodes and timesteps,
        then normalizes in-place. Statistics are stored for potential denormalization.
        """
        # Compute statistics
        self.obs_mean = self.obs.mean(axis=(0, 1))
        self.obs_std = self.obs.std(axis=(0, 1)) + 1e-6
        self.act_mean = self.act.mean(axis=(0, 1))
        self.act_std = self.act.std(axis=(0, 1)) + 1e-6

        # Normalize
        self.obs = (self.obs - self.obs_mean) / self.obs_std
        self.act = (self.act - self.act_mean) / self.act_std

    def _compute_indices(self) -> list[tuple[int, int]]:
        """Compute all valid (episode, start_time) pairs for sequences.

        Returns:
            List of (episode_idx, start_time) tuples
        """
        indices = []
        for e in range(self.E):
            # Valid start times: 0 to T - seq_len (inclusive)
            for t0 in range(0, self.T - self.seq_len + 1, self.stride):
                indices.append((e, t0))
        return indices

    def __len__(self) -> int:
        """Return total number of sequences."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sequence.

        Args:
            idx: Sequence index

        Returns:
            Tuple of (states, frames, actions):
                - states: (seq_len+1, state_dim) float tensor
                - frames: (seq_len+1, 3, H, W) float tensor in [0, 1]
                - actions: (seq_len, action_dim) float tensor
        """
        e, t0 = self.indices[idx]
        t1 = t0 + self.seq_len

        # Extract sequences
        states = torch.from_numpy(self.obs[e, t0:t1+1])   # (seq_len+1, D)
        actions = torch.from_numpy(self.act[e, t0:t1])    # (seq_len, A)

        if self.frames is not None:
            # Convert to torch, permute to CHW, normalize to [0, 1]
            frames = torch.from_numpy(self.frames[e, t0:t1+1])
            frames = frames.permute(0, 3, 1, 2).float() / 255.0
        else:
            # Dummy frames if not available
            frames = torch.zeros(self.seq_len + 1, 3, 1, 1)

        return states, frames, actions


