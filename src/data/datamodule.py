"""Lightning DataModule for PointMaze sequential data.

This module handles:
- Loading and splitting PointMaze trajectory sequences
- Train/validation data partitioning
- Optional epoch-wise subsampling for faster iteration
- Synchronized observation/frame alignment
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, random_split

from src.data.dataset import PointMazeSequences


class PointMazeSequencesDataModule(LightningDataModule):
    """DataModule for loading and managing PointMaze trajectory sequences.

    Wraps PointMazeSequences dataset with train/val splitting and optional
    subsampling capabilities for efficient experimentation.

    Args:
        data_path: Path to .npz dataset file
        val_size: Validation split size as fraction (0-1) or absolute count
        batch_size: Number of sequences per batch
        num_workers: Number of DataLoader worker processes
        pin_memory: Enable pinned memory for faster GPU transfer
        normalize: Apply z-score normalization to observations and actions
        seq_len: Number of timesteps per sequence
        stride: Stride between consecutive sequences
        epoch_fraction: Fraction of training data to use per epoch (0-1)
        seed: Random seed for reproducibility
        device: Optional device for dataset tensors
        frame_size: Target (H, W) for frame resizing

    Example:
        >>> dm = PointMazeSequencesDataModule(
        ...     data_path="data/pointmaze.npz",
        ...     val_size=0.1,
        ...     batch_size=32,
        ...     seq_len=16,
        ... )
        >>> dm.setup()
        >>> train_loader = dm.train_dataloader()
    """

    def __init__(
        self,
        data_path: str,
        val_size: float = 0.1,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        normalize: bool = True,
        seq_len: int = 3,
        stride: int = 3,
        epoch_fraction: float = 1.0,
        seed: int = 0,
        device: Optional[torch.device] = None,
        frame_size: tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.data_path = data_path
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize = normalize
        self.seq_len = seq_len
        self.stride = stride
        self.epoch_fraction = epoch_fraction
        self.seed = seed
        self.device = device
        self.frame_size = frame_size

        # Internal state
        self.ds_full: Optional[PointMazeSequences] = None
        self.ds_train: Optional[Subset] = None
        self.ds_val: Optional[Subset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load dataset and create train/validation splits.

        Args:
            stage: Optional pipeline stage ('fit', 'validate', 'test', 'predict')
        """
        if self.ds_full is not None:
            return

        # Load full dataset
        self.ds_full = PointMazeSequences(
            npz_path=self.data_path,
            seq_len=self.seq_len,
            normalize=self.normalize,
            frame_size=self.frame_size,
            stride=self.stride,
        )

        # Compute split sizes
        n_total = len(self.ds_full)
        n_val = self._compute_val_size(n_total)
        n_train = n_total - n_val

        # Perform split
        generator = torch.Generator().manual_seed(self.seed)
        self.ds_train, self.ds_val = random_split(
            self.ds_full, [n_train, n_val], generator=generator
        )

        print(f"[PointMazeDataModule] Dataset split: {n_train} train / {n_val} val sequences")

    def _compute_val_size(self, n_total: int) -> int:
        """Compute validation set size from fraction or absolute count.

        Args:
            n_total: Total number of sequences

        Returns:
            Number of validation sequences
        """
        if 0 < self.val_size < 1:
            n_val = int(round(n_total * self.val_size))
        else:
            n_val = int(self.val_size)

        # Ensure at least 1 sample in each split
        return min(max(n_val, 1), n_total - 1)

    def _subsample_train(self) -> Subset:
        """Subsample training data for current epoch.

        Uses epoch-dependent random seed for different subsamples per epoch.
        Supports distributed training with rank-specific seeds.

        Returns:
            Subsampled training dataset
        """
        if self.epoch_fraction >= 1.0:
            return self.ds_train

        n_total = len(self.ds_train)
        n_sub = max(1, int(round(n_total * self.epoch_fraction)))

        # Get rank for distributed training
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # Epoch-dependent sampling
        epoch_seed = self.seed + rank + self.trainer.current_epoch
        rng = np.random.default_rng(epoch_seed)
        indices = rng.choice(n_total, size=n_sub, replace=False)

        if rank == 0:
            print(f"[PointMazeDataModule] Epoch {self.trainer.current_epoch}: "
                  f"using {n_sub}/{n_total} sequences ({self.epoch_fraction:.1%})")

        return Subset(self.ds_train, indices)

    def train_dataloader(self) -> DataLoader:
        """Create training data loader with optional subsampling.

        Returns:
            DataLoader for training with shuffling enabled
        """
        train_subset = self._subsample_train()
        return DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation data loader.

        Returns:
            DataLoader for validation without shuffling
        """
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


