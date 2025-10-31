# pointmaze_datamodule.py
# Lightning DataModule for PointMaze transitions
# - Loads a single dataset and splits into train/val inside (val_size)
# - Optionally subsamples each epoch (epoch_fraction)
# - Maintains synchronized obs/frames alignment

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from lightning import LightningDataModule
from typing import Optional

from src.data.dataset import PointMazeTransitions


class PointMazeDataModule(LightningDataModule):
    """
    LightningDataModule wrapping PointMazeTransitions.

    Args:
        data_path: path to .npz dataset
        val_size: fraction (0..1) or int count for validation split
        batch_size: number of transitions per batch
        num_workers: dataloader workers
        pin_memory: whether to use pinned memory for GPU
        normalize: z-score normalize obs/act
        epoch_fraction: fraction of training samples to use per epoch (e.g. 0.1 = 10%)
        seed: random seed for reproducibility
    """

    def __init__(
        self,
        data_path: str,
        val_size: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        normalize: bool = True,
        epoch_fraction: float = 1.0,
        seed: int = 0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize = normalize
        self.epoch_fraction = epoch_fraction
        self.seed = seed
        self.device = device

        # internal containers
        self.ds_full = None
        self.ds_train = None
        self.ds_val = None
        self._train_indices = None
        self._rng = np.random.default_rng(seed)

    # -------------------------------
    # Setup and split
    # -------------------------------

    def setup(self, stage: Optional[str] = None):
        """Load the dataset once and perform a train/val split."""
        if self.ds_full is None:
            self.ds_full = PointMazeTransitions(
                self.data_path,
                normalize=self.normalize,
                device=self.device,
            )

            n_total = len(self.ds_full)
            if 0 < self.val_size < 1:
                n_val = int(round(n_total * self.val_size))
            else:
                n_val = int(self.val_size)
            n_val = min(max(n_val, 1), n_total - 1)
            n_train = n_total - n_val

            # Deterministic split
            gen = torch.Generator().manual_seed(self.seed)
            self.ds_train, self.ds_val = random_split(
                self.ds_full, [n_train, n_val], generator=gen
            )

            print(f"[DataModule] Split {n_total} → {n_train} train / {n_val} val samples")

    # -------------------------------
    # Subsampling logic
    # -------------------------------

    def _subsample_train(self):
        if self.epoch_fraction >= 1.0:
            return self.ds_train

        n_total = len(self.ds_train)
        n_sub = max(1, int(round(n_total * self.epoch_fraction)))

        # different seed per process to diversify subsets across GPUs
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        rng = np.random.default_rng(self.seed + rank + self.trainer.current_epoch)
        idx = rng.choice(n_total, size=n_sub, replace=False)

        subset = Subset(self.ds_train, idx)
        if rank == 0:
            print(f"[DataModule] Using {n_sub}/{n_total} samples on rank 0 (epoch {self.trainer.current_epoch})")
        return subset


    # -------------------------------
    # DataLoaders
    # -------------------------------

    def train_dataloader(self):
        train_subset = self._subsample_train()
        return DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
