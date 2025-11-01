# dataset.py
# PointMazeTransitions dataset — returns ((state, frame), action, (next_state, next_frame))
# Supports optional frame downscaling and normalization.

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
import cv2


class PointMazeTransitions(Dataset):
    def __init__(
        self,
        npz_path: str,
        normalize: bool = True,
        device: Optional[torch.device] = None,
        frame_size: Optional[Tuple[int, int]] = (64, 64),  # e.g., (64, 64)
    ):
        data = np.load(npz_path)
        self.obs = data["obs"].astype(np.float32)          # (E, T+1, D)
        self.act = data["act"].astype(np.float32)          # (E, T, A)
        self.frames = data["frames"].astype(np.uint8) if "frames" in data else None
        self.device = device

        # ----------------------------
        # Optional frame downscaling
        # ----------------------------
        if frame_size is not None and self.frames is not None:
            H, W = frame_size
            E, T1, _, _, C = self.frames.shape
            resized = np.empty((E, T1, H, W, C), dtype=np.uint8)
            for e in range(E):
                for t in range(T1):
                    resized[e, t] = cv2.resize(
                        self.frames[e, t], (W, H), interpolation=cv2.INTER_AREA
                    )
            self.frames = resized
            print(f"[Dataset] Frames resized to {H}×{W}")

        # ----------------------------
        # Prepare transitions
        # ----------------------------
        E, T1, D = self.obs.shape  # T1 = T+1
        T = T1 - 1

        obs_t = self.obs[:, :-1]         # (E, T, D)
        next_obs_t = self.obs[:, 1:]     # (E, T, D)
        act_t = self.act                 # (E, T, A)

        # Flatten episodes
        self.obs = obs_t.reshape(E * T, D)
        self.next_obs = next_obs_t.reshape(E * T, D)
        self.act = act_t.reshape(E * T, -1)

        if self.frames is not None:
            frame_t = self.frames[:, :-1]         # (E, T, H, W, 3)
            next_frame_t = self.frames[:, 1:]     # (E, T, H, W, 3)
            self.frames = frame_t.reshape(E * T, *frame_t.shape[2:])
            self.next_frames = next_frame_t.reshape(E * T, *next_frame_t.shape[2:])
        else:
            self.next_frames = None

        # ----------------------------
        # Optional normalization
        # ----------------------------
        if normalize:
            self.obs_mean = self.obs.mean(axis=0)
            self.obs_std = self.obs.std(axis=0) + 1e-6
            self.act_mean = self.act.mean(axis=0)
            self.act_std = self.act.std(axis=0) + 1e-6

            self.obs = (self.obs - self.obs_mean) / self.obs_std
            self.next_obs = (self.next_obs - self.obs_mean) / self.obs_std
            self.act = (self.act - self.act_mean) / self.act_std

        # Summary
        print(f"[Dataset] Loaded {E} episodes, {E*T} transitions.")
        if self.frames is not None:
            print(f"[Dataset] Frame shape: {self.frames.shape[1:]}")

    # ----------------------------
    # Dataset interface
    # ----------------------------
    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int):
        # Core data
        state = torch.from_numpy(self.obs[idx])
        action = torch.from_numpy(self.act[idx])
        next_state = torch.from_numpy(self.next_obs[idx])

        if self.frames is not None:
            frame = torch.from_numpy(self.frames[idx]).permute(2, 0, 1).float() / 255.0
            next_frame = torch.from_numpy(self.next_frames[idx]).permute(2, 0, 1).float() / 255.0
        else:
            frame = next_frame = torch.zeros(3, 1, 1)

        # JEPA-style tuple output
        return (state, frame), action, (next_state, next_frame)







# class PointMazeSequences(Dataset):
#     """
#     Sequential version of PointMazeTransitions.
#     Returns contiguous subsequences of length `seq_len`
#     (so `seq_len+1` frames/states and `seq_len` actions).

#     Example output:
#         (
#             states: (T+1, D_state),
#             frames: (T+1, 3, H, W),
#         ),
#         actions: (T, D_action)
#     """

#     def __init__(
#         self,
#         npz_path: str,
#         seq_len: int = 8,
#         normalize: bool = True,
#         frame_size=(64, 64),
#     ):
#         super().__init__()
#         data = np.load(npz_path)
#         self.obs = data["obs"].astype(np.float32)          # (E, T+1, D)
#         self.act = data["act"].astype(np.float32)          # (E, T, A)
#         self.frames = data.get("frames", None)
#         if self.frames is not None:
#             self.frames = self.frames.astype(np.uint8)

#         self.seq_len = seq_len

#         # Resize frames if needed
#         if frame_size and self.frames is not None:
#             H, W = frame_size
#             E, T1, _, _, C = self.frames.shape
#             resized = np.empty((E, T1, H, W, C), dtype=np.uint8)
#             for e in range(E):
#                 for t in range(T1):
#                     resized[e, t] = cv2.resize(
#                         self.frames[e, t], (W, H), interpolation=cv2.INTER_AREA
#                     )
#             self.frames = resized
#             print(f"[Dataset] Frames resized to {H}×{W}")

#         # Optional normalization
#         if normalize:
#             obs_mean = self.obs.mean(axis=(0, 1))
#             obs_std = self.obs.std(axis=(0, 1)) + 1e-6
#             act_mean = self.act.mean(axis=(0, 1))
#             act_std = self.act.std(axis=(0, 1)) + 1e-6

#             self.obs = (self.obs - obs_mean) / obs_std
#             self.act = (self.act - act_mean) / act_std

#         self.E, self.T1, self.D = self.obs.shape
#         self.T = self.T1 - 1

#         print(f"[Dataset] Loaded {self.E} episodes, {self.T} steps each.")
#         print(f"[Dataset] Sequence length = {self.seq_len}")

#     def __len__(self):
#         # Number of possible subsequences per episode
#         return self.E * (self.T - self.seq_len)

#     def __getitem__(self, idx):
#         ep = idx // (self.T - self.seq_len)
#         t0 = idx % (self.T - self.seq_len)

#         # Slice contiguous window
#         states = torch.from_numpy(self.obs[ep, t0:t0+self.seq_len+1])  # (T+1, D)
#         actions = torch.from_numpy(self.act[ep, t0:t0+self.seq_len])   # (T, A)

#         if self.frames is not None:
#             frames = torch.from_numpy(self.frames[ep, t0:t0+self.seq_len+1])
#             frames = frames.permute(0, 3, 1, 2).float() / 255.0
#         else:
#             frames = torch.zeros(self.seq_len+1, 3, 1, 1)

#         return states, frames, actions



class PointMazeSequences(Dataset):
    """
    Sequential PointMaze dataset (non-overlapping subsequences).

    Returns contiguous windows of length `seq_len` transitions:
        states:  (seq_len+1, D_state)
        frames:  (seq_len+1, 3, H, W)
        actions: (seq_len,   D_action)
    """

    def __init__(
        self,
        npz_path: str,
        seq_len: int = 8,
        stride: int | None = None,
        normalize: bool = True,
        frame_size=(64, 64),
    ):
        super().__init__()
        data = np.load(npz_path)
        self.obs = data["obs"].astype(np.float32)      # (E, T+1, D)
        self.act = data["act"].astype(np.float32)      # (E, T, A)
        self.frames = data.get("frames", None)
        if self.frames is not None:
            self.frames = self.frames.astype(np.uint8)

        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len  # ← key line

        # resize frames if needed
        if frame_size and self.frames is not None:
            H, W = frame_size
            E, T1, _, _, C = self.frames.shape
            resized = np.empty((E, T1, H, W, C), dtype=np.uint8)
            for e in range(E):
                for t in range(T1):
                    resized[e, t] = cv2.resize(
                        self.frames[e, t], (W, H), interpolation=cv2.INTER_AREA
                    )
            self.frames = resized

        # optional normalization
        if normalize:
            obs_mean = self.obs.mean(axis=(0, 1))
            obs_std  = self.obs.std(axis=(0, 1)) + 1e-6
            act_mean = self.act.mean(axis=(0, 1))
            act_std  = self.act.std(axis=(0, 1)) + 1e-6
            self.obs = (self.obs - obs_mean) / obs_std
            self.act = (self.act - act_mean) / act_std

        self.E, self.T1, self.D = self.obs.shape
        self.T = self.T1 - 1

        # pre-compute valid (episode, start_index) pairs
        self.indices = []
        for e in range(self.E):
            for t0 in range(0, self.T - seq_len + 1, self.stride):
                self.indices.append((e, t0))

        print(
            f"[Dataset] {len(self.indices)} sequences "
            f"from {self.E} episodes, seq_len={self.seq_len}, stride={self.stride}"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        e, t0 = self.indices[idx]

        states  = torch.from_numpy(self.obs[e, t0:t0+self.seq_len+1])  # (T+1, D)
        actions = torch.from_numpy(self.act[e, t0:t0+self.seq_len])    # (T, A)

        if self.frames is not None:
            frames = torch.from_numpy(self.frames[e, t0:t0+self.seq_len+1])
            frames = frames.permute(0, 3, 1, 2).float() / 255.0
        else:
            frames = torch.zeros(self.seq_len+1, 3, 1, 1)

        return states, frames, actions




import torch
from torch.utils.data import Dataset
import numpy as np
import random
import cv2


class PointMazeVICReg(Dataset):
    """
    Dataset for VICReg pretraining on PointMaze frames.
    - Loads only 'frames' from the .npz file.
    - Flattens episodes into a list of images.
    - Applies patch masking augmentation on one branch.
    - Returns (original_image, masked_image).
    """

    def __init__(
        self,
        npz_path: str,
        frame_size=(128, 128),
        mask_ratio: float = 0.7,
        patch_size: int = 16,
        seed: int = 0,
    ):
        super().__init__()
        self.data = np.load(npz_path)
        assert "frames" in self.data, "The provided .npz must contain 'frames'."

        frames = self.data["frames"]  # shape: (E, T+1, H, W, 3)
        E, T1, H, W, C = frames.shape

        # Optional resize
        if frame_size is not None and (H, W) != frame_size:
            newH, newW = frame_size
            resized = np.empty((E, T1, newH, newW, C), dtype=np.uint8)
            for e in range(E):
                for t in range(T1):
                    resized[e, t] = cv2.resize(
                        frames[e, t], (newW, newH), interpolation=cv2.INTER_AREA
                    )
            frames = resized
            print(f"[VICRegDataset] Resized frames to {newH}×{newW}")

        # Flatten all frames: (E * (T+1), H, W, 3)
        self.frames = frames.reshape(-1, *frames.shape[2:])
        self.frames = torch.from_numpy(self.frames).permute(0, 3, 1, 2).float() / 255.0
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.rng = np.random.default_rng(seed)
        print(f"[VICRegDataset] Loaded {len(self.frames)} images.")

    # ----------------------------
    # Augmentation: patch masking
    # ----------------------------
    def _mask_patches(self, img: torch.Tensor) -> torch.Tensor:
        """Randomly zeroes out a fraction of patches."""
        _, H, W = img.shape
        ph, pw = self.patch_size, self.patch_size
        nH, nW = H // ph, W // pw

        # Create binary mask
        num_patches = nH * nW
        num_mask = int(num_patches * self.mask_ratio)
        mask_idx = self.rng.choice(num_patches, num_mask, replace=False)
        mask = torch.ones((nH, nW), dtype=torch.bool)
        mask.view(-1)[mask_idx] = False  # False = masked

        # Apply mask
        img_masked = img.clone()
        for i in range(nH):
            for j in range(nW):
                if not mask[i, j]:
                    img_masked[:, i * ph:(i + 1) * ph, j * pw:(j + 1) * pw] = 0.0

        return img_masked

    # ----------------------------
    # PyTorch dataset interface
    # ----------------------------
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx]
        img_aug = self._mask_patches(img)
        return img, img_aug
