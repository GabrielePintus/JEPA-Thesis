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
        frame_size: Optional[Tuple[int, int]] = (96, 96),  # e.g., (64, 64)
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
