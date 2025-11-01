# trajectory_generator.py
# Generate reward-free trajectories in PointMaze using Von Mises random-walk policy
# MPC-ready schema with explicit episode dimension:
#   obs:   (E, T+1, obs_dim)
#   act:   (E, T,  act_dim)
#   frames:(E, T+1, H, W, 3)  [optional]
# Supports loading a JSON bank of maze layouts (from make_maze_bank.py)
# Usage example:
#   python trajectory_generator.py --episodes 100 --T 100 \
#     --layouts-json banks/maze_bank.json --layout-sample random \
#     --store-images --compress-level 3

from __future__ import annotations
import argparse
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import os
import io
import zipfile

# ----------------------------
# Environment import
# ----------------------------
from envs.pointmaze import PointMazeEnv

os.environ["MUJOCO_GL"] = "egl"


# ----------------------------
# Compression helper
# ----------------------------

def save_npz_deflate(path: str, arrays: Dict[str, np.ndarray], level: int = 3) -> None:
    """Save dict of arrays as .npz using DEFLATE with adjustable compression level.
    level: 1 (fastest, larger) ... 9 (smallest, slower)."""
    with zipfile.ZipFile(
        path, mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=int(level)
    ) as zf:
        for name, arr in arrays.items():
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            zf.writestr(f"{name}.npy", buf.getvalue())


# ----------------------------
# Policies
# ----------------------------

@dataclass
class VonMisesPolicy:
    """Correlated random walk via Von Mises around previous direction (paper Sec. 4.1)."""
    kappa: float = 5.0
    step_max: float = 2.45
    rng: np.random.Generator = np.random.default_rng()

    def reset(self):
        self._theta = float(self.rng.uniform(-math.pi, math.pi))

    def sample(self) -> np.ndarray:
        self._theta = float(self.rng.vonmises(mu=self._theta, kappa=self.kappa))
        r = float(self.rng.uniform(0.0, self.step_max))
        return np.array([r * math.cos(self._theta), r * math.sin(self._theta)], dtype=np.float32)


@dataclass
class UniformDirectionPolicy:
    """Random baseline (Sec. 4.5): θ ~ Uniform(-π, π), r ~ Uniform(0, step_max)."""
    step_max: float = 2.45
    rng: np.random.Generator = np.random.default_rng()

    def reset(self): ...
    def sample(self) -> np.ndarray:
        theta = float(self.rng.uniform(-math.pi, math.pi))
        r = float(self.rng.uniform(0.0, self.step_max))
        return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float32)


# ----------------------------
# Collector
# ----------------------------

@dataclass
class CollectorConfig:
    T: int = 100
    episodes: int = 1000
    action_repeat: int = 1
    seed: Optional[int] = 0
    store_images: bool = False
    clip_to_action_space: bool = True
    action_scale: float = 1.0


def _clip_to_box(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, low), high)


def _scale_and_clip(action_vec: np.ndarray, env) -> np.ndarray:
    a = action_vec.astype(np.float32)
    low = np.array(env.action_space.low, dtype=np.float32)
    high = np.array(env.action_space.high, dtype=np.float32)
    if low.shape != a.shape:
        if low.shape[0] >= 2:
            out = np.zeros_like(low, dtype=np.float32)
            out[:2] = a
            a = out
        else:
            a = a[: low.shape[0]]
    return _clip_to_box(a, low, high)


# ----------------------------
# Maze bank loader
# ----------------------------

def load_maze_bank(path: str, restrict_ids: Optional[List[int]] = None) -> List[List[List[int]]]:
    """Load a JSON maze bank (as produced by make_maze_bank.py)."""
    with open(path, "r") as f:
        data = json.load(f)
    maps = data["maps"]
    if restrict_ids is not None:
        maps = [maps[i] for i in restrict_ids]
    print(f"Loaded {len(maps)} maze layouts from {path}"
          + (f" (restricted to IDs {restrict_ids})" if restrict_ids else ""))
    return maps


def make_env(maze_map=None, seed: Optional[int] = 0) -> PointMazeEnv:
    """Create a PointMazeEnv given an optional maze_map."""
    if maze_map is None:
        from gymnasium_robotics.envs.maze import maps
        maze_map = maps.U_MAZE
    side = max(len(maze_map), len(maze_map[0])) + 2
    env = PointMazeEnv(
        maze_map=maze_map,
        render_goal=False,
        camera_distance=side,
        continuing_task=True,
        reset_target=False,
        render_img_size=(64, 64),
        render_mode="rgb_array",
    )
    env.reset(seed=seed)
    return env


# ----------------------------
# Collector
# ----------------------------

def collect_trajectories(
    cfg: CollectorConfig,
    policy: VonMisesPolicy | UniformDirectionPolicy,
    maze_bank: Optional[List[List[List[int]]]] = None,
    layout_sample: str = "random",
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Collect reward-free trajectories with explicit episode axis.

    If maze_bank is provided, we rebuild the environment each episode
    with a layout sampled from the bank.
    """
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    rng = np.random.default_rng(cfg.seed)

    # Prepare a temporary env to probe shapes
    env = make_env(maze_bank[0] if maze_bank else None, seed=cfg.seed)
    obs_dict, _ = env.reset(seed=cfg.seed)
    obs_dim = int(obs_dict["observation"].size)
    act_dim = int(np.array(env.action_space.sample()).size)
    E, T = cfg.episodes, cfg.T

    obs = np.zeros((E, T + 1, obs_dim), dtype=np.float32)
    act = np.zeros((E, T, act_dim), dtype=np.float32)
    frames = None
    if cfg.store_images:
        img0 = env.render()
        if img0 is None:
            raise RuntimeError("render() returned None; ensure render_mode='rgb_array'.")
        H, W = img0.shape[:2]
        frames = np.zeros((E, T + 1, H, W, 3), dtype=np.uint8)

    ep_iter = range(E)
    if tqdm is not None:
        ep_iter = tqdm(ep_iter, desc="Collecting episodes", unit="ep")

    for ep in ep_iter:
        # Pick maze layout
        if maze_bank:
            if layout_sample == "random":
                idx = int(rng.integers(0, len(maze_bank)))
            elif layout_sample == "cycle":
                idx = ep % len(maze_bank)
            else:
                raise ValueError("--layout-sample must be 'random' or 'cycle'")
            env.close()
            env = make_env(maze_bank[idx], seed=(cfg.seed + ep if cfg.seed is not None else None))

        obs_dict, _ = env.reset(seed=(cfg.seed + ep if cfg.seed is not None else None))
        s = obs_dict["observation"].astype(np.float32)
        obs[ep, 0] = s
        policy.reset()

        if frames is not None:
            img = env.render()
            if img is not None:
                frames[ep, 0] = np.asarray(img, dtype=np.uint8)

        step_iter = range(T)
        if verbose and tqdm is not None:
            step_iter = tqdm(step_iter, leave=False, desc=f"Episode {ep+1}", unit="step")

        for t in step_iter:
            a_vec = policy.sample() * cfg.action_scale
            a = _scale_and_clip(a_vec, env) if cfg.clip_to_action_space else a_vec
            act[ep, t, :len(a)] = a
            # Repeat same action cfg.action_repeat times
            for _ in range(cfg.action_repeat):
                next_obs_dict, _, _, _, _ = env.step(a)

            # After all repeats, record the resulting observation/frame
            s_next = next_obs_dict["observation"].astype(np.float32)
            obs[ep, t + 1] = s_next

            if frames is not None:
                img = env.render()
                if img is not None:
                    frames[ep, t + 1] = np.asarray(img, dtype=np.uint8)

        if tqdm is not None and hasattr(ep_iter, "set_postfix"):
            ep_iter.set_postfix(T=T)

    env.close()
    out: Dict[str, np.ndarray] = {"obs": obs, "act": act}
    if frames is not None:
        out["frames"] = frames
    return out


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser("Collect reward-free trajectories in PointMaze (episode-major schema).")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--T", type=int, default=91)
    parser.add_argument("--kappa", type=float, default=5.0)
    parser.add_argument("--step-max", type=float, default=2.45)
    parser.add_argument("--action-repeat", type=int, default=1, help="Number of repeated environment steps per sampled action.")
    parser.add_argument("--policy", choices=["vonmises", "uniform"], default="vonmises")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="pointmaze_eps_major.npz")
    parser.add_argument("--store-images", action="store_true")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--compress-level", type=int, default=3)

    # NEW: Maze bank options
    parser.add_argument("--layouts-json", type=str, default=None,
                        help="Path to a maze bank JSON file (from make_maze_bank.py)")
    parser.add_argument("--layout-sample", choices=["random", "cycle"], default="random",
                        help="How to pick maze layouts from the bank")
    parser.add_argument("--layout-ids", type=str, default=None,
                        help="Comma-separated layout IDs to restrict to (e.g. '0,1,2')")

    args = parser.parse_args()

    # Prepare maze bank
    maze_bank = None
    if args.layouts_json:
        restrict = None
        if args.layout_ids:
            restrict = [int(x) for x in args.layout_ids.split(",") if x.strip() != ""]
        maze_bank = load_maze_bank(args.layouts_json, restrict_ids=restrict)

    policy = (VonMisesPolicy(kappa=args.kappa, step_max=args.step_max)
              if args.policy == "vonmises"
              else UniformDirectionPolicy(step_max=args.step_max))

    cfg = CollectorConfig(
        T=args.T,
        episodes=args.episodes,
        seed=args.seed,
        store_images=args.store_images,
        clip_to_action_space=not args.no_clip,
        action_scale=args.action_scale,
        action_repeat=args.action_repeat,
    )

    data = collect_trajectories(cfg, policy=policy, maze_bank=maze_bank,
                                layout_sample=args.layout_sample, verbose=args.verbose)

    print(f"Saving dataset to {args.out} ...")
    if args.no_compress:
        np.savez(args.out, **data)
    else:
        save_npz_deflate(args.out, data, level=args.compress_level)

    E, T = args.episodes, args.T
    print(f"Saved dataset with shapes: obs {data['obs'].shape}, act {data['act'].shape}"
          + (f", frames {data['frames'].shape}" if 'frames' in data else ""))


if __name__ == "__main__":
    main()
