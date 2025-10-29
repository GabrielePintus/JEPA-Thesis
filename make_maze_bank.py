# make_maze_bank.py
# Generate a bank of PointMaze layouts and save them to JSON.
# "maps" in JSON are full maps with borders, using the convention: 1 = wall, 0 = free.

from __future__ import annotations
import json
import argparse
import numpy as np
from collections import deque
from typing import List

def _is_connected(free_mask: np.ndarray) -> bool:
    """Check 4-neighborhood connectivity of 1's in free_mask (H×W)."""
    H, W = free_mask.shape
    coords = np.argwhere(free_mask == 1)
    if coords.size == 0:
        return False
    start = tuple(coords[0])
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and free_mask[nr, nc] == 1 and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(free_mask.sum())


def sample_interior_layout(
    rng: np.random.Generator,
    H: int = 4,
    W: int = 4,
    min_free_ratio: float = 0.50,
    max_free_ratio: float = 0.75,
    max_tries: int = 5000,
) -> np.ndarray:
    """Return an H×W interior mask with 1=free, 0=wall, meeting free ratio and connectivity."""
    total = H * W
    min_free = int(np.ceil(min_free_ratio * total))
    max_free = int(np.floor(max_free_ratio * total))
    if min_free > max_free:
        raise ValueError("Inconsistent free ratio bounds (min > max).")
    for _ in range(max_tries):
        k = int(rng.integers(min_free, max_free + 1))
        mask = np.zeros(total, dtype=np.int32)
        mask[:k] = 1
        rng.shuffle(mask)
        free = mask.reshape(H, W)
        if _is_connected(free):
            return free
    raise RuntimeError("Failed to sample connected interior layout within constraints.")


def interior_to_full_with_border(interior: np.ndarray) -> np.ndarray:
    """Convert H×W interior (1=free,0=wall) to (H+2)×(W+2) full map with border, using 1=wall,0=free."""
    H, W = interior.shape
    full = np.ones((H + 2, W + 2), dtype=int)
    full[1:-1, 1:-1] = 1 - interior  # invert: interior 1->free(0), 0->wall(1)
    return full


def main():
    ap = argparse.ArgumentParser("Generate a bank of unique PointMaze layouts to JSON.")
    ap.add_argument("--out", type=str, default="maze_bank.json", help="Output JSON path")
    ap.add_argument("--num-layouts", type=int, default=20, help="Number of unique layouts to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--H", type=int, default=4, help="Interior height")
    ap.add_argument("--W", type=int, default=4, help="Interior width")
    ap.add_argument("--min-free", type=float, default=0.50, help="Min free ratio (0..1)")
    ap.add_argument("--max-free", type=float, default=0.75, help="Max free ratio (0..1)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    maps: List[List[List[int]]] = []
    seen = set()

    for i in range(args.num_layouts):
        for attempt in range(10000):
            interior = sample_interior_layout(
                rng,
                H=args.H, W=args.W,
                min_free_ratio=args.min_free,
                max_free_ratio=args.max_free,
            )
            full = interior_to_full_with_border(interior)
            key = full.tobytes()
            if key not in seen:
                seen.add(key)
                maps.append(full.tolist())
                break
        else:
            raise RuntimeError(
                f"Could not find enough unique layouts after {len(maps)}. "
                f"Try lowering num-layouts or relaxing free ratio bounds."
            )

    payload = {
        "version": 1,
        "interior_size": [args.H, args.W],
        "convention": {"wall": 1, "free": 0},
        "min_free_ratio": args.min_free,
        "max_free_ratio": args.max_free,
        "seed": args.seed,
        "maps": maps,
    }

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(maps)} unique layouts -> {args.out}")


if __name__ == "__main__":
    main()
