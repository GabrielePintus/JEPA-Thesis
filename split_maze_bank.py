# split_maze_bank.py
# Split a maze bank JSON file into train/test subsets.
# Example:
#   python split_maze_bank.py --in banks/maze_bank.json \
#       --out-train banks/train_bank.json --out-test banks/test_bank.json \
#       --test-ratio 0.2 --seed 42

from __future__ import annotations
import json
import argparse
import numpy as np
from typing import List, Dict

def load_bank(path: str) -> Dict:
    """Load a maze bank JSON produced by make_maze_bank.py."""
    with open(path, "r") as f:
        data = json.load(f)
    if "maps" not in data:
        raise KeyError(f"{path} does not contain 'maps' key.")
    return data


def split_indices(n: int, test_ratio: float | None, test_count: int | None, seed: int = 0):
    """Return (train_ids, test_ids) based on ratio or count."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    if test_ratio is not None:
        n_test = int(round(n * test_ratio))
    elif test_count is not None:
        n_test = min(test_count, n)
    else:
        raise ValueError("Either --test-ratio or --test-count must be specified.")
    test_ids = idx[:n_test].tolist()
    train_ids = idx[n_test:].tolist()
    return train_ids, test_ids


def save_split(data: Dict, indices: List[int], out_path: str):
    """Save a subset of the maze bank with only selected maps."""
    subset = dict(data)  # shallow copy metadata
    subset["maps"] = [data["maps"][i] for i in indices]
    subset["num_layouts"] = len(subset["maps"])
    with open(out_path, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"Wrote {len(indices)} layouts -> {out_path}")


def main():
    ap = argparse.ArgumentParser("Split a maze bank JSON into train/test subsets.")
    ap.add_argument("--in", dest="in_path", type=str, required=True, help="Input maze bank JSON file")
    ap.add_argument("--out-train", type=str, required=True, help="Output train JSON path")
    ap.add_argument("--out-test", type=str, required=True, help="Output test JSON path")
    ap.add_argument("--test-ratio", type=float, default=None,
                    help="Fraction of layouts to reserve for test (e.g., 0.2)")
    ap.add_argument("--test-count", type=int, default=None,
                    help="Exact number of layouts to reserve for test (alternative to ratio)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = ap.parse_args()

    data = load_bank(args.in_path)
    n = len(data["maps"])
    print(f"Loaded {n} layouts from {args.in_path}")

    train_ids, test_ids = split_indices(n, args.test_ratio, args.test_count, seed=args.seed)
    save_split(data, train_ids, args.out_train)
    save_split(data, test_ids, args.out_test)

    print(f"Train/Test split: {len(train_ids)}/{len(test_ids)} "
          f"({len(test_ids)/n:.1%} test ratio)")


if __name__ == "__main__":
    main()
