

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic binary datasets -> split -> save as .npz for main.py

Creates train/val/test splits with the exact keys expected by main.py:
  X_train, y_train, X_val, y_val, X_test, y_test

Examples:
  # 1) Moons (classic non-linear separation)
  python data_npz.py --dataset moons --n-samples 5000 --noise 0.25 \
      --standardize --out data/moons_5k.npz

  # 2) Concentric circles (harder margin)
  python data_npz.py --dataset circles --n-samples 6000 --noise 0.15 \
      --standardize --out data/circles_6k.npz

  # 3) Two intertwined spirals (strongly entangled)
  python data_npz.py --dataset spirals --n-samples 4000 --noise 0.4 \
      --standardize --out data/spirals_4k.npz

  # 4) Gaussian blobs (linearly separable baseline)
  python data_npz.py --dataset blobs --n-samples 4000 --std 1.2 \
      --standardize --out data/blobs_4k.npz

Notes
- Labels are {0,1}
- X is float32, y is int64
- Stratified splits; random_state/seed is reproducible
- Standardization is optional (recommended for MLPs)
"""

import argparse
import os
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_blobs

# ------------------------
# Utils
# ------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def save_npz(path: str, X_train, y_train, X_val, y_val, X_test, y_test):
    np.savez_compressed(
        path,
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.int64),
        X_val=X_val.astype(np.float32),
        y_val=y_val.astype(np.int64),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.int64),
    )
    print(f"Saved dataset to: {path}")


# ------------------------
# Generators
# ------------------------

def gen_moons(n_samples: int, noise: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return X.astype(np.float32), y.astype(np.int64)


def gen_circles(n_samples: int, noise: float, factor: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=seed)
    return X.astype(np.float32), y.astype(np.int64)


def gen_blobs(n_samples: int, std: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    # Two Gaussian centers; easy baseline
    centers = np.array([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=std, random_state=seed)
    return X.astype(np.float32), y.astype(np.int64)


def gen_spirals(n_samples: int, noise: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Two intertwined spirals in 2D.
    n_samples is total; split equally into class 0 and 1.
    noise ~ std of gaussian jitter added to (x,y).
    """
    rng = np.random.default_rng(seed)
    n_per = n_samples // 2
    t = np.linspace(0.0, 4*np.pi, n_per, endpoint=False)
    r = np.linspace(0.2, 1.0, n_per)

    x1 = r * np.cos(t)
    y1 = r * np.sin(t)

    x2 = r * np.cos(t + np.pi)
    y2 = r * np.sin(t + np.pi)

    X0 = np.stack([x1, y1], axis=1)
    X1 = np.stack([x2, y2], axis=1)

    if noise > 0:
        X0 += rng.normal(0.0, noise, size=X0.shape)
        X1 += rng.normal(0.0, noise, size=X1.shape)

    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(np.int64)

    # If total n_samples is odd, drop last to keep labels balanced
    if X.shape[0] != n_samples:
        X = X[:n_samples]
        y = y[:n_samples]
    return X, y


# ------------------------
# Main builder
# ------------------------

def build_and_save(dataset: str,
                   n_samples: int,
                   noise: float,
                   factor: float,
                   std: float,
                   test_size: float,
                   val_size: float,
                   standardize: bool,
                   seed: int,
                   out_path: str):
    set_seed(seed)

    if dataset == "moons":
        X, y = gen_moons(n_samples, noise, seed)
    elif dataset == "circles":
        X, y = gen_circles(n_samples, noise, factor, seed)
    elif dataset == "spirals":
        X, y = gen_spirals(n_samples, noise, seed)
    elif dataset == "blobs":
        X, y = gen_blobs(n_samples, std, seed)
    else:
        raise SystemExit(f"Unknown dataset: {dataset}")

    # train/test split (stratified)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # train/val split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=seed, stratify=y_trainval
    )

    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    ensure_dir(out_path)
    save_npz(out_path, X_train, y_train, X_val, y_val, X_test, y_test)


# ------------------------
# CLI
# ------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic binary dataset and save as .npz")
    ap.add_argument("--dataset", choices=["moons", "circles", "spirals", "blobs"], required=True)
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--noise", type=float, default=0.25, help="noise/jitter for moons, circles, spirals")
    ap.add_argument("--factor", type=float, default=0.5, help="inner/outer radius ratio for circles")
    ap.add_argument("--std", type=float, default=1.0, help="cluster std for blobs")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/synth.npz")
    args = ap.parse_args()

    build_and_save(
        dataset=args.dataset,
        n_samples=args.n_samples,
        noise=args.noise,
        factor=args.factor,
        std=args.std,
        test_size=args.test_size,
        val_size=args.val_size,
        standardize=args.standardize,
        seed=args.seed,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
