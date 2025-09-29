#!/usr/bin/env python3
"""Dataset loader utilities for Ricci Flow experiments.

Provides a registry of dataset builders returning train/val/test splits
with float32 features and int64 labels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
from numpy.random import default_rng
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets

Array = np.ndarray
Splits = Dict[str, Array]

_DATA_ROOT = Path("data/raw")


def _ensure_data_root() -> Path:
    _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return _DATA_ROOT


def _split_dataset(
    X: Array,
    y: Array,
    *,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
    standardize: bool = True,
) -> Tuple[Splits, Dict[str, int], StandardScaler | None]:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        random_state=seed,
        stratify=y_trainval,
    )

    scaler: StandardScaler | None = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    splits: Splits = {
        "X_train": X_train.astype(np.float32),
        "y_train": y_train.astype(np.int64),
        "X_val": X_val.astype(np.float32),
        "y_val": y_val.astype(np.int64),
        "X_test": X_test.astype(np.float32),
        "y_test": y_test.astype(np.int64),
    }

    counts = {
        "train": int(len(y_train)),
        "val": int(len(y_val)),
        "test": int(len(y_test)),
    }
    return splits, counts, scaler


def _filter_classes(X: Array, y: Array, classes: Tuple[int, int]) -> Tuple[Array, Array]:
    mask = np.logical_or(y == classes[0], y == classes[1])
    X_f = X[mask]
    y_f = y[mask]
    y_bin = (y_f == classes[1]).astype(np.int64)
    return X_f, y_bin


def _load_mnist_zero_eight(seed: int) -> Tuple[Splits, Dict[str, object]]:
    root = _ensure_data_root()
    train = datasets.MNIST(root=str(root), train=True, download=True)
    test = datasets.MNIST(root=str(root), train=False, download=True)

    X = np.concatenate([train.data.numpy(), test.data.numpy()], axis=0).astype(np.float32)
    y = np.concatenate([train.targets.numpy(), test.targets.numpy()], axis=0).astype(np.int64)

    X, y = _filter_classes(X, y, (0, 8))
    X = (X / 255.0).reshape(len(X), -1)

    splits, counts, scaler = _split_dataset(X, y, seed=seed, standardize=True)
    meta = {
        "classes": {0: "digit-0", 1: "digit-8"},
        "input_dim": int(splits["X_train"].shape[1]),
        "counts": counts,
    }
    return splits, meta


def _load_fashion_top_vs_dress(seed: int) -> Tuple[Splits, Dict[str, object]]:
    root = _ensure_data_root()
    train = datasets.FashionMNIST(root=str(root), train=True, download=True)
    test = datasets.FashionMNIST(root=str(root), train=False, download=True)

    X = np.concatenate([train.data.numpy(), test.data.numpy()], axis=0).astype(np.float32)
    y = np.concatenate([train.targets.numpy(), test.targets.numpy()], axis=0).astype(np.int64)

    # 0: T-shirt/top, 3: Dress
    X, y = _filter_classes(X, y, (0, 3))
    X = (X / 255.0).reshape(len(X), -1)

    splits, counts, scaler = _split_dataset(X, y, seed=seed, standardize=True)
    meta = {
        "classes": {0: "tshirt_top", 1: "dress"},
        "input_dim": int(splits["X_train"].shape[1]),
        "counts": counts,
    }
    return splits, meta


def _load_cifar_cat_vs_dog(seed: int, *, per_class_limit: int | None = None) -> Tuple[Splits, Dict[str, object]]:
    root = _ensure_data_root()
    train = datasets.CIFAR10(root=str(root), train=True, download=True)
    test = datasets.CIFAR10(root=str(root), train=False, download=True)

    X = np.concatenate([train.data, test.data], axis=0).astype(np.float32)
    y = np.concatenate([train.targets, test.targets], axis=0).astype(np.int64)

    # 3: cat, 5: dog
    X, y = _filter_classes(X, y, (3, 5))

    if per_class_limit is not None:
        new_X = []
        new_y = []
        for cls in (0, 1):
            idx = np.where(y == cls)[0]
            rng = default_rng(seed)
            sel = rng.choice(idx, size=min(per_class_limit, len(idx)), replace=False)
            new_X.append(X[sel])
            new_y.append(np.full(len(sel), cls, dtype=np.int64))
        X = np.concatenate(new_X, axis=0)
        y = np.concatenate(new_y, axis=0)

    X = (X / 255.0).reshape(len(X), -1)

    splits, counts, scaler = _split_dataset(X, y, seed=seed, standardize=True)
    meta = {
        "classes": {0: "cat", 1: "dog"},
        "input_dim": int(splits["X_train"].shape[1]),
        "counts": counts,
    }
    return splits, meta


def _load_breast_cancer(seed: int) -> Tuple[Splits, Dict[str, object]]:
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    splits, counts, scaler = _split_dataset(X, y, seed=seed, standardize=True)
    meta = {
        "classes": {0: "malignant", 1: "benign"},
        "input_dim": int(splits["X_train"].shape[1]),
        "feature_names": list(data.feature_names),
        "counts": counts,
    }
    return splits, meta


def _sample_annulus_points(rng: np.random.Generator, n: int, r_inner: float, r_outer: float) -> Array:
    angles = rng.uniform(0.0, 2.0 * math.pi, size=n)
    radii = rng.uniform(r_inner, r_outer, size=n)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)


def _gen_annulus_vs_disk(seed: int, n_samples: int = 6000) -> Tuple[Splits, Dict[str, object]]:
    rng = default_rng(seed)
    n_per = n_samples // 2

    ring = _sample_annulus_points(rng, n_per, 0.7, 1.0)
    disk = _sample_annulus_points(rng, n_per, 0.0, 0.45)

    ring += rng.normal(0.0, 0.02, size=ring.shape)
    disk += rng.normal(0.0, 0.02, size=disk.shape)

    X = np.concatenate([ring, disk], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(n_per, dtype=np.int64), np.ones(n_per, dtype=np.int64)])

    splits, counts, scaler = _split_dataset(X, y, seed=seed, standardize=True)
    meta = {
        "classes": {0: "annulus", 1: "disk"},
        "input_dim": int(splits["X_train"].shape[1]),
        "counts": counts,
        "description": "2D annulus vs inner disk with additive noise",
    }
    return splits, meta


def _gen_torus_vs_sphere(seed: int, n_samples: int = 6000) -> Tuple[Splits, Dict[str, object]]:
    rng = default_rng(seed)
    n_per = n_samples // 2

    # Torus major/minor radii
    R = 1.0
    r = 0.35
    u = rng.uniform(0.0, 2.0 * math.pi, size=n_per)
    v = rng.uniform(0.0, 2.0 * math.pi, size=n_per)
    x_t = (R + r * np.cos(v)) * np.cos(u)
    y_t = (R + r * np.cos(v)) * np.sin(u)
    z_t = r * np.sin(v)
    torus = np.stack([x_t, y_t, z_t], axis=1)
    torus += rng.normal(0.0, 0.05, size=torus.shape)

    # Hollow sphere band with removed poles to introduce holes
    phi = np.arccos(rng.uniform(-0.6, 0.6, size=n_per))
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n_per)
    radius = rng.normal(0.9, 0.03, size=n_per)
    x_s = radius * np.sin(phi) * np.cos(theta)
    y_s = radius * np.sin(phi) * np.sin(theta)
    z_s = radius * np.cos(phi)
    sphere = np.stack([x_s, y_s, z_s], axis=1)
    sphere += rng.normal(0.0, 0.05, size=sphere.shape)

    X = np.concatenate([torus, sphere], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(n_per, dtype=np.int64), np.ones(n_per, dtype=np.int64)])

    splits, counts, scaler = _split_dataset(X, y, seed=seed, standardize=True)
    meta = {
        "classes": {0: "torus", 1: "punctured_sphere_band"},
        "input_dim": int(splits["X_train"].shape[1]),
        "counts": counts,
        "description": "3D torus vs hollow spherical band (both with holes)",
    }
    return splits, meta


@dataclass
class DatasetSpec:
    name: str
    loader: Callable[[int], Tuple[Splits, Dict[str, object]]]


_DATASETS: Dict[str, DatasetSpec] = {
    "mnist_0_vs_8": DatasetSpec("mnist_0_vs_8", _load_mnist_zero_eight),
    "fashion_top_vs_dress": DatasetSpec("fashion_top_vs_dress", _load_fashion_top_vs_dress),
    "cifar_cat_vs_dog": DatasetSpec("cifar_cat_vs_dog", _load_cifar_cat_vs_dog),
    "breast_cancer": DatasetSpec("breast_cancer", _load_breast_cancer),
    "annulus_vs_disk": DatasetSpec("annulus_vs_disk", _gen_annulus_vs_disk),
    "torus_vs_sphere": DatasetSpec("torus_vs_sphere", _gen_torus_vs_sphere),
}


def available_datasets() -> Iterable[str]:
    return sorted(_DATASETS.keys())


def load_dataset(name: str, *, seed: int = 42) -> Tuple[Splits, Dict[str, object]]:
    key = name.lower()
    if key not in _DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {', '.join(available_datasets())}")
    splits, meta = _DATASETS[key].loader(seed)
    meta = dict(meta)
    meta.setdefault("name", key)
    meta["seed"] = seed
    return splits, meta
